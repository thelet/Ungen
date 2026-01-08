import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from Core.Interfaces import IOptimizer, IFeatureExtractor
import math
from Core.Utilities import ImageUtils
from Core.filters.filters import CoolFilter, WarmFilter, GrainyFilter
from Core.filters.mock_filter import MockFilter
import lpips

# Use the import that matches your file structure
try:
    from Core.DiffJPEG.DiffJPEG import DiffJPEG
except ImportError:
    print("DiffJPEG not found, disabling compression simulation.")
    DiffJPEG = None

# ==========================================
# 1. SSIM Implementation (Unchanged - Kept for fallback)
# ==========================================
class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = torch.Tensor(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        if img1.device != self.window.device:
            self.window = self.window.to(img1.device)
        return 1 - self._ssim(img1, img2, self.window, self.window_size, self.channel, self.size_average)


# ==========================================
# 3. Robust Optimizer (Full-Res + Style Guided + LPIPS)
# ==========================================
class RobustPGDOptimizer(IOptimizer):
    def __init__(self, steps=150, epsilon=32/255.0, alpha=2/255.0, device='cuda', debug_output_path=None):
        # NOTE: Epsilon is higher (32/255) to allow visible filter-like color shifts
        self.device = device
        self.steps = steps
        self.epsilon = epsilon
        self.alpha = alpha
        self.debug_output_path = debug_output_path
        
        # 1. Initialize LPIPS (VGG mode is best for quality)
        print("[PGD] Loading LPIPS VGG model...")
        self.lpips_loss = lpips.LPIPS(net='vgg').to(device)

        
        self.filter = WarmFilter(strength=0.4, contrast=1.7, saturation=1.5).to(device)
        self.ssim_loss = SSIMLoss().to(device)

        # Robustness (Augmentation)
        self.aug = nn.Sequential(
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        ).to(device)
        
        if DiffJPEG:
            self.jpeg = DiffJPEG(height=224, width=224, quality=80, differentiable=True).to(device)
            self.has_jpeg = True
        else:
            self.has_jpeg = False

        print(f"[PGD] Initialized (Full-Res + Style Guided Mode). Steps: {steps}, Eps: {epsilon:.3f}")

    def optimize(self, original_img: torch.Tensor, target_features: torch.Tensor, feature_extractor: IFeatureExtractor, landmarks=None) -> torch.Tensor:
        original_img = original_img.to(self.device)
        target_features = target_features.detach().to(self.device)

        # 1. Create the Visual Target (The Filtered Look)
        # We want the result to look like this "Filtered" version, not the original.
        with torch.no_grad():
            visual_ref = self.filter(original_img)
            # Optional: Save reference to check what we are aiming for
           
            debug_dir = str(self.debug_output_path).rsplit('.', 1)[0] + "_ref.png"
            save_image(visual_ref, debug_dir)
            print(f"[PGD] Saved visual reference filter to: {debug_dir}")

        # 2. Init Delta at FULL Resolution
        delta = torch.zeros_like(original_img).uniform_(-0.01, 0.01).to(self.device)
        delta.requires_grad = True
        
        best_sim_loss = float('inf')
        best_delta = delta.detach().clone()

        print(f"[PGD] Starting Optimization (Full Res)...")

        for i in range(self.steps):
            # A. Apply Noise (No Interpolation needed now)
            adv_image = torch.clamp(original_img + delta, 0, 1)

            # B. Robustness (Simulate Upload)
            robust_view = self.aug(adv_image)
            if self.has_jpeg:
                robust_view = self.jpeg(robust_view)

            # C. Extract Features (Attack)
            current_features = feature_extractor.get_features_with_grad(robust_view)

            # --- LOSS FUNCTION ---
            
            # 1. Attack Loss (Similarity to Target Identity)
            sim_loss = 1 - F.cosine_similarity(current_features, target_features).mean()
            
            # 2. Style Limit (MSE to Filtered Target)
            style_loss = F.mse_loss(adv_image, visual_ref)
            
            # 3. Structure Limit (LPIPS to Original)
            adv_scaled = (adv_image * 2) - 1
            orig_scaled = (original_img * 2) - 1
            
            structure_loss = self.ssim_loss(adv_image, original_img)

            # Total Loss
            total_loss = sim_loss + (60.0 * style_loss) + (60.0 * structure_loss)

            # Save best result based on attack success
            if sim_loss.item() < best_sim_loss:
                best_sim_loss = sim_loss.item()
                best_delta = delta.detach().clone()

            if i % 10 == 0:
                print(f"Step {i:<4} | Total: {total_loss.item():.4f} | Sim: {sim_loss.item():.4f} | Style: {style_loss.item():.4f} | SSIM: {structure_loss.item():.4f}")
               

            # Update Gradients
            total_loss.backward()
            grad = delta.grad.detach()
            
            # PGD Update
            delta.data = delta.data - self.alpha * torch.sign(grad)
            
            # Clamp delta to epsilon
            delta.data = torch.clamp(delta.data, -self.epsilon, self.epsilon)
            
            delta.grad.zero_()

        print(f"[PGD] Finished. Restoring best delta.")
        
        # Restore best result
        final_image = torch.clamp(original_img + best_delta, 0, 1)
        print(f"[PGD] Best similarity loss: {best_sim_loss:.4f}")
        
        return final_image