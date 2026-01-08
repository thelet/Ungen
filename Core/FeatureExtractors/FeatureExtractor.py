import torch
from diffusers import AutoencoderKL
from Core.Interfaces import IFeatureExtractor
from Core.Utilities import ImageUtils

class SDVAEFeatureExtractor(IFeatureExtractor):
    def __init__(self, model_id="stabilityai/sd-vae-ft-mse", device='cuda'):
        self.device = device
        print(f"[FeatureExtractor] Loading VAE: {model_id}...")
        self.vae = AutoencoderKL.from_pretrained(model_id).to(self.device)
        self.vae.eval()
        self.vae.requires_grad_(False)

    def get_features(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Standard extraction (no grad)"""
        img_norm = ImageUtils.normalize_sd(image_tensor).to(self.device)
        with torch.no_grad():
            latents = self.vae.encode(img_norm).latent_dist.mode()
        return latents * 0.18215

    def get_features_with_grad(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Extraction enabling gradient flow for PGD"""
        img_norm = ImageUtils.normalize_sd(image_tensor).to(self.device)
        latents = self.vae.encode(img_norm).latent_dist.mode()
        return latents * 0.18215