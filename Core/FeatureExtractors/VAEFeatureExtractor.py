import torch
from diffusers import AutoencoderKL
from Core.Interfaces import IFeatureExtractor

class VAEFeatureExtractor(IFeatureExtractor):
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", device='cuda'):
        self.device = device
        print(f"[UnGen] Loading VAE (FP32): {model_id}...")
        self.model = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)

    def get_features(self, image_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self._extract(image_tensor)

    def get_features_with_grad(self, image_tensor: torch.Tensor) -> torch.Tensor:
        return self._extract(image_tensor)

    def _extract(self, img_tensor: torch.Tensor) -> torch.Tensor:
        img_tensor = img_tensor.to(self.device)
        if img_tensor.shape[-1] != 512:
            img_input = torch.nn.functional.interpolate(img_tensor, size=(512, 512), mode='bicubic', align_corners=False)
        else:
            img_input = img_tensor
        img_normalized = (img_input * 2.0) - 1.0
        posterior = self.model.encode(img_normalized).latent_dist
        return posterior.mean * 0.18215

    # --- NEW: DECODE METHOD ---
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Converts latents back to a viewable image tensor [0, 1]"""
        with torch.no_grad():
            # 1. Unscale the latents (Inverse of 0.18215)
            latents = (1 / 0.18215) * latents
            
            # 2. Decode using VAE Decoder
            image = self.model.decode(latents).sample
            
            # 3. Rescale from [-1, 1] to [0, 1] for saving
            image = (image / 2 + 0.5).clamp(0, 1)
            return image