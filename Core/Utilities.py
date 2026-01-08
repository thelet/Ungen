import torch
import numpy as np
from PIL import Image, ImageOps

class ImageUtils:
    @staticmethod
    def load_image(path, target_size=(512, 512)):
        img = Image.open(path).convert("RGB")
        
        # --- NEW: Letterbox Resize (Pad instead of Squash) ---
        # This keeps the aspect ratio correct so you don't look "weird"
        img.thumbnail(target_size, Image.Resampling.LANCZOS)
        
        # Create a black square canvas
        new_img = Image.new("RGB", target_size, (0, 0, 0))
        
        # Paste your image in the center
        paste_x = (target_size[0] - img.width) // 2
        paste_y = (target_size[1] - img.height) // 2
        new_img.paste(img, (paste_x, paste_y))
        # -----------------------------------------------------

        # Convert to Tensor (0-1 range)
        img_np = np.array(new_img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
        
        return new_img, img_tensor

    @staticmethod
    def tensor_to_pil(tensor):
        img_np = tensor.squeeze().detach().cpu().permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1) * 255
        return Image.fromarray(img_np.astype(np.uint8))

    @staticmethod
    def normalize_sd(tensor):
        # Maps [0, 1] to [-1, 1] for the VAE
        return (tensor * 2.0) - 1.0