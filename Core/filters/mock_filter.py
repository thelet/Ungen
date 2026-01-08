import torch
import torch.nn as nn
import cv2
import numpy as np

class MockFilter(nn.Module):
    """
    Instead of calculating a filter, this simply returns a pre-loaded target image.
    Useful if you edited the photo in FaceApp/Photoshop (e.g. 'stranger.jpg') 
    and want to force the optimizer to match that exact look.
    """
    def __init__(self, target_image_path, device='cuda'):
        super(MockFilter, self).__init__()
        self.device = device
        self.target_tensor = self._load_image(target_image_path)
        print(f"[MockFilter] Loaded target from: {target_image_path}")

    def _load_image(self, path):
        # Load image, convert BGR -> RGB
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"MockFilter could not find image at {path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize 0-1 and convert to float32
        img = img.astype(np.float32) / 255.0
        
        # Convert to Tensor: (H, W, C) -> (C, H, W) -> (1, C, H, W)
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def forward(self, x):
        """
        Returns the pre-loaded target image, resized to match the input 'x'.
        """
        # 1. Resize to match the input dimensions (H, W) if they differ
        if self.target_tensor.shape[-2:] != x.shape[-2:]:
            out = torch.nn.functional.interpolate(
                self.target_tensor, 
                size=x.shape[-2:], 
                mode='bilinear', 
                align_corners=False # Important to prevent alignment shift
            )
        else:
            out = self.target_tensor

        # 2. Expand batch size if necessary
        # If input 'x' has batch size N, we repeat our target N times
        if x.shape[0] != out.shape[0]:
            out = out.expand(x.shape[0], -1, -1, -1)
            
        return out