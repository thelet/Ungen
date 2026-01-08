import os
from Core.Interfaces import ITargetGenerator
from Core.Utilities import ImageUtils


class SimpleTargetLoader(ITargetGenerator):
    def __init__(self, device='cuda'):
        self.device = device
    
    def generate_target(self, user_img_pil, stranger_img_path: str, blend_ratio: float = 1.0):
        print(f"[TargetLoader] Loading pre-generated target from: {stranger_img_path}")
        
        # We ignore 'user_img_pil' because the swap is already done.
        # We just load the file at 'stranger_img_path'.
        _, target_tensor = ImageUtils.load_image(stranger_img_path)
        
        return target_tensor.to(self.device)
        
    def unload_models(self):
        # Nothing to unload
        pass