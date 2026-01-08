import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from torchvision import transforms
from Core.Interfaces import IFeatureExtractor

class CLIPFeatureExtractor(IFeatureExtractor):
    def __init__(self, model_id="openai/clip-vit-large-patch14", device='cuda'):
        self.device = device
        print(f"[FeatureExtractor] Loading CLIP: {model_id}...")
        
        # Load the full CLIP model (we will use the vision tower)
        self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        self.model.eval()
        
        # Freeze the model (we only need gradients for the image, not the model weights)
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Standard CLIP Normalization constants
        self.normalize = transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )

    def get_features(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Standard extraction for the Target Image"""
        return self._extract(image_tensor)

    def get_features_with_grad(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Extraction enabling gradient flow for the Attack"""
        return self._extract(image_tensor)

    def _extract(self, img_tensor: torch.Tensor) -> torch.Tensor:
        # --- FIX: Force input to the correct device (GPU) ---
        img_tensor = img_tensor.to(self.device) 
        # ----------------------------------------------------

        # 1. Resize to CLIP's native resolution (224x224) if needed
        if img_tensor.shape[-1] != 224 or img_tensor.shape[-2] != 224:
            img_resized = torch.nn.functional.interpolate(
                img_tensor, 
                size=(224, 224), 
                mode='bicubic', 
                align_corners=False
            )
        else:
            img_resized = img_tensor

        # 2. Normalize (Shift from [0,1] to CLIP distribution)
        img_norm = self.normalize(img_resized)
        
        # 3. Get Image Embeddings
        output = self.model.get_image_features(pixel_values=img_norm)
        
        # 4. Normalize the feature vector itself
        return output / output.norm(dim=-1, keepdim=True)