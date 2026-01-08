import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class ITargetGenerator(ABC):
    @abstractmethod
    def generate_target(self, user_img_pil, stranger_img_path: str, blend_ratio: float) -> torch.Tensor:
        """Generates the target image tensor."""
        pass

class IFeatureExtractor(ABC):
    @abstractmethod
    def get_features(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Extracts features (latents) from an image tensor."""
        pass

    @abstractmethod
    def get_features_with_grad(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Extracts features while preserving gradients for optimization."""
        pass

class IOptimizer(ABC):
    @abstractmethod
    def optimize(self, original_img: torch.Tensor, target_features: torch.Tensor, feature_extractor: IFeatureExtractor) -> torch.Tensor:
        """Performs the adversarial attack to protect the image."""
        pass


class ISimilarityMetric(nn.Module, ABC):
    """
    Interface for image similarity metrics.
    Must return a scalar tensor acting as a LOSS (lower = better/more similar).
    """
    @abstractmethod
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Calculates the distance/loss between two images.
        """
        pass