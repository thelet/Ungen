import torch
import torch.nn as nn

# ==========================================
# BASE INTERFACE
# ==========================================
class BaseFilter(nn.Module):
    """
    Parent class for all filters. Handles strength, contrast, and saturation.
    """
    def __init__(self, strength=0.5, contrast=1.2, saturation=1.2):
        """
        strength: 0.0 to 1.0 (Blend factor)
        contrast: 1.0 = Original, 1.2 = +20% Contrast
        saturation: 1.0 = Original, 1.2 = +20% Color Boost, 0.0 = Grayscale
        """
        super(BaseFilter, self).__init__()
        self.strength = strength
        self.contrast = contrast
        self.saturation = saturation

    def apply_contrast(self, x):
        """Helper to apply contrast enhancement"""
        # Formula: (Pixel - Mean) * Factor + Mean
        # We assume mean is 0.5 for normalized images
        return (x - 0.5) * self.contrast + 0.5

    def apply_saturation(self, x):
        """Helper to apply saturation (color enhancement)"""
        if self.saturation == 1.0:
            return x
        
        # 1. Calculate Grayscale (Luminance)
        # Rec. 601 weights: 0.299 R + 0.587 G + 0.114 B
        grayscale = x[:, 0] * 0.299 + x[:, 1] * 0.587 + x[:, 2] * 0.114
        grayscale = grayscale.unsqueeze(1).repeat(1, 3, 1, 1) # Expand back to 3 channels

        # 2. Interpolate between Grayscale and Color
        # Saturation > 1 moves AWAY from grayscale (more color)
        # Saturation < 1 moves TOWARDS grayscale (less color)
        return torch.lerp(grayscale, x, self.saturation)

    def blend(self, original, filtered):
        """Helper to blend result based on strength"""
        return torch.lerp(original, filtered, self.strength)

    def forward(self, x):
        raise NotImplementedError("Each filter must implement its own forward method.")

# ==========================================
# CONCRETE FILTERS
# ==========================================

class WarmFilter(BaseFilter):
    """
    Applies a 'Golden Hour' / Sepia look with extra vibrance.
    """
    def forward(self, x):
        r, g, b = x[:, 0], x[:, 1], x[:, 2]
        
        # Warm coefficients
        r_new = r * 1.15 + 0.02
        g_new = g * 1.05 + 0.01
        b_new = b * 0.90
        
        x_colored = torch.stack([r_new, g_new, b_new], dim=1)
        
        # Apply Saturation -> Contrast -> Clamp
        x_sat = self.apply_saturation(x_colored)
        x_con = self.apply_contrast(x_sat)
        x_final = torch.clamp(x_con, 0, 1)

        return self.blend(x, x_final)

class CoolFilter(BaseFilter):
    """
    Applies a 'Cyberpunk' / Winter look with extra vibrance.
    """
    def forward(self, x):
        r, g, b = x[:, 0], x[:, 1], x[:, 2]
        
        # Cool coefficients
        r_new = r * 0.90
        g_new = g * 1.05 + 0.02
        b_new = b * 1.15 + 0.03
        
        x_colored = torch.stack([r_new, g_new, b_new], dim=1)
        
        # Apply Saturation -> Contrast -> Clamp
        x_sat = self.apply_saturation(x_colored)
        x_con = self.apply_contrast(x_sat)
        x_final = torch.clamp(x_con, 0, 1)

        return self.blend(x, x_final)

class GrainyFilter(BaseFilter):
    """
    Applies 'Film Grain' texture. Saturation usually helps film looks.
    """
    def forward(self, x):
        # Generate grain
        noise = torch.randn_like(x[:, 0:1, :, :]) * 0.05
        grain = noise.repeat(1, 3, 1, 1)

        x_grainy = x + grain
        
        # Apply Saturation -> Contrast -> Clamp
        x_sat = self.apply_saturation(x_grainy)
        x_con = self.apply_contrast(x_sat)
        x_final = torch.clamp(x_con, 0, 1)

        return self.blend(x, x_final)