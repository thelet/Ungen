import torch
import os
from Core.Interfaces import ITargetGenerator, IFeatureExtractor, IOptimizer
from Core.Utilities import ImageUtils

class ProtectionPipeline:
    def __init__(
        self, 
        target_generator: ITargetGenerator,
        feature_extractor: IFeatureExtractor,
        optimizer: IOptimizer
    ):
        self.target_gen = target_generator
        self.extractor = feature_extractor
        self.optimizer = optimizer

    def run(
        self, 
        user_img_path: str, 
        stranger_img_path: str, 
        output_path: str,
        debug_target_path: str = None
    ):
        print("--- Starting Protection Pipeline (Robust Global Mode) ---")

        # 1. Load Images
        # Loads image as (Batch, Channel, Height, Width) tensor, normalized 0-1
        user_pil, user_tensor = ImageUtils.load_image(user_img_path)
        
        # 2. Generate Target (The "Stranger" Look)
        print("[Pipeline] Generating target reference...")
        target_tensor = self.target_gen.generate_target(user_pil, stranger_img_path)
        
        if debug_target_path:
            ImageUtils.tensor_to_pil(target_tensor).save(debug_target_path)
            print(f"[Pipeline] Debug target saved to {debug_target_path}")

        # FREE MEMORY: Unload InsightFace if supported to save VRAM for the attack
        if hasattr(self.target_gen, 'unload_models'):
            self.target_gen.unload_models()

        # 3. Extract Target Features
        print("[Pipeline] Extracting target features...")
        target_latents = self.extractor.get_features(target_tensor)

        # 4. Run Optimization
        # STRICT INTERFACE MATCH: We do NOT pass landmarks here.
        print("[Pipeline] Running Robust PGD Optimization...")
        
        protected_tensor = self.optimizer.optimize(
            original_img=user_tensor,
            target_features=target_latents,
            feature_extractor=self.extractor
        )

        # 5. Save Result
        ImageUtils.tensor_to_pil(protected_tensor).save(output_path)
        print(f"--- Pipeline Finished. Saved to {output_path} ---")