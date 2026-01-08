import torch
import cv2
import numpy as np
from Core.Interfaces import ITargetGenerator, IFeatureExtractor, IOptimizer
from Core.Utilities import ImageUtils
import os

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
        print("--- Starting Protection Pipeline ---")

        # 1. Load Images (Tensor for Optimization)
        user_pil, user_tensor = ImageUtils.load_image(user_img_path)
        
        # Save resized base for debugging/diff testing
        base_dir = os.path.dirname(output_path)
        resized_clean_path = os.path.join(base_dir, "resized_clean_base.png")
        ImageUtils.tensor_to_pil(user_tensor).save(resized_clean_path)
        print(f"[Pipeline] Saved resized base image to: {resized_clean_path}")

        # --- NEW: Landmark Detection (Requires OpenCV) ---
        print(f"[Pipeline] Detecting landmarks for mask generation...")
        img_cv2 = cv2.imread(user_img_path)
        # Resize cv2 image to match the tensor size if needed, 
        # but InsightFace usually handles original resolution fine.
        # Ideally, we pass the original cv2 image.
        
        landmarks = None
        # We access the InsightFace 'app' from the target generator
        if hasattr(self.target_gen, 'app'):
            faces = self.target_gen.app.get(img_cv2)
            if len(faces) > 0:
                # Pick largest face
                face = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]))[-1]
                
                if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
                    landmarks = face.landmark_2d_106
                    print(f"[Pipeline] Found 106-point landmarks.")
                else:
                    landmarks = face.kps
                    print(f"[Pipeline] Found 5-point landmarks (Fallback).")
            else:
                print("⚠️ WARNING: No face detected! Optimization masks will be empty.")
        else:
             print("⚠️ WARNING: TargetGenerator has no 'app'. Cannot detect landmarks.")
        # -------------------------------------------------
        
        # 2. Generate Target (The "Stranger" Look)
        target_tensor = self.target_gen.generate_target(user_pil, stranger_img_path)
        
        if debug_target_path:
            ImageUtils.tensor_to_pil(target_tensor).save(debug_target_path)
            print(f"Debug target saved to {debug_target_path}")

        # FREE MEMORY: Unload InsightFace before loading VAE/PGD
        # (We do this AFTER detecting landmarks)
        if hasattr(self.target_gen, 'unload_models'):
            self.target_gen.unload_models()

        # 3. Extract Target Features
        target_latents = self.extractor.get_features(target_tensor)

        # 4. Run Optimization (Attack)
        # PASS LANDMARKS HERE
        protected_tensor = self.optimizer.optimize(
            original_img=user_tensor,
            target_features=target_latents,
            feature_extractor=self.extractor,
            landmarks=landmarks 
        )

        # 5. Save Result
        ImageUtils.tensor_to_pil(protected_tensor).save(output_path)
        print(f"--- Pipeline Finished. Saved to {output_path} ---")