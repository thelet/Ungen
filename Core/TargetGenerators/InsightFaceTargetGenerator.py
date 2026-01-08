import os
import gc
import cv2
import numpy as np
import torch
import insightface
from insightface.app import FaceAnalysis
from Core.Interfaces import ITargetGenerator
from Core.Utilities import ImageUtils

class InsightFaceTargetGenerator(ITargetGenerator):
    def __init__(self, device='cuda'):
        print(f"[TargetGen] Loading InsightFace models on {device}...")
        self.providers = ['CUDAExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        self.device = device
        
        self.app = FaceAnalysis(name='buffalo_l', providers=self.providers)
        self.app.prepare(ctx_id=0 if device == 'cuda' else -1, det_size=(640, 640))
        
        # Load swapper model
        current_dir = os.path.dirname(os.path.abspath(__file__))
        local_model_path = os.path.join(current_dir, "inswapper_128.onnx")

        if os.path.exists(local_model_path):
            self.swapper = insightface.model_zoo.get_model(local_model_path, providers=self.providers)
        else:
            print("[TargetGen] Local model not found, attempting auto-load...")
            self.swapper = insightface.model_zoo.get_model('inswapper_128.onnx', providers=self.providers)

    def generate_target(self, user_img_pil, stranger_img_path, blend_ratio=1.0) -> torch.Tensor:
        print(f"[TargetGen] Processing target (Blend Ratio: {blend_ratio})...")
        user_cv2 = cv2.cvtColor(np.array(user_img_pil), cv2.COLOR_RGB2BGR)
        stranger_cv2 = cv2.imread(stranger_img_path)
        
        if stranger_cv2 is None: raise ValueError(f"Target not found: {stranger_img_path}")

        user_faces = self.app.get(user_cv2)
        stranger_faces = self.app.get(stranger_cv2)

        if len(user_faces) == 0: raise ValueError("No face detected in User image.")
        if len(stranger_faces) == 0: raise ValueError("No face detected in Stranger image.")

        target_face = stranger_faces[0]
        source_face = user_faces[0]
        
        swapped_cv2 = self.swapper.get(user_cv2, source_face, target_face, paste_back=True)
        blended_cv2 = cv2.addWeighted(user_cv2, 1.0 - blend_ratio, swapped_cv2, blend_ratio, 0)

        res_rgb = cv2.cvtColor(blended_cv2, cv2.COLOR_BGR2RGB)
        res_tensor = torch.from_numpy(res_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        return res_tensor

    
    def unload_models(self):
        """Explicitly release models from GPU memory."""
        print("[TargetGen] Cleaning up InsightFace models to free VRAM...")
        if hasattr(self, 'app'):
            del self.app
        if hasattr(self, 'swapper'):
            del self.swapper
        
        self.app = None
        self.swapper = None
        
        # Force Python garbage collection
        gc.collect()
        # Force PyTorch CUDA cache clearing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()





