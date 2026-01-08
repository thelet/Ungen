import torch
import os
import sys
import shutil

# Import concrete implementations
from Core.TargetGenerators.TargetGeneratorWithColors import InsightFaceTargetGenerator
from Core.FeatureExtractors.CLIPFeatureExtractor import CLIPFeatureExtractor
from Core.Pipelines.simple_pipeline import ProtectionPipeline
from Core.SimilarityMetrics.SSIMMetric import SSIMMetric
from Core.Optimizers.filter_v3_optimizer import RobustPGDOptimizer


# ==========================================
# CONFIGURATION
# ==========================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# The Base Directory where numbered folders will be created
BASE_ROOT_DIR = r""

# PGD Attack Parameters
EPSILON = 255 / 255.0   # Max perturbation budget
ALPHA = 0.5 / 255.0     # Step size
STEPS = 150           # Number of iterations

def get_next_run_dir(base_path):
    """
    Scans the base_path for folders named '1', '2', etc.
    Returns the path to the next available number (e.g., .../21).
    """
    if not os.path.exists(base_path):
        try:
            os.makedirs(base_path)
        except OSError as e:
            print(f"Error creating base directory: {e}")
            sys.exit(1)

    i = 1
    while True:
        candidate = os.path.join(base_path, str(i))
        if not os.path.exists(candidate):
            return candidate
        i += 1

def main():
    print(f"--- Initializing Anti-Deepfake Tool on {DEVICE} ---")
    
    # 1. Validate Input Files exist in the BASE directory
    source_user_img = os.path.join(BASE_ROOT_DIR, "original.jpg")
    source_stranger_img = os.path.join(BASE_ROOT_DIR, "stranger.jpg")

    if not os.path.exists(source_user_img):
        print(f"Error: Source file 'original.jpg' not found in {BASE_ROOT_DIR}")
        return
    if not os.path.exists(source_stranger_img):
        print(f"Error: Source file 'stranger.jpg' not found in {BASE_ROOT_DIR}")
        return

    # 2. Create New Run Directory
    run_dir = get_next_run_dir(BASE_ROOT_DIR)
    os.makedirs(run_dir)
    print(f"Created new run directory: {run_dir}")

    # 3. Copy Source Images to New Directory
    current_user_img_path = os.path.join(run_dir, "original.jpg")
    current_stranger_img_path = os.path.join(run_dir, "stranger.jpg")
    
    shutil.copy2(source_user_img, current_user_img_path)
    shutil.copy2(source_stranger_img, current_stranger_img_path)
    print("Copied original and stranger images to run directory.")

    # 4. Define Output Paths in the New Directory

    output_path = os.path.join(run_dir, "protected_output.png")
    debug_target_path = os.path.join(run_dir, "generated_target_swap.png")

    # 5. Instantiate Concrete Modules
    try:
        target_gen = InsightFaceTargetGenerator(device=DEVICE)

        feature_extractor = CLIPFeatureExtractor(
            model_id="openai/clip-vit-large-patch14", 
            device=DEVICE
        )
        
        # Instantiating the Robust Optimizer
        optimizer = RobustPGDOptimizer(
            steps=STEPS, 
            epsilon=EPSILON, 
            alpha=ALPHA, 
            device=DEVICE,
        )

        # 6. Initialize Pipeline
        pipeline = ProtectionPipeline(
            target_generator=target_gen,
            feature_extractor=feature_extractor,
            optimizer=optimizer
        )

        # 7. Run Pipeline
        pipeline.run(
            user_img_path=current_user_img_path,
            stranger_img_path=current_stranger_img_path,
            output_path=output_path,
            debug_target_path=debug_target_path
        )

    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()