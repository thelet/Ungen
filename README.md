# Anti-Deepfake / Anti-Generative-Model Photo Protection

A Python tool that adds **adversarial perturbations** to photos so that **generative AI models** (image generators, face-swappers, and models trained on scraped photos) are less able to use the photo to create convincing new fakes *based on the original identity*.

## What This Project Does

This project is designed to **protect personal photos and creator content** from:

- **Generation attacks**: using your photo as a reference to generate new fake photos/videos “of you”.
- **Training misuse**: using your photos to train or fine-tune a model that can later generate content in your likeness or style.

### The Core Idea (Adversarial Perturbation)

Instead of adding a visible watermark, we add a **small, structured perturbation** to the pixels. The image still looks like you to humans, but it changes what a model “sees” in feature space.

### Target Audience
- **Individuals**: People who want to protect their photos before sharing on social media
- **Content Creators**: Influencers, models, and public figures who share images online
- **Organizations**: Companies protecting employee photos or brand assets
- **Privacy Advocates**: Anyone concerned about AI misuse and digital identity protection
- **Researchers**: Academics studying adversarial robustness and generative-model misuse defenses

## How It Works

The tool uses **Projected Gradient Descent (PGD)** to produce a protected image \(x'\) from an original image \(x\).

### The “Target” Concept (What the Optimizer Pushes Toward)

This project is **targeted**: it first constructs a **target image** (or target representation) \(t\) and then optimizes the protected photo so that, under a feature extractor (CLIP), the protected photo becomes **more similar to the target** than to the original identity.

Intuition: when a generative model (or a pipeline that uses embeddings) processes the protected photo, it is nudged toward the **target identity attributes** rather than the real identity.

### Pipeline (High Level)

1. **Target generation**: generate a human target image \(t\) (commonly via InsightFace-based swapping / reference creation).
2. **Target features**: compute target embedding \(f(t)\) with CLIP.
3. **Optimization**: update a perturbation \(\delta\) so that \(f(x+\delta)\) becomes close to \(f(t)\) (while keeping \(x+\delta\) visually close to \(x\)).
4. **Robustness**: apply augmentations (crop, color jitter, JPEG) inside the loop so the protection survives common upload transforms.
5. **Output**: save the protected image \(x'\).

### What This Achieves

- **For generation**: reference-based generation may drift toward the target (wrong identity cues).
- **For training**: models trained on protected images may learn a representation that is “poisoned” toward the chosen target identity attributes, reducing faithful generation of the real subject.

### Target Selection Recommendations (Important)

Your choice of target strongly affects both **effectiveness** and **visibility** of the perturbation.

- **Pick a human target**: keep the target within the same broad “manifold” (human faces), so a small perturbation can meaningfully shift model features.
- **Different gender often works well**: many models are highly sensitive to gender cues; pushing toward a different gender can more reliably break identity preservation.
- **Different ethnicity can also help**: models are sensitive to ethnicity-related facial feature distributions; shifting toward a different ethnicity can increase confusion.
- **Avoid non-human targets (dog, cartoon, etc.)**: pushing a real face embedding toward a non-human concept usually requires a much larger perturbation, making artifacts more visible. A small epsilon typically won’t be enough.
- **Prefer targets with similar pose/lighting**: closer framing, pose, and illumination can reduce the amount of perturbation needed for the embedding shift (better quality at the same epsilon).

## Features

- **Two Protection Strategies**: Low-resolution and full-resolution optimization approaches
- **Style-Guided Filtering**: Natural-looking protection using warm/cool/grainy filters
- **Robustness Augmentation**: Simulates real-world conditions (compression, cropping, color jitter)
- **CLIP Feature Extraction**: Uses state-of-the-art vision-language models for semantic understanding
- **InsightFace Integration**: Human-face target generation (used to build a target the optimization pulls toward)
- **GPU Acceleration**: CUDA support for faster processing
- **Modular Architecture**: Clean interface-based design for extensibility

## Project Structure

```
Anti_Deepfake_V_1_1/
├── Core/
│   ├── Interfaces.py              # Abstract interfaces for components
│   ├── Utilities.py               # Image utility functions
│   ├── FeatureExtractors/         # CLIP, VAE feature extractors
│   ├── Optimizers/                # Various PGD optimizers
│   ├── Pipelines/                 # Protection pipelines
│   ├── TargetGenerators/          # Target generation utilities (human target construction)
│   ├── SimilarityMetrics/         # SSIM, LPIPS metrics
│   ├── filters/                   # Image filtering components
│   └── tests/                     # Test files
├── simple_main.py                 # Main entry point
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- Windows/Linux/MacOS

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Anti_Deepfake_V_1_1
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

**Note**: The requirements file includes PyTorch with CUDA 12.1 support. For CPU-only or different CUDA versions, adjust the PyTorch installation accordingly.

## Usage

### Basic Usage

1. **Prepare your images**:
   - Place your original image as `original.jpg` in the base directory
   - Place a "stranger" face image as `stranger.jpg` in the base directory
   - Update `BASE_ROOT_DIR` in `simple_main.py` to point to your directory

2. **Configure parameters** in `simple_main.py`:
   ```python
   EPSILON = 32 / 255.0    # Max perturbation budget (0-1). Higher = stronger but more visible
   ALPHA = 2 / 255.0       # Step size for gradient descent
   STEPS = 150             # Number of optimization iterations
   ```

3. **Choose your optimizer** in `simple_main.py`:
   - Import either `RobustPGDOptimizer` from `Core.Optimizers.low_res_Optimizer` (low-res)
   - Or `RobustPGDOptimizer` from `Core.Optimizers.filter_v3_optimizer` (full-res)
   - Update the optimizer instantiation accordingly

4. **Run the protection pipeline**:
   ```bash
   python simple_main.py
   ```

5. **Output**:
   - The tool creates a numbered directory (1, 2, 3, ...) for each run
   - Protected image saved as `protected_output.png`
   - Debug target image saved as `generated_target_swap.png`

## Available Optimizers

The project includes two main optimization strategies, each with different trade-offs:

### 1. Low-Resolution Optimizer (`low_res_Optimizer.py`)

**Strategy**: Optimizes noise at low resolution (256x256) and upscales to full resolution using bicubic interpolation.

**Advantages**:
- **Smooth gradients**: Low-res optimization produces smooth, natural-looking perturbations
- **Memory efficient**: Works with less GPU memory
- **Faster**: Fewer parameters to optimize
- **Less grain**: Avoids high-frequency noise artifacts

**How it works**:
- Creates a 256x256 noise tensor
- Optimizes at this low resolution using PGD
- Upscales the optimized noise to full image resolution
- Applies style-guided filtering (Warm Filter) for natural appearance
- Uses SSIM loss to maintain structural similarity

**Best for**: Users who want smooth, natural-looking protection with lower computational requirements.

### 2. Filter V3 Optimizer (`filter_v3_optimizer.py`)

**Strategy**: Optimizes noise at full image resolution with advanced perceptual constraints.

**Advantages**:
- **Full precision**: Pixel-level control over perturbations
- **Better quality**: LPIPS (Learned Perceptual Image Patch Similarity) for perceptual accuracy
- **Stronger protection**: Can achieve better attack success rates
- **Fine-grained control**: More detailed optimization

**How it works**:
- Optimizes noise directly at full image resolution
- Uses LPIPS loss (VGG-based) for perceptual similarity
- Applies style-guided filtering (Warm Filter) for natural appearance
- Uses SSIM loss as additional structural constraint
- Includes robustness augmentation (JPEG compression, cropping, color jitter)

**Best for**: Users who need maximum protection effectiveness and have sufficient GPU memory.

## Available Filters

Both optimizers use style-guided filtering to make the protected images look natural:

- **Warm Filter**: Applies a "golden hour" / sepia look with enhanced vibrance. Increases red tones, boosts contrast and saturation. Creates a warm, inviting aesthetic.
- **Cool Filter**: Applies a "cyberpunk" / winter look with blue tones. Enhances blue and green channels, creates a cool, modern aesthetic.
- **Grainy Filter**: Adds film grain texture for a vintage, cinematic look. Includes random noise patterns that can help mask adversarial perturbations.
- **Mock Filter**: Instead of computing a filter, it returns a **pre-edited reference image** (resized to match the input). This effectively makes the optimization “pull” the protected image **toward an edited version of itself** (a *pre-edited target look*), which can allow stronger but still plausible pixel changes in key regions.

These filters are not just aesthetic—they guide the optimization process to produce perturbations that look like intentional artistic choices rather than noise.

### Recommendation: Use a Makeup-App Edited Reference (with MockFilter)

For face photos, a very effective workflow is:

- Create a **natural-looking makeup edit** using a makeup app (or Photoshop/FaceApp) as a reference image.
- Use that edited image as the **MockFilter target**.

Why this helps:
- Makeup provides a **realistic reason** for pixel changes around the face (eyes, lips, cheeks), so the optimizer can use more of its perturbation budget in facial areas without looking “noisy”.
- In practice this often improves protection strength **significantly** while keeping modifications looking like normal makeup rather than adversarial artifacts.

## Technical Details

### Protection Pipeline

The protection process follows these steps:

1. **Target Generation**: Builds a *human* target image \(t\) (e.g., via InsightFace-based reference construction)
2. **Feature Extraction**: Extracts target embedding \(f(t)\) using CLIP (vision-language model)
3. **Adversarial Optimization**: Applies PGD to find a perturbation \(\delta\) so that:
   - \(f(x+\delta)\) becomes **similar to** \(f(t)\) (the protected image is “pulled” toward the target in feature space)
   - The image remains visually similar to the original (SSIM / perceptual constraints)
   - The perturbation stays within a small budget (epsilon)
4. **Robustness**: Simulates common transforms (JPEG, crops, color jitter) so the effect survives upload platforms
5. **Output**: Produces a protected image that looks normal but causes embedding-based generation/training to drift toward the target

### Why This Approach Works

Many generative pipelines rely on **feature embeddings** (identity, style, or concept vectors). By making the protected image’s embedding closer to a chosen target embedding, we can:

- **Mislead generation**: “who/what this is” drifts toward the target attributes
- **Reduce training value**: training on protected images can bake in the wrong association (identity/style shifts)

The constraints (SSIM/perceptual + robustness augmentations) aim to keep the perturbation **small**, **less visible**, and **stable under real-world transforms**.

## Key Components

### Core Architecture
- **Interfaces** (`Core/Interfaces.py`): Abstract base classes defining the contract for all components
  - `ITargetGenerator`: Generates target face-swapped images
  - `IFeatureExtractor`: Extracts semantic features from images
  - `IOptimizer`: Performs adversarial optimization
  - `ISimilarityMetric`: Measures image similarity

### Main Components
- **ProtectionPipeline** (`Core/Pipelines/simple_pipeline.py`): Orchestrates the entire protection process
- **CLIPFeatureExtractor** (`Core/FeatureExtractors/CLIPFeatureExtractor.py`): Uses OpenAI CLIP for feature extraction
- **InsightFaceTargetGenerator** (`Core/TargetGenerators/TargetGeneratorWithColors.py`): Generates realistic face-swap targets
- **Optimizers**: Two main optimizers (see "Available Optimizers" section above)
- **Filters** (`Core/filters/filters.py`): Style-guided filters (Warm, Cool, Grainy) for natural appearance

## Parameters Explained

- **EPSILON** (default: 32/255 ≈ 0.125): Maximum allowed perturbation per pixel (0-1 range). 
  - Higher values = stronger protection but potentially more visible changes
  - Both optimizers use epsilon=32/255 to allow for visible filter-like color shifts that look intentional
  - Typical range: 8/255 (subtle) to 64/255 (stronger, more noticeable)

- **ALPHA** (default: 2/255): Step size for PGD gradient descent. 
  - Controls how aggressively the optimizer updates perturbations each iteration
  - Too high = unstable optimization, too low = slow convergence
  - Typical range: 0.5/255 to 4/255

- **STEPS** (default: 150): Number of optimization iterations.
  - More steps = better optimization but longer runtime
  - Low-res optimizer: 100-200 steps typically sufficient
  - Full-res optimizer: 150-300 steps for best results
  - Each step processes the image through the feature extractor and updates perturbations

## Requirements

See `requirements.txt` for full list. Key dependencies:
- PyTorch (>=2.0.0)
- torchvision
- transformers
- diffusers
- insightface
- opencv-python
- pillow
- numpy

## Troubleshooting

### CUDA Out of Memory
- Reduce image resolution
- Use CPU mode (set `DEVICE = 'cpu'`)
- Reduce batch size if applicable

### Missing Models
- CLIP models are downloaded automatically on first use
- InsightFace models may need to be downloaded separately

### Import Errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check that you're in the correct virtual environment

## Contributing

This project uses a modular interface-based architecture. To add new components:

1. Implement the appropriate interface (`ITargetGenerator`, `IFeatureExtractor`, `IOptimizer`)
2. Follow the existing code patterns
3. Test with the pipeline system

## License


## Acknowledgments

- Uses OpenAI CLIP for feature extraction
- InsightFace for face-swapping capabilities
- PyTorch for deep learning operations

## Important Notes

- **Effectiveness**: Protection effectiveness varies across generative models and pipelines. This project optimizes against embedding-based usage (generation and training) by shifting the image toward a chosen target embedding.
- **Visual Quality**: Protected images maintain high visual similarity to originals, with perturbations often appearing as subtle color shifts or filter effects.
- **Processing Time**: Depends on image size, number of steps, and hardware. GPU is strongly recommended (typically 2-5 minutes per image on modern GPUs).
- **Perturbation Budget**: The epsilon parameter controls the maximum allowed change. Higher values provide stronger protection but may be more noticeable.
- **Robustness**: The optimizers include robustness augmentation to ensure protection survives JPEG compression, cropping, and other common image transformations.
- **Ethical Use**: This tool is designed to protect people and creators from unauthorized generation and training on their photos. Use responsibly and in accordance with applicable laws.


