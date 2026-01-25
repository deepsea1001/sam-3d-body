# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SAM 3D Body is a promptable model from Meta AI for single-image full-body 3D human mesh recovery (HMR). It estimates human pose for body, hands, and feet using the Momentum Human Rig (MHR) format with 70 keypoints.

This repo has been adapted to run on Macbook Pro M2 laptop. Local commit to macos-mps-support reflect these updates. 

## Environment Setup

```bash
# Create environment
conda create -n sam_3d_body python=3.11 -y
conda activate sam_3d_body

# Install PyTorch (follow pytorch.org for your platform)
# Then install dependencies:
pip install pytorch-lightning pyrender opencv-python yacs scikit-image einops timm dill pandas rich hydra-core hydra-submitit-launcher hydra-colorlog pyrootutils webdataset chump networkx==3.2.1 roma joblib seaborn wandb appdirs appnope ffmpeg cython jsonlines pytest xtcocotools loguru optree fvcore black pycocotools tensorboard huggingface_hub

# Install detectron2
pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps

# Optional: MoGe for FOV estimation
pip install git+https://github.com/microsoft/MoGe.git
```

## Running Inference

```bash
# Download checkpoints first
hf download facebook/sam-3d-body-dinov3 --local-dir checkpoints/sam-3d-body-dinov3

# Run demo
python demo.py \
    --image_folder <path_to_images> \
    --output_folder <path_to_output> \
    --checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt \
    --mhr_path ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt
```

## CONDA env
code runs in this conda env: sam_3d_body

## Current working command with flags:
python demo.py --image_folder ./data/samples --checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt --mhr_path ./checkpoints/mhr_model.pt --detector_name sam3 --segmentor_name sam3 --use_mask --bbox_thresh 0.3 --mask_thresh 0.3 --debug --export_glb

## Architecture

### Core Components

- **`sam_3d_body/`** - Main package
  - `SAM3DBodyEstimator` (`sam_3d_body_estimator.py`) - High-level inference wrapper that orchestrates detection, segmentation, FOV estimation, and mesh recovery
  - `load_sam_3d_body()` / `load_sam_3d_body_hf()` (`build_models.py`) - Model loading functions

- **`sam_3d_body/models/`**
  - `meta_arch/sam3d_body.py` - Main `SAM3DBody` model class with encoder-decoder architecture
  - `backbones/` - Vision backbones (DINOv3, ViT-H)
  - `decoders/` - Promptable decoder, prompt encoder, keypoint sampler
  - `heads/` - MHR pose head, camera head

- **`tools/`** - External model wrappers and visualization
  - `build_detector.py` - `HumanDetector` class (supports ViTDet, SAM3)
  - `build_fov_estimator.py` - `FOVEstimator` class (MoGe2)
  - `build_sam.py` - `HumanSegmentor` class (SAM2)
  - `vis_utils.py` - 2D skeleton/mesh visualization (uses BGR colors for OpenCV)
  - `json_export.py` - JSON export for web viewer (defines MHR70_BONES with RGB colors)

- **`viewer/`** - Web-based 3D visualization
  - `skeleton_viewer.html` - Three.js skeleton viewer (drag-drop JSON, orbit controls)

- **`notebook/utils.py`** - Helper functions including `setup_sam_3d_body()` for easy initialization

### Data Flow

1. Image -> Human detector (ViTDet/SAM3) -> bounding boxes
2. Boxes -> Human segmentor (SAM2, optional) -> masks
3. Image -> FOV estimator (MoGe2, optional) -> camera intrinsics
4. Cropped regions + prompts -> SAM3DBody model -> MHR pose/mesh outputs

### Output Format

Each detected person returns a dict with:
- `pred_vertices` - 3D mesh vertices
- `pred_keypoints_3d` / `pred_keypoints_2d` - 70 MHR keypoints
- `pred_cam_t` - Camera translation
- `focal_length` - Estimated focal length
- `body_pose_params`, `hand_pose_params`, `shape_params` - MHR model parameters
- `bbox`, `mask` - Detection outputs

### Keypoint Format

Uses MHR70 format (defined in `sam_3d_body/metadata/mhr70.py`):
- 0-4: Face (nose, eyes, ears)
- 5-14: Body (shoulders, elbows, hips, knees, ankles)
- 15-20: Feet (toes, heels)
- 21-41: Right hand (21 keypoints)
- 42-62: Left hand (21 keypoints)
- 63-69: Extra (olecranon, cubital fossa, acromion, neck) - typically skipped in visualization

Virtual keypoints computed for spine/neck visualization:
- 70: midHip (midpoint of indices 9, 10)
- 71: midShoulder (midpoint of indices 5, 6)
- 72: head (midpoint of indices 3, 4)

## Hardware Support

The codebase auto-detects and supports:
- CUDA (NVIDIA GPUs)
- MPS (Apple Silicon)
- CPU fallback

Device selection happens in `demo.py` and `build_models.py`.

## Environment Variables

```bash
SAM3D_MHR_PATH      # Path to MHR asset
SAM3D_DETECTOR_PATH # Path to detector model
SAM3D_SEGMENTOR_PATH # Path to segmentor model
SAM3D_FOV_PATH      # Path to FOV model
```

## Development Notes

### Color Conventions
- `vis_utils.py` uses BGR (OpenCV format)
- `json_export.py` and `skeleton_viewer.html` use RGB (Three.js format)
- When porting colors between Python and web viewer, reverse the order: `[R,G,B]` ↔ `(B,G,R)`

## Pipeline Thresholds

- `--bbox_thresh` (default 0.3) - Human detector confidence threshold
- `--mask_thresh` (default 0.3) - SAM3 segmentor confidence threshold
- Pose estimator has no threshold - processes all detected boxes
- MHR scale/shape params are learned (28 PCA → 68 bone lengths), no hard constraints

## Retargeting Tools

- `retarget_image.py` - Process image → mannequin pose JSON for Three.js viewer
- `tools/retargeting/` - Retargeting module (MHR70 → mannequin joint mapping)
- `tools/export_bind_pose_gltf.py` - Export bind pose skeleton as GLTF
- `tools/export_bind_pose_usd.py` - Export bind pose skeleton as USD (for Maya)
- `tools/blender_bind_pose_armature.py` - Blender script for FBX export

### Blender Headless Export
```bash
/Applications/Blender.app/Contents/MacOS/Blender --background --python tools/blender_bind_pose_armature.py -- --output output/skeleton.fbx --prefix bind_
```

## Gotchas

- `HumanDetector` and `HumanSegmentor` require `path=""` parameter even for SAM3
- Always activate conda: `conda activate sam_3d_body` before running scripts
- MHR module requires float64, falls back to CPU on MPS (Apple Silicon)
