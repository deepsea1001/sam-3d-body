"""Retarget a single image to mannequin pose format.

Usage:
    python retarget_image.py path/to/image.jpg

Output:
    output/<image_name>_mannequin_pose.json
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch

from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from tools.retargeting import retarget_to_mannequin_format


def main():
    parser = argparse.ArgumentParser(description="Retarget image to mannequin pose")
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output JSON path (default: output/<name>_mannequin_pose.json)")
    parser.add_argument("--checkpoint", type=str,
                        default="./checkpoints/sam-3d-body-dinov3/model.ckpt",
                        help="Path to SAM 3D Body checkpoint")
    parser.add_argument("--mhr_path", type=str,
                        default="./checkpoints/mhr_model.pt",
                        help="Path to MHR model")
    parser.add_argument("--bind_pose", type=str,
                        default="./data/bind_poses/default_human.json",
                        help="Path to bind pose JSON")
    parser.add_argument("--resize", type=int, default=1280,
                        help="Resize longest edge to this (0 to disable)")
    parser.add_argument("--bbox_thresh", type=float, default=0.4,
                        help="Detection threshold")
    parser.add_argument("--mask_thresh", type=float, default=0.3,
                        help="Mask/segmentation confidence threshold (SAM3 only)")
    args = parser.parse_args()

    # Check image exists
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)

    # Set output path
    if args.output is None:
        os.makedirs("output", exist_ok=True)
        base_name = os.path.splitext(os.path.basename(args.image))[0]
        args.output = f"output/{base_name}_mannequin_pose.json"

    # Select device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load model
    print("Loading SAM 3D Body model...")
    model, model_cfg = load_sam_3d_body(args.checkpoint, device=device, mhr_path=args.mhr_path)

    # Load detector and segmentor
    from tools.build_detector import HumanDetector
    from tools.build_sam import HumanSegmentor

    human_detector = HumanDetector(name="sam3", device=device, path="")
    human_segmentor = HumanSegmentor(name="sam3", device=device, path="")

    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=human_segmentor,
    )

    # Load and preprocess image
    print(f"Processing: {args.image}")
    img = cv2.imread(args.image)

    if args.resize > 0:
        h, w = img.shape[:2]
        if max(h, w) > args.resize:
            if h > w:
                new_h, new_w = args.resize, int(w * args.resize / h)
            else:
                new_w, new_h = args.resize, int(h * args.resize / w)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            print(f"  Resized: {w}x{h} -> {new_w}x{new_h}")

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Run inference
    print("Running pose estimation...")
    outputs = estimator.process_one_image(
        img_rgb,
        bbox_thr=args.bbox_thresh,
        mask_thr=args.mask_thresh,
        use_mask=True,
    )

    if not outputs:
        print("Error: No person detected in image")
        sys.exit(1)

    print(f"  Detected {len(outputs)} person(s)")

    # Get first person's keypoints
    keypoints_3d = outputs[0]["pred_keypoints_3d"]

    # Retarget to mannequin format
    print("Retargeting to mannequin pose...")
    result = retarget_to_mannequin_format(
        keypoints_3d,
        bind_pose_path=args.bind_pose,
        output_path=args.output,
    )

    print(f"\nSaved mannequin pose to: {args.output}")
    print(f"  Joints: {len(result[0]['pose'])}")


if __name__ == "__main__":
    main()
