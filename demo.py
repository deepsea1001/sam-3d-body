# Copyright (c) Meta Platforms, Inc. and affiliates.
import argparse
import os
from glob import glob

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml", ".sl"],
    pythonpath=True,
    dotenv=True,
)
import time
import cv2
import numpy as np
import torch
from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from tools.vis_utils import visualize_sample, visualize_sample_together, visualize_debug_detections
from tools.json_export import export_from_pipeline_outputs
from tqdm import tqdm


def resize_image(img, max_size):
    """Resize image so longest dimension equals max_size, preserving aspect ratio."""
    h, w = img.shape[:2]
    if max(h, w) <= max_size:
        return img
    if h > w:
        new_h = max_size
        new_w = int(w * max_size / h)
    else:
        new_w = max_size
        new_h = int(h * max_size / w)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def main(args):
    if args.output_folder == "":
        output_folder = os.path.join("./output", os.path.basename(args.image_folder))
    else:
        output_folder = args.output_folder

    os.makedirs(output_folder, exist_ok=True)

    # Use command-line args or environment variables
    mhr_path = args.mhr_path or os.environ.get("SAM3D_MHR_PATH", "")
    detector_path = args.detector_path or os.environ.get("SAM3D_DETECTOR_PATH", "")
    segmentor_path = args.segmentor_path or os.environ.get("SAM3D_SEGMENTOR_PATH", "")
    fov_path = args.fov_path or os.environ.get("SAM3D_FOV_PATH", "")

    # Initialize Skeleton Exporter
    exporter = None
    if args.export_skeleton:
        from tools.export_utils import SkeletonExporter
        exporter = SkeletonExporter(output_folder)
    elif args.export_glb:
        from tools.gltf_export_utils import GLBExporter
        exporter = GLBExporter()

    # Initialize sam-3d-body model and other optional modules
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    model, model_cfg = load_sam_3d_body(
        args.checkpoint_path, device=device, mhr_path=mhr_path
    )

    human_detector, human_segmentor, fov_estimator = None, None, None
    if args.detector_name:
        from tools.build_detector import HumanDetector

        human_detector = HumanDetector(
            name=args.detector_name, device=device, path=detector_path
        )
    
    if (args.segmentor_name == "sam2" and len(segmentor_path)) or args.segmentor_name != "sam2":
        from tools.build_sam import HumanSegmentor

        human_segmentor = HumanSegmentor(
            name=args.segmentor_name, device=device, path=segmentor_path
        )
    if args.fov_name:
        from tools.build_fov_estimator import FOVEstimator

        fov_estimator = FOVEstimator(name=args.fov_name, device=device, path=fov_path)

    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=human_segmentor,
        fov_estimator=fov_estimator,
    )

    image_extensions = [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.gif",
        "*.bmp",
        "*.tiff",
        "*.webp",
    ]
    images_list = sorted(
        [
            image
            for ext in image_extensions
            for image in glob(os.path.join(args.image_folder, ext))
        ]
    )

    pbar = tqdm(images_list)
    for image_path in pbar:
        pbar.set_description(f"Processing {os.path.basename(image_path)}")
        start_time = time.time()

        # Load and optionally resize image
        img = cv2.imread(image_path)
        if args.resize > 0:
            orig_h, orig_w = img.shape[:2]
            img = resize_image(img, args.resize)
            if img.shape[:2] != (orig_h, orig_w):
                tqdm.write(f"[RESIZE] {os.path.basename(image_path)}: {orig_w}x{orig_h} -> {img.shape[1]}x{img.shape[0]}")

        # Convert BGR to RGB for process_one_image (expects RGB when given numpy array)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        outputs = estimator.process_one_image(
            img_rgb,
            bbox_thr=args.bbox_thresh,
            use_mask=args.use_mask,
            det_prompt=args.det_prompt,
        )

        # Export Skeleton Data
        if exporter is not None:
             exporter.add_frame(outputs)

        duration = time.time() - start_time
        pbar.set_postfix(time=f"{duration:.2f}s")

        if args.debug:
            tqdm.write(f"[DEBUG] {os.path.basename(image_path)}: {duration:.3f}s, found {len(outputs)} humans")

        vis_images = visualize_sample_together(img, outputs, estimator.faces)
        
        base_name = os.path.basename(image_path)[:-4]
        for suffix, vis_img in vis_images.items():
            cv2.imwrite(
                f"{output_folder}/{base_name}_{suffix}.jpg",
                vis_img.astype(np.uint8),
            )

        if args.debug:
            debug_img = visualize_debug_detections(img, outputs)
            cv2.imwrite(
                f"{output_folder}/{os.path.basename(image_path)[:-4]}_debug.jpg",
                debug_img.astype(np.uint8),
            )

        # Export JSON for web viewer (one file per image)
        if args.export_json and outputs:
            json_path = f"{output_folder}/{base_name}_skeleton.json"
            export_from_pipeline_outputs(outputs, json_path, os.path.basename(image_path))

    if exporter is not None:
        if args.export_glb:
            exporter.save(os.path.join(output_folder, "skeleton_anim.glb"))
        else:
            exporter.save_motion("skeleton_motion.npz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SAM 3D Body Demo - Single Image Human Mesh Recovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                Examples:
                python demo.py --image_folder ./images --checkpoint_path ./checkpoints/model.ckpt

                Environment Variables:
                SAM3D_MHR_PATH: Path to MHR asset
                SAM3D_DETECTOR_PATH: Path to human detection model folder
                SAM3D_SEGMENTOR_PATH: Path to human segmentation model folder
                SAM3D_FOV_PATH: Path to fov estimation model folder
                """,
    )
    parser.add_argument(
        "--image_folder",
        required=True,
        type=str,
        help="Path to folder containing input images",
    )
    parser.add_argument(
        "--output_folder",
        default="",
        type=str,
        help="Path to output folder (default: ./output/<image_folder_name>)",
    )
    parser.add_argument(
        "--checkpoint_path",
        required=True,
        type=str,
        help="Path to SAM 3D Body model checkpoint",
    )
    parser.add_argument(
        "--detector_name",
        default="vitdet",
        type=str,
        help="Human detection model for demo (Default `vitdet`, add your favorite detector if needed).",
    )
    parser.add_argument(
        "--segmentor_name",
        default="sam2",
        type=str,
        help="Human segmentation model for demo (Default `sam2`, add your favorite segmentor if needed).",
    )
    parser.add_argument(
        "--fov_name",
        default="moge2",
        type=str,
        help="FOV estimation model for demo (Default `moge2`, add your favorite fov estimator if needed).",
    )
    parser.add_argument(
        "--detector_path",
        default="",
        type=str,
        help="Path to human detection model folder (or set SAM3D_DETECTOR_PATH)",
    )
    parser.add_argument(
        "--segmentor_path",
        default="",
        type=str,
        help="Path to human segmentation model folder (or set SAM3D_SEGMENTOR_PATH)",
    )
    parser.add_argument(
        "--fov_path",
        default="",
        type=str,
        help="Path to fov estimation model folder (or set SAM3D_FOV_PATH)",
    )
    parser.add_argument(
        "--mhr_path",
        default="",
        type=str,
        help="Path to MoHR/assets folder (or set SAM3D_mhr_path)",
    )
    parser.add_argument(
        "--bbox_thresh",
        default=0.3,
        type=float,
        help="Bounding box detection threshold",
    )
    parser.add_argument(
        "--use_mask",
        action="store_true",
        default=False,
        help="Use mask-conditioned prediction (segmentation mask is automatically generated from bbox)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Visualize bboxes and masks for debugging",
    )
    parser.add_argument(
        "--det_prompt",
        type=str,
        default="person",
        help="Text prompt for human detection (e.g. 'person')",
    )
    parser.add_argument(
        "--export_skeleton",
        action="store_true",
        help="Export detected 3D skeletons to .npz for animation",
    )
    parser.add_argument(
        "--export_glb",
        action="store_true",
        help="Export detected 3D skeletons to .glb (Binary GLTF) for animation",
    )
    parser.add_argument(
        "--export_json",
        action="store_true",
        help="Export 3D skeletons to JSON for web viewer (one file per image)",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=0,
        help="Resize images so longest dimension equals this value before processing (0 to disable, e.g., --resize 1024)",
    )
    args = parser.parse_args()

    main(args)
