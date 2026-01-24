"""Retargeting module for mapping SAM 3D Body poses to mannequins.

This module provides a clean API for retargeting MHR70 keypoints
to a fixed-proportion mannequin using vector-based rotation computation.

Example usage:
    from tools.retargeting import retarget_to_mannequin

    # From SAM 3D Body output
    outputs = estimator.process_one_image(image)
    keypoints = outputs[0]["pred_keypoints_3d"]

    # Retarget to mannequin and save
    result = retarget_to_mannequin(
        keypoints,
        bind_pose_path="./data/bind_poses/default_human.json",
        output_format="threejs",
        output_path="./output/mannequin_pose.json"
    )

    # Or retarget multiple people from pipeline output
    from tools.retargeting import retarget_from_pipeline

    retarget_from_pipeline(
        outputs,  # List of person dicts from SAM 3D Body
        output_path="./output/poses.json",
        bind_pose_path="./data/bind_poses/default_human.json"
    )
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np

from .core.math_utils import QuaternionMath, VectorMath
from .core.coordinate_systems import CoordinateTransform
from .bind_poses.loader import BindPoseLoader
from .bind_poses.validator import BindPoseValidator
from .retargeters.base import BaseRetargeter
from .retargeters.mhr70_retargeter import MHR70Retargeter
from .exporters.base import BaseExporter
from .exporters.threejs_exporter import ThreeJSExporter
from .exporters.mannequin_exporter import MannequinExporter, MHR70_TO_MANNEQUIN

__all__ = [
    # Public API functions
    "retarget_to_mannequin",
    "retarget_to_mannequin_format",
    "retarget_from_pipeline",
    # Core classes
    "QuaternionMath",
    "VectorMath",
    "CoordinateTransform",
    # Bind pose
    "BindPoseLoader",
    "BindPoseValidator",
    # Retargeters
    "BaseRetargeter",
    "MHR70Retargeter",
    # Exporters
    "BaseExporter",
    "ThreeJSExporter",
    "MannequinExporter",
    "MHR70_TO_MANNEQUIN",
]


def retarget_to_mannequin(
    keypoints_3d: np.ndarray,
    bind_pose_path: Optional[str] = None,
    output_format: str = "threejs",
    output_path: Optional[str] = None,
    include_positions: bool = False,
    metadata: Optional[dict] = None,
) -> dict:
    """High-level API for retargeting MHR70 keypoints to mannequin.

    Computes local joint rotations and bone scales that transform
    the bind pose (T-pose) to match the detected pose.

    Args:
        keypoints_3d: (70, 3) array from SAM 3D Body pred_keypoints_3d
                      in CV coordinates (Y-down)
        bind_pose_path: Path to bind pose JSON. If None, uses default T-pose.
        output_format: Export format - "threejs" (currently only option)
        output_path: Where to save result. If None, returns dict only.
        include_positions: Include joint positions in output (for debugging)
        metadata: Optional metadata to include (e.g., source_image)

    Returns:
        Dict with retargeted pose data:
        - version: Format version
        - coordinate_system: "yup"
        - joints: Dict of {joint_name: {rotation: {...}, scale: float}}

    Raises:
        ValueError: If keypoints_3d has wrong shape
        FileNotFoundError: If bind pose file doesn't exist

    Example:
        >>> result = retarget_to_mannequin(
        ...     keypoints_3d=outputs[0]["pred_keypoints_3d"],
        ...     bind_pose_path="./data/bind_poses/default_human.json",
        ...     output_path="./output/pose.json"
        ... )
    """
    # Validate input
    keypoints_3d = np.array(keypoints_3d)
    if keypoints_3d.shape[0] < 70 or keypoints_3d.shape[1] != 3:
        raise ValueError(
            f"Expected keypoints shape (70, 3), got {keypoints_3d.shape}"
        )

    # Load bind pose
    if bind_pose_path is None:
        bind_pose = BindPoseLoader.get_default_bind_pose()
    else:
        bind_pose = BindPoseLoader.from_json(bind_pose_path)

    # Create retargeter
    retargeter = MHR70Retargeter(bind_pose)

    # Retarget
    rotations, scales, positions = retargeter.retarget(keypoints_3d)

    # Get exporter
    if output_format == "threejs":
        exporter = ThreeJSExporter()
    else:
        raise ValueError(f"Unknown output format: {output_format}")

    # Build metadata
    if metadata is None:
        metadata = {}
    if bind_pose_path:
        metadata["bind_pose"] = Path(bind_pose_path).stem

    # Export or return dict
    if output_path:
        exporter.export(
            rotations=rotations,
            scales=scales,
            output_path=output_path,
            positions=positions if include_positions else None,
            metadata=metadata,
        )

    return exporter.to_dict(
        rotations=rotations,
        scales=scales,
        positions=positions if include_positions else None,
        metadata=metadata,
    )


def retarget_from_pipeline(
    outputs: List[dict],
    output_path: str,
    bind_pose_path: Optional[str] = None,
    output_format: str = "threejs",
    include_positions: bool = False,
    source_image: Optional[str] = None,
) -> str:
    """Retarget multiple people from SAM 3D Body pipeline output.

    Args:
        outputs: List of person dicts from SAM 3D Body, each containing
                 'pred_keypoints_3d' key
        output_path: Path to save result
        bind_pose_path: Path to bind pose JSON. If None, uses default T-pose.
        output_format: Export format - "threejs"
        include_positions: Include joint positions in output
        source_image: Optional source image filename for metadata

    Returns:
        Path to saved file

    Example:
        >>> outputs = estimator.process_one_image(image)
        >>> path = retarget_from_pipeline(
        ...     outputs,
        ...     output_path="./output/multi_person.json",
        ...     source_image="photo.jpg"
        ... )
    """
    # Load bind pose once
    if bind_pose_path is None:
        bind_pose = BindPoseLoader.get_default_bind_pose()
    else:
        bind_pose = BindPoseLoader.from_json(bind_pose_path)

    # Create retargeter
    retargeter = MHR70Retargeter(bind_pose)

    # Process each person
    people = []
    for person_output in outputs:
        if "pred_keypoints_3d" not in person_output:
            continue

        keypoints_3d = person_output["pred_keypoints_3d"]
        if not isinstance(keypoints_3d, np.ndarray):
            keypoints_3d = np.array(keypoints_3d)

        if keypoints_3d.shape[0] < 70:
            continue

        rotations, scales, positions = retargeter.retarget(keypoints_3d)

        person_data = {
            "rotations": rotations,
            "scales": scales,
        }
        if include_positions:
            person_data["positions"] = positions

        people.append(person_data)

    if not people:
        raise ValueError("No valid people found in outputs")

    # Get exporter
    if output_format == "threejs":
        exporter = ThreeJSExporter()
    else:
        raise ValueError(f"Unknown output format: {output_format}")

    # Build metadata
    metadata = {}
    if bind_pose_path:
        metadata["bind_pose"] = Path(bind_pose_path).stem
    if source_image:
        metadata["source_image"] = source_image

    return exporter.export_multi_person(
        people=people,
        output_path=output_path,
        metadata=metadata,
    )


def retarget_to_mannequin_format(
    keypoints_3d: np.ndarray,
    bind_pose_path: Optional[str] = None,
    output_path: Optional[str] = None,
    include_camera_state: bool = True,
) -> dict:
    """Retarget MHR70 keypoints to mannequin pose format.

    Outputs the format matching data/poses/default_poses.json:
    {
        "cameraState": { "position": {...}, "target": {...} },
        "pose": {
            "pelvis": {"isQuaternion": true, "_x": 0, "_y": 0, "_z": 0, "_w": 1},
            "spine_1": {...},
            ...
        }
    }

    Args:
        keypoints_3d: (70, 3) array from SAM 3D Body pred_keypoints_3d
                      in CV coordinates (Y-down)
        bind_pose_path: Path to bind pose JSON. If None, uses default T-pose.
        output_path: Where to save result. If None, returns dict only.
        include_camera_state: Include default camera state in output

    Returns:
        Dict with mannequin pose format (joint names match mannequin skeleton)

    Example:
        >>> result = retarget_to_mannequin_format(
        ...     keypoints_3d=outputs[0]["pred_keypoints_3d"],
        ...     output_path="./output/mannequin_pose.json"
        ... )
        >>> # Result has "pose" with joints like "pelvis", "spine_1", etc.
    """
    # Validate input
    keypoints_3d = np.array(keypoints_3d)
    if keypoints_3d.shape[0] < 70 or keypoints_3d.shape[1] != 3:
        raise ValueError(
            f"Expected keypoints shape (70, 3), got {keypoints_3d.shape}"
        )

    # Load bind pose
    if bind_pose_path is None:
        bind_pose = BindPoseLoader.get_default_bind_pose()
    else:
        bind_pose = BindPoseLoader.from_json(bind_pose_path)

    # Create retargeter
    retargeter = MHR70Retargeter(bind_pose)

    # Retarget
    rotations, scales, positions = retargeter.retarget(keypoints_3d)

    # Use mannequin exporter
    exporter = MannequinExporter()

    # Build metadata with camera state if requested
    metadata = {}
    if include_camera_state:
        metadata["camera_state"] = MannequinExporter.get_default_camera_state()

    # Export or return dict
    if output_path:
        exporter.export(
            rotations=rotations,
            scales=scales,
            output_path=output_path,
            metadata=metadata,
        )

    return exporter.to_dict(
        rotations=rotations,
        scales=scales,
        metadata=metadata,
    )
