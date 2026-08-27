"""Mannequin pose JSON exporter.

Exports to the format used by the Three.js mannequin viewer,
matching data/poses/default_poses.json structure.
"""

import json
from pathlib import Path
from typing import Dict, Optional, List
import numpy as np

from .base import BaseExporter
from ..core.math_utils import QuaternionMath


# Mapping from MHR70 retargeter joint names to mannequin joint names
MHR70_TO_MANNEQUIN = {
    # Spine
    "root": "pelvis",
    "spine1": "spine_1",
    "spine2": "spine_2",
    "neck": "neck",
    "head": "head",

    # Face
    "nose": "nose",
    "left_eye": "left_eye",
    "right_eye": "right_eye",
    "left_ear": "left_ear",
    "right_ear": "right_ear",

    # Left arm
    "left_clavicle": "left_clavicle",
    "left_acromion": None,  # No mannequin equivalent
    "left_shoulder": "left_shoulder",
    "left_elbow": "left_elbow",
    "left_olecranon": None,  # No mannequin equivalent
    "left_cubital_fossa": None,  # No mannequin equivalent
    "left_wrist": "left_wrist",

    # Left hand - thumb (third=base, first=near tip)
    "left_thumb_third": "left_thumb_1",
    "left_thumb_second": "left_thumb_2",
    "left_thumb_first": "left_thumb_3",
    "left_thumb_tip": "left_thumb_tip",

    # Left hand - index
    "left_index_third": "left_index_finger_1",
    "left_index_second": "left_index_finger_2",
    "left_index_first": "left_index_finger_3",
    "left_index_tip": "left_index_finger_tip",

    # Left hand - middle
    "left_middle_third": "left_middle_finger_1",
    "left_middle_second": "left_middle_finger_2",
    "left_middle_first": "left_middle_finger_3",
    "left_middle_tip": "left_middle_finger_tip",

    # Left hand - ring
    "left_ring_third": "left_ring_finger_1",
    "left_ring_second": "left_ring_finger_2",
    "left_ring_first": "left_ring_finger_3",
    "left_ring_tip": "left_ring_finger_tip",

    # Left hand - pinky
    "left_pinky_third": "left_pinky_finger_1",
    "left_pinky_second": "left_pinky_finger_2",
    "left_pinky_first": "left_pinky_finger_3",
    "left_pinky_tip": "left_pinky_finger_tip",

    # Right arm
    "right_clavicle": "right_clavicle",
    "right_acromion": None,  # No mannequin equivalent
    "right_shoulder": "right_shoulder",
    "right_elbow": "right_elbow",
    "right_olecranon": None,  # No mannequin equivalent
    "right_cubital_fossa": None,  # No mannequin equivalent
    "right_wrist": "right_wrist",

    # Right hand - thumb
    "right_thumb_third": "right_thumb_1",
    "right_thumb_second": "right_thumb_2",
    "right_thumb_first": "right_thumb_3",
    "right_thumb_tip": "right_thumb_tip",

    # Right hand - index
    "right_index_third": "right_index_finger_1",
    "right_index_second": "right_index_finger_2",
    "right_index_first": "right_index_finger_3",
    "right_index_tip": "right_index_finger_tip",

    # Right hand - middle
    "right_middle_third": "right_middle_finger_1",
    "right_middle_second": "right_middle_finger_2",
    "right_middle_first": "right_middle_finger_3",
    "right_middle_tip": "right_middle_finger_tip",

    # Right hand - ring
    "right_ring_third": "right_ring_finger_1",
    "right_ring_second": "right_ring_finger_2",
    "right_ring_first": "right_ring_finger_3",
    "right_ring_tip": "right_ring_finger_tip",

    # Right hand - pinky
    "right_pinky_third": "right_pinky_finger_1",
    "right_pinky_second": "right_pinky_finger_2",
    "right_pinky_first": "right_pinky_finger_3",
    "right_pinky_tip": "right_pinky_finger_tip",

    # Left leg
    "left_hip": "left_hip",
    "left_knee": "left_knee",
    "left_ankle": "left_ankle",
    "left_heel": "left_heel",
    "left_big_toe": "left_big_toe",
    "left_small_toe": "left_small_toe",

    # Right leg
    "right_hip": "right_hip",
    "right_knee": "right_knee",
    "right_ankle": "right_ankle",
    "right_heel": "right_heel",
    "right_big_toe": "right_big_toe",
    "right_small_toe": "right_small_toe",
}


class MannequinExporter(BaseExporter):
    """Export rotations to mannequin pose JSON format.

    Output format matches data/poses/default_poses.json:
    {
        "cameraState": { ... },  # optional
        "pose": {
            "pelvis": {"isQuaternion": true, "_x": 0, "_y": 0, "_z": 0, "_w": 1},
            "spine_1": {...},
            ...
        }
    }
    """

    @property
    def file_extension(self) -> str:
        return ".json"

    def _map_joint_name(self, mhr70_name: str) -> Optional[str]:
        """Map MHR70 joint name to mannequin joint name."""
        return MHR70_TO_MANNEQUIN.get(mhr70_name)

    def export(
        self,
        rotations: Dict[str, np.ndarray],
        scales: Dict[str, float],
        output_path: str,
        positions: Optional[Dict[str, np.ndarray]] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """Export retargeted pose to mannequin JSON file.

        Args:
            rotations: Dict mapping joint name -> quaternion [w, x, y, z]
            scales: Dict mapping joint name -> scale factor (ignored for mannequin format)
            output_path: Path to save JSON file
            positions: Optional joint positions (ignored)
            metadata: Optional metadata with 'camera_state' for camera position

        Returns:
            Path to saved file
        """
        data = self.to_dict(rotations, scales, positions, metadata)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        return str(output_path)

    def to_dict(
        self,
        rotations: Dict[str, np.ndarray],
        scales: Dict[str, float],
        positions: Optional[Dict[str, np.ndarray]] = None,
        metadata: Optional[dict] = None,
    ) -> List[dict]:
        """Convert retargeted pose to mannequin dict representation.

        Args:
            rotations: Dict mapping MHR70 joint name -> quaternion [w, x, y, z]
            scales: Dict mapping joint name -> scale factor (ignored)
            positions: Optional joint positions (ignored)
            metadata: Optional metadata with 'camera_state'

        Returns:
            List containing single pose dict (mannequin viewer expects array)
        """
        # Build pose dict with mannequin joint names
        pose = {}
        for mhr70_name, quat in rotations.items():
            mannequin_name = self._map_joint_name(mhr70_name)
            if mannequin_name is not None:
                pose[mannequin_name] = QuaternionMath.to_threejs_dict(quat)

        # Build output structure
        data = {"pose": pose}

        # Add camera state if provided
        if metadata and "camera_state" in metadata:
            data["cameraState"] = metadata["camera_state"]

        # Mannequin viewer expects array of poses
        return [data]

    def export_multi_pose(
        self,
        poses: List[Dict],
        output_path: str,
        camera_states: Optional[List[dict]] = None,
    ) -> str:
        """Export multiple poses as an array (matching default_poses.json format).

        Args:
            poses: List of dicts with 'rotations' key
            output_path: Path to save JSON file
            camera_states: Optional list of camera states (one per pose)

        Returns:
            Path to saved file
        """
        data = []

        for i, pose_data in enumerate(poses):
            pose_dict = self.to_dict(
                rotations=pose_data["rotations"],
                scales=pose_data.get("scales", {}),
            )

            # Add camera state if provided
            if camera_states and i < len(camera_states):
                pose_dict["cameraState"] = camera_states[i]

            data.append(pose_dict)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        return str(output_path)

    @staticmethod
    def get_default_camera_state() -> dict:
        """Get a reasonable default camera state for viewing."""
        return {
            "position": {"x": 0, "y": 9, "z": 50},
            "target": {"x": 0, "y": 9, "z": 0}
        }
