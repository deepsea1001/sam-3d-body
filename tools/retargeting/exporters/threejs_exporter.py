"""Three.js quaternion JSON exporter."""

import json
from pathlib import Path
from typing import Dict, Optional, List
import numpy as np

from .base import BaseExporter
from ..core.math_utils import QuaternionMath


class ThreeJSExporter(BaseExporter):
    """Export rotations to Three.js quaternion JSON format.

    Output format matches Three.js Quaternion serialization:
    {
        "isQuaternion": true,
        "_x": 0.0,
        "_y": 0.0,
        "_z": 0.0,
        "_w": 1.0
    }
    """

    VERSION = "1.0"
    COORDINATE_SYSTEM = "yup"

    @property
    def file_extension(self) -> str:
        return ".json"

    def export(
        self,
        rotations: Dict[str, np.ndarray],
        scales: Dict[str, float],
        output_path: str,
        positions: Optional[Dict[str, np.ndarray]] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """Export retargeted pose to JSON file.

        Args:
            rotations: Dict mapping joint name -> quaternion [w, x, y, z]
            scales: Dict mapping joint name -> scale factor
            output_path: Path to save JSON file
            positions: Optional joint positions for debugging
            metadata: Optional metadata (source_image, bind_pose_name, etc.)

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
    ) -> dict:
        """Convert retargeted pose to dict representation.

        Args:
            rotations: Dict mapping joint name -> quaternion [w, x, y, z]
            scales: Dict mapping joint name -> scale factor
            positions: Optional joint positions for debugging
            metadata: Optional metadata

        Returns:
            Dict with Three.js compatible format
        """
        # Build joints dict with rotations and scales
        joints = {}
        for joint_name in rotations:
            joint_data = {
                "rotation": QuaternionMath.to_threejs_dict(rotations[joint_name]),
            }

            if joint_name in scales:
                joint_data["scale"] = scales[joint_name]

            if positions and joint_name in positions:
                pos = positions[joint_name]
                joint_data["position"] = [float(pos[0]), float(pos[1]), float(pos[2])]

            joints[joint_name] = joint_data

        # Build output structure
        data = {
            "version": self.VERSION,
            "coordinate_system": self.COORDINATE_SYSTEM,
            "joints": joints,
        }

        # Add metadata
        if metadata:
            for key, value in metadata.items():
                if key not in data:  # Don't overwrite core fields
                    data[key] = value

        return data

    def export_multi_person(
        self,
        people: List[Dict],
        output_path: str,
        metadata: Optional[dict] = None,
    ) -> str:
        """Export multiple people's retargeted poses.

        Args:
            people: List of dicts with 'rotations', 'scales', optional 'positions'
            output_path: Path to save JSON file
            metadata: Optional metadata

        Returns:
            Path to saved file
        """
        people_data = []

        for i, person in enumerate(people):
            person_dict = self.to_dict(
                rotations=person["rotations"],
                scales=person["scales"],
                positions=person.get("positions"),
                metadata=None,  # Add to top level only
            )
            person_dict["person_id"] = i
            # Remove redundant top-level fields
            person_dict.pop("version", None)
            person_dict.pop("coordinate_system", None)
            people_data.append(person_dict)

        data = {
            "version": self.VERSION,
            "coordinate_system": self.COORDINATE_SYSTEM,
            "people": people_data,
        }

        if metadata:
            for key, value in metadata.items():
                if key not in data:
                    data[key] = value

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        return str(output_path)
