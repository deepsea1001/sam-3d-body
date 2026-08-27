"""Load bind pose definitions from various formats."""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Union, Optional


class BindPoseLoader:
    """Load and normalize bind pose definitions.

    Supports multiple input formats:
    1. Named joints JSON: {"joints": {"joint_name": [x, y, z], ...}}
    2. Web viewer JSON: {"people": [{"keypoints": [[x,y,z], ...]}]}
    3. Dict with direct joint positions
    """

    # Default bind pose path relative to project root
    DEFAULT_BIND_POSE = "data/bind_poses/default_human.json"

    @staticmethod
    def from_json(path: Union[str, Path]) -> Dict[str, np.ndarray]:
        """Load bind pose from JSON file.

        Args:
            path: Path to JSON file

        Returns:
            Dict mapping joint name -> position [x, y, z]

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If format not recognized
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Bind pose file not found: {path}")

        with open(path) as f:
            data = json.load(f)

        return BindPoseLoader.from_dict(data)

    @staticmethod
    def from_dict(data: dict) -> Dict[str, np.ndarray]:
        """Parse bind pose from dict.

        Supports formats:
        1. {"joints": {"name": [x, y, z], ...}}
        2. {"people": [{"keypoints": [[x,y,z], ...]}]}
        3. Direct dict {"name": [x, y, z], ...}

        Args:
            data: Dict with bind pose data

        Returns:
            Dict mapping joint name -> position numpy array

        Raises:
            ValueError: If format not recognized
        """
        positions = {}

        # Format 1: Named joints (preferred)
        if "joints" in data:
            for name, pos in data["joints"].items():
                positions[name] = np.array(pos, dtype=float)
            return positions

        # Format 2: Web viewer format with keypoints
        if "people" in data and data["people"]:
            person = data["people"][0]
            if "keypoints" in person:
                # Need to convert keypoints to named joints
                # This requires the MHR70 retargeter's joint mapping
                keypoints = np.array(person["keypoints"], dtype=float)
                return BindPoseLoader._keypoints_to_joints(keypoints)

        # Format 3: Direct dict (no "joints" wrapper)
        if data and isinstance(next(iter(data.values())), (list, tuple)):
            for name, pos in data.items():
                if isinstance(pos, (list, tuple)) and len(pos) == 3:
                    positions[name] = np.array(pos, dtype=float)
            if positions:
                return positions

        raise ValueError(
            "Unrecognized bind pose format. Expected 'joints' dict, "
            "'people' with 'keypoints', or direct joint->position dict."
        )

    @staticmethod
    def _keypoints_to_joints(
        keypoints: np.ndarray, coordinate_system: str = "yup"
    ) -> Dict[str, np.ndarray]:
        """Convert raw MHR70 keypoints to named joint positions.

        This method computes derived joints (root, spine, etc.) from
        the 70 keypoints.

        Args:
            keypoints: (70, 3) array of keypoint positions
            coordinate_system: Coordinate system of keypoints

        Returns:
            Dict mapping joint name -> position
        """
        # Import here to avoid circular dependency
        from ..retargeters.mhr70_retargeter import MHR70Retargeter

        return MHR70Retargeter.compute_joint_positions_static(
            keypoints, coordinate_system=coordinate_system
        )

    @staticmethod
    def get_default_bind_pose(project_root: Optional[Path] = None) -> Dict[str, np.ndarray]:
        """Load the default human T-pose bind pose.

        Args:
            project_root: Project root directory (auto-detected if None)

        Returns:
            Dict mapping joint name -> position
        """
        if project_root is None:
            # Try to find project root by looking for CLAUDE.md or .git
            current = Path(__file__).resolve()
            for parent in current.parents:
                if (parent / "CLAUDE.md").exists() or (parent / ".git").exists():
                    project_root = parent
                    break

        if project_root is None:
            project_root = Path.cwd()

        default_path = project_root / BindPoseLoader.DEFAULT_BIND_POSE
        return BindPoseLoader.from_json(default_path)
