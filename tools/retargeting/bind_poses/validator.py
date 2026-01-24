"""Validate bind pose structures."""

import numpy as np
from typing import Dict, List, Optional, Set


class BindPoseValidator:
    """Validate bind pose structure and completeness."""

    @staticmethod
    def validate_structure(
        bind_pose: Dict[str, np.ndarray],
        required_joints: Optional[Set[str]] = None,
    ) -> List[str]:
        """Validate bind pose structure.

        Args:
            bind_pose: Dict mapping joint name -> position
            required_joints: Set of joint names that must be present

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        if not bind_pose:
            errors.append("Bind pose is empty")
            return errors

        # Check all values are valid
        for name, pos in bind_pose.items():
            if not isinstance(pos, np.ndarray):
                errors.append(f"Joint '{name}' position is not a numpy array")
                continue

            if pos.shape != (3,):
                errors.append(f"Joint '{name}' has invalid shape {pos.shape}, expected (3,)")
                continue

            if not np.isfinite(pos).all():
                errors.append(f"Joint '{name}' contains NaN or infinite values")

        # Check required joints
        if required_joints:
            missing = required_joints - set(bind_pose.keys())
            if missing:
                errors.append(f"Missing required joints: {sorted(missing)}")

        return errors

    @staticmethod
    def validate_hierarchy(
        bind_pose: Dict[str, np.ndarray],
        hierarchy: List[tuple],
    ) -> List[str]:
        """Validate that bind pose supports the given hierarchy.

        Args:
            bind_pose: Dict mapping joint name -> position
            hierarchy: List of (joint_name, parent_name, keypoint_idx) tuples

        Returns:
            List of validation errors
        """
        errors = []
        joint_names = set(bind_pose.keys())

        for joint_name, parent_name, _ in hierarchy:
            if joint_name not in joint_names:
                errors.append(f"Hierarchy joint '{joint_name}' not in bind pose")

            if parent_name is not None and parent_name not in joint_names:
                errors.append(f"Parent joint '{parent_name}' not in bind pose")

        return errors

    @staticmethod
    def check_bone_lengths(
        bind_pose: Dict[str, np.ndarray],
        hierarchy: List[tuple],
        min_length: float = 0.001,
    ) -> List[str]:
        """Check that all bones have reasonable length.

        Args:
            bind_pose: Dict mapping joint name -> position
            hierarchy: List of (joint_name, parent_name, keypoint_idx) tuples
            min_length: Minimum acceptable bone length (meters)

        Returns:
            List of warnings for very short bones
        """
        warnings = []

        for joint_name, parent_name, _ in hierarchy:
            if parent_name is None:
                continue

            if joint_name not in bind_pose or parent_name not in bind_pose:
                continue

            bone_vec = bind_pose[joint_name] - bind_pose[parent_name]
            length = np.linalg.norm(bone_vec)

            if length < min_length:
                warnings.append(
                    f"Very short bone: {parent_name} -> {joint_name} "
                    f"(length={length:.4f}m)"
                )

        return warnings

    @staticmethod
    def is_valid(
        bind_pose: Dict[str, np.ndarray],
        required_joints: Optional[Set[str]] = None,
    ) -> bool:
        """Check if bind pose is valid.

        Args:
            bind_pose: Dict mapping joint name -> position
            required_joints: Set of joint names that must be present

        Returns:
            True if valid, False otherwise
        """
        errors = BindPoseValidator.validate_structure(bind_pose, required_joints)
        return len(errors) == 0
