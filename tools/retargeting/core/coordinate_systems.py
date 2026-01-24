"""Coordinate system transformations for retargeting.

SAM 3D Body uses CV coordinates (Y-down, Z-forward).
Target mannequin uses Y-up (Maya/Blender convention).
"""

import numpy as np


class CoordinateTransform:
    """Coordinate system conversions."""

    @staticmethod
    def cv_to_yup(v: np.ndarray) -> np.ndarray:
        """Transform from CV coordinates (Y-down) to Y-up.

        CV coordinates: X-right, Y-down, Z-forward
        Y-up coordinates: X-right, Y-up, Z-forward

        Transform: [x, y, z] -> [x, -y, z]

        Note: This is a simplified transform that just flips Y.
        For Blender's Z-up convention, use cv_to_blender().

        Args:
            v: Vector or array of vectors in CV coordinates

        Returns:
            Vector(s) in Y-up coordinates
        """
        v = np.array(v, dtype=float)
        if v.ndim == 1:
            return np.array([v[0], -v[1], v[2]])
        else:
            result = v.copy()
            result[..., 1] = -result[..., 1]
            return result

    @staticmethod
    def yup_to_cv(v: np.ndarray) -> np.ndarray:
        """Transform from Y-up to CV coordinates (Y-down).

        Inverse of cv_to_yup.

        Args:
            v: Vector or array of vectors in Y-up coordinates

        Returns:
            Vector(s) in CV coordinates
        """
        return CoordinateTransform.cv_to_yup(v)  # Self-inverse

    @staticmethod
    def cv_to_blender(v: np.ndarray) -> np.ndarray:
        """Transform from CV coordinates (Y-down) to Blender (Z-up).

        CV coordinates: X-right, Y-down, Z-forward
        Blender: X-right, Y-forward, Z-up

        Transform: [x, y, z] -> [x, z, -y]

        Args:
            v: Vector or array of vectors in CV coordinates

        Returns:
            Vector(s) in Blender Z-up coordinates
        """
        v = np.array(v, dtype=float)
        if v.ndim == 1:
            return np.array([v[0], v[2], -v[1]])
        else:
            return np.stack([v[..., 0], v[..., 2], -v[..., 1]], axis=-1)

    @staticmethod
    def blender_to_cv(v: np.ndarray) -> np.ndarray:
        """Transform from Blender (Z-up) to CV coordinates (Y-down).

        Inverse of cv_to_blender.

        Args:
            v: Vector or array of vectors in Blender coordinates

        Returns:
            Vector(s) in CV coordinates
        """
        v = np.array(v, dtype=float)
        if v.ndim == 1:
            return np.array([v[0], -v[2], v[1]])
        else:
            return np.stack([v[..., 0], -v[..., 2], v[..., 1]], axis=-1)

    @staticmethod
    def transform_keypoints(
        keypoints: np.ndarray, from_system: str, to_system: str
    ) -> np.ndarray:
        """Transform keypoints between coordinate systems.

        Args:
            keypoints: Array of shape (N, 3) or (3,)
            from_system: Source system ("cv", "yup", "blender")
            to_system: Target system ("cv", "yup", "blender")

        Returns:
            Transformed keypoints

        Raises:
            ValueError: If unknown coordinate system specified
        """
        if from_system == to_system:
            return keypoints.copy()

        # Convert to CV as intermediate
        if from_system == "cv":
            intermediate = keypoints
        elif from_system == "yup":
            intermediate = CoordinateTransform.yup_to_cv(keypoints)
        elif from_system == "blender":
            intermediate = CoordinateTransform.blender_to_cv(keypoints)
        else:
            raise ValueError(f"Unknown source coordinate system: {from_system}")

        # Convert from CV to target
        if to_system == "cv":
            return intermediate
        elif to_system == "yup":
            return CoordinateTransform.cv_to_yup(intermediate)
        elif to_system == "blender":
            return CoordinateTransform.cv_to_blender(intermediate)
        else:
            raise ValueError(f"Unknown target coordinate system: {to_system}")
