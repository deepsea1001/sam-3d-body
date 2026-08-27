"""Pure quaternion and vector math operations for retargeting.

All quaternions use [w, x, y, z] format internally.
"""

import numpy as np
from typing import Tuple


class QuaternionMath:
    """Pure quaternion operations using [w, x, y, z] format."""

    @staticmethod
    def identity() -> np.ndarray:
        """Return identity quaternion [1, 0, 0, 0]."""
        return np.array([1.0, 0.0, 0.0, 0.0])

    @staticmethod
    def multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions q1 * q2.

        Args:
            q1: First quaternion [w, x, y, z]
            q2: Second quaternion [w, x, y, z]

        Returns:
            Product quaternion [w, x, y, z]
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return np.array([w, x, y, z])

    @staticmethod
    def conjugate(q: np.ndarray) -> np.ndarray:
        """Return conjugate of quaternion (inverse for unit quaternions).

        Args:
            q: Quaternion [w, x, y, z]

        Returns:
            Conjugate quaternion [w, -x, -y, -z]
        """
        return np.array([q[0], -q[1], -q[2], -q[3]])

    @staticmethod
    def normalize(q: np.ndarray) -> np.ndarray:
        """Normalize quaternion to unit length.

        Args:
            q: Quaternion [w, x, y, z]

        Returns:
            Unit quaternion
        """
        length = np.linalg.norm(q)
        if length < 1e-10:
            return QuaternionMath.identity()
        return q / length

    @staticmethod
    def from_two_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """Compute quaternion that rotates v1 to align with v2.

        Args:
            v1: Source vector [x, y, z]
            v2: Target vector [x, y, z]

        Returns:
            Rotation quaternion [w, x, y, z]
        """
        v1 = np.array(v1, dtype=float)
        v2 = np.array(v2, dtype=float)

        len1 = np.linalg.norm(v1)
        len2 = np.linalg.norm(v2)

        if len1 < 1e-6 or len2 < 1e-6:
            return QuaternionMath.identity()

        v1 = v1 / len1
        v2 = v2 / len2

        dot = np.dot(v1, v2)

        # Nearly parallel vectors
        if dot > 0.9999:
            return QuaternionMath.identity()

        # Nearly anti-parallel vectors - rotate 180 degrees around perpendicular axis
        if dot < -0.9999:
            # Find perpendicular axis
            perp = np.array([1, 0, 0]) if abs(v1[0]) < 0.9 else np.array([0, 1, 0])
            axis = np.cross(v1, perp)
            axis = axis / np.linalg.norm(axis)
            return np.array([0, axis[0], axis[1], axis[2]])

        # General case: axis = cross(v1, v2), angle = arccos(dot)
        axis = np.cross(v1, v2)

        # Efficient quaternion construction: w = 1 + dot, xyz = cross
        w = 1 + dot
        quat = np.array([w, axis[0], axis[1], axis[2]])
        return quat / np.linalg.norm(quat)

    @staticmethod
    def from_matrix(m: np.ndarray) -> np.ndarray:
        """Rotation matrix -> quaternion [w, x, y, z] (Shepperd's method).

        Branching on the largest diagonal term keeps the divisor away from
        zero; the naive w-first form loses precision near 180 degrees.
        """
        m = np.asarray(m, dtype=float)
        t = m[0, 0] + m[1, 1] + m[2, 2]
        if t > 0.0:
            s = 0.5 / np.sqrt(t + 1.0)
            q = np.array([0.25 / s,
                          (m[2, 1] - m[1, 2]) * s,
                          (m[0, 2] - m[2, 0]) * s,
                          (m[1, 0] - m[0, 1]) * s])
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            q = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s,
                          (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
        elif m[1, 1] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s,
                          0.25 * s, (m[1, 2] + m[2, 1]) / s])
        else:
            s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s,
                          (m[1, 2] + m[2, 1]) / s, 0.25 * s])
        return QuaternionMath.normalize(q)

    @staticmethod
    def rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Rotate vector v by quaternion q.

        Args:
            q: Rotation quaternion [w, x, y, z]
            v: Vector [x, y, z]

        Returns:
            Rotated vector [x, y, z]
        """
        # Convert vector to quaternion with w=0
        v_quat = np.array([0, v[0], v[1], v[2]])

        # Rotate: q * v * q^-1
        q_conj = QuaternionMath.conjugate(q)
        result = QuaternionMath.multiply(
            QuaternionMath.multiply(q, v_quat), q_conj
        )

        return np.array([result[1], result[2], result[3]])

    @staticmethod
    def slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
        """Spherical linear interpolation between two quaternions.

        Args:
            q1: Start quaternion [w, x, y, z]
            q2: End quaternion [w, x, y, z]
            t: Interpolation factor [0, 1]

        Returns:
            Interpolated quaternion [w, x, y, z]
        """
        q1 = QuaternionMath.normalize(q1)
        q2 = QuaternionMath.normalize(q2)

        dot = np.dot(q1, q2)

        # Ensure shortest path
        if dot < 0:
            q2 = -q2
            dot = -dot

        # Linear interpolation for very similar quaternions
        if dot > 0.9995:
            result = q1 + t * (q2 - q1)
            return QuaternionMath.normalize(result)

        # Spherical interpolation
        theta = np.arccos(np.clip(dot, -1, 1))
        sin_theta = np.sin(theta)

        s1 = np.sin((1 - t) * theta) / sin_theta
        s2 = np.sin(t * theta) / sin_theta

        return s1 * q1 + s2 * q2

    @staticmethod
    def to_threejs_dict(q: np.ndarray) -> dict:
        """Convert quaternion to Three.js format.

        Args:
            q: Quaternion [w, x, y, z]

        Returns:
            Dict with {isQuaternion: true, _x, _y, _z, _w}
        """
        return {
            "isQuaternion": True,
            "_x": float(q[1]),
            "_y": float(q[2]),
            "_z": float(q[3]),
            "_w": float(q[0]),
        }

    @staticmethod
    def from_threejs_dict(d: dict) -> np.ndarray:
        """Convert Three.js format to quaternion array.

        Args:
            d: Dict with {_x, _y, _z, _w}

        Returns:
            Quaternion [w, x, y, z]
        """
        return np.array([d["_w"], d["_x"], d["_y"], d["_z"]])

    @staticmethod
    def to_axis_angle(q: np.ndarray) -> Tuple[np.ndarray, float]:
        """Convert quaternion to axis-angle representation.

        Args:
            q: Quaternion [w, x, y, z]

        Returns:
            (axis [x, y, z], angle in radians)
        """
        q = QuaternionMath.normalize(q)
        w = q[0]
        xyz = q[1:4]

        # Handle identity quaternion
        sin_half = np.linalg.norm(xyz)
        if sin_half < 1e-10:
            return np.array([0, 1, 0]), 0.0

        angle = 2 * np.arctan2(sin_half, w)
        axis = xyz / sin_half

        return axis, float(angle)

    @staticmethod
    def from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
        """Create quaternion from axis-angle representation.

        Args:
            axis: Rotation axis [x, y, z] (will be normalized)
            angle: Rotation angle in radians

        Returns:
            Quaternion [w, x, y, z]
        """
        axis = np.array(axis, dtype=float)
        length = np.linalg.norm(axis)
        if length < 1e-10:
            return QuaternionMath.identity()

        axis = axis / length
        half_angle = angle / 2
        w = np.cos(half_angle)
        xyz = axis * np.sin(half_angle)

        return np.array([w, xyz[0], xyz[1], xyz[2]])


class VectorMath:
    """Vector operations for retargeting."""

    @staticmethod
    def normalize(v: np.ndarray) -> np.ndarray:
        """Normalize vector to unit length.

        Args:
            v: Vector [x, y, z]

        Returns:
            Unit vector, or zero vector if input is too small
        """
        v = np.array(v, dtype=float)
        length = np.linalg.norm(v)
        if length < 1e-10:
            return np.zeros(3)
        return v / length

    @staticmethod
    def project_onto_plane(v: np.ndarray, normal: np.ndarray) -> np.ndarray:
        """Project vector onto plane perpendicular to normal.

        Args:
            v: Vector to project [x, y, z]
            normal: Plane normal [x, y, z]

        Returns:
            Projected vector [x, y, z]
        """
        v = np.array(v, dtype=float)
        normal = VectorMath.normalize(normal)
        return v - np.dot(v, normal) * normal

    @staticmethod
    def signed_angle(v1: np.ndarray, v2: np.ndarray, normal: np.ndarray) -> float:
        """Compute signed angle between two vectors in a plane.

        Args:
            v1: First vector [x, y, z]
            v2: Second vector [x, y, z]
            normal: Plane normal defining positive rotation direction

        Returns:
            Signed angle in radians
        """
        v1 = VectorMath.normalize(v1)
        v2 = VectorMath.normalize(v2)
        normal = VectorMath.normalize(normal)

        dot = np.clip(np.dot(v1, v2), -1, 1)
        angle = np.arccos(dot)

        # Determine sign using cross product
        cross = np.cross(v1, v2)
        if np.dot(cross, normal) < 0:
            angle = -angle

        return float(angle)

    @staticmethod
    def swing_twist_decompose(
        q: np.ndarray, twist_axis: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Decompose quaternion into swing and twist components.

        Swing: Rotation perpendicular to twist_axis
        Twist: Rotation around twist_axis

        Args:
            q: Quaternion to decompose [w, x, y, z]
            twist_axis: Axis for twist component [x, y, z]

        Returns:
            (swing_quat, twist_quat) both as [w, x, y, z]
        """
        twist_axis = VectorMath.normalize(twist_axis)

        # Project quaternion's vector part onto twist axis
        q_xyz = q[1:4]
        projection = np.dot(q_xyz, twist_axis) * twist_axis

        # Twist quaternion: rotation around twist_axis
        twist = QuaternionMath.normalize(np.array([q[0], projection[0], projection[1], projection[2]]))

        # Swing = q * twist^-1
        twist_inv = QuaternionMath.conjugate(twist)
        swing = QuaternionMath.multiply(q, twist_inv)

        return swing, twist
