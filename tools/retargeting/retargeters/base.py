"""Abstract base class for skeleton retargeting."""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
import numpy as np

from ..core.math_utils import QuaternionMath, VectorMath


class BaseRetargeter(ABC):
    """Abstract base for skeleton retargeting.

    Subclasses must implement:
    - validate_bind_pose(): Ensure bind pose has required joints
    - compute_joint_positions(): Map source keypoints to target joints
    - get_hierarchy(): Return joint hierarchy for FK traversal
    """

    def __init__(self, bind_pose: Dict[str, np.ndarray]):
        """Initialize retargeter with bind pose.

        Args:
            bind_pose: Dict mapping joint names to 3D positions (Y-up)
        """
        self.bind_pose = bind_pose
        self.validate_bind_pose()

    @abstractmethod
    def validate_bind_pose(self):
        """Ensure bind pose contains required joints.

        Raises:
            ValueError: If required joints are missing
        """
        pass

    @abstractmethod
    def compute_joint_positions(
        self, keypoints_3d: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Compute target skeleton joint positions from source keypoints.

        Args:
            keypoints_3d: Source keypoints (N, 3) in CV coordinates

        Returns:
            Dict mapping joint name -> 3D position in Y-up
        """
        pass

    @abstractmethod
    def get_hierarchy(self) -> List[Tuple[str, Optional[str]]]:
        """Get joint hierarchy for FK traversal.

        Returns:
            List of (joint_name, parent_name) tuples in traversal order.
            Root joints have parent_name=None.
        """
        pass

    @abstractmethod
    def get_children_map(self) -> Dict[str, List[str]]:
        """Get mapping from joint to its children.

        Returns:
            Dict mapping joint name -> list of child joint names
        """
        pass

    def _fit_world_rotation(
        self,
        joint_name: str,
        children: List[str],
        current_positions: Dict[str, np.ndarray],
        bind_dir: np.ndarray,
        curr_dir: np.ndarray,
    ) -> np.ndarray:
        """World rotation for *joint_name*, fitted to every usable child.

        With two or more children this is the least-squares rotation carrying
        the children's bind directions onto their current ones (Kabsch/SVD).
        With one it falls back to the minimal rotation ``bind_dir -> curr_dir``:
        a single direction pair does not constrain the twist about itself, so
        an SVD fit there would invent one and pass it to every descendant.
        """
        usable = []
        for child in children:
            if child not in current_positions or child not in self.bind_pose:
                continue
            b = self.bind_pose[child] - self.bind_pose[joint_name]
            c = current_positions[child] - current_positions[joint_name]
            nb, nc = np.linalg.norm(b), np.linalg.norm(c)
            if nb < 1e-6 or nc < 1e-6:
                continue
            usable.append((b / nb, c / nc))

        if len(usable) < 2:
            return QuaternionMath.from_two_vectors(bind_dir, curr_dir)

        a = np.array([u[0] for u in usable])
        b = np.array([u[1] for u in usable])
        u_, _s, vt = np.linalg.svd(a.T @ b)
        # Reflection guard: without it a degenerate configuration can yield a
        # determinant of -1, i.e. a mirrored "rotation".
        d = np.sign(np.linalg.det(vt.T @ u_.T)) or 1.0
        return QuaternionMath.from_matrix(vt.T @ np.diag([1.0, 1.0, d]) @ u_.T)

    def compute_rotations_and_scales(
        self, current_positions: Dict[str, np.ndarray]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        """Compute local rotations and scales via vector-based FK.

        Uses the algorithm:
        1. For each joint, get bone vector to first child
        2. Compute world-space rotation from bind to current direction
        3. Convert to local rotation: local = parent_inv * world
        4. Compute scale = current_length / bind_length

        Args:
            current_positions: Current joint positions (Y-up coordinates)

        Returns:
            Tuple of:
            - rotations: Dict mapping joint name -> quaternion [w, x, y, z]
            - scales: Dict mapping joint name -> scale factor (float)
        """
        rotations = {}
        scales = {}
        world_rotations = {}

        children_map = self.get_children_map()
        hierarchy = self.get_hierarchy()

        for joint_name, parent_name in hierarchy:
            # Get first child to determine bone direction
            children = children_map.get(joint_name, [])
            first_child = children[0] if children else None

            # Leaf joints: no bone to rotate/scale
            if first_child is None:
                rotations[joint_name] = QuaternionMath.identity()
                scales[joint_name] = 1.0
                world_rotations[joint_name] = world_rotations.get(
                    parent_name, QuaternionMath.identity()
                )
                continue

            # Check positions exist
            if joint_name not in current_positions or first_child not in current_positions:
                rotations[joint_name] = QuaternionMath.identity()
                scales[joint_name] = 1.0
                world_rotations[joint_name] = world_rotations.get(
                    parent_name, QuaternionMath.identity()
                )
                continue

            if joint_name not in self.bind_pose or first_child not in self.bind_pose:
                rotations[joint_name] = QuaternionMath.identity()
                scales[joint_name] = 1.0
                world_rotations[joint_name] = world_rotations.get(
                    parent_name, QuaternionMath.identity()
                )
                continue

            # Compute bone vectors: joint -> first_child
            bind_vec = self.bind_pose[first_child] - self.bind_pose[joint_name]
            curr_vec = current_positions[first_child] - current_positions[joint_name]

            # Bone lengths for scaling
            bind_length = np.linalg.norm(bind_vec)
            curr_length = np.linalg.norm(curr_vec)

            # Scale factor: current_length / bind_length
            if bind_length > 1e-6:
                scales[joint_name] = float(curr_length / bind_length)
            else:
                scales[joint_name] = 1.0

            # Handle zero-length bones
            if bind_length < 1e-6 or curr_length < 1e-6:
                rotations[joint_name] = QuaternionMath.identity()
                world_rotations[joint_name] = world_rotations.get(
                    parent_name, QuaternionMath.identity()
                )
                continue

            # Normalize bone directions
            bind_dir = bind_vec / bind_length
            curr_dir = curr_vec / curr_length

            # Get parent's accumulated world rotation
            parent_world_rot = world_rotations.get(
                parent_name, QuaternionMath.identity()
            )

            # Compute world-space rotation. A joint with several children is
            # fitted to ALL of them: aiming only at the first leaves every
            # other child unaimed, which in the MHR-70 rig is both hips
            # (root's first child is spine1), both forearms (elbow's is
            # olecranon) and both clavicle->shoulder (neck's is head).
            world_rot = self._fit_world_rotation(
                joint_name, children, current_positions, bind_dir, curr_dir
            )

            # Local rotation = parent^-1 * world_rot
            parent_inv = QuaternionMath.conjugate(parent_world_rot)
            local_rot = QuaternionMath.multiply(parent_inv, world_rot)

            rotations[joint_name] = local_rot

            # Track world rotation for children
            world_rotations[joint_name] = world_rot

        return rotations, scales

    def retarget(
        self, keypoints_3d: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, np.ndarray]]:
        """Full retargeting pipeline.

        Args:
            keypoints_3d: Source keypoints (N, 3) in CV coordinates

        Returns:
            Tuple of:
            - rotations: Dict mapping joint name -> quaternion [w, x, y, z]
            - scales: Dict mapping joint name -> scale factor
            - positions: Dict mapping joint name -> 3D position (for debugging)
        """
        # Compute joint positions (handles coordinate transform)
        positions = self.compute_joint_positions(keypoints_3d)

        # Compute rotations and scales
        rotations, scales = self.compute_rotations_and_scales(positions)

        return rotations, scales, positions
