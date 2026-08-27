"""MHR70 to mannequin retargeter.

Maps 70 MHR keypoints to 54 mannequin joints using vector-based retargeting.
"""

from typing import Dict, List, Tuple, Optional, Set
import numpy as np

from .base import BaseRetargeter
from ..core.coordinate_systems import CoordinateTransform


class MHR70Retargeter(BaseRetargeter):
    """Retarget MHR70 (70 keypoints) to mannequin skeleton.

    Joint hierarchy covers:
    - Body: root, spine (3), neck, head, shoulders, elbows
    - Legs: hips, knees, ankles, feet
    - Hands: full finger articulation (21 joints per hand)
    """

    # Joint hierarchy: (joint_name, parent_name, keypoint_idx or None for computed)
    JOINT_HIERARCHY = [
        # Spine
        ("root", None, None),  # computed: midpoint of hips
        ("spine1", "root", None),  # computed: 1/3 from root to neck
        ("spine2", "spine1", None),  # computed: 2/3 from root to neck
        ("neck", "spine2", 69),  # pit of neck

        # Head
        ("head", "neck", None),  # computed: midpoint of ears
        ("left_ear", "head", 3),
        ("right_ear", "head", 4),
        ("left_eye", "head", 1),
        ("right_eye", "head", 2),
        ("nose", "head", 0),

        # Left arm chain
        ("left_clavicle", "neck", None),  # computed: 5% from neck to acromion
        ("left_acromion", "left_clavicle", 67),
        ("left_shoulder", "left_acromion", 5),
        ("left_elbow", "left_shoulder", 7),
        ("left_olecranon", "left_elbow", 63),
        ("left_cubital_fossa", "left_elbow", 65),
        ("left_wrist", "left_elbow", 62),

        # Left hand
        ("left_thumb_third", "left_wrist", 45),
        ("left_thumb_second", "left_thumb_third", 44),
        ("left_thumb_first", "left_thumb_second", 43),
        ("left_thumb_tip", "left_thumb_first", 42),
        ("left_index_third", "left_wrist", 49),
        ("left_index_second", "left_index_third", 48),
        ("left_index_first", "left_index_second", 47),
        ("left_index_tip", "left_index_first", 46),
        ("left_middle_third", "left_wrist", 53),
        ("left_middle_second", "left_middle_third", 52),
        ("left_middle_first", "left_middle_second", 51),
        ("left_middle_tip", "left_middle_first", 50),
        ("left_ring_third", "left_wrist", 57),
        ("left_ring_second", "left_ring_third", 56),
        ("left_ring_first", "left_ring_second", 55),
        ("left_ring_tip", "left_ring_first", 54),
        ("left_pinky_third", "left_wrist", 61),
        ("left_pinky_second", "left_pinky_third", 60),
        ("left_pinky_first", "left_pinky_second", 59),
        ("left_pinky_tip", "left_pinky_first", 58),

        # Right arm chain
        ("right_clavicle", "neck", None),  # computed: 5% from neck to acromion
        ("right_acromion", "right_clavicle", 68),
        ("right_shoulder", "right_acromion", 6),
        ("right_elbow", "right_shoulder", 8),
        ("right_olecranon", "right_elbow", 64),
        ("right_cubital_fossa", "right_elbow", 66),
        ("right_wrist", "right_elbow", 41),

        # Right hand
        ("right_thumb_third", "right_wrist", 24),
        ("right_thumb_second", "right_thumb_third", 23),
        ("right_thumb_first", "right_thumb_second", 22),
        ("right_thumb_tip", "right_thumb_first", 21),
        ("right_index_third", "right_wrist", 28),
        ("right_index_second", "right_index_third", 27),
        ("right_index_first", "right_index_second", 26),
        ("right_index_tip", "right_index_first", 25),
        ("right_middle_third", "right_wrist", 32),
        ("right_middle_second", "right_middle_third", 31),
        ("right_middle_first", "right_middle_second", 30),
        ("right_middle_tip", "right_middle_first", 29),
        ("right_ring_third", "right_wrist", 36),
        ("right_ring_second", "right_ring_third", 35),
        ("right_ring_first", "right_ring_second", 34),
        ("right_ring_tip", "right_ring_first", 33),
        ("right_pinky_third", "right_wrist", 40),
        ("right_pinky_second", "right_pinky_third", 39),
        ("right_pinky_first", "right_pinky_second", 38),
        ("right_pinky_tip", "right_pinky_first", 37),

        # Left leg
        ("left_hip", "root", 9),
        ("left_knee", "left_hip", 11),
        ("left_ankle", "left_knee", 13),
        ("left_heel", "left_ankle", 17),
        ("left_big_toe", "left_ankle", 15),
        ("left_small_toe", "left_ankle", 16),

        # Right leg
        ("right_hip", "root", 10),
        ("right_knee", "right_hip", 12),
        ("right_ankle", "right_knee", 14),
        ("right_heel", "right_ankle", 20),
        ("right_big_toe", "right_ankle", 18),
        ("right_small_toe", "right_ankle", 19),
    ]

    # Required joints that must exist in bind pose
    REQUIRED_JOINTS: Set[str] = {
        "root", "spine1", "spine2", "neck", "head",
        "left_shoulder", "left_elbow", "left_wrist",
        "right_shoulder", "right_elbow", "right_wrist",
        "left_hip", "left_knee", "left_ankle",
        "right_hip", "right_knee", "right_ankle",
    }

    def __init__(self, bind_pose: Dict[str, np.ndarray]):
        """Initialize MHR70 retargeter.

        Args:
            bind_pose: Dict mapping joint names to 3D positions (Y-up)
        """
        super().__init__(bind_pose)
        self._children_map = self._build_children_map()

    def _build_children_map(self) -> Dict[str, List[str]]:
        """Build mapping from joint to its children."""
        children = {}
        for joint, parent, _ in self.JOINT_HIERARCHY:
            if parent is not None:
                if parent not in children:
                    children[parent] = []
                children[parent].append(joint)
        return children

    def validate_bind_pose(self):
        """Ensure bind pose has required joints."""
        missing = self.REQUIRED_JOINTS - set(self.bind_pose.keys())
        if missing:
            raise ValueError(f"Bind pose missing required joints: {sorted(missing)}")

    def get_hierarchy(self) -> List[Tuple[str, Optional[str]]]:
        """Get joint hierarchy for FK traversal."""
        return [(j, p) for j, p, _ in self.JOINT_HIERARCHY]

    def get_children_map(self) -> Dict[str, List[str]]:
        """Get mapping from joint to its children."""
        return self._children_map

    def get_pass_through_joints(self) -> frozenset:
        """The acromia carry no rotation the mannequin can receive.

        `MHR70_TO_MANNEQUIN` maps both acromia to None because the mannequin
        skeleton wires the girdle `clavicle -> shoulder` direct. Solving them
        as ordinary joints put the shoulder's local in a frame the payload
        never carries, so the exporter's skip silently cost a MEDIAN ~95 deg
        (see test_acromion_passthrough.py). As pass-throughs their local is
        identity and the shoulder comes out clavicle-relative, so the skip is
        lossless and the shoulder's world orientation is unchanged.

        The joints stay in the hierarchy: keypoints 67/68 still place them, and
        the acromion POSITION is the usable scapula signal. Only the artifact
        rotation fitted to the short, noisy acromion->shoulder segment goes.
        """
        return frozenset({"left_acromion", "right_acromion"})

    def compute_joint_positions(
        self, keypoints_3d: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Compute joint positions from MHR70 keypoints.

        Transforms from CV coordinates (Y-down) to Y-up.

        Args:
            keypoints_3d: (70, 3) array of keypoint positions in CV coordinates

        Returns:
            Dict mapping joint name -> 3D position in Y-up coordinates
        """
        return self.compute_joint_positions_static(
            keypoints_3d, coordinate_system="cv"
        )

    @staticmethod
    def compute_joint_positions_static(
        keypoints_3d: np.ndarray,
        coordinate_system: str = "cv",
    ) -> Dict[str, np.ndarray]:
        """Compute joint positions from MHR70 keypoints (static method).

        Can be called without an instance, useful for bind pose loading.

        Args:
            keypoints_3d: (70, 3) array of keypoint positions
            coordinate_system: Input coordinate system ("cv" or "yup")

        Returns:
            Dict mapping joint name -> 3D position in Y-up coordinates
        """
        kps = np.array(keypoints_3d)

        # Transform to Y-up if needed
        if coordinate_system == "cv":
            kps = CoordinateTransform.cv_to_yup(kps)

        positions = {}

        # Direct keypoint mappings
        for joint_name, parent_name, kp_idx in MHR70Retargeter.JOINT_HIERARCHY:
            if kp_idx is not None:
                positions[joint_name] = kps[kp_idx].copy()

        # Computed joints
        left_hip = kps[9]
        right_hip = kps[10]
        neck = kps[69]
        left_ear = kps[3]
        right_ear = kps[4]
        left_acromion = kps[67]
        right_acromion = kps[68]

        # Root = midpoint of hips
        root = (left_hip + right_hip) / 2
        positions["root"] = root

        # Spine joints - interpolate between root and neck
        spine_vec = neck - root
        positions["spine1"] = root + spine_vec / 3
        positions["spine2"] = root + spine_vec * 2 / 3

        # Head = midpoint of ears
        positions["head"] = (left_ear + right_ear) / 2

        # Clavicles = 5% from neck toward acromion
        positions["left_clavicle"] = neck + 0.05 * (left_acromion - neck)
        positions["right_clavicle"] = neck + 0.05 * (right_acromion - neck)

        return positions

    @classmethod
    def get_all_joint_names(cls) -> List[str]:
        """Get list of all joint names in hierarchy order."""
        return [j for j, _, _ in cls.JOINT_HIERARCHY]

    @classmethod
    def get_keypoint_mapping(cls) -> Dict[str, Optional[int]]:
        """Get mapping from joint name to keypoint index.

        Returns:
            Dict mapping joint name -> keypoint index (None for computed joints)
        """
        return {j: k for j, _, k in cls.JOINT_HIERARCHY}
