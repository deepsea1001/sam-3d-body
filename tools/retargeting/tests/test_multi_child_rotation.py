"""A joint with several children must be fitted to all of them, not just one.

`compute_rotations_and_scales` aims each joint's rotation at its FIRST child.
For a joint with one child that is exactly right. For a joint with several it
leaves every other child unaimed -- and in the real MHR-70 rig that is six of
the thirteen full-weight bones: both hips (root's first child is spine1), both
forearms (elbow's is olecranon) and both clavicle->shoulder (neck's is head).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from retargeting.core.math_utils import QuaternionMath  # noqa: E402
from retargeting.retargeters.base import BaseRetargeter  # noqa: E402


class _TwoChildRig(BaseRetargeter):
    """joint -> {childA along +X, childB along +Y}. The smallest rig that can
    show the difference: a twist about childA's axis moves childB and leaves
    childA fixed, so aiming at childA alone cannot see it."""

    def validate_bind_pose(self):
        pass

    def compute_joint_positions(self, keypoints_3d):
        raise NotImplementedError("not used by this test")

    def get_hierarchy(self):
        return [("joint", None), ("childA", "joint"), ("childB", "joint")]

    def get_children_map(self):
        return {"joint": ["childA", "childB"]}


BIND = {
    "joint": np.array([0.0, 0.0, 0.0]),
    "childA": np.array([1.0, 0.0, 0.0]),
    "childB": np.array([0.0, 1.0, 0.0]),
}


def _rotated_by_90_about_x():
    """childA is ON the rotation axis, so it does not move; childB does."""
    return {
        "joint": np.array([0.0, 0.0, 0.0]),
        "childA": np.array([1.0, 0.0, 0.0]),
        "childB": np.array([0.0, 0.0, 1.0]),
    }


def test_second_child_direction_is_recovered_for_a_multi_child_joint():
    rotations, _scales = _TwoChildRig(BIND).compute_rotations_and_scales(
        _rotated_by_90_about_x()
    )
    got = QuaternionMath.rotate_vector(rotations["joint"], BIND["childB"])
    assert np.allclose(got, [0.0, 0.0, 1.0], atol=1e-6), (
        f"childB should land on +Z; the first-child fit leaves it at {got}"
    )


def test_first_child_direction_is_still_recovered():
    """The fit must not trade one child for the other."""
    rotations, _scales = _TwoChildRig(BIND).compute_rotations_and_scales(
        _rotated_by_90_about_x()
    )
    got = QuaternionMath.rotate_vector(rotations["joint"], BIND["childA"])
    assert np.allclose(got, [1.0, 0.0, 0.0], atol=1e-6)
