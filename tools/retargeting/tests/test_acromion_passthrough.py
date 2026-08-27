"""The acromion is a PASS-THROUGH: it contributes no rotation of its own.

MHR70's arm chain is `clavicle -> acromion -> shoulder`, so the FK in
`retargeters/base.py` expresses the shoulder's LOCAL in the acromion's frame
(`local = conj(parent_world) . world`). The mannequin skeleton has no acromion
bone -- `MHR70_TO_MANNEQUIN` maps both sides to None -- and
`MannequinExporter.to_dict` is a pure name-map-and-skip, so the acromion's
local is DROPPED and the consumer composes the shoulder directly under the
clavicle.

Correct composition is

    W(shoulder) = W(clavicle) . L(acromion) . L(shoulder)

but the payload only supports

    W(shoulder) = W(clavicle) . L(shoulder)

so the two agree if and only if `L(acromion)` is identity. Before this change
it was not: measured over these same fixture rows, the discarded rotation ran
to a MEDIAN of ~95 deg (max 160.8), i.e. the same order as the elbow's own
local -- not a rounding error. Every mannequin shoulder was misoriented by it.

The fix collapses the two-link chain honestly rather than deleting a link:
the acromion inherits its parent's world frame, which makes its local exactly
identity AND re-expresses the shoulder's local relative to the CLAVICLE. The
shoulder's world orientation is unchanged; only the frame the payload reports
it in moves. Dropping an identity costs nothing, so the export becomes lossless.

Why a pass-through and not a fold into the child: the acromion KEYPOINT (67 /
68) stays in the hierarchy for future scapula work. What it does not carry is a
usable rotation -- measured against arm elevation over these rows the local
correlates r=+0.228 (left) / -0.066 (right), inconsistent in magnitude and sign
between sides, i.e. a chain artifact rather than scapulohumeral rhythm. The
acromion's POSITION is the real signal (up to 26.6% of torso length of travel);
its orientation is not, so nothing is lost by zeroing it.

Quaternions are the solver's [w,x,y,z] throughout.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from retargeting.bind_poses.loader import BindPoseLoader
from retargeting.core.math_utils import QuaternionMath as QM
from retargeting.exporters.mannequin_exporter import MHR70_TO_MANNEQUIN
from retargeting.retargeters.mhr70_retargeter import MHR70Retargeter

FIX = Path(__file__).parent / "fixtures"
ROWS = json.loads((FIX / "mhr_npz_rows.json").read_text())["rows"]
SINGLE = json.loads((FIX / "mhr70_row.json").read_text())

SIDES = ("left", "right")

# Comparing two separately-accumulated chains, not a value against an exact
# constant: `2*arccos(dot)` is ill-conditioned as dot -> 1, so ~7 quaternion
# multiplications leave a few MICROdegrees of float residue (measured 1.7e-6 to
# 4.5e-6). A millidegree is three orders above that residue and five below the
# ~95 deg defect these tests exist to catch, so the bound still fails loudly on
# any real regression. The identity check below keeps the tighter 1e-6 bound
# because it compares against a directly-assigned identity.
CHAIN_EPS_DEG = 1e-3

RETARGETER = MHR70Retargeter(BindPoseLoader.get_default_bind_pose())
PARENTS = {j: p for j, p in RETARGETER.get_hierarchy()}


def _all_rows():
    rows = [("mhr70_row", np.asarray(SINGLE["mhr70_xyz"], float).reshape(70, 3))]
    rows += [(rid, np.asarray(r["kp70"], np.float32).reshape(70, 3))
             for rid, r in ROWS.items() if "kp70" in r]
    return rows


ALL_ROWS = _all_rows()


def _quat_deg(a, b):
    d = abs(float(np.dot(np.asarray(a, float), np.asarray(b, float))))
    return float(np.degrees(2.0 * np.arccos(np.clip(d, 0.0, 1.0))))


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def _vec_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(float(np.dot(_unit(a), _unit(b))), -1.0, 1.0))))


def _chain_to(joint, skip=()):
    """Root-first list of joints from the hierarchy root down to `joint`."""
    out, j = [], joint
    while j is not None:
        if j not in skip:
            out.append(j)
        j = PARENTS.get(j)
    return list(reversed(out))


def _world_along(rotations, chain):
    q = QM.identity()
    for name in chain:
        q = QM.multiply(q, rotations[name])
    return q


@pytest.mark.parametrize("side", SIDES)
def test_acromion_has_no_mannequin_bone_to_receive_a_rotation(side):
    """The premise: the payload genuinely has nowhere to put this rotation."""
    assert MHR70_TO_MANNEQUIN[f"{side}_acromion"] is None


@pytest.mark.parametrize("side", SIDES)
def test_shoulder_hangs_off_the_acromion_in_the_solver_hierarchy(side):
    """The other half of the premise: the shoulder's local IS acromion-relative."""
    assert PARENTS[f"{side}_shoulder"] == f"{side}_acromion"
    assert PARENTS[f"{side}_acromion"] == f"{side}_clavicle"


@pytest.mark.parametrize("row_id,kp", ALL_ROWS)
@pytest.mark.parametrize("side", SIDES)
def test_acromion_local_is_identity(side, row_id, kp):
    """A pass-through joint rotates by nothing, so dropping it loses nothing."""
    rotations, _, _ = RETARGETER.retarget(kp)
    assert _quat_deg(rotations[f"{side}_acromion"], QM.identity()) < 1e-6


@pytest.mark.parametrize("row_id,kp", ALL_ROWS)
@pytest.mark.parametrize("side", SIDES)
def test_dropping_the_acromion_preserves_shoulder_world_orientation(side, row_id, kp):
    """What the exporter ships must equal what the solver actually solved."""
    rotations, _, _ = RETARGETER.retarget(kp)
    shoulder = f"{side}_shoulder"
    solved = _world_along(rotations, _chain_to(shoulder))
    shipped = _world_along(rotations, _chain_to(shoulder, skip={f"{side}_acromion"}))
    assert _quat_deg(solved, shipped) < CHAIN_EPS_DEG


class _AcromiaSolvedNormally(MHR70Retargeter):
    """The pre-fix solver: acromia fitted as ordinary joints."""

    def get_pass_through_joints(self) -> frozenset:
        return frozenset()


@pytest.mark.parametrize("row_id,kp", ALL_ROWS)
@pytest.mark.parametrize("side", SIDES)
def test_control_the_drop_really_is_lossy_without_the_pass_through(side, row_id, kp):
    """Falsifiability control: the checks above must be able to FAIL.

    `CHAIN_EPS_DEG` is loose enough to absorb float residue, so pin that it is
    still nowhere near loose enough to absorb the defect. Solve the acromia the
    old way and the same comparison has to blow past the bound by orders of
    magnitude -- if this ever stops failing, the tests above have gone blind.
    """
    rotations, _, _ = _AcromiaSolvedNormally(
        BindPoseLoader.get_default_bind_pose()).retarget(kp)
    shoulder = f"{side}_shoulder"
    solved = _world_along(rotations, _chain_to(shoulder))
    shipped = _world_along(rotations, _chain_to(shoulder, skip={f"{side}_acromion"}))
    assert _quat_deg(solved, shipped) > 30.0


@pytest.mark.parametrize("row_id,kp", ALL_ROWS)
@pytest.mark.parametrize("side", SIDES)
def test_mannequin_chain_still_aims_the_upper_arm_at_the_elbow(side, row_id, kp):
    """The behavioural consequence: the arm points where the subject's arm points.

    Composed the way the viewer composes it -- acromion absent -- the shoulder's
    world rotation must carry the bind upper-arm direction onto the detected one.
    This is what was off by ~95 deg before the acromion became a pass-through.
    """
    rotations, _, positions = RETARGETER.retarget(kp)
    shoulder, elbow = f"{side}_shoulder", f"{side}_elbow"
    bind = RETARGETER.bind_pose

    bind_dir = bind[elbow] - bind[shoulder]
    curr_dir = positions[elbow] - positions[shoulder]
    if np.linalg.norm(bind_dir) < 1e-6 or np.linalg.norm(curr_dir) < 1e-6:
        pytest.skip("degenerate upper-arm segment in this row")

    shipped = _world_along(rotations, _chain_to(shoulder, skip={f"{side}_acromion"}))
    assert _vec_deg(QM.rotate_vector(shipped, bind_dir), curr_dir) < 1e-3
