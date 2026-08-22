"""Which mapping drives the two mannequin spine bones, as a switch.

v16 task 4 shipped ONE mapping when `mhr_rots` is present: spine_1 <-
Delta(c_spine1, row 35) and spine_2 <- Delta(c_spine3, row 37). Measurement
against the hand-posed captures put that mapping BEHIND v15 on spine_2, and
suggested row 36 (c_spine2) for spine_1 -- so the module now carries three
named mappings behind `SPINE_SOURCE` instead of one hardcoded branch:

    SPINE_SOURCE_MHR     spine_1 <- Delta(35), spine_2 <- Delta(37)   (shipped)
    SPINE_SOURCE_HYBRID  spine_1 <- Delta(36), spine_2 <- v15's neck+clavicle
                         construction
    SPINE_SOURCE_V15     both bones from v15, even with mhr_rots present

Which one SHIPS is not settled here and these tests deliberately do not
settle it: the hand-posed captures were authored on top of a v15-applied
pose (the review server caches retargetVersion 15 at import), so v15's
measured error is a LOWER BOUND, not an estimate, and the instrument cannot
currently separate "v16 is worse" from "v16 differs from v15". The default
therefore stays where it shipped and the alternatives stay reachable and
tested, so the comparison is reproducible the moment uncontaminated ground
truth exists.

Quaternions are the solver's [w,x,y,z] throughout; the PoseGoblin captures
and three.js use [x,y,z,w] (converted at the boundary only).
"""
import json
from pathlib import Path

import numpy as np
import pytest

from retargeting.core.math_utils import QuaternionMath as QM
from retargeting.retargeters import posegoblin_rig as PG
from retargeting.retargeters.posegoblin_rig import (
    fk_world_orientations, load_mhr_rest, load_rig, rig_targets_from_mhr70,
    solve_rig_locals)

ROWS = json.loads(
    (Path(__file__).parent / "fixtures" / "mhr_npz_rows.json").read_text())["rows"]
DEV_ROW = "a6566802a6c9ddd63340ccb4520e0001"

RIG = load_rig()
I = RIG.index_of_name

MHR_C_SPINE1 = 35
MHR_C_SPINE2 = 36
MHR_C_SPINE3 = 37


def _wxyz_to_mat(q):
    q = np.asarray(q, float)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _quat_deg(a, b):
    d = abs(float(np.dot(np.asarray(a, float), np.asarray(b, float))))
    return float(np.degrees(2.0 * np.arccos(np.clip(d, 0.0, 1.0))))


def _rots(row_id):
    return np.asarray(ROWS[row_id]["joint_global_rots"], float)


def _kp(row_id):
    return np.asarray(ROWS[row_id]["kp70"], np.float32).reshape(70, 3)


def _solve(row_id, mhr_rots):
    return solve_rig_locals(RIG, rig_targets_from_mhr70(_kp(row_id)), mhr_rots=mhr_rots)


def _world(local_q):
    return fk_world_orientations(RIG, {**RIG.rest_local_q, **local_q})


def test_default_is_the_relative_per_joint_mapping():
    """The default is pinned, and pinned to the mapping that measured best
    on the spine-zeroed captures: relative per-joint, spine_1 from c_spine2.

    28.0 deg mean total chest-vs-pelvis error against v15's 39.3 and the
    originally-shipped absolute mapping's 38.0 -- see the module docstring
    and the task-6b report. A default that drifts silently is exactly what
    this pin exists to stop."""
    assert PG.SPINE_SOURCE == PG.SPINE_SOURCE_REL_PERJOINT
    assert PG.SPINE_PERJOINT_SPINE1_ROW == 36
    assert {PG.SPINE_SOURCE_MHR, PG.SPINE_SOURCE_HYBRID, PG.SPINE_SOURCE_V15,
            PG.SPINE_SOURCE_REAL_TOTAL, PG.SPINE_SOURCE_REL_PERJOINT} == \
        {"mhr", "hybrid", "v15", "real_total", "rel_perjoint"}


def test_unknown_spine_source_raises(monkeypatch):
    """A typo must be loud. Silently falling through to v15 would ship a
    different spine than the constant names -- and nothing downstream could
    tell (CLAUDE.md rule 1)."""
    monkeypatch.setattr(PG, "SPINE_SOURCE", "hybird")
    with pytest.raises(ValueError, match="hybird"):
        _solve(DEV_ROW, _rots(DEV_ROW))


def test_hybrid_drives_spine_1_from_c_spine2(monkeypatch):
    """Under SPINE_SOURCE_HYBRID, spine_1's WORLD orientation is exactly
    Delta(row 36) applied to its rest -- not row 35's, and not row 37's.

    Built as matrices, independent of the solver's quaternion order and sign
    conventions: W(bone) == R_pose(row) @ R_rest(row).T @ W_rest(bone)."""
    monkeypatch.setattr(PG, "SPINE_SOURCE", PG.SPINE_SOURCE_HYBRID)
    R_pose = _rots(DEV_ROW)
    R_rest = np.stack([_wxyz_to_mat(q) for q in load_mhr_rest()["q_wxyz"]])
    Wr = fk_world_orientations(RIG, RIG.rest_local_q)
    W = _world(_solve(DEV_ROW, R_pose))

    got = _wxyz_to_mat(W[I["spine_1"]])
    want = R_pose[MHR_C_SPINE2] @ R_rest[MHR_C_SPINE2].T @ _wxyz_to_mat(Wr[I["spine_1"]])
    assert np.abs(got - want).max() < 1e-6, "spine_1 is not driven by row 36"
    # Positive control: the check is row-SPECIFIC. Its neighbours must be
    # visibly wrong for the same bone, or "row 36" above means nothing.
    for other in (MHR_C_SPINE1, MHR_C_SPINE3):
        alt = R_pose[other] @ R_rest[other].T @ _wxyz_to_mat(Wr[I["spine_1"]])
        assert np.abs(got - alt).max() > 1e-3, \
            f"rows {MHR_C_SPINE2} and {other} are indistinguishable here"


def test_hybrid_keeps_spine_2_on_v15s_landmark_anchor(monkeypatch):
    """Under SPINE_SOURCE_HYBRID, spine_2's WORLD orientation is bit-for-bit
    v15's neck+clavicle construction -- while its LOCAL is NOT v15's.

    Both halves are the point. The world half says the hybrid really did
    keep v15's spine_2 anchor rather than a real delta. The local half is
    the chain coupling that makes this whole comparison un-predictable from
    a table of per-joint errors: spine_1's world orientation moved, so the
    local that holds spine_2's world FIXED had to move with it."""
    v15 = _solve(DEV_ROW, None)
    monkeypatch.setattr(PG, "SPINE_SOURCE", PG.SPINE_SOURCE_HYBRID)
    hyb = _solve(DEV_ROW, _rots(DEV_ROW))

    w15, whyb = _world(v15), _world(hyb)
    assert _quat_deg(w15[I["spine_2"]], whyb[I["spine_2"]]) < 1e-6, \
        "spine_2's world anchor is not v15's construction under the hybrid"
    moved = _quat_deg(v15[I["spine_1"]], hyb[I["spine_1"]])
    assert moved > 1.0, f"spine_1 moved only {moved:.4f} deg -- the branch did not switch"
    coupled = _quat_deg(v15[I["spine_2"]], hyb[I["spine_2"]])
    assert coupled > 1.0, (
        f"spine_2's LOCAL moved only {coupled:.4f} deg although its parent moved "
        f"{moved:.2f} deg -- the chain coupling this comparison rests on is absent")


def test_v15_source_reproduces_the_none_path_spine_exactly(monkeypatch):
    """SPINE_SOURCE_V15 with mhr_rots present == the mhr_rots=None spine,
    exactly -- while the fingers still solve from those same rotations.

    This is the fallback the decision rule reserves ("v16 ships its fingers
    and its plumbing, the spine unchanged"), so it must be a real identity
    and not merely a close one."""
    v15 = _solve(DEV_ROW, None)
    monkeypatch.setattr(PG, "SPINE_SOURCE", PG.SPINE_SOURCE_V15)
    both = _solve(DEV_ROW, _rots(DEV_ROW))

    # Index-keyed, never name-keyed: two rig bones share the name "joint7".
    changed = {i for i in v15 if not np.allclose(v15[i], both[i], atol=1e-12)}
    assert changed == set(), \
        f"v15 source changed {sorted(RIG.name[i] for i in changed)}"
    phalanges = {i for i in RIG.order
                 if ("_thumb_" in RIG.name[i] or "_finger_" in RIG.name[i])
                 and not RIG.name[i].endswith("_tip")}
    assert len(phalanges) == 30                                   # positive control
    assert set(both) - set(v15) == phalanges, \
        "the fingers stopped solving when the spine fell back to v15"


def test_the_three_sources_are_three_different_spines(monkeypatch):
    """Positive control for every monkeypatch above: the switch actually
    switches. Each pair of mappings must put spine_1 somewhere visibly
    different on a real row -- otherwise the tests above could all be
    passing against one unchanged code path."""
    got = {}
    for src in (PG.SPINE_SOURCE_MHR, PG.SPINE_SOURCE_HYBRID, PG.SPINE_SOURCE_V15):
        monkeypatch.setattr(PG, "SPINE_SOURCE", src)
        got[src] = _world(_solve(DEV_ROW, _rots(DEV_ROW)))[I["spine_1"]]
    pairs = [(a, b) for a in got for b in got if a < b]
    assert len(pairs) == 3                                        # positive control
    for a, b in pairs:
        d = _quat_deg(got[a], got[b])
        assert d > 1.0, f"{a} and {b} put spine_1 {d:.4f} deg apart -- not distinct"
