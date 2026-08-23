"""SPINE_SOURCE_REL_TOTAL: the model's root-relative TOTAL, split 65/35.

The defect this mapping exists for, measured on row 0693dd37 (a deep pike
fold, Scott's captures -37 solver / -38 correction):

    MHR's root-relative delta at c_spine2 (row 36) is 87.8 deg.
    MHR's root-relative delta at c_spine3 (row 37) is only 66.5 deg.

The human's spine curvature is NOT monotonic along its own column on that
row, and it is not a fluke -- 7 of the 17 fixture rows are like it, and 490
of the 1800-row motion corpus (27.2%). SPINE_SOURCE_REL_PERJOINT gives each of the
mannequin's two spine bones its own MHR row, so it dumps all 87.8 deg into
spine_1 and then has to drive spine_2 BACKWARDS to land the chest on row
37: measured -20.9 deg of EXTENSION where Scott hand-posed +21.5 of
flexion. The chest lands right; the column reads wrong.

The total, meanwhile, is right: MHR's 66.5 deg against Scott's own 61.3.
REL_TOTAL therefore keeps REL_PERJOINT's chest EXACTLY -- the same
root-relative row-37 delta composed onto the pelvis we solved, bit for bit,
which is what preserves the 28.0 deg total-error result and the
spine_2->clavicle machine edge -- and distributes that total across the two
bones with the rig's own 65/35 convention instead of a second MHR row.

That convention is measured, not chosen: Scott's six spine-zeroed captures
put spine_1 at 0.650 +/- 0.002 of the total local bend across -37 deg of
extension, +76 of flexion and 51 of lateral bend. Under REL_PERJOINT the
same quantity ranges 0.598 to 0.967 over these fixture rows.

Quaternions are the solver's [w,x,y,z]; the PoseGoblin captures are
three.js [x,y,z,w] (converted at the boundary only).
"""
import json
from pathlib import Path

import numpy as np
import pytest

from retargeting.core.math_utils import QuaternionMath as QM
from retargeting.retargeters import posegoblin_rig as PG
from retargeting.retargeters.posegoblin_rig import (
    fk_world_orientations, fk_world_positions, load_rig, rig_targets_from_mhr70,
    solve_rig_locals)

ROWS = json.loads(
    (Path(__file__).parent / "fixtures" / "mhr_npz_rows.json").read_text())["rows"]

# The deep pike fold. |rel(36)| 87.8 > |rel(37)| 66.5 -- the non-monotonic
# family this mapping exists for.
PIKE_ROW = "0693dd37755e0d6eb9012857f5e3b405"
DEV_ROW = "a6566802a6c9ddd63340ccb4520e0001"
MHR_ROOT, MHR_SPINE2, MHR_SPINE3 = 1, 36, 37

RIG = load_rig()
I = RIG.index_of_name
WREST = fk_world_orientations(RIG, RIG.rest_local_q)

_conj = QM.conjugate
_unit = lambda v: np.asarray(v, float) / np.linalg.norm(v)


def _ang(q):
    q = np.asarray(q, float)
    if q[0] < 0:
        q = -q
    return float(np.degrees(2 * np.arctan2(np.linalg.norm(q[1:]), q[0])))


def _rotvec_deg(q):
    """Rotation VECTOR (axis * angle) in degrees, [w,x,y,z] in."""
    q = np.asarray(q, float)
    if q[0] < 0:
        q = -q
    n = np.linalg.norm(q[1:])
    if n < 1e-12:
        return np.zeros(3)
    return (q[1:] / n) * np.degrees(2 * np.arctan2(n, q[0]))


# Anatomical axes in the pelvis's OWN rest frame, read off the rig rather
# than assumed -- the same construction the ground-truth captures were
# characterised with (right = r_hip->l_hip = +X exactly; up = pelvis->
# spine_1; forward = right x up). Flexion is + about right.
_RIGHT = np.array([1.0, 0.0, 0.0])
_UP = QM.rotate_vector(_conj(WREST[I["pelvis"]]),
                       _unit(RIG.rest_world_p[I["spine_1"]] - RIG.rest_world_p[I["pelvis"]]))
_FORWARD = np.cross(_RIGHT, _UP)


def _rots(row_id):
    return np.asarray(ROWS[row_id]["joint_global_rots"], float)


def _solve(row_id):
    return solve_rig_locals(RIG, rig_targets_from_mhr70(
        np.asarray(ROWS[row_id]["kp70"], np.float32).reshape(70, 3)),
        mhr_rots=_rots(row_id))


def _local_delta(L, name):
    """A bone's local rotation away from its rest local -- the quantity the
    hand-posed captures were measured in, and the one the 65/35 share is a
    share OF."""
    return QM.multiply(np.asarray(L[I[name]], float),
                       _conj(RIG.rest_local_q[I[name]]))


def _flex_deg(L, name):
    return float(_rotvec_deg(_local_delta(L, name)) @ _RIGHT)


def _rel_delta(rots, row):
    """MHR's rotation at *row* RELATIVE TO ITS OWN ROOT -- exactly the
    quantity both relative mappings transfer."""
    return QM.multiply(_conj(PG._mhr_delta_q(rots, MHR_ROOT)),
                       PG._mhr_delta_q(rots, row))


def _world(L):
    return fk_world_orientations(RIG, {**RIG.rest_local_q, **L})


def _at(source, monkeypatch, row_id):
    monkeypatch.setattr(PG, "SPINE_SOURCE", source)
    return _solve(row_id)


# --------------------------------------------------------------------------
# 1. The defect, and the fix, on the row it was reported from.
# --------------------------------------------------------------------------

def test_mhr_curvature_is_non_monotonic_on_the_pike_row():
    """The premise, straight off the committed fixture: MHR's own spine
    bends MORE at c_spine2 than at c_spine3 on this row.

    Nothing downstream is meaningful if this is not true -- a two-bone
    mannequin spine can represent a monotonic column per-joint perfectly
    well. Stated with a positive control (a row where it IS monotonic) so
    the comparison cannot be passing on a broken accessor."""
    rots = _rots(PIKE_ROW)
    a36, a37 = _ang(_rel_delta(rots, MHR_SPINE2)), _ang(_rel_delta(rots, MHR_SPINE3))
    assert a36 == pytest.approx(87.8, abs=0.1)
    assert a37 == pytest.approx(66.5, abs=0.1)
    assert a36 > a37, f"|rel(36)|={a36:.1f} is not above |rel(37)|={a37:.1f}"
    # Positive control: the same measurement on a monotonic row.
    mono = _rots("1e6a7a60b4539a8fd09867d7e2fa3db6")
    assert _ang(_rel_delta(mono, MHR_SPINE2)) < _ang(_rel_delta(mono, MHR_SPINE3))
    # ...and it is not rare. Seven of seventeen fixture rows are like the pike.
    bad = sum(_ang(_rel_delta(_rots(r), MHR_SPINE2)) > _ang(_rel_delta(_rots(r), MHR_SPINE3))
              for r in ROWS)
    assert bad == 7, f"{bad}/{len(ROWS)} fixture rows non-monotonic, expected 7"


def test_the_pike_row_bends_both_spine_bones_forward(monkeypatch):
    """THE REGRESSION. spine_2 must flex FORWARD on the pike fold, not
    extend backwards.

    Scott hand-posed +39.8 / +21.5 (capture -38, correcting the solver's
    own -37). REL_TOTAL lands +43.2 / +23.3: 0.65 and 0.35 of the model's
    66.5 deg total, which the model under-reports against Scott's 61.3 by
    the usual ~19%.

    The sign on spine_2 is the whole point and is asserted separately from
    the magnitudes -- a mapping that got the numbers close with the wrong
    sign would still read as a broken back."""
    L = _at(PG.SPINE_SOURCE_REL_TOTAL, monkeypatch, PIKE_ROW)
    f1, f2 = _flex_deg(L, "spine_1"), _flex_deg(L, "spine_2")
    assert f2 > 0.0, f"spine_2 is EXTENDING at {f2:.1f} deg on a forward fold"
    assert f1 == pytest.approx(43.2, abs=0.6), f1
    assert f2 == pytest.approx(23.0, abs=0.6), f2
    # Positive control -- the defect this replaces, on the same row, same
    # measurement. Without it "spine_2 > 0" could be true of any mapping.
    P = _at(PG.SPINE_SOURCE_REL_PERJOINT, monkeypatch, PIKE_ROW)
    assert _flex_deg(P, "spine_1") == pytest.approx(87.6, abs=0.6)
    assert _flex_deg(P, "spine_2") == pytest.approx(-21.2, abs=0.6)


# --------------------------------------------------------------------------
# 2. The chest end is preserved EXACTLY -- that is what makes this a
#    redistribution and nothing more.
# --------------------------------------------------------------------------

def test_rel_total_keeps_rel_perjoints_chest_on_every_row(monkeypatch):
    """spine_2's WORLD orientation is bit-identical to REL_PERJOINT's on
    every fixture row.

    This is load-bearing twice over: the 28.0 deg mean total chest-vs-pelvis
    error that chose REL_PERJOINT is a function of spine_2 and the pelvis
    only, so it carries over unchanged; and both clavicles and the neck hang
    off spine_2's world, so the machine-edge costs are unchanged too.

    Asserted alongside its own positive control -- spine_1 really did move --
    so a monkeypatch that failed to take cannot pass this."""
    moved = []
    for row_id in ROWS:
        rel = _world(_at(PG.SPINE_SOURCE_REL_PERJOINT, monkeypatch, row_id))
        tot = _world(_at(PG.SPINE_SOURCE_REL_TOTAL, monkeypatch, row_id))
        d = _ang(QM.multiply(tot[I["spine_2"]], _conj(rel[I["spine_2"]])))
        assert d < 1e-9, f"{row_id[:8]}: chest moved {d:.3e} deg"
        moved.append(_ang(QM.multiply(tot[I["spine_1"]], _conj(rel[I["spine_1"]]))))
    assert len(moved) == 17                                   # positive control
    assert min(moved) > 1.0, f"spine_1 barely moved on some row: min {min(moved):.4f} deg"
    assert max(moved) == pytest.approx(44.5, abs=1.0), max(moved)


def test_rel_total_moves_spine_1s_world_and_spine_2s_local_and_nothing_else(monkeypatch):
    """The blast radius, as an exact set of bone INDICES.

    Everything downstream of spine_2 -- neck, head, both clavicles, both
    shoulders -- reads `A[spine_2]`, which is unchanged, so their WORLD
    orientations must be untouched and their LOCALS with them. Only
    spine_1's world moves, and spine_2's local moves to absorb it.

    Index-keyed, never name-keyed: two rig bones share the name "joint7"."""
    rel = _at(PG.SPINE_SOURCE_REL_PERJOINT, monkeypatch, PIKE_ROW)
    tot = _at(PG.SPINE_SOURCE_REL_TOTAL, monkeypatch, PIKE_ROW)
    assert set(rel) == set(tot)
    changed = {i for i in rel if not np.allclose(rel[i], tot[i], atol=1e-12)}
    assert changed == {I["spine_1"], I["spine_2"]}, \
        sorted(RIG.name[i] for i in changed ^ {I["spine_1"], I["spine_2"]})
    wrel, wtot = _world(rel), _world(tot)
    for name in ("spine_2", "neck", "head", "left_clavicle", "right_clavicle",
                 "left_shoulder", "right_shoulder"):
        d = _ang(QM.multiply(wtot[I[name]], _conj(wrel[I[name]])))
        assert d < 1e-9, f"{name}'s world moved {d:.3e} deg"
    assert _ang(QM.multiply(wtot[I["spine_1"]], _conj(wrel[I["spine_1"]]))) > 1.0


def test_the_clavicle_machine_edge_is_unchanged(monkeypatch):
    """Spot check on the machine-side cost the 6b report priced: the
    spine_2->clavicle direction, scored against MHR's own, is bit-identical
    between the two relative mappings.

    Read off FK POSITIONS rather than orientations, because that is how the
    poseforge3d harness scores it -- a spine_1 that moved could in principle
    have translated spine_2 even with its orientation fixed, and this is the
    check that says it did not."""
    P_mhr = np.asarray(ROWS[PIKE_ROW]["pred_joint_coords"], float)
    cam_to_rig = np.array([1.0, -1.0, -1.0])
    got = {}
    for src in (PG.SPINE_SOURCE_REL_PERJOINT, PG.SPINE_SOURCE_REL_TOTAL):
        P = fk_world_positions(RIG, {**RIG.rest_local_q,
                                     **_at(src, monkeypatch, PIKE_ROW)})
        for bone, row in PG._MHR_CLAVICLE_ROW.items():
            want = _unit((P_mhr[row] - P_mhr[PG._MHR_CHEST]) * cam_to_rig)
            got.setdefault(bone, []).append(
                float(np.dot(_unit(P[I[bone]] - P[I["spine_2"]]), want)))
    for bone, (a, b) in got.items():
        assert a == pytest.approx(b, abs=1e-12), f"{bone}: {a} -> {b}"
        assert 0.0 < a < 1.0                                  # positive control


# --------------------------------------------------------------------------
# 3. The distribution: Scott's rig convention, by construction.
# --------------------------------------------------------------------------

def test_spine_1_takes_exactly_65_percent_of_the_local_bend(monkeypatch):
    """spine_1's share of the total local bend is _SPINE1_SHARE on EVERY
    row, to machine precision -- lateral bend and twist included, not just
    flexion.

    That is what a geodesic split of one rotation buys: both bones turn
    about the same axis, so the angles are 0.65 and 0.35 of the total by
    construction rather than by fit. Scott's spine-zeroed captures measure
    0.650 +/- 0.002 across every axis, which is what makes this the right
    construction rather than merely a tidy one.

    Positive control: the mapping this replaces has no such property -- its
    share ranges 0.598 to 0.967 over the same rows."""
    shares, ref = [], []
    for row_id in ROWS:
        for src, sink in ((PG.SPINE_SOURCE_REL_TOTAL, shares),
                          (PG.SPINE_SOURCE_REL_PERJOINT, ref)):
            L = _at(src, monkeypatch, row_id)
            a1, a2 = _ang(_local_delta(L, "spine_1")), _ang(_local_delta(L, "spine_2"))
            assert a1 + a2 > 5.0, f"{row_id[:8]}: {src} bend too small to divide"
            sink.append(a1 / (a1 + a2))
    assert len(shares) == 17                                  # positive control
    assert max(abs(s - PG._SPINE1_SHARE) for s in shares) < 1e-9, \
        f"share range {min(shares):.9f}..{max(shares):.9f}"
    assert max(abs(s - PG._SPINE1_SHARE) for s in ref) > 0.1, \
        "REL_PERJOINT already splits 65/35 -- this test proves nothing"


def test_the_split_follows_the_share_constant(monkeypatch):
    """The 0.65 is READ, not hardcoded twice: move `_SPINE1_SHARE` and the
    solved split moves with it, to machine precision.

    Without this the test above pins a number that could have been baked
    into the branch, and the one-constant revert the design promises would
    not be real."""
    monkeypatch.setattr(PG, "_SPINE1_SHARE", 0.5)
    L = _at(PG.SPINE_SOURCE_REL_TOTAL, monkeypatch, PIKE_ROW)
    a1, a2 = _ang(_local_delta(L, "spine_1")), _ang(_local_delta(L, "spine_2"))
    assert a1 / (a1 + a2) == pytest.approx(0.5, abs=1e-9)
    # 1e-6 rather than 1e-9 on the raw angles: the two are read back through
    # different lengths of the rig's parent chain, which costs about 1e-8 deg
    # of composition error. The RATIO above is the tight statement.
    assert a1 == pytest.approx(a2, abs=1e-6)


def test_the_total_is_the_models_root_relative_row_37(monkeypatch):
    """Where the total comes from, as a contract: the local bend spine_1 and
    spine_2 divide between them is EXACTLY MHR's root-relative rotation at
    c_spine3 -- the same quantity REL_PERJOINT hands to spine_2 alone.

    Positive controls on both halves: rows 35 and 36 are visibly different
    numbers on this row, and so is the ABSOLUTE (non-root-relative) form."""
    rots = _rots(PIKE_ROW)
    L = _at(PG.SPINE_SOURCE_REL_TOTAL, monkeypatch, PIKE_ROW)
    total = _ang(_local_delta(L, "spine_1")) + _ang(_local_delta(L, "spine_2"))
    assert total == pytest.approx(_ang(_rel_delta(rots, MHR_SPINE3)), abs=1e-6)
    for other in (35, MHR_SPINE2):
        assert abs(total - _ang(_rel_delta(rots, other))) > 5.0
    assert abs(total - _ang(PG._mhr_delta_q(rots, MHR_SPINE3))) > 5.0
