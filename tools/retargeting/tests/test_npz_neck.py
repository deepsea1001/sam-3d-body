"""The mannequin's neck driven by the MHR model's own c_neck rotation.

Until task-relpeak the neck carried only a Y-twist half-bridge: half the
chest->head rotation's Y component, with the head's world-exact anchor
absorbing everything else. That construction was built for rows with NO
rotations, where twist is all the landmarks can support -- but MHR's c_neck
holds 20-50 deg of real local bend on deep folds (21.9 deg on the pike
fixture row: folds are heavily cervical), and the bridge dropped all of it
into the head's local, where it reads as a hinged skull instead of a bent
neck.

On rows WITH `mhr_rots` the neck now takes the model's own c_neck rotation
CHEST-RELATIVE -- `A[neck] = A[spine_2] . conj(Delta(c_spine3)) .
Delta(c_neck)` -- the exact pattern the clavicles use, and well posed for
the same reason: MHR's c_neck hangs off c_spine3 precisely as the
mannequin's neck hangs off spine_2. Because the transfer is chest-relative
it is SOURCE-INDEPENDENT: the same neck local under rel_peak, rel_total, or
any chest. The head anchor is untouched -- head world orientation still
comes from the nose + eye line, and its local absorbs the new neck exactly
as it absorbed the bridge.

Rows without the blob keep the Y-bridge bit-identically; test_neck_head.py
keeps pinning that path's purity, and the bit-identity is asserted here
against an independent reconstruction of the bridge.

Quaternions are the solver's [w,x,y,z] throughout.
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

PIKE_ROW = "0693dd37755e0d6eb9012857f5e3b405"   # rel c_spine3->c_neck = 21.9 deg
DEV_ROW = "a6566802a6c9ddd63340ccb4520e0001"
MHR_SPINE3 = 37

RIG = load_rig()
I = RIG.index_of_name
WREST = fk_world_orientations(RIG, RIG.rest_local_q)
MHR_NECK = load_mhr_rest()["names"].index("c_neck")

_conj = QM.conjugate


def _ang(q):
    q = np.asarray(q, float)
    if q[0] < 0:
        q = -q
    return float(np.degrees(2 * np.arctan2(np.linalg.norm(q[1:]), q[0])))


def _rots(row_id):
    return np.asarray(ROWS[row_id]["joint_global_rots"], float)


def _targets(row_id):
    return rig_targets_from_mhr70(
        np.asarray(ROWS[row_id]["kp70"], np.float32).reshape(70, 3))


def _anchors(row_id, mhr_rots):
    return PG._anchor_deltas(RIG, _targets(row_id), WREST, mhr_rots)


def _rel_neck(rots):
    """The transferred quantity, straight off the fixture blob."""
    return QM.multiply(_conj(PG._mhr_delta_q(rots, MHR_SPINE3)),
                       PG._mhr_delta_q(rots, MHR_NECK))


def _bridge_neck(A):
    """The v15 Y-twist half-bridge, reconstructed independently of the
    solver: kappa is the Y component of the chest->head rotation expressed
    in the neck's rest frame, and the neck takes half of it."""
    nk = I["neck"]
    r_ln = QM.multiply(
        _conj(WREST[nk]),
        QM.multiply(_conj(A[I["spine_2"]]), QM.multiply(A[I["head"]], WREST[nk])))
    if r_ln[0] < 0:
        r_ln = -np.asarray(r_ln, float)
    kappa = 2.0 * float(np.arctan2(r_ln[2], r_ln[0]))
    half = np.array([np.cos(kappa / 4.0), 0.0, np.sin(kappa / 4.0), 0.0])
    return QM.multiply(A[I["spine_2"]],
                       QM.multiply(WREST[nk], QM.multiply(half, _conj(WREST[nk]))))


# --------------------------------------------------------------------------
# 1. The transfer, as an invariant of the anchor dict.
# --------------------------------------------------------------------------

def test_neck_is_the_models_c_neck_chest_relative_under_both_sources(
        monkeypatch):
    """`conj(A[spine_2]) . A[neck]` equals `conj(Delta(37)) . Delta(c_neck)`
    to 1e-9 -- under BOTH rel_peak and rel_total, because the transfer is
    chest-relative and cancels whichever chest the spine source built
    (source-independence, the same algebra the clavicles rely on).

    Positive controls: the transferred quantity is a real rotation on these
    rows (not a vacuous identity), and the WRONG rows are visibly wrong --
    c_head (what a careless "neck" mapping would grab; 50.7/20.9 deg off on
    these rows) on both, and c_neck's own procedural twist neighbours
    (c_neck_twist{0,1}_proc, rows 112/111) on the dev row. The twist procs
    carry a FRACTION of c_neck's twist and nothing else, so they separate
    only where the neck actually twists -- 3.5/6.9 deg on the dev row,
    0.2/0.4 on the pike, whose neck bend is almost pure flexion -- which is
    why the pike row cannot host that half of the control."""
    controls = {PIKE_ROW: ("c_head",),
                DEV_ROW: ("c_head", "c_neck_twist1_proc", "c_neck_twist0_proc")}
    names = load_mhr_rest()["names"]
    for row_id in (PIKE_ROW, DEV_ROW):
        rots = _rots(row_id)
        want = _rel_neck(rots)
        assert _ang(want) > 5.0, f"{row_id[:8]}: nothing to transfer"
        for source in (PG.SPINE_SOURCE_REL_PEAK, PG.SPINE_SOURCE_REL_TOTAL):
            monkeypatch.setattr(PG, "SPINE_SOURCE", source)
            A = _anchors(row_id, rots)
            got = QM.multiply(_conj(A[I["spine_2"]]), A[I["neck"]])
            d = _ang(QM.multiply(got, _conj(want)))
            assert d < 1e-9, f"{row_id[:8]} {source}: neck off by {d:.3e} deg"
            for other_name in controls[row_id]:
                alt = QM.multiply(_conj(PG._mhr_delta_q(rots, MHR_SPINE3)),
                                  PG._mhr_delta_q(rots, names.index(other_name)))
                assert _ang(QM.multiply(got, _conj(alt))) > 1.0, (
                    f"{row_id[:8]}: c_neck and {other_name} are "
                    f"indistinguishable here")


def test_the_neck_local_is_source_independent(monkeypatch):
    """The consequence of chest-relativity one derivative on: the neck's
    LOCAL is bit-for-bit the same under rel_peak and rel_total even where
    their chests differ by 20 deg -- `conj(W(spine_2)) . W(neck)` cancels
    the chest algebraically, exactly as the clavicle transfer's local does.

    Positive control: the chests really do differ on this row, so the
    equality is a property of the transfer and not of the sources
    agreeing."""
    monkeypatch.setattr(PG, "SPINE_SOURCE", PG.SPINE_SOURCE_REL_TOTAL)
    Lt = solve_rig_locals(RIG, _targets(PIKE_ROW), mhr_rots=_rots(PIKE_ROW))
    monkeypatch.setattr(PG, "SPINE_SOURCE", PG.SPINE_SOURCE_REL_PEAK)
    Lp = solve_rig_locals(RIG, _targets(PIKE_ROW), mhr_rots=_rots(PIKE_ROW))
    assert np.allclose(Lt[I["neck"]], Lp[I["neck"]], atol=1e-9), \
        "the neck local moved with the spine source"
    assert not np.allclose(Lt[I["spine_2"]], Lp[I["spine_2"]], atol=1e-3), \
        "positive control: the two chests agree on the pike row"


# --------------------------------------------------------------------------
# 2. The fallback: no blob, the Y-bridge, bit-identically.
# --------------------------------------------------------------------------

def test_no_npz_rows_keep_the_y_bridge_bit_identically():
    """`mhr_rots=None` still produces the v15 Y-twist half-bridge EXACTLY,
    asserted against an independent reconstruction of that construction --
    so the new branch demonstrably did not touch the fallback.

    Positive control in the same test: on the same row WITH rotations the
    neck is NOT the bridge (21.9 deg of real cervical bend replaces the
    twist-only construction), so equality on the None path cannot mean the
    branch never switched."""
    A_none = _anchors(PIKE_ROW, None)
    want = _bridge_neck(A_none)
    d = _ang(QM.multiply(A_none[I["neck"]], _conj(want)))
    assert d < 1e-9, f"the None-path neck is {d:.3e} deg off the Y-bridge"

    A_npz = _anchors(PIKE_ROW, _rots(PIKE_ROW))
    moved = _ang(QM.multiply(A_npz[I["neck"]],
                             _conj(_bridge_neck(A_npz))))
    assert moved > 5.0, (
        f"positive control: with rotations the neck sits {moved:.2f} deg "
        f"from the bridge -- the branch did not switch")


# --------------------------------------------------------------------------
# 3. The head: world-exact before, world-exact after.
# --------------------------------------------------------------------------

def test_head_world_anchor_is_preserved_through_the_neck_swap():
    """The head's world orientation on an npz row equals the anchor
    construction's own output (`A[head] . W_rest(head)`) after FK through
    the new neck -- the head local absorbs the neck change, so the
    world-exact nose + eye anchor survives untouched.

    Positive control: the neck's WORLD moved >5 deg against the bridge on
    this row, so 'the head did not move' is a statement about absorption,
    not about a neck that stayed put."""
    rots = _rots(PIKE_ROW)
    A = _anchors(PIKE_ROW, rots)
    L = solve_rig_locals(RIG, _targets(PIKE_ROW), mhr_rots=rots)
    W = fk_world_orientations(RIG, {**RIG.rest_local_q, **L})
    want = QM.multiply(A[I["head"]], WREST[I["head"]])
    d = _ang(QM.multiply(W[I["head"]], _conj(want)))
    assert d < 1e-9, f"head world drifted {d:.3e} deg off its anchor"
    moved = _ang(QM.multiply(A[I["neck"]], _conj(_bridge_neck(A))))
    assert moved > 5.0                                       # positive control
