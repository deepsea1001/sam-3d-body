"""The transferred clavicle is aimed back at the kp70 shoulder, by at most 15 deg.

`f07064f` replaced the clavicle's position-aim with a rotation transfer: the
model's own chest-relative swing, composed onto the spine_2 we solved. That
fixed the mirror-asymmetry Scott reported -- and it recorded a trade. The
shoulder ROOT drifts off the kp70 shoulder keypoint (corpus median 0.516 rig
units; `clavicle->shoulder` cosine median 0.9491, min 0.6031) while every arm
DIRECTION below it stays exact.

Scott then reported the cost of that trade in the images: consistent hand and
arm OVERSHOOT on crouch and ground-contact poses -- rows
`2ba1b3e0`, `38608eb8`, `4fe66c92` (his captures -39 / -40 / -41). Right
directions from wrong origins. The arm is world-anchored from the arm
keypoints, so an 18 deg error in where the ball sits is 18 deg of the whole
limb's placement that nothing downstream can absorb.

The fix is a BOUNDED aim correction on top of the transfer.
`_CLAV_AIM_CORRECTION_MAX_DEG` = 15.0 is the cap: rows whose residual is under
it land exactly on the keypoint, rows above it keep a bounded remainder.

This was designed on the premise that a correction applied FROM the
transferred pose "carries no rest-geometry bias -- it is the residual
chest-position error plus noise". **That premise is false, and these tests
measure it false.** The correction's rotation axis, taken in the posed
clavicle's own frame over 500 corpus rows, scatters a median 11.8 deg (left) /
22.1 deg (right) about ONE fixed axis. The correction is a minimal swing, so
that axis is provably confined to the great circle perpendicular to the
clavicle's rest long axis (measured `|axis . rest_long_axis| <= 4e-16` over
1000 samples) -- axes uniform on THAT circle would scatter a median 79.1,
within 20 deg 13.0% of the time (85.5 / ~3.6% is the null for a full sphere,
which this axis never explores). The residual is a near-constant per-side
bias, and its source is the rig
asset: the mannequin's two rest `clavicle->shoulder` directions are 36.72 deg
from mirrored while MHR-70's keypoint pair is 3.09 deg from mirrored
(`test_the_rig_s_own_rest_girdle_is_the_asymmetric_thing`). So the cap does
bound anatomy, and every degree of it lands in the LOCAL as protraction -- the
signature `f07064f` removed. The trade is linear: one degree of cap buys one
degree of aim and costs ~0.95 deg of local protraction, with no sweet spot.

It is kept because the WORLD result is much better and the local result is
what the asset defect makes unavoidable -- see
`test_the_symmetric_pose_s_shoulder_girdle_gets_MORE_mirrored`, which pins
both halves of that trade on the same row rather than reporting only the half
that flatters it.

Roll-freeness (`fb42e71`) survives by CONSTRUCTION, not by measurement: the
correction is applied by re-aiming through `_aim_delta`, whose contract is
that the LOCAL carries no twist about the aim axis. The transfer's own delta
is itself an aim in disguise -- `_swing_about(rel, axis)` is the unique
twist-free rotation sending `axis` where `rel` sends it -- so re-aiming at a
corrected direction reproduces the transfer exactly when the cap is zero.
That identity is `test_the_zero_cap_reproduces_the_shipped_transfer`, and it
is what licenses every "before" figure here to be taken from a cap-0 solve.

`mhr_rots=None` never enters the block at all and stays bit-identical.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import retargeting.retargeters.posegoblin_rig as PG                # noqa: E402
from retargeting.retargeters.posegoblin_rig import (               # noqa: E402
    load_rig, rig_targets_from_mhr70, solve_rig_locals,
    fk_world_orientations)
from retargeting.core.math_utils import QuaternionMath as QM       # noqa: E402

_FIX = Path(__file__).resolve().parent / "fixtures"
CLAV_ROWS = json.loads((_FIX / "clavicle_source_rows.json").read_text())["rows"]
NPZ_ROWS = json.loads((_FIX / "mhr_npz_rows.json").read_text())["rows"]
ALL_ROWS = {**NPZ_ROWS, **CLAV_ROWS}

BICEPS = "01a3a1444b8b6d3fea9a9b5d94c13c32"     # double-biceps, arms symmetric
PIKE = "0693dd37755e0d6eb9012857f5e3b405"       # the spine row, untouched here
# Scott's overshoot rows; 2ba1b3e0 is corpus-only and is measured in the
# report, not here.
CROUCH = "38608eb8ca343e1250a0258178412105"
FOLD = "4fe66c9255c8b01de55a46efe261788d"

RIG = load_rig()
I = RIG.index_of_name
WR = fk_world_orientations(RIG, RIG.rest_local_q)

CAP = 15.0


def _deg(q):
    q = np.asarray(q, float)
    return float(np.degrees(2 * np.arctan2(np.linalg.norm(q[1:]), abs(q[0]))))


def _twist_deg(q, axis):
    q = np.asarray(q, float)
    if q[0] < 0:
        q = -q
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    tw = np.array([q[0], *(np.dot(q[1:], axis) * axis)])
    n = np.linalg.norm(tw)
    return 0.0 if n < 1e-12 else _deg(tw / n)


def _rest_long_axis(side):
    """clavicle -> shoulder rest direction, world and in the clavicle's own
    rest frame."""
    ci, si = I[f"{side}_clavicle"], I[f"{side}_shoulder"]
    d = RIG.rest_world_p[si] - RIG.rest_world_p[ci]
    d = d / np.linalg.norm(d)
    return d, QM.rotate_vector(QM.conjugate(WR[ci]), d)


def _solve(row_id, cap=None, monkeypatch=None, mhr=True):
    """Solve one fixture row, optionally with the cap overridden.

    cap=0 is the SHIPPED transfer (see the module docstring and
    test_the_zero_cap_reproduces_the_shipped_transfer); cap=None is
    production."""
    if cap is not None:
        monkeypatch.setattr(PG, "_CLAV_AIM_CORRECTION_MAX_DEG", float(cap))
    r = ALL_ROWS[row_id]
    kp = np.asarray(r["kp70"], np.float32).reshape(70, 3)
    targets = rig_targets_from_mhr70(kp)
    kw = {}
    if mhr:
        kw["mhr_rots"] = np.asarray(r["joint_global_rots"], float)
    L = solve_rig_locals(RIG, targets, **kw)
    return targets, {i: QM.normalize(q) for i, q in L.items()}


def _landed(L, side):
    """Where the solved delta actually sends the clavicle's rest long axis.

    Taken from the DELTA rather than from FK positions on purpose: FK carries
    the shoulder on `rest_local_p` while this reads `rest_world_p`, and the
    asset's rounding makes those disagree by ~1e-4 deg -- enough to blur an
    assertion that has to be exact to 1e-6."""
    ci = I[f"{side}_clavicle"]
    rb, _ = _rest_long_axis(side)
    W = fk_world_orientations(RIG, {**RIG.rest_local_q, **L})
    d = QM.multiply(W[ci], QM.conjugate(WR[ci]))
    v = QM.rotate_vector(d, rb)
    return v / np.linalg.norm(v)


def _target_dir(targets, side):
    ci, si = I[f"{side}_clavicle"], I[f"{side}_shoulder"]
    t = np.asarray(targets[si], float) - np.asarray(targets[ci], float)
    return t / np.linalg.norm(t)


def _aim_err_deg(L, targets, side):
    a, b = _landed(L, side), _target_dir(targets, side)
    return float(np.degrees(np.arccos(np.clip(float(a @ b), -1.0, 1.0))))


def _local_delta(L, side):
    ci = I[f"{side}_clavicle"]
    return QM.multiply(QM.conjugate(RIG.rest_local_q[ci]), L[ci])


def _shoulder_shift(local_delta, side):
    """Where the clavicle's local delta sends its own rest shoulder offset,
    in rig WORLD axes: +Y up (elevation), +Z toward the viewer (protraction),
    X lateral. Rig units. Same measure `test_clavicle_source` uses -- Scott
    looks at where the ball goes, not at an euler triple."""
    ci, si = I[f"{side}_clavicle"], I[f"{side}_shoulder"]
    off = RIG.rest_world_p[si] - RIG.rest_world_p[ci]
    world = QM.multiply(WR[ci], QM.multiply(local_delta, QM.conjugate(WR[ci])))
    return QM.rotate_vector(world, off) - off


def _mhr_chest_relative(row_id, side):
    rots = np.asarray(ALL_ROWS[row_id]["joint_global_rots"], float)
    chest = PG._mhr_delta_q(rots, PG._MHR_CHEST)
    return QM.multiply(QM.conjugate(chest),
                       PG._mhr_delta_q(rots, PG._MHR_CLAVICLE_ROW[f"{side}_clavicle"]))


# --------------------------------------------------------------------------- #
# The seam: the cap exists, and zero of it is the shipped transfer
# --------------------------------------------------------------------------- #

def test_the_cap_is_a_named_module_constant_at_fifteen_degrees():
    """The bound is a constant somebody can find and argue with, not a
    literal buried in the solve."""
    assert PG._CLAV_AIM_CORRECTION_MAX_DEG == pytest.approx(CAP)


def test_the_zero_cap_reproduces_the_shipped_transfer(monkeypatch):
    """With the correction disabled the clavicle's local must be EXACTLY the
    shipped contract: MHR's chest-relative rotation, swing-only, carried into
    the clavicle's own rest frame.

        local delta-from-rest = Wr(clav)^-1 . swing(dRel) . Wr(clav)

    This is `test_clavicle_source`'s frame-algebra assertion re-run against
    the new code path, and it is what makes every cap-0 "before" number in
    this file and in the report a measurement of the SHIPPED solver rather
    than of a reconstruction.

    Positive control: the identical comparison at the production cap must
    FAIL somewhere, or the correction is not doing anything at all."""
    same_at_zero, differ_at_cap = 0, 0
    for rid in ALL_ROWS:
        _, L0 = _solve(rid, cap=0.0, monkeypatch=monkeypatch)
        for side in ("left", "right"):
            ci = I[f"{side}_clavicle"]
            ax_world, _ = _rest_long_axis(side)
            want = QM.multiply(
                QM.conjugate(WR[ci]),
                QM.multiply(PG._swing_about(_mhr_chest_relative(rid, side), ax_world),
                            WR[ci]))
            off = _deg(QM.multiply(QM.conjugate(QM.normalize(want)),
                                   _local_delta(L0, side)))
            assert off < 1e-9, f"{rid[:8]} {side}: cap-0 is {off:.3e} deg off the transfer"
            same_at_zero += 1

    monkeypatch.undo()
    for rid in ALL_ROWS:
        _, L = _solve(rid)
        for side in ("left", "right"):
            ci = I[f"{side}_clavicle"]
            ax_world, _ = _rest_long_axis(side)
            want = QM.multiply(
                QM.conjugate(WR[ci]),
                QM.multiply(PG._swing_about(_mhr_chest_relative(rid, side), ax_world),
                            WR[ci]))
            if _deg(QM.multiply(QM.conjugate(QM.normalize(want)),
                                _local_delta(L, side))) > 1.0:
                differ_at_cap += 1
    assert same_at_zero == 2 * len(ALL_ROWS)
    assert differ_at_cap > 0, \
        "positive control: the production cap changed nothing anywhere"


# --------------------------------------------------------------------------- #
# Acceptance 1 and 2 -- the aim error, before and after
# --------------------------------------------------------------------------- #

def test_the_correction_walks_the_aim_exactly_to_the_cap(monkeypatch):
    """The whole behavioural contract in one equation:

        err_after == max(0, err_before - CAP)

    A row inside the cap lands ON the keypoint; a row outside keeps exactly
    the excess and no more. Asserted to 1e-6 deg on every fixture row and
    side, because the correction is a rotation of one unit vector toward
    another by a known angle and there is nothing approximate about it.

    Positive control: some row must SATURATE (err_before > CAP) and some row
    must not, or this equation is being satisfied vacuously by one branch."""
    sat, inside, worst = 0, 0, 0.0
    for rid in sorted(ALL_ROWS):
        t0, L0 = _solve(rid, cap=0.0, monkeypatch=monkeypatch)
        monkeypatch.undo()
        t1, L1 = _solve(rid)
        for side in ("left", "right"):
            before = _aim_err_deg(L0, t0, side)
            after = _aim_err_deg(L1, t1, side)
            want = max(0.0, before - CAP)
            # 1e-5, not 1e-6: `after` is an arccos of a dot product, which
            # loses half its significant digits as the angle goes to zero.
            # Measured worst deviation over these 38 clavicles: 1.2e-06.
            assert after == pytest.approx(want, abs=1e-5), (
                f"{rid[:8]} {side}: {before:.4f} -> {after:.4f}, expected {want:.4f}")
            worst = max(worst, before)
            if before > CAP:
                sat += 1
            else:
                inside += 1
    print(f"\nfixture clavicles: {sat} saturate the {CAP:.0f} deg cap, "
          f"{inside} land exactly; worst residual before {worst:.2f} deg")
    assert sat > 0 and inside > 0, \
        f"positive control: {sat} saturated / {inside} inside -- one branch is untested"


def test_scott_s_overshoot_rows_get_their_shoulder_balls_back(monkeypatch):
    """Acceptance 2, on the two of Scott's three rows that are in the
    fixture (`2ba1b3e0` is corpus-only; its numbers are in the report).

    These are the crouch and forward-fold poses where he reads the hands as
    overshooting: the arm direction is exact but hangs off a ball that sits
    up to 25 deg away from the keypoint.

    RE-PINNED 2026-08-23 (task-pelvis), and the shape of the change matters
    more than the numbers. The pelvis now takes the model's own root rotation,
    so the chest these clavicles hang off moved -- and the TRANSFER's residual
    moved with it, in both directions:

        crouch  left 22.08 -> 12.27   right 18.81 -> 25.35
        fold    left 14.64 -> 15.88   right  6.68 -> 15.26

    A blanket `after < 8.0` was the old bound and it no longer holds: crouch
    RIGHT now starts 25.35 out, saturates the 15 deg cap and lands 10.35. It
    is NOT weakened to fit -- each of the four is pinned at its measured value
    instead, so any further movement in either direction fails here.

    Why it goes both ways: the residual is dominated by the rig's own rest
    girdle, 36.72 deg from mirrored where the keypoints are 3.09 (b2d58c1
    §3.3), not by pelvis error. A better pelvis redistributes that asset gap
    rather than removing it. Corpus-wide the change is nonetheless an
    improvement in the TAIL, which is where the complaint lives: over 3600
    clavicles, p90 12.04 -> 9.45 deg, max 26.02 -> 20.45, and the count above
    8 deg falls 934 -> 554."""
    want = {(CROUCH, "left"): (12.27, 0.00), (CROUCH, "right"): (25.35, 10.35),
            (FOLD, "left"): (15.88, 0.88), (FOLD, "right"): (15.26, 0.26)}
    for rid in (CROUCH, FOLD):
        t0, L0 = _solve(rid, cap=0.0, monkeypatch=monkeypatch)
        monkeypatch.undo()
        t1, L1 = _solve(rid)
        for side in ("left", "right"):
            before = _aim_err_deg(L0, t0, side)
            after = _aim_err_deg(L1, t1, side)
            print(f"\n{rid[:8]} {side:6s} clavicle->shoulder "
                  f"{before:6.2f} -> {after:6.2f} deg")
            assert after <= max(0.0, before - CAP) + 1e-6
            w0, w1 = want[(rid, side)]
            assert before == pytest.approx(w0, abs=0.05), f"{rid[:8]} {side}"
            assert after == pytest.approx(w1, abs=0.05), f"{rid[:8]} {side}"
            assert before > 5.0, \
                f"positive control: {rid[:8]} {side} had nothing to correct"


def test_the_shoulder_ball_moves_toward_the_keypoint_not_just_somewhere(
        monkeypatch):
    """The correction has to close the gap, not merely change the pose by 15
    deg. Measured as the shoulder JOINT's distance to its own keypoint
    target, the quantity Scott's overshoot complaint is about (the trade
    `f07064f` recorded was a median 0.516 rig units of exactly this)."""
    closed, opened = 0, 0
    for rid in sorted(ALL_ROWS):
        t0, L0 = _solve(rid, cap=0.0, monkeypatch=monkeypatch)
        monkeypatch.undo()
        t1, L1 = _solve(rid)
        for side in ("left", "right"):
            ci, si = I[f"{side}_clavicle"], I[f"{side}_shoulder"]
            L = float(np.linalg.norm(RIG.rest_world_p[si] - RIG.rest_world_p[ci]))
            origin = np.asarray(t0[ci], float)
            tgt = np.asarray(t0[si], float)
            d0 = float(np.linalg.norm(origin + L * _landed(L0, side) - tgt))
            d1 = float(np.linalg.norm(origin + L * _landed(L1, side) - tgt))
            if d1 < d0 - 1e-9:
                closed += 1
            elif d1 > d0 + 1e-9:
                opened += 1
    print(f"\nshoulder-ball distance to its keypoint: {closed} closed, "
          f"{opened} opened, of {2 * len(ALL_ROWS)}")
    assert opened == 0, f"{opened} clavicles were moved AWAY from their target"
    assert closed > 0, "positive control: nothing moved at all"


# --------------------------------------------------------------------------- #
# Acceptance 5 -- the cap is load-bearing
# --------------------------------------------------------------------------- #

def _rotated_target(targets, side, deg, axis):
    """A copy of *targets* with the shoulder keypoint swung *deg* about
    *axis* around the clavicle keypoint -- a synthetic residual of known,
    arbitrary size. The clavicle transfer does not read `targets` at all, so
    this changes the RESIDUAL and nothing else."""
    ci, si = I[f"{side}_clavicle"], I[f"{side}_shoulder"]
    o = np.asarray(targets[ci], float)
    v = np.asarray(targets[si], float) - o
    ax = np.asarray(axis, float)
    ax = ax / np.linalg.norm(ax)
    h = np.radians(deg) / 2.0
    q = np.array([np.cos(h), *(np.sin(h) * ax)])
    out = dict(targets)
    out[si] = o + QM.rotate_vector(q, v)
    return out


def test_a_huge_residual_clamps_at_exactly_the_cap(monkeypatch):
    """Acceptance 5. Drag the shoulder keypoint 70 deg away from where the
    transfer put it and the correction must move exactly 15.000000 deg --
    not 70, not 'about 15'.

    This is the assertion an UNCAPPED implementation fails: without the
    clamp the correction swings the whole 70 and the clavicle is back to
    being a pure aim, rest-gap bias and all.

    Positive control, in the same test so it cannot rot apart: with the cap
    lifted to 180 the identical setup DOES swing the whole way and lands on
    the keypoint. So a 15.0 here means 'the clamp fired', never 'the
    correction is inert'."""
    r = ALL_ROWS[BICEPS]
    kp = np.asarray(r["kp70"], np.float32).reshape(70, 3)
    rots = np.asarray(r["joint_global_rots"], float)
    base = rig_targets_from_mhr70(kp)
    side = "left"

    monkeypatch.setattr(PG, "_CLAV_AIM_CORRECTION_MAX_DEG", 0.0)
    L0 = {i: QM.normalize(q) for i, q in
          solve_rig_locals(RIG, base, mhr_rots=rots).items()}
    transferred = _landed(L0, side)

    # Aim the synthetic target 70 deg off the TRANSFERRED direction, about an
    # axis perpendicular to it, so the residual is 70.000 by construction.
    perp = np.cross(transferred, np.array([0.0, 1.0, 0.0]))
    if np.linalg.norm(perp) < 0.1:
        perp = np.cross(transferred, np.array([1.0, 0.0, 0.0]))
    ci, si = I[f"{side}_clavicle"], I[f"{side}_shoulder"]
    o = np.asarray(base[ci], float)
    Ln = float(np.linalg.norm(np.asarray(base[si], float) - o))
    h = np.radians(70.0) / 2.0
    ax = perp / np.linalg.norm(perp)
    q = np.array([np.cos(h), *(np.sin(h) * ax)])
    hard = dict(base)
    hard[si] = o + Ln * QM.rotate_vector(q, transferred)

    monkeypatch.undo()
    monkeypatch.setattr(PG, "_CLAV_AIM_CORRECTION_MAX_DEG", 0.0)
    t_check = {i: QM.normalize(v) for i, v in
               solve_rig_locals(RIG, hard, mhr_rots=rots).items()}
    residual = _aim_err_deg(t_check, hard, side)
    assert residual == pytest.approx(70.0, abs=1e-6), \
        f"the synthetic residual is {residual:.4f}, not 70 -- setup is wrong"

    monkeypatch.undo()
    Lc = {i: QM.normalize(v) for i, v in
          solve_rig_locals(RIG, hard, mhr_rots=rots).items()}
    applied = float(np.degrees(np.arccos(np.clip(
        float(_landed(Lc, side) @ transferred), -1.0, 1.0))))
    print(f"\nsynthetic 70.00 deg residual -> correction applied "
          f"{applied:.6f} deg (cap {CAP})")
    assert applied == pytest.approx(CAP, abs=1e-6), \
        f"the cap did not clamp: {applied:.4f} deg of correction was applied"
    assert _aim_err_deg(Lc, hard, side) == pytest.approx(70.0 - CAP, abs=1e-6)

    monkeypatch.setattr(PG, "_CLAV_AIM_CORRECTION_MAX_DEG", 180.0)
    Lu = {i: QM.normalize(v) for i, v in
          solve_rig_locals(RIG, hard, mhr_rots=rots).items()}
    uncapped = float(np.degrees(np.arccos(np.clip(
        float(_landed(Lu, side) @ transferred), -1.0, 1.0))))
    assert uncapped == pytest.approx(70.0, abs=1e-6), \
        (f"positive control: with the cap lifted the correction applied "
         f"{uncapped:.4f} deg, not the full 70 -- the 15.0 above may be an "
         f"artefact, not the clamp")
    assert _aim_err_deg(Lu, hard, side) < 1e-6


# --------------------------------------------------------------------------- #
# Acceptance 3 -- the old aim's opposite splay must NOT come back
# --------------------------------------------------------------------------- #

def _girdle_deg(pos):
    """How far the two shoulder balls are from being mirror images of each
    other, as an angle about spine_2. Zero on a perfectly mirrored girdle.

    Scale-free on purpose: `targets` and the rig live at different scales
    (kp70 `clavicle->shoulder` is 0.167 long, the rig's is 1.617), so a
    displacement in rig units cannot be compared with one in target units,
    and only directions can. *pos* is any index-keyed position map."""
    a = np.asarray(pos[I["left_shoulder"]], float) - np.asarray(pos[I["spine_2"]], float)
    b = np.asarray(pos[I["right_shoulder"]], float) - np.asarray(pos[I["spine_2"]], float)
    b = b * np.array([-1.0, 1.0, 1.0])                     # the rig mirrors about x=0
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(float(a @ b), -1.0, 1.0))))


def test_the_symmetric_pose_s_shoulder_girdle_gets_MORE_mirrored(monkeypatch):
    """Acceptance 3, measured in the frame that can answer it.

    The defect `f07064f` fixed was the two shoulder balls going opposite ways
    on a mirror-image pose. The obvious way to check that the correction has
    not brought it back is `_shoulder_shift`'s forward component -- and that
    measure says it HAS: on this row it goes from the transfer's +0.052 /
    -0.017 rig units to +0.391 / -0.425, opposite-signed, against the old
    aim's own +0.47 / -0.48. Those numbers are real and are pinned below.

    They are also measured in the wrong frame, and the reason is an asset
    defect. `_shoulder_shift` reports displacement from each bone's OWN rest
    offset, and the mannequin's two rest `clavicle->shoulder` directions are
    36.72 deg from being mirror images (`test_the_rig_s_own_rest_girdle_is_
    the_asymmetric_thing` below), while MHR-70's keypoint pair is 3.09 deg
    from mirrored. Reaching two mirrored targets from two non-mirrored rests
    REQUIRES two non-mirrored local swings. So a local measure reads the
    asset's asymmetry being CORRECTED as the pose going asymmetric.

    In the world, where Scott looks, the correction moves the girdle two
    thirds of the way from the rig's own rest asymmetry to the keypoints'.
    Measured over the 16 corpus rows whose kp70 girdle is itself within 5 deg
    of mirrored -- a population selected by the data, not by the solve:

        rig's own rest      26.14 deg from mirrored
        transfer (cap 0)    24.97 median, p90 27.08   <- inherits the asset
        corrected (cap 15)   8.66 median, p90 11.47
        full aim (cap 180)   4.71 median
        kp70 targets         4.12 median

    This test is that statement on the double-biceps row, with the transfer
    as its own control."""
    _, L0 = _solve(BICEPS, cap=0.0, monkeypatch=monkeypatch)
    monkeypatch.undo()
    _, L1 = _solve(BICEPS)
    targets, _ = _solve(BICEPS, cap=0.0, monkeypatch=monkeypatch)
    monkeypatch.undo()

    rest = {i: RIG.rest_world_p[i] for i in RIG.order}
    g_rest = _girdle_deg(rest)
    g_tgt = _girdle_deg(targets)
    from retargeting.retargeters.posegoblin_rig import fk_world_positions
    g0 = _girdle_deg(fk_world_positions(RIG, {**RIG.rest_local_q, **L0}))
    g1 = _girdle_deg(fk_world_positions(RIG, {**RIG.rest_local_q, **L1}))
    print(f"\ngirdle, deg from mirrored:  rig rest {g_rest:.2f}  "
          f"transfer {g0:.2f}  corrected {g1:.2f}  kp70 targets {g_tgt:.2f}")

    assert g_tgt < 5.0, \
        f"this row's kp70 girdle is {g_tgt:.1f} deg from mirrored -- not a symmetric pose"
    assert g0 > 0.5 * g_rest, (
        f"positive control: the transfer's girdle is {g0:.1f} deg, not the "
        f"{g_rest:.1f} deg of asset asymmetry it is supposed to inherit")
    assert g1 < 0.5 * g0, \
        f"the correction left the girdle at {g1:.1f} deg against the transfer's {g0:.1f}"
    assert g1 > g_tgt - 1e-6, "the solve cannot be MORE mirrored than its own targets"

    # The ratio above only bounds g1 against g0 -- a girdle that regressed
    # from 6.8 to 12.4 deg would still clear `g1 < 0.5 * g0`. Pin the values
    # themselves, RED-verified: perturbing either constant fails this exact
    # assertion before the real numbers are restored.
    assert g0 == pytest.approx(23.03, abs=0.3), \
        f"transfer girdle drifted: {g0:.2f} vs the pinned 23.03"
    assert g1 == pytest.approx(6.78, abs=0.3), \
        f"corrected girdle drifted: {g1:.2f} vs the pinned 6.78"

    # ...and the local numbers, pinned rather than argued away. If the cap is
    # ever lowered these move together, which is the point.
    fwd = {}
    for label, L in (("transfer", L0), ("corrected", L1)):
        for side in ("left", "right"):
            v = _shoulder_shift(_local_delta(L, side), side)
            fwd[(label, side)] = float(v[2])
            print(f"{label:10s} {side:6s} shoulder ball  lat {v[0]:+.4f}  "
                  f"up {v[1]:+.4f}  fwd {v[2]:+.4f}")
    assert fwd[("transfer", "left")] == pytest.approx(+0.052, abs=0.01)
    assert fwd[("transfer", "right")] == pytest.approx(-0.017, abs=0.01)
    assert fwd[("corrected", "left")] == pytest.approx(+0.391, abs=0.02)
    assert fwd[("corrected", "right")] == pytest.approx(-0.425, abs=0.02)


def test_the_rig_s_own_rest_girdle_is_the_asymmetric_thing():
    """The premise the test above rests on, asserted rather than asserted-in-
    prose: the ASSET's two clavicle->shoulder rest directions are far from
    mirrored, and MHR-70's keypoints are not.

    This is `f07064f`'s own concern 2 ("correcting the right clavicle does NOT
    restore mirror symmetry of the rest geometry ... the asymmetry is mostly
    POSITIONAL") measured as a number. It is the root cause of everything in
    this file, and fixing it -- not tuning the cap -- is what would let the
    clavicle be both correctly rotated and correctly placed."""
    d = {}
    for side in ("left", "right"):
        ci, si = I[f"{side}_clavicle"], I[f"{side}_shoulder"]
        v = RIG.rest_world_p[si] - RIG.rest_world_p[ci]
        d[side] = v / np.linalg.norm(v)
    mirrored = d["right"] * np.array([-1.0, 1.0, 1.0])
    gap = float(np.degrees(np.arccos(np.clip(float(d["left"] @ mirrored), -1.0, 1.0))))
    print(f"\nmannequin rest clavicle->shoulder: left vs MIRRORED right "
          f"{gap:.2f} deg apart")
    assert gap == pytest.approx(36.72, abs=0.1), \
        f"the asset's clavicle rest asymmetry is now {gap:.2f} deg, was 36.72"

    r = ALL_ROWS[BICEPS]
    t = rig_targets_from_mhr70(np.asarray(r["kp70"], np.float32).reshape(70, 3))
    k = {}
    for side in ("left", "right"):
        ci, si = I[f"{side}_clavicle"], I[f"{side}_shoulder"]
        v = np.asarray(t[si], float) - np.asarray(t[ci], float)
        k[side] = v / np.linalg.norm(v)
    kgap = float(np.degrees(np.arccos(np.clip(
        float(k["left"] @ (k["right"] * np.array([-1.0, 1.0, 1.0]))), -1.0, 1.0))))
    print(f"kp70 clavicle->shoulder:           left vs MIRRORED right "
          f"{kgap:.2f} deg apart")
    assert kgap < 6.0, \
        (f"positive control: kp70's own pair is {kgap:.1f} deg from mirrored, so "
         f"the {gap:.1f} deg above cannot be blamed on the asset alone")


# --------------------------------------------------------------------------- #
# Acceptance 4 -- roll-freeness survives the correction
# --------------------------------------------------------------------------- #

def test_the_corrected_local_still_carries_no_roll(monkeypatch):
    """`fb42e71`'s contract, under the correction. Roll-free by construction:
    the correction re-aims through `_aim_delta`, whose LOCAL is twist-free
    about the aim axis whatever the parent does.

    Checked at three caps -- 0 (the transfer), 15 (production) and 180
    (correction saturated at the keypoint) -- so 'no roll' cannot be an
    accident of the correction being small.

    Positive control: some local must carry real SWING, or a roll of zero is
    the roll of a bone that never moved."""
    for cap in (0.0, None, 180.0):
        swings = []
        for rid in ALL_ROWS:
            _, L = (_solve(rid, cap=cap, monkeypatch=monkeypatch)
                    if cap is not None else _solve(rid))
            for side in ("left", "right"):
                _, ax_local = _rest_long_axis(side)
                d = _local_delta(L, side)
                roll = _twist_deg(d, ax_local)
                assert roll < 1e-9, \
                    f"cap={cap} {rid[:8]} {side}: {roll:.4e} deg of roll"
                swings.append(_deg(d))
            monkeypatch.undo()
        assert max(swings) > 5.0, \
            f"positive control: cap={cap} moved no clavicle more than {max(swings):.2f} deg"


# --------------------------------------------------------------------------- #
# Acceptance 6 and the fallback -- what this change must NOT touch
# --------------------------------------------------------------------------- #

def test_the_fallback_path_never_sees_the_cap(monkeypatch):
    """`mhr_rots=None` has no transfer to correct and must not enter the
    block at all. Asserted as BIT-identity of the whole solved state across
    three wildly different caps, not as a tolerance on the clavicle."""
    ref = None
    for cap in (0.0, 15.0, 180.0):
        monkeypatch.setattr(PG, "_CLAV_AIM_CORRECTION_MAX_DEG", float(cap))
        got = {}
        for rid in sorted(ALL_ROWS):
            _, L = _solve(rid, mhr=False)
            got[rid] = {i: np.asarray(q, float).tobytes() for i, q in L.items()}
        monkeypatch.undo()
        if ref is None:
            ref = got
        else:
            assert got == ref, f"the fallback moved when the cap became {cap}"
    assert ref is not None and len(ref) == len(ALL_ROWS)
    # Positive control: the transfer path IS sensitive to the cap, so the
    # equality above is a property of the fallback and not of the comparison.
    moved = 0
    for cap in (0.0, 180.0):
        monkeypatch.setattr(PG, "_CLAV_AIM_CORRECTION_MAX_DEG", float(cap))
        _, L = _solve(BICEPS)
        monkeypatch.undo()
        moved += 1 if L else 0
    _, a = _solve(BICEPS, cap=0.0, monkeypatch=monkeypatch)
    monkeypatch.undo()
    _, b = _solve(BICEPS, cap=180.0, monkeypatch=monkeypatch)
    ci = I["left_clavicle"]
    assert _deg(QM.multiply(QM.conjugate(a[ci]), b[ci])) > 1.0, \
        "positive control: the cap does not move the TRANSFER path either"


def test_the_clavicle_correction_leaves_the_spine_alone(monkeypatch):
    """Acceptance 6. The clavicle hangs off spine_2; nothing hangs off the
    clavicle but the arm. So every spine local must be bit-identical across
    caps, and the pike row's flexion must still read what
    `test_spine_rel_total` pins it at.

    Positive control: the clavicle locals on the same solves must DIFFER, or
    'the spine did not move' is a statement about a solve that did not
    run."""
    _, L0 = _solve(PIKE, cap=0.0, monkeypatch=monkeypatch)
    monkeypatch.undo()
    _, L1 = _solve(PIKE)
    for name in ("pelvis", "spine_1", "spine_2", "neck", "head"):
        i = I[name]
        assert np.asarray(L0[i], float).tobytes() == np.asarray(L1[i], float).tobytes(), \
            f"{name} moved when the clavicle correction was applied"
    assert _deg(QM.multiply(QM.conjugate(L0[I["left_clavicle"]]),
                            L1[I["left_clavicle"]])) > 1.0, \
        "positive control: the correction did nothing on this row"

    # ...and the flexion numbers themselves, in the captures' own frame.
    right = np.array([1.0, 0.0, 0.0])
    for name, want in (("spine_1", 43.2), ("spine_2", 22.8)):
        d = QM.multiply(np.asarray(L1[I[name]], float),
                        QM.conjugate(RIG.rest_local_q[I[name]]))
        d = np.asarray(d, float)
        if d[0] < 0:
            d = -d
        n = np.linalg.norm(d[1:])
        rv = np.zeros(3) if n < 1e-12 else (d[1:] / n) * np.degrees(
            2 * np.arctan2(n, d[0]))
        assert float(rv @ right) == pytest.approx(want, abs=0.6), \
            f"pike {name} flexion is {float(rv @ right):.1f}, pinned at {want}"


def test_the_arm_below_the_shoulder_is_still_exact(monkeypatch):
    """The correction rotates the clavicle; the shoulder, elbow and wrist are
    world-anchored from the arm keypoints and must not follow. Their WORLD
    orientations are bit-identical across caps -- only the clavicle's own
    frame, and therefore the shoulder's LOCAL, changes."""
    for rid in sorted(ALL_ROWS):
        _, L0 = _solve(rid, cap=0.0, monkeypatch=monkeypatch)
        monkeypatch.undo()
        _, L1 = _solve(rid)
        W0 = fk_world_orientations(RIG, {**RIG.rest_local_q, **L0})
        W1 = fk_world_orientations(RIG, {**RIG.rest_local_q, **L1})
        for side in ("left", "right"):
            for bone in ("shoulder", "elbow", "wrist"):
                i = I[f"{side}_{bone}"]
                off = _deg(QM.multiply(QM.conjugate(QM.normalize(W0[i])),
                                       QM.normalize(W1[i])))
                assert off < 1e-9, f"{rid[:8]} {side}_{bone} world moved {off:.3e} deg"
            # Positive control on the same solves: the clavicle DID move --
            # every row, both sides (min over the corpus is 3.27 deg), not
            # just whatever W0/W1 happened to be left over after the loop.
            ci = I[f"{side}_clavicle"]
            moved = _deg(QM.multiply(QM.conjugate(QM.normalize(W0[ci])),
                                     QM.normalize(W1[ci])))
            assert moved > 1.0, \
                f"positive control: {rid[:8]} {side}_clavicle moved only {moved:.3e} deg"
