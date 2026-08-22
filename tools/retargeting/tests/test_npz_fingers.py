"""The mannequin's ten finger chains driven by the MHR model's own hand rotations.

The first design for this transfer composed MHR's world rotation DELTA onto
each phalange, exactly as v16 task 4 does for the spine. That is correct only
while the two rigs' rests nearly agree -- 7.6-31.6 deg for the spine -- and the
two rests' FINGER directions are 37-75 deg apart (cosine 0.258-0.804). Measured
against the rig's own joint contract the delta transfer produced Y-dominant
middle phalanges and Z-dominant distals where the rig bends about X: it bent
fingers sideways and twisted them, and rendered as something that does not read
as a hand.

What replaces it is a signed joint-ANGLE transfer onto the rig's own axes:

  1. the rig's per-bone flexion axis is MEASURED, not assumed, from PoseGoblin's
     shipped hand presets (`bind_poses/mannequin_finger_axes.json`, extracted by
     `tools_extract_finger_axes.py`). It is not +X everywhere -- the thumb's
     three joints come out mixed-XY, mixed-XZ and pure -Z;
  2. MHR supplies only the joint's BEND, `conj(Delta(parent)) o Delta(row)`,
     whose magnitude is frame-independent and therefore immune to the rest
     mismatch that broke the delta transfer;
  3. its sign (flexion vs extension) comes from projecting that rotation onto a
     reference axis carried across from the mannequin to the MHR skeleton
     through an anatomical hand frame each skeleton builds from its OWN
     landmarks (wrist, middle_1, index_1, pinky_1) -- so the hands' chirality is
     derived, never hardcoded;
  4. the angle is applied about the rig's measured axis, composed onto the
     bone's rest local.

Fist is the ground truth this buys: feeding each bone its stored fist angle must
land on the stored Fist preset, and feeding zeros must land on rest. The delta
transfer could never be checked that way.

Quaternions are the solver's [w,x,y,z] throughout; PoseGoblin's state.pose and
three.js use [x,y,z,w], and the only conversion is at serialization
(`QuaternionMath.to_threejs_dict`).
"""
import json
from pathlib import Path

import numpy as np
import pytest

from retargeting.core.math_utils import QuaternionMath as QM
from retargeting.retargeters.posegoblin_rig import (
    _FINGER_BEND_LIMIT_DEG, _FINGER_PHALANGE_NAMES, _MHR_FINGER_ROWS,
    _MHR_WRIST_ROW, _finger_bends, _finger_digits_passing, _finger_flex_refs,
    _mhr_delta_q, load_finger_axes, load_mhr_rest, load_rig, rig_state_from_mhr70,
    rig_targets_from_mhr70, solve_rig_locals, solved_indices)

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mhr_npz_rows.json"
ROWS = json.loads(_FIXTURE.read_text())["rows"]
DEV_ROW = "a6566802a6c9ddd63340ccb4520e0001"
HAND_POSES = Path("/Users/scotteaton/Dropbox/CODE/poseGoblin/poses/hand_poses.json")

RIG = load_rig()
I = RIG.index_of_name

DIGITS = ("thumb", "index", "middle", "ring", "pinky")
SIDES = ("left", "right")


def bone(side, digit, k):
    return f"{side}_thumb_{k}" if digit == "thumb" else f"{side}_{digit}_finger_{k}"


def digit_bones(side, digit):
    return [bone(side, digit, k) for k in (1, 2, 3)]


def rots(row_id):
    return np.asarray(ROWS[row_id]["joint_global_rots"], float)


def kp(row_id):
    return np.asarray(ROWS[row_id]["kp70"], float)


def quat_deg(a, b):
    d = abs(float(np.dot(QM.normalize(np.asarray(a, float)), QM.normalize(np.asarray(b, float)))))
    return float(np.degrees(2 * np.arccos(min(1.0, d))))


def assert_same_quat(a, b, what):
    """Component equality, sign-normalised. STRICTER than quat_deg, and used
    wherever the claim is "identical" rather than "close": arccos(1-eps) has a
    precision floor around 1.7e-6 deg, so quat_deg cannot express exactness at
    all -- it returns 1.7e-6 for a quaternion compared with itself."""
    a = QM.normalize(np.asarray(a, float))
    b = QM.normalize(np.asarray(b, float))
    if float(np.dot(a, b)) < 0.0:
        b = -b
    assert np.allclose(a, b, atol=1e-9), f"{what}: {a} != {b}"


def random_rotations(n, seed):
    """n proper rotations, Gaussian -> QR, sign-fixed so det is exactly +1.

    These are VALID rotations. The orthonormality half of the gate cannot see
    them; only the bend bound can -- which is the point of using them.
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        q, r = np.linalg.qr(rng.normal(size=(3, 3)))
        q = q @ np.diag(np.sign(np.diag(r)))
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1.0
        out.append(q)
    return out


def sabotage_left_hand(mhr_rots, seed):
    """Replace the left wrist + 15 left finger rows with random rotations."""
    bad = mhr_rots.copy()
    left_rows = [_MHR_WRIST_ROW["left"]] + [_MHR_FINGER_ROWS[b]
                                            for d in DIGITS for b in digit_bones("left", d)]
    for row, m in zip(left_rows, random_rotations(len(left_rows), seed)):
        bad[row] = m
    return bad, left_rows


# --------------------------------------------------------------------------
# The fixture: the rig's own measured joint contract
# --------------------------------------------------------------------------

def test_finger_axis_fixture_covers_thirty_bones_with_unit_axes():
    ax = load_finger_axes()
    assert set(ax["axis"]) == set(_FINGER_PHALANGE_NAMES)
    assert len(ax["axis"]) == 30
    for name, a in ax["axis"].items():
        assert np.isclose(np.linalg.norm(a), 1.0, atol=1e-6), name
        assert 5.0 < ax["fist_angle_deg"][name] < 130.0, (name, ax["fist_angle_deg"][name])


def test_measured_axes_match_the_rigs_contract_and_are_not_plus_x_everywhere():
    """Axis conformance, re-asserted independently of the extractor: the four
    fingers' middle and distal phalanges bend about local +X to 3 decimals on
    BOTH hands, and the thumb -- which is why the axis had to be measured
    rather than assumed -- does not."""
    ax = load_finger_axes()["axis"]
    for side in SIDES:
        for d in ("index", "middle", "ring", "pinky"):
            for k in (2, 3):
                a = ax[bone(side, d, k)]
                assert a[0] == pytest.approx(1.0, abs=1e-3), (bone(side, d, k), a)
    # positive control: the thumb genuinely disagrees, so the +X assertion above
    # is a real constraint and not something every bone satisfies.
    assert abs(ax[bone("left", "thumb", 3)][0]) < 0.05
    assert abs(ax[bone("left", "thumb", 3)][2]) == pytest.approx(1.0, abs=1e-3)
    assert abs(ax[bone("left", "thumb", 1)][1]) > 0.5


def test_full_flexion_reproduces_the_stored_fist_and_zero_reproduces_rest():
    """Fist is a reproducible ground truth -- the whole point of measuring the
    axes. Feeding each bone its own stored fist angle must land on PoseGoblin's
    stored Fist preset on BOTH hands; feeding zeros must land on rest."""
    presets = {e["name"]: e["pose"] for e in json.loads(HAND_POSES.read_text())}
    ax = load_finger_axes()
    for side in SIDES:
        for d in DIGITS:
            for name in digit_bones(side, d):
                rest = RIG.rest_local_q[I[name]]
                want = QM.from_threejs_dict(presets["Fist"][name.replace(side, "left", 1)])
                got = QM.multiply(rest, QM.from_axis_angle(
                    ax["axis"][name], np.radians(ax["fist_angle_deg"][name])))
                assert_same_quat(got, want, f"{name} at full flexion vs stored Fist")
                zero = QM.multiply(rest, QM.from_axis_angle(ax["axis"][name], 0.0))
                assert_same_quat(zero, rest, f"{name} at zero flexion vs rest")


# --------------------------------------------------------------------------
# MHR side: bends, and the chirality-derived reference axes
# --------------------------------------------------------------------------

def test_reference_axes_align_with_the_corpus_own_finger_rotations():
    """The design's positive control. The reference axes come from two sources
    that know nothing about each other -- PoseGoblin's authored Fist preset and
    the MHR template skeleton's rest geometry -- so the corpus's ACTUAL finger
    rotations landing on them is evidence the correspondence is right. Median
    |alignment| must be high AND the signs must be predominantly positive
    (MHR's fingers flexing the way the rig calls flexion, not the reverse)."""
    refs = _finger_flex_refs()
    aligns = []
    for row_id in ROWS:
        mr = rots(row_id)
        for side in SIDES:
            for d in DIGITS:
                prev = _MHR_WRIST_ROW[side]
                for name in digit_bones(side, d):
                    row = _MHR_FINGER_ROWS[name]
                    rel = QM.multiply(QM.conjugate(_mhr_delta_q(mr, prev)), _mhr_delta_q(mr, row))
                    axis, _ = QM.to_axis_angle(rel if rel[0] >= 0 else -rel)
                    aligns.append(float(np.dot(axis, refs[name])))
                    prev = row
    aligns = np.array(aligns)
    assert len(aligns) == 180
    assert np.median(np.abs(aligns)) > 0.90
    assert np.median(aligns) > 0.90, "MHR flexes the way the rig calls flexion"
    # positive control: swapping each hand's reference for the other hand's
    # must destroy the agreement, or the check above is measuring nothing.
    swapped = []
    for row_id in ROWS:
        mr = rots(row_id)
        for side in SIDES:
            other = "right" if side == "left" else "left"
            for d in DIGITS:
                prev = _MHR_WRIST_ROW[side]
                for k in (1, 2, 3):
                    row = _MHR_FINGER_ROWS[bone(side, d, k)]
                    rel = QM.multiply(QM.conjugate(_mhr_delta_q(mr, prev)), _mhr_delta_q(mr, row))
                    axis, _ = QM.to_axis_angle(rel if rel[0] >= 0 else -rel)
                    swapped.append(float(np.dot(axis, refs[bone(other, d, k)])))
                    prev = row
    assert np.median(swapped) < 0.0, (
        "cross-hand references must NOT agree -- if they do, the reference "
        "construction is not chirality-sensitive and proves nothing")


def test_real_bends_are_far_below_the_gate_bound():
    """The bound exists to catch garbage, not to trim real data. Every one of
    the 180 real per-joint bends must sit well under it."""
    worst = 0.0
    for row_id in ROWS:
        bends = _finger_bends(rots(row_id))
        assert len(bends) == 30
        for name, (theta, deg) in bends.items():
            assert np.isfinite(theta) and np.isfinite(deg), name
            worst = max(worst, deg)
    assert worst < 0.5 * _FINGER_BEND_LIMIT_DEG, (
        f"largest real bend {worst:.1f} deg is not comfortably under the "
        f"{_FINGER_BEND_LIMIT_DEG} deg bound")


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------

def test_no_real_digit_is_gated_out_on_any_fixture_row():
    all_digits = {(s, d) for s in SIDES for d in DIGITS}
    for row_id in ROWS:
        assert _finger_digits_passing(rots(row_id)) == all_digits, row_id
    # positive control: the gate is capable of returning less than everything.
    bad, _ = sabotage_left_hand(rots(DEV_ROW), seed=20260822)
    assert _finger_digits_passing(bad) != all_digits


def test_sabotaged_hand_falls_back_to_rest_and_the_other_hand_is_untouched():
    """The gate's positive control. Every left row is replaced by a VALID random
    rotation (QR, det +1) -- so the orthonormality check cannot be what rejects
    them; the bend bound has to be. Every left digit must fall back to its rest
    locals and no right digit may move."""
    clean = rots(DEV_ROW)
    bad, left_rows = sabotage_left_hand(clean, seed=20260822)

    for row in left_rows:                       # the sabotage really is valid rotations
        m = bad[row]
        assert np.allclose(m.T @ m, np.eye(3), atol=1e-9)
        assert np.linalg.det(m) == pytest.approx(1.0, abs=1e-9)

    passing = _finger_digits_passing(bad)
    assert passing == {("right", d) for d in DIGITS}, passing

    st_bad = rig_state_from_mhr70(kp(DEV_ROW), bad)
    st_clean = rig_state_from_mhr70(kp(DEV_ROW), clean)
    for d in DIGITS:
        for name in digit_bones("left", d):
            assert_same_quat(QM.from_threejs_dict(st_bad["pose"][name]),
                             RIG.rest_local_q[I[name]], f"gated {name} vs rest")
        for name in digit_bones("right", d):
            assert_same_quat(QM.from_threejs_dict(st_bad["pose"][name]),
                             QM.from_threejs_dict(st_clean["pose"][name]),
                             f"ungated {name} vs the clean solve")


def test_each_integrity_rejection_reason_fires_on_its_own():
    """Three deterministic controls, one per rejection reason -- so no reason is
    carried by the probabilistic random-rotation control alone."""
    all_digits = {(s, d) for s in SIDES for d in DIGITS}

    nan = rots(DEV_ROW).copy()
    nan[_MHR_FINGER_ROWS["left_ring_finger_2"]] = np.nan
    assert _finger_digits_passing(nan) == all_digits - {("left", "ring")}

    skew = rots(DEV_ROW).copy()
    skew[_MHR_FINGER_ROWS["right_pinky_finger_3"]] *= 1.5      # orthonormal * scale
    assert _finger_digits_passing(skew) == all_digits - {("right", "pinky")}

    bent = rots(DEV_ROW).copy()
    over = QM.from_axis_angle(np.array([0.0, 0.0, 1.0]), np.radians(170.0))
    bent[_MHR_FINGER_ROWS["left_index_finger_3"]] = (
        _wxyz_to_mat(over) @ bent[_MHR_FINGER_ROWS["left_index_finger_3"]])
    assert _finger_digits_passing(bent) == all_digits - {("left", "index")}


def _wxyz_to_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


# --------------------------------------------------------------------------
# solved_indices stays the single source of truth
# --------------------------------------------------------------------------

def test_solved_indices_without_rotations_is_the_v15_set_and_takes_one_argument():
    """The poseforge3d harness calls solved_indices(rig) today; that call must
    keep working and keep returning exactly the v15 34."""
    v15 = solved_indices(RIG)
    assert len(v15) == 34
    assert not any(RIG.name[i] in _FINGER_PHALANGE_NAMES for i in v15)
    assert solved_indices(RIG, None) == v15


def test_solved_indices_includes_the_phalanges_when_rotations_are_present():
    got = solved_indices(RIG, rots(DEV_ROW))
    assert len(got) == 64
    assert got == sorted(got)                              # still in rig.order
    assert {RIG.name[i] for i in got} >= _FINGER_PHALANGE_NAMES


def test_solved_indices_drops_exactly_the_gated_digits():
    bad, _ = sabotage_left_hand(rots(DEV_ROW), seed=20260822)
    got = {RIG.name[i] for i in solved_indices(RIG, bad)}
    assert got & _FINGER_PHALANGE_NAMES == {
        b for d in DIGITS for b in digit_bones("right", d)}


def test_solve_rig_locals_returns_exactly_solved_indices():
    """One function's answer, not two. Whatever solved_indices says, with and
    without rotations and with a gated hand, is what the solver returns."""
    targets = rig_targets_from_mhr70(kp(DEV_ROW))
    bad, _ = sabotage_left_hand(rots(DEV_ROW), seed=20260822)
    for mr in (None, rots(DEV_ROW), bad):
        assert set(solve_rig_locals(RIG, targets, mr)) == set(solved_indices(RIG, mr))


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------

def test_real_row_actually_poses_the_fingers_and_leaves_tips_at_rest():
    st = rig_state_from_mhr70(kp(DEV_ROW), rots(DEV_ROW))
    assert len(st["pose"]) == 73
    moved = [n for n in _FINGER_PHALANGE_NAMES
             if quat_deg(QM.from_threejs_dict(st["pose"][n]), RIG.rest_local_q[I[n]]) > 1.0]
    assert len(moved) >= 25, f"only {len(moved)} of 30 phalanges moved off rest"
    for side in SIDES:
        for d in DIGITS:
            tip = f"{side}_thumb_tip" if d == "thumb" else f"{side}_{d}_finger_tip"
            assert_same_quat(QM.from_threejs_dict(st["pose"][tip]),
                             RIG.rest_local_q[I[tip]], f"{tip} vs rest")


def test_no_rotations_leaves_all_forty_finger_bones_at_rest():
    """v15 behaviour, unchanged: without the npz blob nothing about the hands
    may move."""
    st = rig_state_from_mhr70(kp(DEV_ROW))
    for i in range(74):
        n = RIG.name[i]
        if "_thumb_" not in n and "_finger_" not in n:
            continue
        assert_same_quat(QM.from_threejs_dict(st["pose"][n]), RIG.rest_local_q[i],
                         f"{n} with mhr_rots=None vs rest")


def test_posed_fingers_stay_within_the_rigs_own_range_of_motion():
    """Sanity on the applied angle, not on pose quality: no phalange may end up
    further from its rest than its own measured full-fist arc plus a margin --
    the failure mode of the delta transfer this replaced was fingers thrown far
    outside any pose the rig can reach."""
    ax = load_finger_axes()
    for row_id in ROWS:
        st = rig_state_from_mhr70(kp(row_id), rots(row_id))
        for name in _FINGER_PHALANGE_NAMES:
            moved = quat_deg(QM.from_threejs_dict(st["pose"][name]), RIG.rest_local_q[I[name]])
            assert moved <= ax["fist_angle_deg"][name] + 15.0, (row_id, name, moved)


# --------------------------------------------------------------------------
# Fist as end-to-end ground truth, through the real transfer
# --------------------------------------------------------------------------

def synthetic_rots(angles):
    """A (127,3,3) `joint_global_rots` whose finger bends are EXACTLY *angles*
    (radians, keyed by mannequin bone name) about each bone's own reference
    axis, and whose every other joint sits at rest (delta = identity).

    Inverts the transfer: `Delta(row) = R_pose(row) @ R_rest(row)^T`, so
    `R_pose(row) = Delta(row) @ R_rest(row)`, and the deltas accumulate down
    the digit from an identity wrist -- which makes
    `conj(Delta(parent)) o Delta(row)` come out as the planted rotation.
    """
    rest = load_mhr_rest()["q_wxyz"]
    refs = _finger_flex_refs()
    out = np.stack([_wxyz_to_mat(q) for q in rest])            # every delta identity
    for side in SIDES:
        for d in DIGITS:
            acc = QM.identity()
            for name in digit_bones(side, d):
                acc = QM.multiply(acc, QM.from_axis_angle(refs[name], angles[name]))
                out[_MHR_FINGER_ROWS[name]] = _wxyz_to_mat(QM.multiply(acc, rest[_MHR_FINGER_ROWS[name]]))
    return out


def test_full_flexion_fed_through_the_solver_lands_on_the_stored_fist():
    """Ruling step 5, end to end: synthetic full-flexion bends pushed through
    `_finger_bends` -> `_finger_locals` must land on PoseGoblin's stored Fist
    preset, on both hands. This is the pass/fail the world-delta transfer this
    replaced could never have."""
    presets = {e["name"]: e["pose"] for e in json.loads(HAND_POSES.read_text())}
    fist_deg = load_finger_axes()["fist_angle_deg"]
    mr = synthetic_rots({n: np.radians(fist_deg[n]) for n in _MHR_FINGER_ROWS})

    solved = solve_rig_locals(RIG, rig_targets_from_mhr70(kp(DEV_ROW)), mr)
    for side in SIDES:
        for d in DIGITS:
            for name in digit_bones(side, d):
                assert_same_quat(
                    solved[I[name]],
                    QM.from_threejs_dict(presets["Fist"][name.replace(side, "left", 1)]),
                    f"{name} driven to full flexion vs the stored Fist preset")


def test_zero_bend_fed_through_the_solver_lands_on_rest():
    """The other half: an MHR hand sitting exactly at ITS rest must leave the
    mannequin's fingers at exactly THEIR rest -- no drift from the two rests
    being different shapes, which is the whole reason a world-delta transfer
    could not be used here."""
    mr = synthetic_rots({n: 0.0 for n in _MHR_FINGER_ROWS})
    solved = solve_rig_locals(RIG, rig_targets_from_mhr70(kp(DEV_ROW)), mr)
    for name in _FINGER_PHALANGE_NAMES:
        assert_same_quat(solved[I[name]], RIG.rest_local_q[I[name]],
                         f"{name} at zero MHR bend vs rest")


def test_half_flexion_is_monotone_between_rest_and_fist():
    """A sanity band on the whole pipeline, not just its endpoints: half the
    fist angle must land strictly between rest and Fist for every bone."""
    fist_deg = load_finger_axes()["fist_angle_deg"]
    mr = synthetic_rots({n: np.radians(0.5 * fist_deg[n]) for n in _MHR_FINGER_ROWS})
    solved = solve_rig_locals(RIG, rig_targets_from_mhr70(kp(DEV_ROW)), mr)
    for name in _FINGER_PHALANGE_NAMES:
        from_rest = quat_deg(solved[I[name]], RIG.rest_local_q[I[name]])
        assert from_rest == pytest.approx(0.5 * fist_deg[name], abs=1e-4), name


def test_axes_are_refused_on_a_rig_they_were_not_measured_against():
    """A re-rig must fail loudly here rather than silently producing a wrong
    hand from stale axes."""
    import dataclasses
    other = dataclasses.replace(RIG, version="posegoblin_rig_v99")
    with pytest.raises(ValueError, match="tools_extract_finger_axes"):
        solve_rig_locals(other, rig_targets_from_mhr70(kp(DEV_ROW)), rots(DEV_ROW))
    # positive control: the same call on the real rig does NOT raise
    assert solve_rig_locals(RIG, rig_targets_from_mhr70(kp(DEV_ROW)), rots(DEV_ROW))


def test_flex_refs_honour_an_explicitly_passed_rig():
    """The per-version cache must not answer for a rig it was not asked about."""
    a = _finger_flex_refs()
    b = _finger_flex_refs(RIG)
    assert set(a) == set(b) and all(np.allclose(a[k], b[k]) for k in a)
