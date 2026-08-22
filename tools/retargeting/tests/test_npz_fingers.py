"""The mannequin's ten finger chains driven by the MHR model's own hand rotations.

The first design for this transfer composed MHR's world rotation DELTA onto
each phalange, exactly as v16 task 4 does for the spine. That is correct only
while the two rigs' rests nearly agree -- 7.6-31.6 deg for the spine -- and the
two rests' FINGER directions are 3-49 deg apart, median 27 (in each skeleton's
own anatomical hand frame; 3-161 deg before R14 re-based the finger rest). Measured
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
    _FINGER_BEND_LIMIT_DEG, _FINGER_PHALANGE_NAMES, _FINGER_REFS, _MHR_FINGER_ROWS,
    _MHR_WRIST_ROW, _finger_bends, _finger_digits_passing, _finger_flex_refs,
    _mhr_delta_q, load_finger_axes, load_mhr_rest, load_rig, rig_state_from_mhr70,
    rig_targets_from_mhr70, solve_rig_locals, solved_indices)

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mhr_npz_rows.json"
ROWS = json.loads(_FIXTURE.read_text())["rows"]
DEV_ROW = "a6566802a6c9ddd63340ccb4520e0001"
# Sibling repo, not a runtime dependency -- repo-relative (from this file's
# own location), not the absolute /Users/scotteaton/... path this used to
# hardcode. Same fix as tools_extract_finger_axes.py's HAND_POSES and
# consistent with e0e9426's ruling against baked-in cross-repo absolute
# paths.
HAND_POSES = Path(__file__).resolve().parents[4] / "poseGoblin" / "poses" / "hand_poses.json"
# v1: a genuinely DIFFERENT rig -- different version string, different finger
# rest (its right hand was captured curled into a fist, which is what R14
# removed from v2), finger islands still disconnected at `parent: None`.
# Its flexion references sit 11.6-139.4 deg away from v2's on all 30 bones.
_V1_PATH = Path(__file__).resolve().parent.parent / "bind_poses" / "posegoblin_rig_v1.json"

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
# The axes, corroborated by a pose they were NOT measured from
# --------------------------------------------------------------------------

# The nine joints per side whose flexion axis is a genuine single hinge, so a
# second, independently authored pose has to rotate them about the SAME axis:
# the eight interphalangeal joints of the four fingers, plus the thumb IP.
_SINGLE_HINGE = [f"{stem}_{k}" for stem in ("index_finger", "middle_finger",
                                            "ring_finger", "pinky_finger")
                 for k in (2, 3)] + ["thumb_3"]

# The six per side that are NOT single hinges, with the |dot| each actually
# reaches against Point. The four MCPs abduct as well as flex; the thumb CMC
# and MCP are the two joints Fist alone genuinely cannot determine -- which is
# the same thumb_1 that the corpus alignment finds inverted (see _KNOWN_INVERTED).
# Pinned, not smoothed: if any of these moves, the axis model changed.
_MULTI_DOF_POINT_DOT = {
    "thumb_1": 0.0949, "thumb_2": 0.6721, "index_finger_1": 0.6923,
    "middle_finger_1": 1.0000, "ring_finger_1": 0.9324, "pinky_finger_1": 0.9562,
}


def _preset_axis_and_arc(name, pose_name, presets):
    """(axis, degrees) of `conj(rest) o preset` for one bone, in the bone's own
    rest-local frame -- the same construction tools_extract_finger_axes.py uses
    for Fist. The preset is authored on LEFT bone names and applied to both
    hands verbatim, so the right hand reads the left entry against its own rest."""
    side = "left" if name.startswith("left_") else "right"
    rest = RIG.rest_local_q[I[name]]
    rest = -rest if rest[0] < 0 else rest
    want = QM.from_threejs_dict(presets[pose_name][name.replace(side, "left", 1)])
    want = -want if want[0] < 0 else want
    rel = QM.multiply(QM.conjugate(QM.normalize(rest)), QM.normalize(want))
    axis, ang = QM.to_axis_angle(rel if rel[0] >= 0 else -rel)
    return axis, float(np.degrees(ang))


def test_point_and_spread_corroborate_the_axes_measured_from_fist():
    """Review finding: the axes are derived from Fist ALONE, so every one of
    them rests on a single chord with nothing to check it against. `Point` and
    `Spread` are two more hand poses PoseGoblin ships, authored independently of
    Fist, and they are a real constraint -- a single-hinge joint must rotate
    about the same axis in all three.

    Measured agreement on the eighteen single-hinge bones (nine per side):
    Point |1 - |dot|| <= 3.3e-16 -- machine precision, on all eighteen. Spread
    <= 1.6e-3. The thumb IP (`thumb_3`) is in that set, so two of the thumb's
    three axes now have independent corroboration.

    The other twelve DISAGREE, and are reported rather than averaged away:
    the four MCPs abduct as well as flex (|dot| 0.69-1.00), and the thumb CMC
    and MCP come out at 0.0949 and 0.6721 -- Fist alone does not determine a
    single axis for them. That is the same weakest-joint finding the transfer
    already carries, arrived at from a second direction."""
    presets = {e["name"]: e["pose"] for e in json.loads(HAND_POSES.read_text())}
    assert {"Fist", "Point", "Spread"} <= set(presets)
    fist_axis = load_finger_axes()["axis"]

    corroborated = [f"{side}_{nm}" for side in SIDES for nm in _SINGLE_HINGE]
    assert len(corroborated) == 18
    for name in corroborated:
        # (pose, |1-|dot|| bound, arc floor). Spread's arcs are the shorter of
        # the two -- 3.71 deg at `ring_finger_2` -- and still land on the same
        # axis to 1.6e-3, which is the point: a short chord is not a vague one.
        for pose_name, tol, arc_floor in (("Point", 1e-9, 10.0), ("Spread", 1e-2, 3.0)):
            axis, arc = _preset_axis_and_arc(name, pose_name, presets)
            assert arc > arc_floor, (name, pose_name, arc)  # the arc determines an axis
            dot = abs(float(np.dot(axis, fist_axis[name])))
            assert 1.0 - dot <= tol, (name, pose_name, dot)

    # Positive control: |dot| ~ 1 is a real constraint, not something any pair of
    # axes satisfies. The twelve multi-DOF joints are measured against the same
    # Point pose and do NOT reach it -- and their disagreement is pinned.
    for side in SIDES:
        for nm, want in _MULTI_DOF_POINT_DOT.items():
            axis, _arc = _preset_axis_and_arc(f"{side}_{nm}", "Point", presets)
            dot = abs(float(np.dot(axis, fist_axis[f"{side}_{nm}"])))
            assert dot == pytest.approx(want, abs=1e-3), (side, nm, dot)
    assert min(_MULTI_DOF_POINT_DOT.values()) < 0.2      # thumb_1 really is ~orthogonal


def test_left_and_right_share_one_measured_axis_after_r14():
    """R14's own regression guard on the fixture. The rig asset's two hands are
    now at the same rest, so `conj(rest) o Fist` must give the same axis AND the
    same arc on both sides. Before R14 the arcs were uniformly 0.6300 of the
    left's on all fifteen bones -- a constant ratio, the fingerprint of a right
    hand captured 37% closed.

    The extractor asserts this too and refuses to write; this is the same claim
    pinned against the committed fixture, so a hand-edited file cannot slip past."""
    ax = load_finger_axes()
    for nm in [f"{stem}_{k}" for stem in ("thumb", "index_finger", "middle_finger",
                                          "ring_finger", "pinky_finger")
               for k in (1, 2, 3)]:
        la, ra = ax["axis"][f"left_{nm}"], ax["axis"][f"right_{nm}"]
        deg = float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(la, ra)))))))
        assert deg < 0.1, (nm, deg)                      # measured worst 0.0023
        assert ax["fist_angle_deg"][f"left_{nm}"] == pytest.approx(
            ax["fist_angle_deg"][f"right_{nm}"], abs=1e-3)   # measured worst 1.4e-4
    # positive control: the arcs are not all the same number, so "equal" above is
    # a constraint on the pairing and not on a constant.
    arcs = [ax["fist_angle_deg"][f"left_{nm}"] for nm in ("thumb_2", "index_finger_2")]
    assert abs(arcs[0] - arcs[1]) > 50.0, arcs


# --------------------------------------------------------------------------
# MHR side: bends, and the chirality-derived reference axes
# --------------------------------------------------------------------------

def _per_bone_alignment_medians(refs):
    """{bone name -> median over EVERY fixture row of dot(the corpus's OWN
    bend axis, *refs*[bone])}. 30 bones, one sample per row each.

    PER BONE, deliberately (review finding). The pooled median over all
    30*len(ROWS) samples cannot see a single mis-signed digit -- 12 samples do
    not move a median -- and a mis-signed digit reads downstream as
    hyperextension, which the range-of-motion test tolerates. Per bone, one
    inverted digit is three bones at -1 and impossible to miss.

    The sample count below tracks `len(ROWS)` rather than a literal: the
    fixture grew from 6 rows to 16 when the ten spine captures were added
    (2026-08-22). It still asserts one sample per bone per row --
    a row that silently failed to contribute is still a failure here."""
    per: dict = {}
    for row_id in ROWS:
        mr = rots(row_id)
        for side in SIDES:
            for d in DIGITS:
                prev = _MHR_WRIST_ROW[side]
                for name in digit_bones(side, d):
                    row = _MHR_FINGER_ROWS[name]
                    rel = QM.multiply(QM.conjugate(_mhr_delta_q(mr, prev)), _mhr_delta_q(mr, row))
                    axis, _ = QM.to_axis_angle(rel if rel[0] >= 0 else -rel)
                    per.setdefault(name, []).append(float(np.dot(axis, refs[name])))
                    prev = row
    assert set(per) == set(_MHR_FINGER_ROWS) and all(len(v) == len(ROWS) for v in per.values())
    return {n: float(np.median(v)) for n, v in per.items()}


# The thumb CMC is the one joint where the corpus's rotation runs OPPOSITE to
# the rig's declared flexion. Measured medians: left -0.889, right -0.953 --
# consistent, on both hands, and unchanged by R14 (identical before and after
# the rest correction), so it is not a chirality or transport error. The
# mannequin's `thumb_1` maps to MHR's `thumb1`, whose own parent `thumb0` (the
# CMC) has no mannequin counterpart, so `conj(Delta(wrist)) o Delta(thumb1)`
# absorbs thumb0's rotation -- largely opposition/abduction, which a single
# hinge cannot express. Pinned here rather than tolerated: it is NAMED, so a
# new inversion cannot hide behind it and a future fix must re-pin this list.
_KNOWN_INVERTED = {"left_thumb_1", "right_thumb_1"}


def test_every_bones_reference_axis_agrees_in_sign_with_the_corpus():
    """The design's positive control, per bone. The reference axes come from two
    sources that know nothing about each other -- PoseGoblin's authored Fist
    preset and the MHR template skeleton's rest geometry -- so the corpus's
    ACTUAL finger rotations landing on them is evidence the correspondence is
    right."""
    med = _per_bone_alignment_medians(_finger_flex_refs())

    inverted = {n for n, v in med.items() if v < 0.0}
    assert inverted == _KNOWN_INVERTED, (
        f"finger bones whose corpus rotation runs against the rig's flexion "
        f"axis changed: {sorted(inverted ^ _KNOWN_INVERTED)}")

    agreeing = {n: v for n, v in med.items() if n not in _KNOWN_INVERTED}
    assert len(agreeing) == 28
    worst = min(agreeing.items(), key=lambda kv: kv[1])
    assert worst[1] > 0.60, worst          # measured 0.665 (both thumb_2 bones)
    # the pinned exception is a real inversion, not a near-zero wobble
    assert max(med[n] for n in _KNOWN_INVERTED) < -0.70, {n: med[n] for n in _KNOWN_INVERTED}

    # left and right must agree in SIGN on every one of the 15 bone pairs: a
    # reference transported onto the wrong hand breaks one side, not both.
    for n in med:
        if n.startswith("left_"):
            assert np.sign(med[n]) == np.sign(med[n.replace("left", "right", 1)]), (
                n, med[n], med[n.replace("left", "right", 1)])


def test_the_sign_check_names_a_single_inverted_digit_and_a_swapped_hand():
    """Two positive controls for the check above (CLAUDE.md rule 1).

    The first is the exact failure mode the review named -- ONE mis-signed
    digit, which the old pooled median could not see at all. It must be named,
    and nothing else may be."""
    refs = _finger_flex_refs()

    flipped = dict(refs)
    for name in digit_bones("right", "ring"):
        flipped[name] = -refs[name]
    got = {n for n, v in _per_bone_alignment_medians(flipped).items() if v < 0.0}
    assert got == _KNOWN_INVERTED | set(digit_bones("right", "ring")), sorted(got)

    # ...and swapping each hand's references for the other hand's must destroy
    # the agreement wholesale, or the construction is not chirality-sensitive.
    swapped = {n: refs[n.replace("left", "right", 1) if n.startswith("left_")
                       else n.replace("right", "left", 1)] for n in refs}
    med = _per_bone_alignment_medians(swapped)
    n_inverted = sum(1 for v in med.values() if v < 0.0)
    assert n_inverted >= 20, n_inverted     # measured 24 of 30; the six thumb
                                            # bones survive, the 24 finger bones do not


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

def _phalange_movement(row_id):
    """{phalange name -> degrees its solved local sits off its rest} for one row."""
    st = rig_state_from_mhr70(kp(row_id), rots(row_id))
    return {n: quat_deg(QM.from_threejs_dict(st["pose"][n]), RIG.rest_local_q[I[n]])
            for n in _FINGER_PHALANGE_NAMES}


def test_every_phalange_moves_off_rest_on_a_real_row():
    """PER BONE, not a count (review finding). `len(moved) >= 25` permitted five
    of thirty phalanges to sit dead at rest and still pass -- and a bone sitting
    dead at rest is exactly what a mis-transported reference axis produces, since
    the applied angle is `bend * dot(axis, ref)` and a wrong `ref` drives that
    dot to zero. A count cannot name the offender; this can."""
    st = rig_state_from_mhr70(kp(DEV_ROW), rots(DEV_ROW))
    assert len(st["pose"]) == 73

    moved = _phalange_movement(DEV_ROW)
    assert len(moved) == 30
    dead = {n: v for n, v in moved.items() if v <= 5.0}
    assert dead == {}, dead              # measured floor on this row: 8.89 deg
                                          # (left_thumb_3); the largest is 41.14

    for side in SIDES:
        for d in DIGITS:
            tip = f"{side}_thumb_tip" if d == "thumb" else f"{side}_{d}_finger_tip"
            assert_same_quat(QM.from_threejs_dict(st["pose"][tip]),
                             RIG.rest_local_q[I[tip]], f"{tip} vs rest")


def test_no_phalange_is_ever_dead_across_the_whole_fixture():
    """The same check widened to every fixture row, with the floor the real
    data actually supports rather than the one the dev row alone would allow.

    Measured over the ORIGINAL six rows: worst per-bone MEDIAN 9.96 deg
    (left_thumb_3); worst per-bone-per-ROW minimum 0.659 deg, at
    `right_thumb_1` -- small because that joint's rotation is nearly
    ORTHOGONAL to the rig's flexion axis there (alignment -0.069 on that row),
    so the projection correctly applies almost none of it. That is the
    mechanism working, not a dead bone; a bone the transfer never reaches sits
    at ~1e-9, eight orders of magnitude below. Re-measured over all sixteen
    rows once the ten spine captures were added: worst median 9.42 deg
    (right_thumb_1), worst single sample unchanged at 0.659 deg -- both still
    clear of the floors, which are NOT moved here."""
    per_bone: dict = {}
    for row_id in ROWS:
        for n, v in _phalange_movement(row_id).items():
            per_bone.setdefault(n, []).append(v)
    assert len(per_bone) == 30 and all(len(v) == len(ROWS) for v in per_bone.values())

    quiet = {n: float(np.median(v)) for n, v in per_bone.items() if np.median(v) <= 5.0}
    assert quiet == {}, quiet            # worst measured median 9.42 deg over 16 rows
    never = {n: min(v) for n, v in per_bone.items() if min(v) <= 0.25}
    assert never == {}, never            # worst measured single sample 0.297 deg


def test_the_movement_checks_name_the_bone_an_orthogonalised_reference_kills():
    """Positive control for the two checks above (CLAUDE.md rule 1). `dead ==
    {}`, `quiet == {}` and `never == {}` have no committed evidence the
    5.0/0.25 deg floors can ever fire -- a detector broken back to always
    returning `{}` would pass both vacuously, and read as coverage.

    The perturbation is `left_index_finger_2`'s own reference axis ROTATED 90
    deg within the plane it shares with world +X, landing perpendicular to
    where it started -- the shape of "mis-transported" the projection
    `bend * dot(axis, ref)` is defenceless against. It does not even reach
    zero: the corpus's own bend axis is not perfectly aligned with the
    reference it is projected onto (per-bone median |alignment| 0.87-0.996,
    never 1.0 -- see `_per_bone_alignment_medians`), so a small component
    survives the near-cancellation. Measured 2.36 deg on the dev row,
    comfortably under the 5.0 deg floor both checks use.

    Built in memory for this test only -- never a committed fixture on disk.
    `_FINGER_REFS` is restored in a `finally` so no other test can see it."""
    name = "left_index_finger_2"
    ref = _finger_flex_refs()[name]
    twist_axis = np.cross(ref, np.array([1.0, 0.0, 0.0]))
    twist_axis = twist_axis / np.linalg.norm(twist_axis)
    orthogonalised = np.cross(twist_axis, ref)
    orthogonalised = orthogonalised / np.linalg.norm(orthogonalised)
    assert abs(np.dot(orthogonalised, ref)) < 1e-9      # genuinely perpendicular

    saved = dict(_FINGER_REFS[RIG.version])
    _FINGER_REFS[RIG.version] = {**saved, name: orthogonalised}
    try:
        dead = {n: v for n, v in _phalange_movement(DEV_ROW).items() if v <= 5.0}

        per_bone: dict = {}
        for row_id in ROWS:
            for n, v in _phalange_movement(row_id).items():
                per_bone.setdefault(n, []).append(v)
        quiet = {n: float(np.median(v)) for n, v in per_bone.items() if np.median(v) <= 5.0}
    finally:
        _FINGER_REFS[RIG.version] = saved

    assert dead == {name: pytest.approx(2.36, abs=0.01)}, dead     # comfortably under the 5.0 deg floor
    # Re-pinned 2026-08-22 from 3.15 (six rows) to 2.78: the fixture gained
    # the ten spine captures, so this median is taken over sixteen rows now.
    # The dev-row number above is unchanged, which is the control that only
    # the POPULATION moved and not the measurement.
    assert quiet == {name: pytest.approx(2.78, abs=0.01)}, quiet   # median across all 16 rows


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
    worst = (-1e9, None)
    for row_id in ROWS:
        st = rig_state_from_mhr70(kp(row_id), rots(row_id))
        for name in _FINGER_PHALANGE_NAMES:
            moved = quat_deg(QM.from_threejs_dict(st["pose"][name]), RIG.rest_local_q[I[name]])
            over = moved - ax["fist_angle_deg"][name]
            if over > worst[0]:
                worst = (over, (row_id, name, moved))
            # 10, re-pinned after R14 from the 15 the pre-R14 arcs were given.
            # Measured worst overshoot 7.14 deg at left_thumb_2 (arc 19.21, moved
            # 26.36) -- the thumb MCP again, the one joint a single hinge cannot
            # express. R14 barely moved this: it was 7.19 deg before.
            assert over <= 10.0, (row_id, name, moved, ax["fist_angle_deg"][name])
    assert worst[0] > 0.0, worst   # positive control: something DOES exceed its
                                     # own fist arc, so the bound is a real bound


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
    """The per-version cache must not answer for a rig it was not asked about.

    Review finding: this used to compare `_finger_flex_refs()` against
    `_finger_flex_refs(RIG)` -- where RIG *is* the default rig. It asserted that
    two identical things were identical and could not fail for the reason it
    named. It now passes a genuinely different rig (v1: different version
    string, different finger rest, finger islands still disconnected), whose
    references really do land somewhere else."""
    other = load_rig(_V1_PATH)
    assert other.version != RIG.version                      # genuinely different

    a = _finger_flex_refs()
    b = _finger_flex_refs(other)
    assert set(a) == set(b) == set(_MHR_FINGER_ROWS)
    angles = sorted(float(np.degrees(np.arccos(np.clip(float(np.dot(a[n], b[n])), -1.0, 1.0))))
                    for n in a)
    # measured 11.643 / 107.346 / 139.357 deg (min/median/max); all 30 differ.
    assert angles[0] > 5.0, angles[:3]
    # ...and asking again for the default must still give the DEFAULT's answer,
    # not v1's -- the cache miss must not have overwritten the shared entry.
    again = _finger_flex_refs()
    assert all(np.array_equal(a[n], again[n]) for n in a)


def test_finger_bends_uses_the_rig_it_is_given():
    """Latent bug (review finding): `_finger_bends` called `_finger_flex_refs()`
    with no argument, so the runtime path always projected onto the DEFAULT
    rig's axes no matter which rig was passed down from `solved_indices` /
    `_finger_locals`. Nothing could see it while only one rig existed.

    The two halves below are what make this test able to fail for its own
    reason: the |bend| MAGNITUDE is frame-independent and must NOT move (that
    is the positive control -- it proves both calls saw the same MHR data),
    while the SIGNED angle is a projection onto the rig's axes and must."""
    other = load_rig(_V1_PATH)
    mr = rots(DEV_ROW)
    a = _finger_bends(mr)
    b = _finger_bends(mr, other)
    assert set(a) == set(b) and len(a) == 30

    for n in a:
        assert b[n][1] == pytest.approx(a[n][1], abs=1e-9), f"{n} magnitude moved"
    delta = {n: abs(float(np.degrees(a[n][0] - b[n][0]))) for n in a}
    # With the bug, every one of these is exactly 0. Measured on the dev row:
    # 23 of 30 move more than 1 deg (21-26 across the six rows), worst 52.8 deg.
    assert max(delta.values()) > 20.0, delta
    moved = [n for n, v in delta.items() if v > 1.0]
    assert len(moved) >= 20, (
        f"only {len(moved)} of 30 signed angles changed with a rig whose flexion "
        f"references sit 11.6-139.4 deg away -- the rig is being ignored")
