"""Ankles rotate about Z (plantar/dorsiflexion) plus X (inversion/eversion),
never Y.

Scott (2026-08-21): "ankle is incorrect - axis is +/-Z". The rig agrees
(config/joint-limits.js ankle: z [-81, 106], x [-75, 75], no y entry), and
his hand-posed captures confirm it: every posed left-ankle delta is pure Z
to two decimals; right ankles are Z+X mixes; none carry Y.

A rotation of the form Rz(theta)*Rx(psi) has quaternion [c1c2, c1s2, s1s2,
s1c2], so w*y - x*z == 0 identically -- the checkable no-Y invariant.
Written to FAIL against the free-orientation Kabsch ankle.
"""
import json
from pathlib import Path

import numpy as np

from retargeting.core.math_utils import QuaternionMath as QM
from retargeting.retargeters.posegoblin_rig import (
    load_rig, rig_targets_from_mhr70, solve_rig_locals, fk_world_orientations)

FIX = json.loads(
    (Path(__file__).parent / "fixtures" / "ground_truth_captures.json").read_text())
RIG = load_rig()
I = RIG.index_of_name
conj = lambda q: np.array([q[0], -q[1], -q[2], -q[3]])
unit = lambda v: np.asarray(v, float) / np.linalg.norm(v)


def _delta(bone, local_q):
    d = QM.multiply(conj(RIG.rest_local_q[I[bone]]), np.asarray(local_q, float))
    return -d if d[0] < 0 else d


def _angle(q):
    return float(np.degrees(2 * np.arctan2(np.linalg.norm(q[1:]), q[0])))


def _solve(row):
    return solve_rig_locals(RIG, rig_targets_from_mhr70(
        np.asarray(row["kp70"], np.float32).reshape(70, 3)))


def test_solved_ankles_have_no_y_component():
    checked = 0
    for row in FIX:
        L = _solve(row)
        for bone in ("left_ankle", "right_ankle"):
            d = _delta(bone, L[I[bone]])
            if _angle(d) < 10:
                continue
            checked += 1
            inv = float(d[0] * d[2] - d[1] * d[3])
            assert abs(inv) < 1e-6, (
                f"{row['tag']}/{bone}: delta {_angle(d):.1f} deg carries a Y "
                f"component (w*y - x*z = {inv:+.4f})")
    assert checked >= 6, f"positive control: only {checked} posed ankles checked"


def test_solved_ankle_reproduces_the_target_foot_axis():
    # The Z*X fit aims heel->toes exactly whenever the pitch is reachable.
    Wr = fk_world_orientations(RIG, RIG.rest_local_q)
    checked = 0
    for row in FIX:
        T = rig_targets_from_mhr70(np.asarray(row["kp70"], np.float32).reshape(70, 3))
        L = _solve(row)
        for side in ("left", "right"):
            ai, hi = I[f"{side}_ankle"], I[f"{side}_heel"]
            bi, si = I[f"{side}_big_toe"], I[f"{side}_small_toe"]
            if not all(k in T for k in (ai, hi, bi, si)):
                continue
            # world orientation of the ankle from the solved locals
            w = None
            j = ai
            chain = []
            while j is not None:
                chain.append(j)
                j = RIG.parent[j]
            for j in reversed(chain):
                q = np.asarray(L[j], float)
                w = q if w is None else QM.multiply(w, q)
            toe_mid_rest = 0.5 * (RIG.rest_world_p[bi] + RIG.rest_world_p[si])
            rest_axis = unit(toe_mid_rest - RIG.rest_world_p[hi])
            # world foot axis = (world delta of ankle) applied to rest axis
            D_a = QM.multiply(w, conj(Wr[ai]))
            got = unit(QM.rotate_vector(D_a, rest_axis))
            toe_mid_t = 0.5 * (np.asarray(T[bi], float) + np.asarray(T[si], float))
            want = unit(toe_mid_t - np.asarray(T[hi], float))
            checked += 1
            cos = float(np.dot(got, want))
            assert cos > 0.995, (
                f"{row['tag']}/{side}_ankle: foot axis off by "
                f"{np.degrees(np.arccos(np.clip(cos, -1, 1))):.1f} deg")
    assert checked >= 8, f"positive control: only {checked} feet checked"


def test_ankle_roll_bias_applies_at_state_assembly():
    """Scott's calibration (2026-08-22): from the roll ladder on row
    c5cf2013 he picked +30 deg about the foot axis. The detector's foot
    labels carry a roll bias only an eye can calibrate; the correction
    ramps with pointedness (flat planted feet -- long approved -- stay
    untouched) and lives in rig_state_from_mhr70, NOT solve_rig_locals,
    so the rest-roundtrip and the machine gate stay pure."""
    from retargeting.retargeters.posegoblin_rig import rig_state_from_mhr70
    REG = json.loads(
        (Path(__file__).parent / "fixtures" / "regression_rows.json").read_text())
    kp = np.asarray(REG[0]["kp70"], np.float32).reshape(70, 3)
    T = rig_targets_from_mhr70(kp)
    L = solve_rig_locals(RIG, T)          # pure solve, no bias
    st = rig_state_from_mhr70(kp)         # assembled state, bias applied
    Wr = fk_world_orientations(RIG, RIG.rest_local_q)

    def wdelta_from(locals_by, ai, name_keyed):
        w = None
        chain = []
        j = ai
        while j is not None:
            chain.append(j)
            j = RIG.parent[j]
        for j in reversed(chain):
            if name_keyed:
                q4 = locals_by[RIG.name[j]]
                q = np.array([q4["_w"], q4["_x"], q4["_y"], q4["_z"]], float)
            else:
                q = np.asarray(locals_by[j], float)
            w = q if w is None else QM.multiply(w, q)
        return QM.multiply(w, conj(Wr[ai]))

    # Scott (2026-08-22): "the left foot needs the opposite sign" -- the
    # bias mirrors anatomically: +30 right, -30 left, about each foot's own
    # heel->toe axis. Signed checks per side; row 0 pins the right (+),
    # row 1 (pointed LEFT foot, 0.92) pins the left (-).
    def check(row_idx, side, lo, hi, sign):
        kp2 = np.asarray(REG[row_idx]["kp70"], np.float32).reshape(70, 3)
        L2 = solve_rig_locals(RIG, rig_targets_from_mhr70(kp2))
        st2 = rig_state_from_mhr70(kp2)
        ai = I[f"{side}_ankle"]
        d_pure = wdelta_from(L2, ai, name_keyed=False)
        d_state = wdelta_from(st2["pose"], ai, name_keyed=True)
        diff = QM.multiply(d_state, conj(d_pure))
        if diff[0] < 0:
            diff = -np.asarray(diff, float)
        ang = float(np.degrees(2 * np.arctan2(np.linalg.norm(diff[1:]), diff[0])))
        assert lo <= ang <= hi, (
            f"row{row_idx}/{side}: state-vs-solve differs {ang:.1f} deg, "
            f"expected [{lo}, {hi}]")
        if ang > 5:
            hi_, bi, si = (I[f"{side}_{x}"] for x in ("heel", "big_toe", "small_toe"))
            axis = unit(np.asarray(QM.rotate_vector(
                d_pure, unit(0.5 * (RIG.rest_world_p[bi] + RIG.rest_world_p[si])
                             - RIG.rest_world_p[hi_])), float))
            d = float(np.dot(unit(diff[1:]), axis))
            assert d * sign > 0.99, (
                f"row{row_idx}/{side}: bias sign wrong (dot={d:+.3f}, want "
                f"sign {'+' if sign > 0 else '-'})")
    from retargeting.retargeters.posegoblin_rig import _ANKLE_ROLL_BIAS_DEG as B
    check(0, "right", B - 3.0, B + 3.0, +1)   # Scott's calibration (currently 20)
    check(0, "left", 0.0, 5.0, +1)            # planted: no bias
    check(1, "left", B - 3.0, B + 3.0, -1)    # pointed left: mirrored sign
