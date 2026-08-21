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
