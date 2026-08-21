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
            # Pointed feet (axis ~along the shin) deliberately trade axis
            # exactness for the toe line -- the sole's roll is what the eye
            # reads there. Assert the reference the solve aims.
            ki = RIG.parent[ai]
            shin = unit(np.asarray(T[ai], float) - np.asarray(T[ki], float))
            pointed = abs(float(np.dot(want, shin)))
            checked += 1
            if pointed > 0.55:
                tl_o = unit(QM.rotate_vector(D_a, unit(RIG.rest_world_p[bi] - RIG.rest_world_p[si])))
                tl_t = unit(np.asarray(T[bi], float) - np.asarray(T[si], float))
                cos = float(np.dot(tl_o, tl_t))
                assert cos > 0.98, (
                    f"{row['tag']}/{side}_ankle (pointed {pointed:.2f}): toe line off by "
                    f"{np.degrees(np.arccos(np.clip(cos, -1, 1))):.1f} deg")
            else:
                cos = float(np.dot(got, want))
                assert cos > 0.995, (
                    f"{row['tag']}/{side}_ankle: foot axis off by "
                    f"{np.degrees(np.arccos(np.clip(cos, -1, 1))):.1f} deg")
    assert checked >= 8, f"positive control: only {checked} feet checked"


def test_pointed_foot_toe_line_is_not_mirrored():
    """Scott (2026-08-22): inversion/eversion reads OPPOSITE on the pointed
    trailing foot of the walking pose. Root cause: the near-straight-leg
    twist derives the foot's azimuth from the heel->toe axis, which is
    nearly parallel to the shin on a pointed foot -- its perpendicular
    projection is noise, and noise mirrors. The toe LINE (big<->small toe)
    stays perpendicular to the leg regardless of pointing, so the solved
    world toe line must agree with the hand-posed capture's."""
    Wr = fk_world_orientations(RIG, RIG.rest_local_q)

    def world_dirs(local_by_name_or_idx, name_keyed):
        out = {}
        for side in ("left", "right"):
            bi, si = I[f"{side}_big_toe"], I[f"{side}_small_toe"]
            ai, hi = I[f"{side}_ankle"], I[f"{side}_heel"]
            P = {}
            for j0 in (bi, si, ai, hi):
                w = None
                pos = None
                chain = []
                j = j0
                while j is not None:
                    chain.append(j)
                    j = RIG.parent[j]
                for j in reversed(chain):
                    if name_keyed:
                        nm = RIG.name[j]
                        q4 = local_by_name_or_idx[nm]
                        q = np.array([q4[3], q4[0], q4[1], q4[2]], float)
                    else:
                        q = np.asarray(local_by_name_or_idx[j], float)
                    if w is None:
                        w, pos = q, RIG.rest_local_p[j].copy()
                    else:
                        pos = pos + np.asarray(QM.rotate_vector(w, RIG.rest_local_p[j]), float)
                        w = QM.multiply(w, q)
                P[j0] = pos
            out[side] = {"toe_line": unit(P[bi] - P[si]),
                         "foot_axis": unit(0.5 * (P[bi] + P[si]) - P[hi]),
                         "shin_ref": None}
        return out

    checked = 0
    for row in FIX:
        L = solve_rig_locals(RIG, rig_targets_from_mhr70(
            np.asarray(row["kp70"], np.float32).reshape(70, 3)))
        ours = world_dirs(L, name_keyed=False)
        scott = world_dirs(row["scott_pose"], name_keyed=True)
        for side in ("left", "right"):
            d = float(np.dot(ours[side]["toe_line"], scott[side]["toe_line"]))
            checked += 1
            assert d > 0.0, (
                f"{row['tag']}/{side}: solved toe line is MIRRORED vs the "
                f"hand-posed capture (dot={d:+.2f})")
    assert checked >= 10, f"positive control: only {checked} feet checked"


def test_pointed_foot_roll_follows_the_toe_line():
    """Regression row from Scott's report (2026-08-22): pointed trailing
    foot whose sole rolled ~48 deg against MHR's own toe-line evidence
    (dot 0.667) because the psi-branch tiebreak used the heel ray -- nearly
    parallel to the leg on a pointed foot, hence noise. The toe line is the
    well-conditioned roll reference and must win the branch."""
    REG = json.loads(
        (Path(__file__).parent / "fixtures" / "regression_rows.json").read_text())
    Wr = fk_world_orientations(RIG, RIG.rest_local_q)
    checked = 0
    for row in REG:
        kp = np.asarray(row["kp70"], np.float32).reshape(70, 3)
        T = rig_targets_from_mhr70(kp)
        L = solve_rig_locals(RIG, T)
        for side in ("left", "right"):
            bi, si = I[f"{side}_big_toe"], I[f"{side}_small_toe"]
            if bi not in T or si not in T:
                continue
            w = None
            P = {}
            for j0 in (bi, si):
                q = None
                pos = None
                chain = []
                j = j0
                while j is not None:
                    chain.append(j)
                    j = RIG.parent[j]
                for j in reversed(chain):
                    lq = np.asarray(L[j], float)
                    if q is None:
                        q, pos = lq, RIG.rest_local_p[j].copy()
                    else:
                        pos = pos + np.asarray(QM.rotate_vector(q, RIG.rest_local_p[j]), float)
                        q = QM.multiply(q, lq)
                P[j0] = pos
            tl_o = unit(P[bi] - P[si])
            tl_t = unit(np.asarray(T[bi], float) - np.asarray(T[si], float))
            d = float(np.dot(tl_o, tl_t))
            checked += 1
            assert d > 0.8, (
                f"{row['point_id'][:8]}/{side}: solved toe line {np.degrees(np.arccos(np.clip(d,-1,1))):.1f} "
                f"deg off MHR's toe-line evidence (dot={d:+.3f})")
    assert checked >= 2, f"positive control: only {checked} feet checked"
