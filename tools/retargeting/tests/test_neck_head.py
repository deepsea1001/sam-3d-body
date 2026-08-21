"""Neck nods about local Y and shares the nod with the head.

Scott (2026-08-21): "head and neck forward tilt in Y." His captures agree to
two decimals: every posed neck delta is pure Y (+38.4 looking down in the
crouch, -28.1 in the inverted hoop), and he splits a look-down roughly
half-and-half between neck and head (crouch: neck +38 / head +36). The
config's head entry says the same (y = nod, x = turn, z = tip).

Written to FAIL against the minimal-rotation neck, which aims the head bone
and smears the nod across whatever axis that lands on.
"""
import json
from pathlib import Path

import numpy as np

from retargeting.core.math_utils import QuaternionMath as QM
from retargeting.retargeters.posegoblin_rig import (
    fk_world_orientations, load_rig, rig_targets_from_mhr70, solve_rig_locals)

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


def test_solved_neck_is_pure_y():
    checked = 0
    for row in FIX:
        L = _solve(row)
        d = _delta("neck", L[I["neck"]])
        # purity is about AXIS, not magnitude -- check any engaged neck. The
        # solve derives the nod from MHR (chest->head Y-twist / 2), which is
        # smaller than Scott's hand-split, so few fixtures exceed 8 deg.
        if _angle(d) < 3:
            continue
        checked += 1
        ax = d[1:] / np.linalg.norm(d[1:])
        assert abs(ax[1]) > 0.99, (
            f"{row['tag']}: neck delta {_angle(d):.1f} deg about "
            f"axis=({ax[0]:+.2f},{ax[1]:+.2f},{ax[2]:+.2f}) -- not the Y nod")
    assert checked >= 2, f"positive control: only {checked} posed necks checked"


def test_head_world_orientation_still_hits_nose_and_eyes():
    # The neck taking half the nod must not move the head off its anchor.
    Wr = fk_world_orientations(RIG, RIG.rest_local_q)
    checked = 0
    for row in FIX:
        T = rig_targets_from_mhr70(np.asarray(row["kp70"], np.float32).reshape(70, 3))
        L = _solve(row)
        hi, ni = I["head"], I["nose"]
        le, re_ = I["left_eye"], I["right_eye"]
        if not all(k in T for k in (hi, ni, le, re_)):
            continue
        w = None
        j = hi
        chain = []
        while j is not None:
            chain.append(j)
            j = RIG.parent[j]
        for j in reversed(chain):
            w = np.asarray(L[j], float) if w is None else QM.multiply(w, np.asarray(L[j], float))
        D_h = QM.multiply(w, conj(Wr[hi]))
        for a, b, tag in (
            (RIG.rest_world_p[ni] - RIG.rest_world_p[hi],
             np.asarray(T[ni], float) - np.asarray(T[hi], float), "nose"),
            (RIG.rest_world_p[le] - RIG.rest_world_p[re_],
             np.asarray(T[le], float) - np.asarray(T[re_], float), "eyeline"),
        ):
            cos = float(np.dot(unit(QM.rotate_vector(D_h, unit(a))), unit(b)))
            assert cos > 0.999, f"{row['tag']}: head {tag} off (cos={cos:.4f})"
        checked += 1
    assert checked >= 5, f"positive control: only {checked} heads checked"
