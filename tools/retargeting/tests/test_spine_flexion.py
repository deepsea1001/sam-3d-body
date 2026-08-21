"""The spine distributes the pelvis->chest rotation instead of staying a rod.

Root cause (2026-08-21, mhr70_retargeter.py:26-27): MHR-70 has no mid-spine
keypoints, so spine1/spine2 targets are LINEAR INTERPOLATION root->neck --
collinear by construction, zero curvature in every pose. Aiming them made
the column "stiff as a board" (Scott), with all torso pitch slammed into
the two end anchors.

Fix: spine_1's world delta is slerp(D_pelvis, D_chest, 0.65) -- exactly 65%
of the total torso rotation, the remainder absorbed by spine_2's local
(whose world anchor is untouched). The 0.65 lumbar share is measured from
Scott's own captures: crouch 60.7/32.8, hoop 37.0/19.9 -- both ~0.65.
Written to FAIL against the rod-aiming solve.
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


def _world_delta(L, Wr, idx):
    w = None
    j = idx
    chain = []
    while j is not None:
        chain.append(j)
        j = RIG.parent[j]
    for j in reversed(chain):
        w = np.asarray(L[j], float) if w is None else QM.multiply(w, np.asarray(L[j], float))
    return QM.multiply(w, conj(Wr[idx]))


def _axis_angle(q):
    q = np.asarray(q, float)
    if q[0] < 0:
        q = -q
    s = np.linalg.norm(q[1:])
    if s < 1e-9:
        return np.zeros(3), 0.0
    return q[1:] / s, float(np.degrees(2 * np.arctan2(s, q[0])))


def test_spine_splits_the_torso_rotation_065_035():
    Wr = fk_world_orientations(RIG, RIG.rest_local_q)
    checked = 0
    for row in FIX:
        L = solve_rig_locals(RIG, rig_targets_from_mhr70(
            np.asarray(row["kp70"], np.float32).reshape(70, 3)))
        Dp = _world_delta(L, Wr, I["pelvis"])
        D1 = _world_delta(L, Wr, I["spine_1"])
        D2 = _world_delta(L, Wr, I["spine_2"])
        ax_t, ang_t = _axis_angle(QM.multiply(conj(Dp), D2))
        if ang_t < 15:
            continue
        checked += 1
        ax_a, ang_a = _axis_angle(QM.multiply(conj(Dp), D1))
        ax_b, ang_b = _axis_angle(QM.multiply(conj(D1), D2))
        assert abs(ang_a - 0.65 * ang_t) < 1.0, (
            f"{row['tag']}: spine_1 carries {ang_a:.1f} deg of a {ang_t:.1f} deg "
            f"torso rotation -- expected 65% = {0.65*ang_t:.1f}")
        assert abs(ang_b - 0.35 * ang_t) < 1.0, (
            f"{row['tag']}: spine_2 local carries {ang_b:.1f} deg, expected "
            f"35% = {0.35*ang_t:.1f}")
        for ax, tag in ((ax_a, "spine_1"), (ax_b, "spine_2")):
            assert float(np.dot(ax, ax_t)) > 0.99, (
                f"{row['tag']}: {tag} rotates about a different axis than the "
                f"total (dot={np.dot(ax, ax_t):.3f})")
    assert checked >= 2, f"positive control: only {checked} bent torsos checked"
