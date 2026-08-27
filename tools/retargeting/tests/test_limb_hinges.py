"""Limb chains must respect the rig's hinge anatomy.

Scott (2026-08-21): "the Elbows need to rotate around +/-Z. internal/external
rotation of shoulder is X." PoseGoblin's own config agrees: every limb chain
declares hingeAxis 'z' (config/ik-config.js), elbow z-range [-148, 20]
(flexion is -Z), knee z-range [-20.5, 151.5] (flexion is +Z)
(config/joint-limits.js). The hand-posed captures confirm it empirically:
every posed elbow/knee delta from rest is Z-dominant to two decimals.

Written to FAIL against the minimal-rotation solve, which invents twist: it
aims each bone at its child and leaves the bend axis wherever that lands.
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

MID_JOINTS = ("left_elbow", "right_elbow", "left_knee", "right_knee")
CHAIN = {"left_elbow": ("left_shoulder", "left_wrist"),
         "right_elbow": ("right_shoulder", "right_wrist"),
         "left_knee": ("left_hip", "left_ankle"),
         "right_knee": ("right_hip", "right_ankle")}

conj = lambda q: np.array([q[0], -q[1], -q[2], -q[3]])
unit = lambda v: np.asarray(v, float) / np.linalg.norm(v)


def _axis_angle(q):
    q = np.asarray(q, float)
    if q[0] < 0:
        q = -q
    s = np.linalg.norm(q[1:])
    if s < 1e-9:
        return np.zeros(3), 0.0
    return q[1:] / s, float(np.degrees(2 * np.arctan2(s, q[0])))


def _delta_from_rest(bone, local_q):
    return QM.multiply(conj(RIG.rest_local_q[I[bone]]), np.asarray(local_q, float))


def _solve(row):
    return solve_rig_locals(RIG, rig_targets_from_mhr70(
        np.asarray(row["kp70"], np.float32).reshape(70, 3)))


def test_solved_elbows_and_knees_bend_purely_about_local_z():
    checked = 0
    for row in FIX:
        L = _solve(row)
        for bone in MID_JOINTS:
            ax, ang = _axis_angle(_delta_from_rest(bone, L[I[bone]]))
            if ang < 15:
                continue
            checked += 1
            assert abs(ax[2]) > 0.99, (
                f"{row['tag']}/{bone}: bends {ang:.1f} deg about "
                f"axis=({ax[0]:+.2f},{ax[1]:+.2f},{ax[2]:+.2f}) -- not the Z hinge")
    assert checked >= 8, f"positive control: only {checked} bent mid-joints checked"


def test_solved_bend_sign_matches_the_hand_posed_captures():
    # A joint Scott clearly flexed must come out flexed, not hyperextended.
    checked = 0
    for row in FIX:
        L = _solve(row)
        for bone in MID_JOINTS:
            s_ax, s_ang = _axis_angle(_delta_from_rest(bone, [
                row["scott_pose"][bone][3], row["scott_pose"][bone][0],
                row["scott_pose"][bone][1], row["scott_pose"][bone][2]]))
            o_ax, o_ang = _axis_angle(_delta_from_rest(bone, L[I[bone]]))
            if s_ang < 25 or o_ang < 15 or abs(s_ax[2]) < 0.9:
                continue
            checked += 1
            assert np.sign(o_ax[2]) == np.sign(s_ax[2]), (
                f"{row['tag']}/{bone}: ours bends {o_ang:.1f} deg about "
                f"z={o_ax[2]:+.2f}, Scott's {s_ang:.1f} deg about z={s_ax[2]:+.2f}")
    assert checked >= 4, f"positive control: only {checked} sign comparisons ran"


def test_shoulder_twist_lays_elbow_hinge_on_the_bend_plane():
    # World-space invariant of the anchored chain: the shoulder delta carries
    # the elbow's rest +Z onto the bend-plane normal (sign: elbow flexion -Z).
    Wr = fk_world_orientations(RIG, RIG.rest_local_q)
    checked = 0
    for row in FIX:
        T = rig_targets_from_mhr70(
            np.asarray(row["kp70"], np.float32).reshape(70, 3))
        L = _solve(row)
        for mid, (root, end) in CHAIN.items():
            if "elbow" not in mid:
                continue
            u = T[I[mid]] - T[I[root]]
            f = T[I[end]] - T[I[mid]]
            n_raw = np.cross(unit(u), unit(f))
            if np.linalg.norm(n_raw) < 0.26:    # < ~15 deg bend: no stable plane
                continue
            checked += 1
            n = -unit(n_raw)                     # elbow flexion is local -Z
            # D_root = W_root * Wr_root^-1 ; W_root = W_parent * L_root
            w = None
            chain_idx = []
            j = I[root]
            while j is not None:
                chain_idx.append(j)
                j = RIG.parent[j]
            for j in reversed(chain_idx):
                q = np.asarray(L[j], float)
                w = q if w is None else QM.multiply(w, q)
            D_root = QM.multiply(w, conj(Wr[I[root]]))
            hinge_world = QM.rotate_vector(
                D_root, QM.rotate_vector(Wr[I[mid]], np.array([0.0, 0.0, 1.0])))
            d = float(np.dot(unit(hinge_world), n))
            assert d > 0.98, (
                f"{row['tag']}/{mid}: shoulder leaves hinge {np.degrees(np.arccos(np.clip(d,-1,1))):.1f} "
                f"deg off the bend normal")
    assert checked >= 4, f"positive control: only {checked} bent elbows checked"
