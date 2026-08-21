"""Wrists carry the palm orientation from the hand keypoints.

Scott (2026-08-21): "do we do anything with the wrists? ... or pronation
supination of wrist?" Before this: nothing -- the wrist inherited the
elbow's frame rigidly, so palm-up vs palm-down never reached the mannequin.
MHR-70 carries all knuckle keypoints, so the wrist gets the head treatment:
a two-vector anchor from the hand axis (wrist -> middle knuckle) and the
knuckle line (index_1 <-> pinky_1), which encodes pronation/supination,
flexion, and deviation together. Written to FAIL against the inherit-elbow
fallback.
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


def test_wrist_world_hits_hand_axis_and_knuckle_line():
    Wr = fk_world_orientations(RIG, RIG.rest_local_q)
    checked = 0
    for row in FIX:
        T = rig_targets_from_mhr70(np.asarray(row["kp70"], np.float32).reshape(70, 3))
        L = solve_rig_locals(RIG, rig_targets_from_mhr70(
            np.asarray(row["kp70"], np.float32).reshape(70, 3)))
        for side in ("left", "right"):
            wi = I[f"{side}_wrist"]
            mi = I[f"{side}_middle_finger_1"]
            ii = I[f"{side}_index_finger_1"]
            pi = I[f"{side}_pinky_finger_1"]
            if not all(k in T for k in (wi, mi, ii, pi)):
                continue
            w = None
            j = wi
            chain = []
            while j is not None:
                chain.append(j)
                j = RIG.parent[j]
            for j in reversed(chain):
                w = np.asarray(L[j], float) if w is None else QM.multiply(
                    w, np.asarray(L[j], float))
            D_w = QM.multiply(w, conj(Wr[wi]))
            for a, b, tag in (
                (RIG.rest_world_p[mi] - RIG.rest_world_p[wi],
                 np.asarray(T[mi], float) - np.asarray(T[wi], float), "hand axis"),
                (RIG.rest_world_p[ii] - RIG.rest_world_p[pi],
                 np.asarray(T[ii], float) - np.asarray(T[pi], float), "knuckle line"),
            ):
                cos = float(np.dot(unit(QM.rotate_vector(D_w, unit(a))), unit(b)))
                assert cos > 0.995, (
                    f"{row['tag']}/{side}_wrist {tag} off by "
                    f"{np.degrees(np.arccos(np.clip(cos, -1, 1))):.1f} deg")
            checked += 1
    assert checked >= 8, f"positive control: only {checked} wrists checked"
