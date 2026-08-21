"""Pelvis facing against the hand-posed ground-truth captures.

Written to fail against the mirrored + direction-only implementation
(2026-08-21): cv_to_yup negates only Y -- a det(-1) reflection -- so the
labeled anatomy of every target cloud is chirally mirrored (measured: 117/118
corpus rows, median chirality -0.883), and the pelvis's near-coplanar
children (|triple product| = 0.165) let Kabsch pay the reflection off in
facing: solved pelvis landed 174.8 deg from person-forward with 0.2 deg IQR.
The fixture is the five hand-posed captures -- the spec for what "facing the
right way" means.
"""
import json
from pathlib import Path

import numpy as np

from retargeting.core.math_utils import QuaternionMath as QM
from retargeting.retargeters.posegoblin_rig import (
    _orthonormal_frame_from_hips_and_up, load_rig, rig_targets_from_mhr70,
    solve_rig_locals)

FIX = json.loads(
    (Path(__file__).parent / "fixtures" / "ground_truth_captures.json").read_text())
RIG = load_rig()
I = RIG.index_of_name


def _unit(v):
    return np.asarray(v, float) / np.linalg.norm(v)


def _kp(row):
    return np.asarray(row["kp70"], np.float32).reshape(70, 3)


def _pelvis_facing(q_wxyz):
    """World facing of the pelvis: rest-local +Z carried by q ([w,x,y,z])."""
    rest = RIG.rest_local_q[I["pelvis"]]
    fwd_local = QM.rotate_vector(QM.conjugate(rest), np.array([0.0, 0.0, 1.0]))
    return _unit(QM.rotate_vector(np.asarray(q_wxyz, float), fwd_local))


def test_targets_are_right_handed():
    # Chirality: labeled (up x right-side) must agree with the face direction.
    # Upright captures only -- a folded/inverted body can legitimately put the
    # face near the hip plane, where this sign is not meaningful.
    for row in FIX:
        if row["tag"] not in ("martial-arts", "walking", "louis"):
            continue
        T = rig_targets_from_mhr70(_kp(row))
        mid = 0.5 * (T[I["left_hip"]] + T[I["right_hip"]])
        s = float(np.dot(
            np.cross(_unit(T[I["neck"]] - mid),
                     _unit(T[I["right_hip"]] - T[I["left_hip"]])),
            _unit(T[I["nose"]] - T[I["head"]])))
        assert s > 0.2, f"{row['tag']}: targets are mirrored (chirality s={s:.3f})"


def test_pelvis_faces_like_the_hand_posed_captures():
    for row in FIX:
        T = rig_targets_from_mhr70(_kp(row))
        L = solve_rig_locals(RIG, T)
        ours = _pelvis_facing(L[I["pelvis"]])          # root: local == world
        sq = row["scott_pose"]["pelvis"]               # three.js [x,y,z,w]
        scott = _pelvis_facing([sq[3], sq[0], sq[1], sq[2]])
        d = float(np.dot(ours, scott))
        deg = float(np.degrees(np.arccos(np.clip(d, -1, 1))))
        assert d > 0.0, (
            f"{row['tag']}: solved pelvis facing is {deg:.1f} deg from the "
            f"hand-posed capture's -- wrong hemisphere")


def test_pelvis_anchor_reproduces_target_frame():
    # The pelvis world delta must map the rig's rest hip frame EXACTLY onto
    # the target hip frame -- facing is a constraint, not a leftover.
    row = FIX[0]
    T = rig_targets_from_mhr70(_kp(row))
    L = solve_rig_locals(RIG, T)
    D = QM.multiply(np.asarray(L[I["pelvis"]], float),
                    QM.conjugate(RIG.rest_local_q[I["pelvis"]]))
    fr = _orthonormal_frame_from_hips_and_up(
        RIG.rest_world_p, I["left_hip"], I["right_hip"], I["spine_1"])
    ft = _orthonormal_frame_from_hips_and_up(
        T, I["left_hip"], I["right_hip"], I["spine_1"])
    for a, b, name in zip(fr[:3], ft[:3], ("right", "up", "fwd")):
        got = _unit(QM.rotate_vector(D, a))
        cos = float(np.dot(got, _unit(b)))
        assert cos > 0.999, f"pelvis anchor: {name} axis off (cos={cos:.4f})"
