"""The PoseGoblin rig as a versioned asset, and the rig-native solve.

Why this exists: the shipped rig rests in an A-pose, authored Z-up, and its
rest directions agree with default_human.json at mean cosine +0.065. No fixed
per-bone correction can bridge two different rest GEOMETRIES (measured spread
61 deg across five hand-matched poses), so poses must be solved against the
rig's own rest. See the design doc named in the plan header.

Bones are keyed by INDEX, not name. PoseGoblin's own captureCurrentState/
RecallPoseCommand key bone state by `state.pose[child.name]`, and one pair of
bones -- the toe-adjacent joint under each heel -- legitimately shares the
literal name "joint7" (both collapse to the same state.pose entry on the live
rig; matching that behaviour, not diverging from it, is correct). Index
identity is therefore load-bearing for FK: `Rig.parent`, `rest_local_q`,
`rest_local_p`, `rest_world_p` and `children` are all index-keyed so the two
`joint7` bones keep independent data instead of one silently overwriting the
other. `index_of_name` resolves the 72 genuinely-unique names to an index and
deliberately omits ambiguous ones -- looking up "joint7" by name raises
KeyError rather than silently returning one of the two.

Ten bones (five fingers/thumb x two hands; 40 bones counting descendants) have
`parent: None` in the capture. On the live rig they are actually parented
under a Group carrying its own non-identity rotation AND a 0.1 scale -- the
container-transform contingency this module's tests anticipate -- but
modelling rotation+scale propagation through FK for them was ruled out of
scope (finger articulation is not a search input; hand pose is invisible at
display scale). Their `solve` flag is False, and fk_world_orientations/
fk_world_positions skip them. Nothing is deleted from the asset -- a future
re-rig may reconnect them -- but the FK self-consistency test pins the solved
set to exactly what's reachable from pelvis, so a re-rig that changes this
shows up as a failing assertion, never as silence.
"""
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np

from ..core.math_utils import QuaternionMath

_ASSET = Path(__file__).resolve().parent.parent / "bind_poses" / "posegoblin_rig_v1.json"


@dataclass(frozen=True)
class Rig:
    version: str
    order: list          # parent-first bone INDICES (int), all 74, solved and unsolved
    name: dict            # index -> bone name (str); NOT unique, see module docstring
    parent: dict          # index -> parent index | None
    rest_local_q: dict    # index -> (4,) [w,x,y,z]
    rest_local_p: dict    # index -> (3,)
    rest_world_p: dict    # index -> (3,)
    children: dict        # index -> [child indices], asset order
    solve: dict           # index -> bool; False for the finger/thumb islands
    index_of_name: dict   # unique name -> index; ambiguous names (e.g. "joint7") omitted


def load_rig(path: Path = _ASSET) -> Rig:
    d = json.loads(Path(path).read_text())
    bones = d["bones"]
    order = list(range(len(bones)))  # asset position IS the index; verified parent-first
    name = {i: b["name"] for i, b in enumerate(bones)}
    parent = {i: b["parent"] for i, b in enumerate(bones)}
    children: dict = {i: [] for i in order}
    for i, b in enumerate(bones):
        if b["parent"] is not None:
            children[b["parent"]].append(i)
    name_counts = Counter(name.values())
    index_of_name = {b["name"]: i for i, b in enumerate(bones) if name_counts[b["name"]] == 1}
    return Rig(
        version=d["version"], order=order, name=name, parent=parent,
        rest_local_q={i: np.asarray(b["rest_local_q"], float) for i, b in enumerate(bones)},
        rest_local_p={i: np.asarray(b["rest_local_p"], float) for i, b in enumerate(bones)},
        rest_world_p={i: np.asarray(b["rest_world_p"], float) for i, b in enumerate(bones)},
        children=children,
        solve={i: bool(b["solve"]) for i, b in enumerate(bones)},
        index_of_name=index_of_name,
    )


def fk_world_orientations(rig: Rig, local_q: dict) -> dict:
    """World orientations for the SOLVED set only (see module docstring)."""
    W: dict = {}
    for i in rig.order:
        if not rig.solve[i]:
            continue
        p = rig.parent[i]
        q = np.asarray(local_q[i], float)
        W[i] = q if p is None else QuaternionMath.multiply(W[p], q)
    return W


def fk_world_positions(rig: Rig, local_q: dict) -> dict:
    """World positions for the SOLVED set only (see module docstring)."""
    Wq = fk_world_orientations(rig, local_q)
    P: dict = {}
    for i in rig.order:
        if not rig.solve[i]:
            continue
        p = rig.parent[i]
        if p is None:
            P[i] = rig.rest_local_p[i].copy()
        else:
            P[i] = P[p] + QuaternionMath.rotate_vector(Wq[p], rig.rest_local_p[i])
    return P


def _kabsch_q(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Least-squares rotation (as [w,x,y,z]) taking unit rows a -> b."""
    u, _s, vt = np.linalg.svd(a.T @ b)
    d = float(np.sign(np.linalg.det(vt.T @ u.T))) or 1.0
    return QuaternionMath.from_matrix(vt.T @ np.diag([1.0, 1.0, d]) @ u.T)


def solve_rig_locals(rig: Rig, targets: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """Solve absolute local quaternions posing the rig onto *targets*.

    Per bone, a world DELTA D(b) rotates the rig's rest bone directions onto
    the target directions: all-children Kabsch for >=2 usable children,
    minimal rotation for exactly 1 (a single pair leaves twist unconstrained;
    an SVD there would invent one), parent's delta when the bone or its
    children have no targets. The consumer REPLACES locals, and most rig rest
    locals are non-identity, so deltas are composed with the rest:

        L(b) = (D(p) * Wr(p))^-1 * D(b) * Wr(b)      (root: D * Wr)

    `targets` is keyed by bone INDEX (subset OK; world positions, Y-up).
    Returns absolute local quaternions keyed by bone INDEX for the SOLVED set
    only (see module docstring) -- fk_world_orientations/fk_world_positions
    are themselves solved-set-only, so Wr below has no entries for unsolved
    bones. `targets` may carry entries for unsolved bones; they are never
    read, since no solved bone has an unsolved child (the finger/thumb
    islands are disconnected at their captured parent: None roots). Unsolved
    bones are Task 3's concern (rest locals, unchanged).
    """
    Wr = fk_world_orientations(rig, rig.rest_local_q)
    solved = [i for i in rig.order if rig.solve[i]]
    D: dict = {}
    for n in solved:
        pairs = []
        if n in targets:
            for c in rig.children[n]:
                if c not in targets:
                    continue
                rb = rig.rest_world_p[c] - rig.rest_world_p[n]
                tb = np.asarray(targets[c], float) - np.asarray(targets[n], float)
                nr, nt = np.linalg.norm(rb), np.linalg.norm(tb)
                if nr > 1e-6 and nt > 1e-6:
                    pairs.append((rb / nr, tb / nt))
        p = rig.parent[n]
        if len(pairs) >= 2:
            D[n] = _kabsch_q(np.array([x[0] for x in pairs]),
                             np.array([x[1] for x in pairs]))
        elif len(pairs) == 1:
            D[n] = QuaternionMath.from_two_vectors(pairs[0][0], pairs[0][1])
        else:
            D[n] = D[p] if p is not None else QuaternionMath.identity()

    L: dict = {}
    for n in solved:
        p = rig.parent[n]
        wb = QuaternionMath.multiply(D[n], Wr[n])
        if p is None:
            L[n] = wb
        else:
            wp = QuaternionMath.multiply(D[p], Wr[p])
            L[n] = QuaternionMath.multiply(QuaternionMath.conjugate(wp), wb)
    return L


ROOT_DISPLAY_YAW_DEG: float = 0.0   # fitted against ground-truth captures (Task 5); inert until then


def rig_targets_from_mhr70(kp_cam: np.ndarray) -> dict[int, np.ndarray]:
    """(70,3) CV camera keypoints -> rig-bone-INDEX-keyed world targets, Y-up.

    Reuses MHR70Retargeter.compute_joint_positions for cv_to_yup, the
    keypoint mapping, and the synthesised joints (root = hip midpoint, head =
    ear midpoint, interpolated spine). This loads default_human.json only to
    construct a valid MHR70Retargeter instance -- compute_joint_positions
    never reads self.bind_pose, so nothing here is SOLVED against that
    asset; the solve itself (solve_rig_locals) never touches it either.

    Index-keyed, not name-keyed (ruling 6): solve_rig_locals and
    fk_world_positions are index-keyed throughout because the rig has two
    bones sharing the name "joint7", so rig.index_of_name resolves the 72
    genuinely unique names and deliberately raises KeyError for "joint7".
    MHR-70 names are unique and none of them maps to a mannequin bone named
    "joint7", so this lookup is safe here -- but if it ever DID raise, it
    must raise, not silently drop the joint.
    """
    from .mhr70_retargeter import MHR70Retargeter
    from ..bind_poses.loader import BindPoseLoader
    from ..exporters.mannequin_exporter import MHR70_TO_MANNEQUIN

    rig = load_rig()
    pos = MHR70Retargeter(BindPoseLoader.get_default_bind_pose()) \
        .compute_joint_positions(np.asarray(kp_cam, np.float32))
    targets: dict[int, np.ndarray] = {}
    for j, p in pos.items():
        bone_name = MHR70_TO_MANNEQUIN.get(j)
        if bone_name is None:
            continue                                  # no mannequin equivalent: deliberate skip
        targets[rig.index_of_name[bone_name]] = np.asarray(p, float)   # KeyError propagates, never caught

    # ruling 9 (amended): MHR-70's `root` is the exact hip MIDPOINT -- level
    # with the hips, so pelvis->left_hip and pelvis->right_hip are always
    # exactly 180 deg apart in this raw target. The rig's own pelvis sits
    # ABOVE the hip line (105.2 deg apart in rest): same topology, different
    # height, and a rotation cannot change the angle between two vectors, so
    # a hip-midpoint pelvis target makes the pelvis's own Kabsch fit
    # unsatisfiable by construction. Replace it with the rig's own
    # above-the-hip-line offset, rebuilt in the target's own frame.
    targets[rig.index_of_name["pelvis"]] = synthesize_pelvis_target(rig, targets)
    return targets


def _leg_len(points: dict, hip: int, knee: int, ankle: int) -> float:
    """hip->knee->ankle chain length. *points* and the three args share one
    index space (rig bone index): both rig.rest_world_p and the targets from
    rig_targets_from_mhr70 are index-keyed (ruling 6)."""
    return (float(np.linalg.norm(points[knee] - points[hip]))
            + float(np.linalg.norm(points[ankle] - points[knee])))


def _orthonormal_frame_from_hips_and_up(points: dict, left_hip: int, right_hip: int,
                                         up_ref: int) -> tuple:
    """right = direction from right_hip to left_hip; up = direction from the
    hip midpoint to *up_ref*, re-orthonormalised against right (Gram-Schmidt
    via cross products, so it's exact even when the raw up reference isn't
    perfectly perpendicular to the hip line); fwd completes a right-handed
    orthonormal basis. Returns (right, up, fwd, hip_mid)."""
    hip_mid = (points[left_hip] + points[right_hip]) / 2
    right = points[left_hip] - points[right_hip]
    right = right / np.linalg.norm(right)
    up_raw = points[up_ref] - hip_mid
    fwd = np.cross(right, up_raw)
    fwd = fwd / np.linalg.norm(fwd)
    up = np.cross(fwd, right)
    up = up / np.linalg.norm(up)
    return right, up, fwd, hip_mid


def synthesize_pelvis_target(rig: Rig, targets: dict[int, np.ndarray]) -> np.ndarray:
    """Ruling 9 (amended) -- rebuild the rig's own "pelvis sits above the
    hip line" offset in the TARGET's own frame, instead of leaving the
    pelvis target at the raw hip midpoint MHR70Retargeter computes.

    1. Rig rest: how far above the hip line the pelvis sits, expressed
       scale-free (as a fraction of the rig's own leg length) in the rig's
       own world-aligned right/up/forward axes.
    2. Target: an orthonormal frame built from the TARGETS themselves --
       never the rig's world axes, since the subject may be lying down or
       inverted -- so the reconstructed offset points the anatomically
       correct way regardless of the subject's orientation in camera space.
    3. Re-express the rig's offset in that frame and scale by the target's
       own leg length.

    `targets` must already contain left_hip, right_hip, left/right knee and
    ankle, and (spine_1 or neck); everything but the pelvis entry is read,
    not written.
    """
    li = rig.index_of_name
    lhip, rhip = li["left_hip"], li["right_hip"]
    lknee, rknee = li["left_knee"], li["right_knee"]
    lankle, rankle = li["left_ankle"], li["right_ankle"]
    pelvis_i = li["pelvis"]

    rest = rig.rest_world_p
    hip_mid_rest = (rest[lhip] + rest[rhip]) / 2
    offset_rest = rest[pelvis_i] - hip_mid_rest
    rig_leg_len = np.mean([_leg_len(rest, lhip, lknee, lankle),
                            _leg_len(rest, rhip, rknee, rankle)])
    offset_frac = offset_rest / rig_leg_len

    up_ref = li["spine_1"] if li["spine_1"] in targets else li["neck"]
    right_t, up_t, fwd_t, hip_mid_t = _orthonormal_frame_from_hips_and_up(
        targets, lhip, rhip, up_ref)
    tgt_leg_len = np.mean([_leg_len(targets, lhip, lknee, lankle),
                            _leg_len(targets, rhip, rknee, rankle)])

    offset_target = (offset_frac[0] * right_t + offset_frac[1] * up_t
                      + offset_frac[2] * fwd_t) * tgt_leg_len
    return hip_mid_t + offset_target


def rig_state_from_mhr70(kp_cam: np.ndarray) -> dict:
    """(70,3) CV camera keypoints -> full mannequinState for the PoseGoblin
    viewer: every rig bone posed (rig units), ready to serialize."""
    rig = load_rig()
    li = rig.index_of_name
    targets = rig_targets_from_mhr70(kp_cam)
    solved = solve_rig_locals(rig, targets)   # index-keyed, the 34 SOLVED bones ONLY (ruling 7)

    if ROOT_DISPLAY_YAW_DEG:
        t = np.radians(ROOT_DISPLAY_YAW_DEG)
        yaw = np.array([np.cos(t / 2), 0.0, np.sin(t / 2), 0.0])
        pelvis_i = li["pelvis"]
        solved[pelvis_i] = QuaternionMath.multiply(yaw, solved[pelvis_i])

    # ruling 7: solve_rig_locals deliberately covers only the solved bones --
    # padding its own output would claim to have solved bones it never
    # touched. Assembly owns the merge: every bone must land in the state, so
    # the 40 unsolved finger/thumb bones fall back to their REST locals.
    full_by_index = {**rig.rest_local_q, **solved}

    # ruling 5: PoseGoblin reads state.pose[child.name], so the emitted pose
    # is NAME-keyed, not index-keyed. The two "joint7" bones collapse to a
    # single entry -- last-one-wins by iterating rig.order in parent-first
    # order, matching PoseGoblin's own captureCurrentState/RecallPoseCommand
    # contract instead of diverging from it.
    pose = {rig.name[i]: QuaternionMath.to_threejs_dict(full_by_index[i]) for i in rig.order}

    # scale: subject leg length (meters, pose-invariant) -> rig units
    rig_leg = np.mean([
        _leg_len(rig.rest_world_p, li["left_hip"], li["left_knee"], li["left_ankle"]),
        _leg_len(rig.rest_world_p, li["right_hip"], li["right_knee"], li["right_ankle"]),
    ])
    tgt_leg = np.mean([
        _leg_len(targets, li["left_hip"], li["left_knee"], li["left_ankle"]),
        _leg_len(targets, li["right_hip"], li["right_knee"], li["right_ankle"]),
    ])
    s = rig_leg / max(tgt_leg, 1e-6)

    # Feet mannequin bone names verified against MHR70_TO_MANNEQUIN's VALUES
    # (not guessed): left_heel/right_heel/left_big_toe/right_big_toe.
    feet_idx = [li[n] for n in ("left_heel", "right_heel", "left_big_toe", "right_big_toe")]
    feet = [targets[i][1] for i in feet_idx if i in targets]

    # ruling 10: pelvisPosition is the rig's REST pelvis position, not the
    # subject's camera-space position scaled to rig units. A pose viewer
    # shows the POSE, not the subject's translation through space -- the
    # scaled camera-space position could (and for this fixture's Cyr-wheel
    # capture, did) land the figure almost inside the fixed default camera.
    # `s` is still needed for groundY, so it stays computed above.
    pelvis = rig.rest_world_p[li["pelvis"]]

    from ..exporters.mannequin_exporter import MannequinExporter
    return {
        "pose": pose,
        "pelvisPosition": {"x": float(pelvis[0]), "y": float(pelvis[1]), "z": float(pelvis[2])},
        "groundY": float(min(feet) * s) if feet else 0.0,
        "cameraState": MannequinExporter.get_default_camera_state(),
        "rigVersion": rig.version,
        "retargetVersion": 2,
    }
