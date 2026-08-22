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

Ten finger/thumb chains (40 bones counting the tip) were, in v1, `parent:
None` orphans -- disconnected from the rig they are actually rendered on.
v2 (the default since v16 task 3) reconnects them through the ten Group
nodes they hang off in the live scene (`transform4`..`transform13`,
appended at indices 74..83): a fixed rotation plus a 0.1 uniform scale
each, applied to every finger/thumb bone beneath it. FK walks the full
84-node topology now (`Rig.topo_order`, NOT `Rig.order` -- the groups sit
at HIGHER indices than their finger children, so a pass over plain index
order hits an unprocessed parent) and threads that scale through node
POSITION (`fk_world_positions`; orientation is scale-invariant, so
`fk_world_orientations` carries no scale term). The 30 finger PHALANGES
(chain root + two interior joints; the ten tips stay `solve: False`) carry
`solve: True` in the v2 asset -- their chains are reachable from pelvis
now -- but MHR-70 supplies no finger ROTATION data, only raw keypoints that
`rig_targets_from_mhr70` maps onto them like any other bone. Posing them
from that data with no per-digit quality gate would be a silent, unreviewed
behavior change, so `solve_rig_locals` deliberately excludes them
(`_FINGER_PHALANGE_NAMES`) and they stay at their rest locals. A later task
adds the real finger path (MHR rotations, gated per digit) and lifts that
exclusion deliberately. Nothing is deleted from the asset, and both
versions stay loadable (`load_rig(path=...)`) -- the FK self-consistency
tests pin exactly what's reachable from pelvis and what's actually solved,
so a future re-rig that changes either shows up as a failing assertion,
never as silence.
"""
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np

from ..core.math_utils import QuaternionMath

_ASSET = Path(__file__).resolve().parent.parent / "bind_poses" / "posegoblin_rig_v2.json"


@dataclass(frozen=True)
class Rig:
    version: str
    order: list           # asset-order bone INDICES (int), ALL nodes. NOT guaranteed
                          # parent-first under v2 (see topo_order) -- kept as asset
                          # order because state serialization depends on it (ruling 6)
    topo_order: list      # parent-first bone INDICES (int), ALL nodes, groups included.
                          # v2's groups sit at 74..83 while their finger children sit at
                          # 14..57, so this is the order every FK walk must use instead
                          # of `order` (ruling 6)
    name: dict            # index -> bone name (str); NOT unique, see module docstring
    parent: dict          # index -> parent index | None
    rest_local_q: dict    # index -> (4,) [w,x,y,z]
    rest_local_p: dict    # index -> (3,)
    rest_world_p: dict    # index -> (3,)
    scale: np.ndarray     # (N,) float; per-node LOCAL uniform scale. 1.0 everywhere
                          # except the ten v2 groups, which carry 0.1
    is_group: np.ndarray  # (N,) bool; True for the ten constant group nodes (74..83).
                          # solve:False; excluded from the emitted pose -- the viewer
                          # already owns them as a fixed container transform
    children: dict        # index -> [child indices], asset order
    solve: dict           # index -> bool; False for the ten finger/thumb TIPS (and, in
                          # v1, the whole finger/thumb islands)
    index_of_name: dict   # unique name -> index; ambiguous names (e.g. "joint7") omitted


def _topological_order(order: list, parent: dict, children: dict) -> list:
    """Parent-before-child visitation order over every node in *order*
    (ruling 6): v2's index layout is deliberately non-topological -- the
    ten group nodes sit at 74..83 while their finger children sit at
    14..57 -- so a single forward pass over `range(len(order))` raises
    KeyError on exactly those ten chains the moment a parent lookup lands
    on an unprocessed group. Pre-order DFS from each root (`parent is
    None`), visiting children in ascending index order for a deterministic
    result."""
    visited: set = set()
    result: list = []

    def visit(i: int) -> None:
        if i in visited:
            return
        visited.add(i)
        result.append(i)
        for c in children[i]:
            visit(c)

    for i in order:
        if parent[i] is None:
            visit(i)
    # Review finding (v16 task 3): a dangling parent (points at an index
    # never visited as a root or reached as a child) or a cycle would
    # silently yield a SHORT order here -- every downstream FK walk would
    # just quietly cover fewer nodes, caught previously only by a
    # hardcoded set(range(84)) in one test. Fail at the source instead.
    assert set(result) == set(order), (
        f"topological order missed {set(order) - set(result)} -- a "
        "dangling or cyclic parent pointer left some node(s) unreached")
    return result


def load_rig(path: Path = _ASSET) -> Rig:
    d = json.loads(Path(path).read_text())
    bones = d["bones"]
    order = list(range(len(bones)))  # asset position IS the index; NOT necessarily
                                      # parent-first under v2 -- see topo_order
    name = {i: b["name"] for i, b in enumerate(bones)}
    parent = {i: b["parent"] for i, b in enumerate(bones)}
    children: dict = {i: [] for i in order}
    for i, b in enumerate(bones):
        if b["parent"] is not None:
            children[b["parent"]].append(i)
    name_counts = Counter(name.values())
    index_of_name = {b["name"]: i for i, b in enumerate(bones) if name_counts[b["name"]] == 1}
    return Rig(
        version=d["version"], order=order, topo_order=_topological_order(order, parent, children),
        name=name, parent=parent,
        rest_local_q={i: np.asarray(b["rest_local_q"], float) for i, b in enumerate(bones)},
        rest_local_p={i: np.asarray(b["rest_local_p"], float) for i, b in enumerate(bones)},
        rest_world_p={i: np.asarray(b["rest_world_p"], float) for i, b in enumerate(bones)},
        # v1.json predates scale/is_group (every bone implicitly scale 1.0,
        # not a group); .get(..., default) loads both versions uniformly.
        scale=np.array([float(b.get("scale", 1.0)) for b in bones], dtype=float),
        is_group=np.array([bool(b.get("is_group", False)) for b in bones], dtype=bool),
        children=children,
        solve={i: bool(b["solve"]) for i, b in enumerate(bones)},
        index_of_name=index_of_name,
    )


def fk_world_orientations(rig: Rig, local_q: dict) -> dict:
    """World orientations for EVERY node in the rig, solved or not (ruling
    2) -- not just the SOLVED set this function used to cover. A node
    absent from *local_q* falls back to its own rest local
    (`local_q.get(i, rig.rest_local_q[i])`), so every existing caller, which
    passes a solved-ONLY dict (`solve_rig_locals`'s own `rig.rest_local_q`
    seed; `rig_state_from_mhr70`'s `{**rig.rest_local_q, **solved}`; the
    poseforge3d harness's `fk_world_positions(rig, solved)`), keeps working
    unchanged -- the unsolved remainder just resolves to rest.

    Traversal follows `rig.topo_order`, never `rig.order`: under v2 the ten
    group nodes sit at indices 74..83 while their finger children sit at
    14..57, so a single pass over `rig.order` would look up an unprocessed
    parent (KeyError) for those ten chains the moment a finger bone is
    reached. Uniform scale never affects orientation, so this function
    carries no scale term -- see `fk_world_positions` for where the v2
    groups' 0.1 scale actually enters the computation.
    """
    W: dict = {}
    for i in rig.topo_order:
        p = rig.parent[i]
        q = np.asarray(local_q.get(i, rig.rest_local_q[i]), float)
        W[i] = q if p is None else QuaternionMath.multiply(W[p], q)
    return W


def fk_world_positions(rig: Rig, local_q: dict) -> dict:
    """World positions for EVERY node in the rig, solved or not (ruling 2)
    -- see `fk_world_orientations` for the local_q fallback-to-rest rule
    and why traversal must follow `rig.topo_order`.

    Scale-aware (v16 task 3): the ten v2 group nodes carry a 0.1 uniform
    scale that shrinks every rest offset beneath them, so position FK
    threads a cumulative scale down the chain --

        cum_scale(i) = cum_scale(parent) * rig.scale[i]
        W_p(i) = W_p(parent) + rotate(W_R(parent), cum_scale(parent) * rest_local_p(i))

    -- rather than the old scale-is-always-1 `P[p] + rotate(Wq[p],
    rest_local_p[i])`. Uniform scale never affects orientation, so
    `fk_world_orientations` (source of W_R above) needs no equivalent term
    -- but dropping this one puts finger tips ten times too far out
    (verified: left_index_finger_1 sits 0.994 rig units from left_wrist;
    without the group's 0.1 scale it lands at 9.94).
    """
    Wq = fk_world_orientations(rig, local_q)
    P: dict = {}
    cum_scale: dict = {}
    for i in rig.topo_order:
        p = rig.parent[i]
        if p is None:
            P[i] = rig.rest_local_p[i].copy()
            cum_scale[i] = float(rig.scale[i])
        else:
            P[i] = P[p] + QuaternionMath.rotate_vector(Wq[p], cum_scale[p] * rig.rest_local_p[i])
            cum_scale[i] = cum_scale[p] * float(rig.scale[i])
    return P


def _kabsch_q(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Least-squares rotation (as [w,x,y,z]) taking unit rows a -> b."""
    u, _s, vt = np.linalg.svd(a.T @ b)
    d = float(np.sign(np.linalg.det(vt.T @ u.T))) or 1.0
    return QuaternionMath.from_matrix(vt.T @ np.diag([1.0, 1.0, d]) @ u.T)


# Limb chains and their hinge sign, per the rig's own contract
# (poseGoblin config/ik-config.js: every chain hingeAxis 'z';
# config/joint-limits.js: elbow z [-148, 20] => flexion is LOCAL -Z,
# knee z [-20.5, 151.5] => flexion is LOCAL +Z; Scott 2026-08-21: "Elbows
# need to rotate around +/-Z. internal/external rotation of shoulder is X").
# The hand-posed captures confirm it: every posed elbow/knee delta from rest
# is Z-dominant to two decimals. sign s below maps the bend-plane normal
# n = unit(u x f) onto the mid-joint's LOCAL +Z world image: rotating the
# upper segment u onto the lower f is +theta about n; a flexion of -theta
# about local +Z is +theta about -(local +Z), so elbows need +Z -> -n.
_LIMB_CHAINS = (
    ("left_shoulder", "left_elbow", "left_wrist", -1.0),
    ("right_shoulder", "right_elbow", "right_wrist", -1.0),
    ("left_hip", "left_knee", "left_ankle", +1.0),
    ("right_hip", "right_knee", "right_ankle", +1.0),
)
_MIN_BEND_SIN = 0.05     # < ~3 deg bend: no stable bend plane; generic solve
_PLANE_TRUST_SIN = 0.26  # < ~15 deg bend: plane too noisy for LEG twist; foot decides
_SPINE1_SHARE = 0.65     # lumbar share of the pelvis->chest rotation (Scott's captures)
# Ankle roll bias: Scott's pick from the roll ladder on row c5cf2013
# (2026-08-22) -- the detector's tiny foot labels carry a sole-roll bias
# only an eye can calibrate. Applied about the solved foot axis, ramped by
# pointedness so flat planted feet (long approved) stay untouched. Lives in
# rig_state_from_mhr70, never in solve_rig_locals: the rest-roundtrip and
# the machine gate stay pure. One-row calibration -- revisit per his eye.
_ANKLE_ROLL_ENABLED = False   # DISABLED per Scott 2026-08-22 ("leave the code in
                              # place") pending review of more pointed-toe poses;
                              # flip to True to restore his calibration below.
_ANKLE_ROLL_BIAS_DEG = 20.0   # Scott dialed down from his initial +30 ladder pick
_ANKLE_ROLL_RAMP = (0.25, 0.60)   # pointedness: 0 bias below, full above


def _unit_or_none(v):
    n = np.linalg.norm(v)
    return None if n < _FRAME_EPS else np.asarray(v, float) / n


def _slerp(q0, q1, t):
    """Geodesic interpolation between [w,x,y,z] quaternions."""
    q0 = np.asarray(q0, float); q1 = np.asarray(q1, float)
    d = float(np.dot(q0, q1))
    if d < 0.0:
        q1, d = -q1, -d
    if d > 0.9995:                      # nearly parallel: lerp + normalise
        out = q0 + t * (q1 - q0)
        return out / np.linalg.norm(out)
    th = np.arccos(np.clip(d, -1.0, 1.0))
    return (np.sin((1 - t) * th) * q0 + np.sin(t * th) * q1) / np.sin(th)


def _axis_angle_q(axis, angle):
    axis = np.asarray(axis, float)
    h = angle / 2.0
    return np.array([np.cos(h), *(np.sin(h) * axis)])


def _twist_to_meet_cone(w, axis, f_t, c, prefer):
    """Angle psi rotating w about *axis* so dot(rotated_w, f_t) == c.

    Rotating the upper segment's swing brings the hinge to w; the remaining
    freedom is twist about the segment (*axis*). A pure local-Z bend keeps
    the hinge/lower-segment angle at its rest value (cos = c), so the twist
    must land the hinge on that cone. Two solutions exist; the one whose
    hinge lies on the anatomical bend side (max dot with *prefer*) wins.
    Unreachable c is clamped to the nearest attainable value. Returns None
    only when the geometry is degenerate."""
    wpar = np.dot(w, axis) * axis
    wperp = w - wpar
    A_ = float(np.dot(wperp, f_t))
    B_ = float(np.dot(np.cross(axis, wperp), f_t))
    C_ = c - float(np.dot(wpar, f_t))
    R = float(np.hypot(A_, B_))
    if R < 1e-9:
        return None
    base = float(np.arctan2(B_, A_))
    if abs(C_) <= R:
        d = float(np.arccos(np.clip(C_ / R, -1.0, 1.0)))
        cands = (base + d, base - d)
    else:
        cands = (base if C_ > 0 else base + np.pi,)
    best, best_score = None, -2.0
    for psi in cands:
        h = QuaternionMath.rotate_vector(_axis_angle_q(axis, psi), w)
        score = float(np.dot(h, prefer))
        if score > best_score:
            best, best_score = psi, score
    return best


def _pair_delta(a1, a2, b1, b2):
    """World rotation mapping direction pair (a1, a2) exactly onto (b1, b2).

    Orthonormal-frame alignment with Gram-Schmidt on the second vector of
    each pair. Returns None when either pair is degenerate (second vector
    parallel to the first) -- callers fall back to the generic solve rather
    than fabricate an axis.
    """
    out = []
    for v1, v2 in ((a1, a2), (b1, b2)):
        e1 = np.asarray(v1, float)
        n1 = np.linalg.norm(e1)
        if n1 < _FRAME_EPS:
            return None
        e1 = e1 / n1
        e2 = np.asarray(v2, float) - np.dot(v2, e1) * e1
        n2 = np.linalg.norm(e2)
        if n2 < 1e-3:
            return None
        e2 = e2 / n2
        out.append(np.column_stack([e1, e2, np.cross(e1, e2)]))
    return QuaternionMath.from_matrix(out[1] @ out[0].T)


_ANKLE_FOOT = {
    "left_ankle": ("left_heel", "left_big_toe", "left_small_toe"),
    "right_ankle": ("right_heel", "right_big_toe", "right_small_toe"),
}


def _try_ankle_delta(rig: Rig, targets: dict, Wr: dict, D: dict, n: int):
    """World delta posing the ankle as Rz(theta)*Rx(psi) ONLY -- never Y.

    Scott (2026-08-21): "ankle is incorrect - axis is +/-Z"; the rig's own
    limits agree (ankle: z [-81, 106] plantar/dorsiflexion, x [-75, 75]
    inversion/eversion, no y entry), and every posed capture ankle is a Z or
    Z+X composition. The free-orientation Kabsch this replaces invented Y
    twist from the heel/toe fit.

    Aims the foot axis (heel -> toe midpoint): psi is solved so the axis's
    local-Z height is reachable (Rz preserves z), theta closes the remaining
    angle in the XY plane. Exact whenever the target pitch lies inside the
    rig's reachable cone; clamped to the nearest pitch otherwise. Returns
    None (caller falls back to the generic solve) if the foot targets are
    absent or the geometry is degenerate."""
    names = _ANKLE_FOOT.get(rig.name[n])
    p = rig.parent[n]
    if names is None or p is None or p not in D:
        return None
    li = rig.index_of_name
    hi, bi, si = (li[x] for x in names)
    if not all(k in targets for k in (hi, bi, si)):
        return None
    rest = rig.rest_world_p
    axis_rest = _unit_or_none(0.5 * (rest[bi] + rest[si]) - rest[hi])
    axis_tgt = _unit_or_none(0.5 * (np.asarray(targets[bi], float) + np.asarray(targets[si], float))
                             - np.asarray(targets[hi], float))
    if axis_rest is None or axis_tgt is None:
        return None
    v_w, u_w = axis_rest, axis_tgt
    pre = QuaternionMath.multiply(D[p], Wr[n])          # ankle frame before its own delta
    v = QuaternionMath.rotate_vector(QuaternionMath.conjugate(Wr[n]), v_w)
    u = QuaternionMath.rotate_vector(QuaternionMath.conjugate(pre), u_w)
    R = float(np.hypot(v[1], v[2]))
    if R < 1e-6:
        return None
    phi0 = float(np.arctan2(v[1], v[2]))
    c = float(np.clip(u[2], -R, R))                     # unreachable pitch: clamp
    d = float(np.arccos(np.clip(c / R, -1.0, 1.0)))
    # Both psi branches aim the foot axis exactly; they are +/- mirrors of
    # the sole's roll, scored by the toe-line + heel evidence. NOTE
    # (2026-08-22): Scott reports the roll reads reversed on real images; a
    # blanket branch inversion was tried and REJECTED -- it breaks the
    # rest-roundtrip sanity (rest evidence is correct evidence), proving
    # the rule cannot be a global flip. Resolution pending his A/B pick on
    # live rows; see the session ledger.
    toe_rest = _unit_or_none(rest[bi] - rest[si])
    toe_tgt = _unit_or_none(np.asarray(targets[bi], float)
                            - np.asarray(targets[si], float))
    heel_rest = _unit_or_none(rest[hi] - rest[n])
    heel_tgt = _unit_or_none(np.asarray(targets[hi], float)
                             - np.asarray(targets[n], float))
    best = None
    for psi in (phi0 + d, phi0 - d):
        psi = float(np.arctan2(np.sin(psi), np.cos(psi)))
        w = np.array([v[0],
                      v[1] * np.cos(psi) - v[2] * np.sin(psi),
                      v[1] * np.sin(psi) + v[2] * np.cos(psi)])
        if np.hypot(w[0], w[1]) < 1e-6 or np.hypot(u[0], u[1]) < 1e-6:
            continue                                    # foot axis along the hinge
        theta = float(np.arctan2(u[1], u[0]) - np.arctan2(w[1], w[0]))
        delta = QuaternionMath.multiply(
            _axis_angle_q(np.array([0.0, 0.0, 1.0]), theta),
            _axis_angle_q(np.array([1.0, 0.0, 0.0]), psi))
        d_world = QuaternionMath.multiply(
            D[p], QuaternionMath.multiply(
                Wr[n], QuaternionMath.multiply(delta, QuaternionMath.conjugate(Wr[n]))))
        score = 0.0
        if toe_rest is not None and toe_tgt is not None:
            score += float(np.dot(
                QuaternionMath.rotate_vector(d_world, toe_rest), toe_tgt))
        if heel_rest is not None and heel_tgt is not None:
            score += 0.25 * float(np.dot(
                QuaternionMath.rotate_vector(d_world, heel_rest), heel_tgt))
        if best is None or score > best[0]:
            best = (score, d_world)
    return None if best is None else best[1]


def _anchor_deltas(rig: Rig, targets: dict[int, np.ndarray],
                   Wr: dict) -> dict[int, np.ndarray]:
    """World deltas for the anchored bones: pelvis (hip frame), the four limb
    chains (bend-plane hinge), the head (nose + eye line) and spine_2
    (neck + clavicle line). Facing and twist become constraints here; the
    generic per-bone direction solve cannot see either."""
    li = rig.index_of_name
    rest = rig.rest_world_p
    A: dict[int, np.ndarray] = {}

    if all(k in targets for k in (li["left_hip"], li["right_hip"], li["spine_1"])):
        fr = _orthonormal_frame_from_hips_and_up(
            rest, li["left_hip"], li["right_hip"], li["spine_1"])
        ft = _orthonormal_frame_from_hips_and_up(
            targets, li["left_hip"], li["right_hip"], li["spine_1"])
        A[li["pelvis"]] = QuaternionMath.from_matrix(
            np.column_stack(ft[:3]) @ np.column_stack(fr[:3]).T)

    for root, mid, end, sgn in _LIMB_CHAINS:
        ri, mi, ei = li[root], li[mid], li[end]
        if not all(k in targets for k in (ri, mi, ei)):
            continue
        u_t = _unit_or_none(np.asarray(targets[mi], float) - np.asarray(targets[ri], float))
        f_t = _unit_or_none(np.asarray(targets[ei], float) - np.asarray(targets[mi], float))
        if u_t is None or f_t is None:
            continue
        u_r = _unit_or_none(rest[mi] - rest[ri])
        f_r = _unit_or_none(rest[ei] - rest[mi])
        z_r = QuaternionMath.rotate_vector(Wr[mi], np.array([0.0, 0.0, 1.0]))
        if u_r is None or f_r is None:
            continue
        n_raw = np.cross(u_t, f_t)
        bend_sin = float(np.linalg.norm(n_raw))
        foot = _ANKLE_FOOT.get(rig.name[ei])
        if bend_sin < _PLANE_TRUST_SIN and foot is not None:
            # Near-straight LEG: the bend plane is noise (measured: knee bends
            # under 5 deg put 24/29 corpus heels below the floor, q10 cosine
            # 0.31), but a straight leg's twist is exactly what orients the
            # foot -- so the foot axis chooses it instead of the plane.
            hi2, bi2, si2 = (li[x] for x in foot)
            if all(k in targets for k in (hi2, bi2, si2)):
                fa_r = _unit_or_none(0.5 * (rest[bi2] + rest[si2]) - rest[hi2])
                fa_t = _unit_or_none(
                    0.5 * (np.asarray(targets[bi2], float) + np.asarray(targets[si2], float))
                    - np.asarray(targets[hi2], float))
                if fa_r is not None and fa_t is not None:
                    swing = QuaternionMath.from_two_vectors(u_r, u_t)
                    a0 = QuaternionMath.rotate_vector(swing, fa_r)
                    pa = a0 - np.dot(a0, u_t) * u_t
                    pb = fa_t - np.dot(fa_t, u_t) * u_t
                    if np.linalg.norm(pa) > 1e-3 and np.linalg.norm(pb) > 1e-3:
                        pa, pb = pa / np.linalg.norm(pa), pb / np.linalg.norm(pb)
                        psi = float(np.arctan2(np.dot(u_t, np.cross(pa, pb)),
                                               np.dot(pa, pb)))
                        d_root = QuaternionMath.multiply(_axis_angle_q(u_t, psi), swing)
                        h = _unit_or_none(QuaternionMath.rotate_vector(d_root, z_r))
                        a = QuaternionMath.rotate_vector(d_root, f_r)
                        if h is not None:
                            a_p = a - np.dot(a, h) * h
                            b_p = f_t - np.dot(f_t, h) * h
                            if np.linalg.norm(a_p) > 1e-6 and np.linalg.norm(b_p) > 1e-6:
                                a_p, b_p = (a_p / np.linalg.norm(a_p),
                                            b_p / np.linalg.norm(b_p))
                                phi = float(np.arctan2(
                                    np.dot(h, np.cross(a_p, b_p)), np.dot(a_p, b_p)))
                                A[ri] = d_root
                                A[mi] = QuaternionMath.multiply(
                                    _axis_angle_q(h, phi), d_root)
            continue
        if bend_sin < _MIN_BEND_SIN:
            continue                      # near-straight arm: hinge undefined
        prefer = sgn * n_raw / bend_sin
        # The rig's own rest cone: the angle between the hinge and the lower
        # segment is preserved by any pure-Z bend, so the shoulder/hip twist
        # must place the hinge where the cone passes through the target.
        c = float(np.dot(z_r, f_r))
        swing = QuaternionMath.from_two_vectors(u_r, u_t)
        w = QuaternionMath.rotate_vector(swing, z_r)
        psi = _twist_to_meet_cone(w, u_t, f_t, c, prefer)
        if psi is None:
            d_root = _pair_delta(u_r, z_r, u_t, prefer)   # degenerate: frame fallback
            d_mid = _pair_delta(f_r, z_r, f_t, prefer)
            if d_root is not None:
                A[ri] = d_root
            if d_mid is not None:
                A[mi] = d_mid
            continue
        d_root = QuaternionMath.multiply(_axis_angle_q(u_t, psi), swing)
        h = QuaternionMath.rotate_vector(d_root, z_r)
        a = QuaternionMath.rotate_vector(d_root, f_r)
        a_p = a - np.dot(a, h) * h
        b_p = f_t - np.dot(f_t, h) * h
        na, nb = np.linalg.norm(a_p), np.linalg.norm(b_p)
        if na < _FRAME_EPS or nb < _FRAME_EPS:
            A[ri] = d_root
            continue                      # forearm along hinge: bend angle undefined
        a_p, b_p = a_p / na, b_p / nb
        phi = float(np.arctan2(np.dot(h, np.cross(a_p, b_p)), np.dot(a_p, b_p)))
        A[ri] = d_root
        A[mi] = QuaternionMath.multiply(_axis_angle_q(h, phi), d_root)

    hi, ni_ = li["head"], li["nose"]
    le, re_ = li["left_eye"], li["right_eye"]
    if all(k in targets for k in (hi, ni_, le, re_)):
        d = _pair_delta(rest[ni_] - rest[hi], rest[le] - rest[re_],
                        np.asarray(targets[ni_], float) - np.asarray(targets[hi], float),
                        np.asarray(targets[le], float) - np.asarray(targets[re_], float))
        if d is not None:
            A[hi] = d

    s2, nk = li["spine_2"], li["neck"]
    lc, rc = li["left_clavicle"], li["right_clavicle"]
    if all(k in targets for k in (s2, nk, lc, rc)):
        d = _pair_delta(rest[nk] - rest[s2], rest[lc] - rest[rc],
                        np.asarray(targets[nk], float) - np.asarray(targets[s2], float),
                        np.asarray(targets[lc], float) - np.asarray(targets[rc], float))
        if d is not None:
            A[s2] = d

    # Spine flexion (Scott 2026-08-21: "stiff as a board"). MHR-70 has no
    # mid-spine keypoints -- spine1/spine2 targets are LINEAR INTERPOLATION
    # root->neck (mhr70_retargeter.py JOINT_HIERARCHY), collinear by
    # construction, so aiming them produced a straight rod with all torso
    # pitch at the two end anchors (worse: the rod-aim over-rotated spine_1
    # toward the chord, measured 84 deg of a 29 deg total). Instead spine_1
    # takes exactly 65% of the pelvis->chest rotation along its geodesic --
    # the lumbar share measured from Scott's own captures (crouch 60.7/32.8,
    # hoop 37.0/19.9, both ~0.65) -- and spine_2's local absorbs the rest
    # (its world anchor above is untouched).
    pv = li["pelvis"]
    if pv in A and s2 in A:
        # Split only the SWING (pitch + lateral) of the pelvis->chest
        # rotation; keep its TWIST about the torso axis concentrated at the
        # chest, as the pre-split solve did. Both anchors take their
        # vertical from the same hipmid->neck chord (no mid-spine keypoints
        # exist), so their relative rotation is dominated by hips-vs-
        # clavicles TWIST -- and distributing twist along the column
        # corkscrews it (Scott's crow-pose report: 59.5 deg about -Y read
        # as an arch bending backwards). Swing alone curves the column the
        # way flexion looks.
        r_rel = QuaternionMath.multiply(QuaternionMath.conjugate(A[pv]), A[s2])
        if r_rel[0] < 0:
            r_rel = -np.asarray(r_rel, float)
        chord = _unit_or_none(np.asarray(targets[nk], float)
                              - 0.5 * (np.asarray(targets[li["left_hip"]], float)
                                       + np.asarray(targets[li["right_hip"]], float)))
        if chord is not None:
            axis_p = QuaternionMath.rotate_vector(QuaternionMath.conjugate(A[pv]), chord)
            d_par = float(np.dot(r_rel[1:], axis_p))
            tw = np.array([r_rel[0], *(d_par * np.asarray(axis_p, float))])
            ntw = np.linalg.norm(tw)
            if ntw > 1e-9:
                tw = tw / ntw
                swing = QuaternionMath.multiply(r_rel, QuaternionMath.conjugate(tw))
                part = _slerp(np.array([1.0, 0.0, 0.0, 0.0]), swing, _SPINE1_SHARE)
                A[li["spine_1"]] = QuaternionMath.multiply(A[pv], part)
            else:
                A[li["spine_1"]] = _slerp(A[pv], A[s2], _SPINE1_SHARE)
        else:
            A[li["spine_1"]] = _slerp(A[pv], A[s2], _SPINE1_SHARE)

    # Neck: pure local-Y nod carrying HALF the chest->head rotation's Y
    # component (Scott 2026-08-21: "head and neck forward tilt in Y"; his
    # captures split a look-down ~half/half -- crouch neck +38 / head +36 --
    # and every posed neck is pure Y to two decimals). The head anchor above
    # is world-exact, so its local absorbs the remainder exactly.
    # Wrists: palm orientation from the hand keypoints (Scott 2026-08-21:
    # "do we do anything with the wrists? ... or pronation supination?" --
    # before this, the wrist inherited the elbow rigidly and pro/sup was
    # lost). Hand axis (wrist -> middle knuckle) + knuckle line (index_1 <->
    # pinky_1) encode pronation/supination, flexion and deviation together.
    # Rest directions come straight from rest_world_p, which was captured
    # from the live scene and therefore already includes the finger groups'
    # 0.1 scale (fk_world_positions reproduces these too since v16 task 3 --
    # rest_world_p is simply the more direct source, already on hand here).
    # Degenerate/missing knuckles: _pair_delta returns None and the wrist
    # keeps the inherit-elbow fallback.
    for side in ("left", "right"):
        wi = li[f"{side}_wrist"]
        mi2 = li[f"{side}_middle_finger_1"]
        ii2 = li[f"{side}_index_finger_1"]
        pi2 = li[f"{side}_pinky_finger_1"]
        if not all(k in targets for k in (wi, mi2, ii2, pi2)):
            continue
        d = _pair_delta(rest[mi2] - rest[wi], rest[ii2] - rest[pi2],
                        np.asarray(targets[mi2], float) - np.asarray(targets[wi], float),
                        np.asarray(targets[ii2], float) - np.asarray(targets[pi2], float))
        if d is not None:
            A[wi] = d

    hi = li["head"]
    if s2 in A and hi in A and rig.name[nk] == "neck":
        r_ln = QuaternionMath.multiply(
            QuaternionMath.conjugate(Wr[nk]),
            QuaternionMath.multiply(
                QuaternionMath.conjugate(A[s2]),
                QuaternionMath.multiply(A[hi], Wr[nk])))
        if r_ln[0] < 0:
            r_ln = -np.asarray(r_ln, float)
        kappa = 2.0 * float(np.arctan2(r_ln[2], r_ln[0]))   # Y twist of the total
        dy = _axis_angle_q(np.array([0.0, 1.0, 0.0]), kappa / 2.0)
        A[nk] = QuaternionMath.multiply(
            A[s2], QuaternionMath.multiply(
                Wr[nk], QuaternionMath.multiply(dy, QuaternionMath.conjugate(Wr[nk]))))

    return A


# Ruling 3 (v16 task 3): the v2 asset flips these 30 finger phalanges (chain
# root + two interior joints per digit; the ten tips stay solve:False) to
# solve:True, since their chains are reachable from pelvis now. But MHR-70
# supplies no finger ROTATION data -- only raw keypoints, which
# rig_targets_from_mhr70 maps onto them like any other bone -- so left in
# `solved` below, the generic aim-based branch would silently start posing
# fingers from noisy, ungated keypoint data purely as a side effect of the
# asset switch. Excluded here by NAME rather than a structural walk: unlike
# "joint7" elsewhere in this rig (module docstring), all 30 of these are
# uniquely named, and solve_rig_locals already resolves everything else by
# name via rig.index_of_name. A later task adds the real finger path (MHR
# rotations, gated per digit) and lifts this exclusion deliberately --
# search for this constant's name when that happens.
_FINGER_PHALANGE_NAMES = frozenset({
    "left_thumb_1", "left_thumb_2", "left_thumb_3",
    "left_index_finger_1", "left_index_finger_2", "left_index_finger_3",
    "left_middle_finger_1", "left_middle_finger_2", "left_middle_finger_3",
    "left_ring_finger_1", "left_ring_finger_2", "left_ring_finger_3",
    "left_pinky_finger_1", "left_pinky_finger_2", "left_pinky_finger_3",
    "right_thumb_1", "right_thumb_2", "right_thumb_3",
    "right_index_finger_1", "right_index_finger_2", "right_index_finger_3",
    "right_middle_finger_1", "right_middle_finger_2", "right_middle_finger_3",
    "right_ring_finger_1", "right_ring_finger_2", "right_ring_finger_3",
    "right_pinky_finger_1", "right_pinky_finger_2", "right_pinky_finger_3",
})


def solved_indices(rig: Rig) -> list:
    """The bone indices `solve_rig_locals` actually solves, in `rig.order`.

    The SINGLE authoritative source for "solved" (ruling 3): `rig.solve`
    alone is no longer enough under v2, since the 30 finger phalanges carry
    solve:True but are excluded from the generic aim-based path
    (`_FINGER_PHALANGE_NAMES` above). `solve_rig_locals` calls this itself
    rather than re-deriving the filter inline, and so must every OTHER
    reader of "which bones will solve_rig_locals return" -- including
    outside this module. (Review finding, v16 task 3: poseforge3d's
    validate_rig_retarget.py machine gate read `rig.solve` directly and, as
    a result, silently started gating the 30 unposed finger phalanges
    against real keypoint targets the moment v2 shipped -- every row
    failing. A direct `rig.solve` read is exactly the bug this function
    exists to make impossible to repeat.)
    """
    return [i for i in rig.order
            if rig.solve[i] and rig.name[i] not in _FINGER_PHALANGE_NAMES]


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
    Returns absolute local quaternions keyed by bone INDEX for the SOLVED
    set only: the 34 bones the rig has always solved, unchanged by v16 task
    3 -- ruling 3 (`solved_indices`/`_FINGER_PHALANGE_NAMES` above) keeps
    the 30 finger phalanges out of `solved` below even though the v2 asset
    flags them solve:True. `fk_world_orientations`/`fk_world_positions`
    cover every node now (ruling 2), so Wr below DOES have entries for
    bones this function never solves -- a solved wrist has an unsolved
    GROUP child under v2 (the finger islands are reconnected there now,
    not disconnected at a captured `parent: None` root as in v1), and that
    is fine: this function never looks up Wr, D or L for a group or a
    finger phalange. `targets` may carry entries for both (MHR-70 supplies
    raw finger keypoints, and rig_targets_from_mhr70 maps them onto the
    phalanges like any other bone); they are never read, since neither
    group nor phalange is ever `n` here, nor a solved bone's direct child
    (a solved wrist's only children are its five groups, and a group with
    no target is dropped from `pairs` before it can influence anything).
    Unsolved bones fall back to their rest locals; that fallback is the
    caller's job (`rig_state_from_mhr70`), not this function's.
    """
    Wr = fk_world_orientations(rig, rig.rest_local_q)
    solved = solved_indices(rig)
    # Anchored bones get exact frame deltas (facing and hinge twist are
    # constraints there); everything else falls through to the generic
    # child-direction solve below. See _anchor_deltas.
    anchors = _anchor_deltas(rig, targets, Wr)
    D: dict = {}
    for n in solved:
        if n in anchors:
            D[n] = anchors[n]
            continue
        d_ankle = _try_ankle_delta(rig, targets, Wr, D, n)
        if d_ankle is not None:
            D[n] = d_ankle
            continue
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


ROOT_DISPLAY_YAW_DEG: float = 0.0   # obsolete under the chirality fix below; kept inert at 0

# cv_to_yup negates only Y -- a det(-1) REFLECTION of camera space. The
# labeled anatomy of its output is chirally mirrored: over 118 corpus rows,
# dot(cross(up, right_hip - left_hip), nose - head) came out negative on 117
# (median -0.883) -- labeled left/right disagree with face-and-up on
# essentially every real detection. A proper rotation can never fit a mirror
# (_kabsch_q forces det +1), so the near-coplanar pelvis fit paid the
# reflection off in the one axis it barely constrains: facing. Measured
# before this fix: solved pelvis 174.8 deg from person-forward, IQR 0.2 deg,
# deterministic and corpus-wide. Negating Z as well restores det(+1) -- the
# composite cv->rig map is then a 180-deg rotation of camera space about X --
# and lands camera-facing subjects facing +Z, the viewer: the convention
# every hand-posed ground-truth capture was made in.
_CV_YUP_TO_RIG = np.array([1.0, 1.0, -1.0])


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
        targets[rig.index_of_name[bone_name]] = (
            np.asarray(p, float) * _CV_YUP_TO_RIG)    # KeyError propagates, never caught

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


_FRAME_EPS = 1e-6  # matches solve_rig_locals's own "vector is degenerate" tolerance
                    # elsewhere in this module; far below any real bone length in
                    # metres (MHR-70 targets) or rig units (rest positions), but large
                    # enough to catch genuine coincidence/collinearity rather than noise.


def _orthonormal_frame_from_hips_and_up(points: dict, left_hip: int, right_hip: int,
                                         up_ref: int) -> tuple:
    """right = direction from right_hip to left_hip; up = direction from the
    hip midpoint to *up_ref*, re-orthonormalised against right (Gram-Schmidt
    via cross products, so it's exact even when the raw up reference isn't
    perfectly perpendicular to the hip line); fwd completes a right-handed
    orthonormal basis. Returns (right, up, fwd, hip_mid).

    Raises ValueError naming the degenerate condition and the offending
    joint positions if any basis vector's norm is below _FRAME_EPS before
    normalising -- never silently falls back to world axes, which would
    produce a plausible-looking but wrong pose for an inverted or supine
    subject. SAM-3D output is noisy and this runs over thousands of real
    detections; a bad one must fail loudly, not emit a NaN that only a
    numpy RuntimeWarning would hint at.
    """
    hip_mid = (points[left_hip] + points[right_hip]) / 2

    right_raw = points[left_hip] - points[right_hip]
    right_norm = np.linalg.norm(right_raw)
    if right_norm < _FRAME_EPS:
        raise ValueError(
            f"degenerate pelvis frame: left_hip and right_hip are coincident "
            f"(separation {right_norm:.3e} < {_FRAME_EPS:.0e}) -- "
            f"left_hip={points[left_hip]!r}, right_hip={points[right_hip]!r}")
    right = right_raw / right_norm

    up_raw = points[up_ref] - hip_mid
    up_raw_norm = np.linalg.norm(up_raw)
    if up_raw_norm < _FRAME_EPS:
        raise ValueError(
            f"degenerate pelvis frame: the up reference coincides with the hip "
            f"midpoint (separation {up_raw_norm:.3e} < {_FRAME_EPS:.0e}) -- "
            f"up_ref={points[up_ref]!r}, hip_mid={hip_mid!r}")

    fwd_raw = np.cross(right, up_raw)
    fwd_norm = np.linalg.norm(fwd_raw)
    if fwd_norm < _FRAME_EPS:
        raise ValueError(
            f"degenerate pelvis frame: the hip line and the up direction are "
            f"collinear (cross-product norm {fwd_norm:.3e} < {_FRAME_EPS:.0e}) -- "
            f"left_hip={points[left_hip]!r}, right_hip={points[right_hip]!r}, "
            f"up_ref={points[up_ref]!r}")
    fwd = fwd_raw / fwd_norm

    up_result_raw = np.cross(fwd, right)
    up_result_norm = np.linalg.norm(up_result_raw)
    if up_result_norm < _FRAME_EPS:
        # Unreachable if the two guards above hold: fwd and right are then
        # unit vectors that are exactly orthogonal by construction (fwd was
        # built as a cross product involving right), so this cross
        # product's norm is exactly 1. Kept as defense in depth, per
        # "check each basis vector's norm before normalising."
        raise ValueError(
            f"degenerate pelvis frame: fwd x right had near-zero norm "
            f"({up_result_norm:.3e} < {_FRAME_EPS:.0e}), which should be "
            f"unreachable -- fwd={fwd!r}, right={right!r}")
    up = up_result_raw / up_result_norm

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
    ankle, and spine_1; everything but the pelvis entry is read, not
    written. (No neck fallback: MHR70Retargeter.compute_joint_positions_static
    unconditionally computes spine1, so rig_targets_from_mhr70's output
    always has it -- a fallback for "spine_1 missing" would be a branch
    with no reachable input, not a guard against the degenerate cases
    above, which are handled explicitly in
    _orthonormal_frame_from_hips_and_up instead.)
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

    right_t, up_t, fwd_t, hip_mid_t = _orthonormal_frame_from_hips_and_up(
        targets, lhip, rhip, li["spine_1"])
    tgt_leg_len = np.mean([_leg_len(targets, lhip, lknee, lankle),
                            _leg_len(targets, rhip, rknee, rankle)])

    offset_target = (offset_frac[0] * right_t + offset_frac[1] * up_t
                      + offset_frac[2] * fwd_t) * tgt_leg_len
    return hip_mid_t + offset_target


def _assert_all_finite(state: dict) -> None:
    """Belt to _orthonormal_frame_from_hips_and_up's guard-clause braces: a
    future edit that introduces a non-finite value by some OTHER route must
    still fail loudly here, before json.dumps ever gets a chance to emit a
    bare `NaN`/`Infinity` -- invalid JSON, and something JavaScript's
    JSON.parse rejects on the consuming end. Raises ValueError naming the
    first offending bone/field, not a summary of all of them."""
    for bone, q in state["pose"].items():
        for k in ("_x", "_y", "_z", "_w"):
            if not np.isfinite(q[k]):
                raise ValueError(f"non-finite value in pose[{bone!r}][{k!r}]: {q[k]}")
    for k, v in state["pelvisPosition"].items():
        if not np.isfinite(v):
            raise ValueError(f"non-finite value in pelvisPosition[{k!r}]: {v}")
    if not np.isfinite(state["groundY"]):
        raise ValueError(f"non-finite value in groundY: {state['groundY']}")


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

    # Ankle roll bias (see _ANKLE_ROLL_BIAS_DEG above): applied here so the
    # pure solve stays bias-free.
    Wr_all = None if not _ANKLE_ROLL_ENABLED else fk_world_orientations(rig, rig.rest_local_q)
    Wq = None if Wr_all is None else fk_world_orientations(rig, {**rig.rest_local_q, **solved})
    for side in ("left", "right") if _ANKLE_ROLL_ENABLED else ():
        ai = li[f"{side}_ankle"]
        ki, hi2 = rig.parent[ai], li[f"{side}_heel"]
        bi2, si2 = li[f"{side}_big_toe"], li[f"{side}_small_toe"]
        if not (ai in solved and ki in solved
                and all(k in targets for k in (ki, ai, hi2, bi2, si2))):
            continue
        fa_t = np.asarray(targets[bi2], float) + np.asarray(targets[si2], float)
        fa_t = fa_t * 0.5 - np.asarray(targets[hi2], float)
        shin = np.asarray(targets[ai], float) - np.asarray(targets[ki], float)
        nf, ns = np.linalg.norm(fa_t), np.linalg.norm(shin)
        if nf < _FRAME_EPS or ns < _FRAME_EPS:
            continue
        pointed = abs(float(np.dot(fa_t / nf, shin / ns)))
        lo, hi_r = _ANKLE_ROLL_RAMP
        scale = min(1.0, max(0.0, (pointed - lo) / (hi_r - lo)))
        if scale <= 0.0:
            continue
        # Anatomically mirrored (Scott: "the left foot needs the opposite
        # sign"): a consistent inversion/eversion bias flips sign across the
        # midline. +30 right, -30 left, about each foot's own heel->toe axis.
        if side == "left":
            scale = -scale
        d_a = QuaternionMath.multiply(Wq[ai], QuaternionMath.conjugate(Wr_all[ai]))
        rest_axis = (0.5 * (rig.rest_world_p[bi2] + rig.rest_world_p[si2])
                     - rig.rest_world_p[hi2])
        na = np.linalg.norm(rest_axis)
        if na < _FRAME_EPS:
            continue
        axis_now = QuaternionMath.rotate_vector(d_a, rest_axis / na)
        bias = _axis_angle_q(np.asarray(axis_now, float) / np.linalg.norm(axis_now),
                             np.radians(_ANKLE_ROLL_BIAS_DEG * scale))
        w_new = QuaternionMath.multiply(bias, Wq[ai])
        solved[ai] = QuaternionMath.multiply(QuaternionMath.conjugate(Wq[ki]), w_new)


    # ruling 7: solve_rig_locals deliberately covers only the solved bones --
    # padding its own output would claim to have solved bones it never
    # touched. Assembly owns the merge: every bone must land in the state, so
    # the 40 unsolved finger/thumb bones (30 phalanges excluded per ruling 3
    # / _FINGER_PHALANGE_NAMES, plus the 10 tips, which are solve:False
    # outright) AND the ten constant group nodes all fall back to their REST
    # locals -- the pose comprehension below then drops the groups entirely
    # (the viewer owns them; see there).
    full_by_index = {**rig.rest_local_q, **solved}

    # ruling 5: PoseGoblin reads state.pose[child.name], so the emitted pose
    # is NAME-keyed, not index-keyed. The two "joint7" bones collapse to a
    # single entry -- last-one-wins by iterating rig.order, which keeps the
    # original 74 bones (indices 0..73) in their v1 parent-first order,
    # matching PoseGoblin's own captureCurrentState/RecallPoseCommand
    # contract instead of diverging from it. The ten v2 group nodes
    # (`transform4`..`transform13`, is_group=True) are excluded here: the
    # viewer already owns them as a fixed container transform it never
    # poses, so sending it ten `transform*` entries would be sending
    # something it never asked for (v16 task 3) -- 73 keys out, same as v1.
    pose = {rig.name[i]: QuaternionMath.to_threejs_dict(full_by_index[i])
            for i in rig.order if not rig.is_group[i]}

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
    state = {
        "pose": pose,
        "pelvisPosition": {"x": float(pelvis[0]), "y": float(pelvis[1]), "z": float(pelvis[2])},
        "groundY": float(min(feet) * s) if feet else 0.0,
        "cameraState": MannequinExporter.get_default_camera_state(),
        "rigVersion": rig.version,
        "retargetVersion": 15,
    }
    _assert_all_finite(state)   # belt: no non-finite value reaches the wire, regardless of cause
    return state
