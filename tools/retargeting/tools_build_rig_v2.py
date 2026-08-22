#!/usr/bin/env python
"""Build posegoblin_rig_v2.json: reconnect the ten orphaned finger islands
(`*_thumb_1`/`*_finger_1`, `parent: null` in v1) through the ten constant
Group nodes they actually hang off in the live three.js scene
(`transform4`..`transform13`, each a fixed rotation + 0.1 uniform scale).

No torch: pure numpy/json, reading v1, both rig probes
(bind_poses/capture-rigprobe-09.json, capture-rigrestprobe-10.json) and the
base-pose probe (capture-rigbase2-11.json -- see R14 below).

Quaternion conventions (do not mix these up -- see the same note in
tests/test_rig_v2_asset.py):
  - the ASSET (`rest_local_q` on every node, in both v1 and v2) is
    [w,x,y,z] -- fed straight into this repo's QuaternionMath.
  - the PROBES (everything under `nodes[].quat`) are raw three.js dumps,
    [x,y,z,w].
The 34 BODY bones are copied verbatim from v1 (already [w,x,y,z],
untouched). The 40 finger/thumb bones' `rest_local_q` and the ten new
groups' come from a probe in [x,y,z,w] and are converted to [w,x,y,z]
before they are written to the asset.

capture-rigprobe-09/-rigrestprobe-10 are POSED STATES, not rest poses --
never read rest geometry off their world matrices. They are used only to
(a) learn topology (which group each `_1` bone hangs from), (b) read the ten
groups' constant local transforms, and (c) validate, by FK-reproducing each
probe's OWN locals, that the topology + group constants this script wrote
are actually correct (the gate at the bottom of main()).

**R14 -- the finger rest comes from the base-pose probe, not from v1.**
v1's finger rest descends from `capture-rigrest-02.json`, which was captured
with the RIGHT hand curled ~34 deg into a fist (mean 33.87 deg / max 44.72
deg from the rig's true base pose, on all 15 phalanges; the left hand is
5.72 / 7.56). That curl has sat in the asset as "rest" ever since. It did
not matter while nothing posed the fingers; v16 task 5 applies
`rest_local o R(axis, angle)`, so a posed rest over-curls every right hand
by exactly that amount. `capture-rigbase2-11.json` is the rig reset to its
true base pose (2026-08-22), where the two hands are the SAME local
quaternion on all 20 bones per side to 1.79e-4 deg -- PoseGoblin authors a
hand pose as one set of absolute local quaternions applied to both hands
verbatim, so symmetric-means-identical is the convention, not an assumption.
Only the 40 finger/thumb bones are re-based; the 34 body bones keep v1's
values, which remain the asset of record for them.

The 40 finger bones' `rest_world_p` is therefore RECOMPUTED, by the same
scale-aware FK the ten groups use (`_fk_world_positions`), since their world
geometry moves with the corrected locals. The `_1` bones do not move (their
world position depends on the wrist chain and their own rest_local_p, not on
their own rotation); `_2`/`_3`/`_tip` do. Only the ten new groups otherwise
need a `rest_world_p`; each group's local position reads as ~0 (three.js
float noise on what is conceptually an identity offset), so it equals its
wrist's rest_world_p -- computed via FK from v1 rest locals and self-checked
against the wrist's own stored value below.
"""
import json
from pathlib import Path

import numpy as np

BP = Path(__file__).resolve().parent / "bind_poses"
V1 = BP / "posegoblin_rig_v1.json"
V2 = BP / "posegoblin_rig_v2.json"
PROBE_PATHS = [BP / "capture-rigprobe-09.json", BP / "capture-rigrestprobe-10.json"]
# R14: the rig reset to its TRUE base pose. Source of the 40 finger/thumb
# bones' rest_local_q -- and, since R15, the two clavicles'. Its BODY is
# otherwise at whatever pose the live app was showing (right_shoulder alone
# reads 105.9 deg from v1's rest), so reading anything else off it would be
# wrong; see CLAVICLE_BONE_NAMES for what earns the clavicle exception.
BASE_PROBE_PATH = BP / "capture-rigbase2-11.json"

# R15 (2026-08-22): the two CLAVICLES come off the base probe too. That is a
# deliberate exception to the sentence above and it is earned by a STRUCTURAL
# check, not by trusting the probe's body: in the base probe the pair is an
# exact mirror -- [0.707107, -0.707107, 0, 0] and [0.707107, +0.707107, 0, 0],
# the +/-90 deg about X that the mirrored bone offsets require -- and the LEFT
# one already equals v1's to v1's own rounding. v1's RIGHT clavicle is 6.077
# deg off that: Scott's incidental posing recorded as rest, the same defect
# R14 removed from the right hand. It matters because the solver emits
# `rest_local . delta`, so a rest frame 6 deg off the app's own splits a
# symmetric clavicle rotation asymmetrically between elevation and
# protraction -- what Scott reported as a dislocated-looking shoulder ball.
CLAVICLE_BONE_NAMES = ("left_clavicle", "right_clavicle")
# Bones whose rest_world_p must follow the corrected clavicle. The clavicle's
# OWN world position is fixed by spine_2 and its own offset, so it does not
# move; everything below it does. The right hand's twenty bones are already on
# the R14 recompute path and pick the change up there, and the five right-hand
# groups inherit right_wrist's value.
CLAVICLE_DEPENDENT_NAMES = ("right_shoulder", "right_elbow", "right_wrist")
CLAVICLE_MIRROR_TOL_DEG = 1e-4       # base probe L vs mirrored R (measured: 0.0 exactly)
CLAVICLE_LEFT_AGREES_TOL_DEG = 1e-3  # base probe left vs v1 left (measured: 4e-5, v1's rounding)
CLAVICLE_RIGHT_MIN_DEG = 1.0         # v1's right must really be wrong (measured: 6.077)

# transform4..8 hang off left_wrist, transform9..13 off right_wrist (binding wiring, from probes)
GROUP_NAMES = [f"transform{k}" for k in range(4, 14)]

# The 40 finger/thumb bones, by the name structure the whole module uses
# ("_thumb_" / "_finger_", tips included). Sides are appended in main().
_DIGIT_STEMS = ("thumb", "index_finger", "middle_finger", "ring_finger", "pinky_finger")
FINGER_BONE_NAMES = [f"{side}_{stem}_{k}"
                     for side in ("left", "right")
                     for stem in _DIGIT_STEMS
                     for k in ("1", "2", "3", "tip")]

# Tolerances below are generous relative to what's actually measured (see this script's own
# printed gate output, and the two cross-checks in main()); each comment gives the measured value.
BONE_SCALE_TOL = 1e-6              # probe Bone nodes must read scale ~= 1.0
CROSS_PROBE_TOL = 1e-9             # a group's local is pose-independent (measured: 0.0 exactly)
CROSS_GROUP_TOL = 1e-6             # the ten groups share one fixed local (measured: ~5.3e-15)
# FK group world pos vs its wrist's stored rest_world_p. Generous: v1's own 6-decimal rounding
# alone accounts for ~1.7e-5 of chain drift (measured: ~9.4e-6 left, ~5.2e-6 right).
GROUP_WORLD_SELFCHECK_TOL = 1e-3
GATE_TOL = 1e-6                    # build-time reproduction gate (mirrors test_rig_v2_asset.py)
# R14 tolerances.
BASE_OFFSET_TOL = 1e-5             # base probe's finger bone OFFSETS must equal v1's, since a
                                   # bone's local position is pose-independent (measured: 5.0e-7,
                                   # which is just v1's own 6-decimal rounding)
BASE_SYMMETRY_TOL_DEG = 0.01       # left vs right finger local in the base probe (measured: 1.8e-4)
FK_BODY_SELFCHECK_TOL = 1e-4       # _fk_world_positions reproducing v1's stored body rest_world_p
                                   # (measured worst: 1.9e-5, v1's rounding drift down the chain)


def _mat_local(pos, quat_xyzw, scale):
    """Local TRS matrix. `quat_xyzw` is three.js/probe convention [x,y,z,w]."""
    x, y, z, w = quat_xyzw
    R = np.array([[1 - 2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                  [2*(x*y+w*z), 1 - 2*(x*x+z*z), 2*(y*z-w*x)],
                  [2*(x*z-w*y), 2*(y*z+w*x), 1 - 2*(x*x+y*y)]])
    M = np.eye(4)
    M[:3, :3] = R * scale
    M[:3, 3] = pos
    return M


def _wxyz_to_xyzw(q):
    w, x, y, z = q
    return [x, y, z, w]


def _xyzw_to_wxyz(q):
    x, y, z, w = q
    return [w, x, y, z]


def _load(path):
    return json.loads(path.read_text())


def _probe_groups(probe):
    """{group name -> probe node} for the ten transform4..13 Group nodes."""
    out = {n["name"]: n for n in probe["nodes"]
           if n["type"] == "Group" and n["name"] in GROUP_NAMES}
    missing = set(GROUP_NAMES) - set(out)
    assert not missing, f"probe missing group nodes: {sorted(missing)}"
    return out


def _finger1_to_group(probe):
    """{finger/thumb `_1` bone name -> group name}, from probe topology: a
    Bone whose parent_uuid resolves to one of the ten transform Groups."""
    group_uuid_to_name = {g["uuid"]: name for name, g in _probe_groups(probe).items()}
    return {n["name"]: group_uuid_to_name[n["parent_uuid"]]
            for n in probe["nodes"]
            if n["type"] == "Bone" and n["parent_uuid"] in group_uuid_to_name}


def _quat_deg(a_wxyz, b_wxyz):
    """Angle between two [w,x,y,z] rotations, sign-normalised."""
    a = np.asarray(a_wxyz, float); a = a / np.linalg.norm(a)
    b = np.asarray(b_wxyz, float); b = b / np.linalg.norm(b)
    return float(np.degrees(2 * np.arccos(min(1.0, abs(float(a @ b))))))


def _unique_bone_nodes(probe):
    """{name -> probe Bone node} for names that are UNAMBIGUOUS in the probe.

    This asset's naming is not unique -- "joint7" names two unrelated bones
    (see posegoblin_rig.py's module docstring) -- so ambiguous names are
    dropped rather than silently resolved to whichever came last. Every name
    this script actually looks up is asserted present at the call site."""
    counts = {}
    for n in probe["nodes"]:
        if n["type"] == "Bone":
            counts[n["name"]] = counts.get(n["name"], 0) + 1
    return {n["name"]: n for n in probe["nodes"]
            if n["type"] == "Bone" and counts[n["name"]] == 1}


def _fk_world_positions(bones):
    """{index -> (3,) world position} for an assembled v2 bone list, by
    scale-aware FK from the bones' own rest locals.

    Mirrors `posegoblin_rig.fk_world_positions` (kept in sync by hand): the
    ten groups carry a 0.1 uniform scale that shrinks every offset beneath
    them, which `_mat_local`'s `R * scale` in the upper 3x3 propagates down
    the chain automatically as the matrices compose.

    Resolved by memoized recursion, not a forward index pass: the ten
    reconnected finger `_1` bones (14..57) parent to a group at 74..83,
    ABOVE their own index."""
    worlds = {}

    def world_of(i):
        if i not in worlds:
            b = bones[i]
            M = _mat_local(b["rest_local_p"], _wxyz_to_xyzw(b["rest_local_q"]), b["scale"])
            worlds[i] = M if b["parent"] is None else world_of(b["parent"]) @ M
        return worlds[i]

    return {i: world_of(i)[:3, 3] for i in range(len(bones))}


def _fk_world_matrices(v1_bones):
    """World matrix for each of the 74 v1 bones, via pure FK from rest
    locals (scale 1.0 throughout, verified against the probes above). Used
    only to locate the wrists in world space for the new groups; v1's own
    stored `rest_world_p` is left untouched everywhere else."""
    world = {}
    for i, b in enumerate(v1_bones):
        M = _mat_local(b["rest_local_p"], _wxyz_to_xyzw(b["rest_local_q"]), 1.0)
        world[i] = M if b["parent"] is None else world[b["parent"]] @ M
    return world


def _reproduce(doc, probe):
    """Mirrors tests/test_rig_v2_asset.py::_reproduce (kept in sync by hand
    -- update both on any change to either). FK each v2 node using the
    PROBE's own local TRS but the V2 asset's topology (parent indices, and
    the asset's stored group locals for group nodes) -> worst
    |world - probe.world| over all 84 nodes.

    World matrices are resolved by memoized recursion, not a single forward
    pass over `bones`: the ten reconnected finger `_1` bones (indices in
    14..57) now parent to a group at 74..83, ABOVE their own index, which a
    plain `for i in range(84): worlds[i] = worlds[parent] @ local` pass
    would hit before `worlds[parent]` exists (KeyError). Recursing to the
    parent on demand handles a parent index on either side of the child's."""
    bones = doc["bones"]
    by_uuid = {n["uuid"]: n for n in probe["nodes"]}
    pn = {}
    for n in probe["nodes"]:
        if n["type"] in ("Bone", "Group"):
            par = by_uuid.get(n["parent_uuid"])
            pn.setdefault((n["name"], par["name"] if par else None), n)

    def probe_node(i):
        b = bones[i]
        par = bones[b["parent"]]["name"] if b["parent"] is not None else None
        n = pn.get((b["name"], par)) or pn.get((b["name"], ""))
        assert n is not None, (b["name"], par)
        return n

    worlds = {}

    def world_of(i):
        if i not in worlds:
            b = bones[i]
            n = probe_node(i)
            if b.get("is_group"):
                # asset convention is [w,x,y,z]; _mat_local wants three.js
                # [x,y,z,w] (probe convention) -- reorder before use.
                w, x, y, z = b["rest_local_q"]
                M = _mat_local(b["rest_local_p"], [x, y, z, w], b["scale"])
            else:
                M = _mat_local(n["pos"], n["quat"], n["scale"][0])
            worlds[i] = M if b["parent"] is None else world_of(b["parent"]) @ M
        return worlds[i]

    worst, anchor = 0.0, None
    for i, b in enumerate(bones):
        n = probe_node(i)
        W = world_of(i)
        Wp = np.asarray(n["world"], float).reshape(4, 4).T  # column-major
        if anchor is None:
            anchor = Wp @ np.linalg.inv(W)
        worst = max(worst, float(np.abs(anchor @ W - Wp).max()))
    return worst


def main():
    v1_doc = _load(V1)
    v1_bones = v1_doc["bones"]
    assert len(v1_bones) == 74
    name_to_i_v1 = {b["name"]: i for i, b in enumerate(v1_bones)}

    probes = [_load(p) for p in PROBE_PATHS]
    base_nodes = _unique_bone_nodes(_load(BASE_PROBE_PATH))

    # --- R14: the base probe really is a symmetric, same-skeleton hand ------
    missing = [n for n in FINGER_BONE_NAMES if n not in base_nodes]
    assert not missing, f"{BASE_PROBE_PATH.name} is missing (or duplicates) {missing}"
    assert len(FINGER_BONE_NAMES) == 40

    # (a) the bone OFFSETS must be v1's -- a local position is pose-independent,
    #     so if these disagree the probe is not this skeleton and nothing below
    #     is meaningful.
    worst_offset = max(
        float(np.abs(np.array(v1_bones[name_to_i_v1[n]]["rest_local_p"], float)
                     - np.array(base_nodes[n]["pos"], float)).max())
        for n in FINGER_BONE_NAMES)
    assert worst_offset <= BASE_OFFSET_TOL, (
        f"base probe finger offsets differ from v1 by {worst_offset} -- not the same skeleton")

    # (b) the two hands must carry the SAME local quaternion. This is the
    #     R14 gate: it is exactly what capture-rigrest-02 (v1's source) failed,
    #     with the right hand curled 34 deg into a fist.
    base_finger_q = {n: _xyzw_to_wxyz(base_nodes[n]["quat"]) for n in FINGER_BONE_NAMES}
    worst_sym, worst_sym_name = 0.0, None
    for n in FINGER_BONE_NAMES:
        if not n.startswith("left_"):
            continue
        d = _quat_deg(base_finger_q[n], base_finger_q[n.replace("left", "right", 1)])
        if d > worst_sym:
            worst_sym, worst_sym_name = d, n
    assert worst_sym <= BASE_SYMMETRY_TOL_DEG, (
        f"base probe hands are NOT symmetric: {worst_sym_name} differs by "
        f"{worst_sym:.4f} deg -- this probe was captured with a hand posed, "
        f"which is the exact defect R14 removed from v1")

    # --- R15: the clavicle pair, from the base probe ------------------------
    # Three gates, and the first two are what make reading the probe's BODY
    # legitimate here where R14's note forbids it in general.
    missing = [n for n in CLAVICLE_BONE_NAMES if n not in base_nodes]
    assert not missing, f"{BASE_PROBE_PATH.name} is missing (or duplicates) {missing}"
    base_clav_q = {n: _xyzw_to_wxyz(base_nodes[n]["quat"]) for n in CLAVICLE_BONE_NAMES}

    # (a) offsets are pose-independent: if these disagree it is not this skeleton.
    worst_clav_offset = max(
        float(np.abs(np.array(v1_bones[name_to_i_v1[n]]["rest_local_p"], float)
                     - np.array(base_nodes[n]["pos"], float)).max())
        for n in CLAVICLE_BONE_NAMES)
    assert worst_clav_offset <= BASE_OFFSET_TOL, (
        f"base probe clavicle offsets differ from v1 by {worst_clav_offset} -- "
        f"not the same skeleton")

    # (b) STRUCTURAL: the pair must be an exact mirror. The two bone offsets
    #     are exact negatives of each other, which forces the two rest frames
    #     to differ by 180 deg about X -- i.e. w and x equal up to x's sign,
    #     y and z zero. A probe caught mid-pose cannot satisfy this by luck,
    #     so this is the check that stands in for trusting the probe's body.
    lq, rq = base_clav_q["left_clavicle"], base_clav_q["right_clavicle"]
    mirrored = np.array([rq[0], -rq[1], rq[2], rq[3]])
    mirror_dev = _quat_deg(lq, mirrored)
    assert mirror_dev <= CLAVICLE_MIRROR_TOL_DEG, (
        f"base probe clavicles are NOT mirrored: {mirror_dev:.6f} deg -- this probe "
        f"was captured with a clavicle posed, the defect R14/R15 exist to remove")
    assert max(abs(lq[2]), abs(lq[3]), abs(rq[2]), abs(rq[3])) <= 1e-9, (
        f"base probe clavicles carry y/z components: {lq}, {rq}")

    # (c) the LEFT must already agree with v1 (positive control -- this
    #     migration is about the right one) and the RIGHT must not.
    dev_l = _quat_deg(lq, v1_bones[name_to_i_v1["left_clavicle"]]["rest_local_q"])
    dev_r = _quat_deg(rq, v1_bones[name_to_i_v1["right_clavicle"]]["rest_local_q"])
    assert dev_l <= CLAVICLE_LEFT_AGREES_TOL_DEG, (
        f"base probe LEFT clavicle is {dev_l:.4f} deg off v1 -- the two sources were "
        f"expected to agree there, so this probe is not comparable to v1 at all")
    assert dev_r >= CLAVICLE_RIGHT_MIN_DEG, (
        f"v1's RIGHT clavicle is only {dev_r:.4f} deg off the base probe -- R15 has "
        f"nothing to correct and this build is silently a no-op")

    # --- probe sanity: every Bone node reads scale ~= 1.0 -------------------
    worst_bone_scale = max(
        float(np.max(np.abs(np.array(n["scale"], float) - 1.0)))
        for probe in probes for n in probe["nodes"] if n["type"] == "Bone"
    )
    assert worst_bone_scale <= BONE_SCALE_TOL, (
        f"probe Bone scale deviates from 1.0 by {worst_bone_scale}")

    # --- extract each probe's ten group locals (pos, quat_xyzw, scale) ------
    def group_local(node):
        s = np.array(node["scale"], float)
        assert (s.max() - s.min()) <= CROSS_GROUP_TOL, (
            f"{node['name']} scale not uniform: {node['scale']}")
        return {"pos": list(node["pos"]), "quat_xyzw": list(node["quat"]), "scale": float(s.mean())}

    locals_by_probe = [{name: group_local(node) for name, node in _probe_groups(probe).items()}
                        for probe in probes]

    def _vec(gl):
        return np.array(gl["pos"] + gl["quat_xyzw"] + [gl["scale"]])

    # cross-probe: a given group's local must not depend on which pose the
    # probe was captured in -- it's a constant container transform.
    worst_cross_probe = max(
        float(np.abs(_vec(locals_by_probe[0][name]) - _vec(locals_by_probe[1][name])).max())
        for name in GROUP_NAMES
    )
    assert worst_cross_probe <= CROSS_PROBE_TOL, (
        f"group locals differ between probes: worst {worst_cross_probe}")

    # cross-group: within one probe, all ten groups share the same fixed
    # rotation + scale (measured worst spread: ~5.3e-15, see CROSS_GROUP_TOL above).
    worst_cross_group = 0.0
    for gl in locals_by_probe:
        vecs = [_vec(gl[name]) for name in GROUP_NAMES]
        for a in vecs:
            for b in vecs:
                worst_cross_group = max(worst_cross_group, float(np.abs(a - b).max()))
    assert worst_cross_group <= CROSS_GROUP_TOL, (
        f"group locals differ across the ten groups: worst {worst_cross_group}")

    # --- topology: which group each finger/thumb `_1` bone hangs from -------
    f2g = [_finger1_to_group(probe) for probe in probes]
    assert f2g[0] == f2g[1], "finger->group wiring disagrees between probes"
    finger_to_group = f2g[0]
    assert len(finger_to_group) == len(GROUP_NAMES) == 10
    v1_orphans = {b["name"] for b in v1_bones if b["parent"] is None and b["name"] != "pelvis"}
    assert set(finger_to_group) == v1_orphans, (set(finger_to_group), v1_orphans)

    # --- R15: correct the clavicle rest, then re-derive what hangs off it ---
    # Done on a copy of v1 BEFORE the fingers are reparented, so the plain
    # 74-bone FK below is valid, and before the ten groups are appended, so
    # they inherit right_wrist's corrected value rather than v1's stale one.
    v1_bones = [dict(b) for b in v1_bones]
    for n in CLAVICLE_BONE_NAMES:
        v1_bones[name_to_i_v1[n]]["rest_local_q"] = base_clav_q[n]
    clav_world = _fk_world_matrices(v1_bones)
    clav_moved = []
    for n in CLAVICLE_DEPENDENT_NAMES:
        i = name_to_i_v1[n]
        was = np.array(v1_bones[i]["rest_world_p"], float)
        now = [round(float(v), 6) for v in clav_world[i][:3, 3]]
        v1_bones[i]["rest_world_p"] = now
        clav_moved.append((float(np.linalg.norm(np.array(now) - was)), n))
    # Positive control: the clavicle's own world position must NOT move (its
    # own rotation cannot move it), and the LEFT arm must not move at all --
    # if either did, this is reaching further than the right clavicle.
    for n in ("right_clavicle", "left_clavicle", "left_shoulder", "left_elbow", "left_wrist"):
        i = name_to_i_v1[n]
        dev = float(np.abs(clav_world[i][:3, 3]
                           - np.array(v1_bones[i]["rest_world_p"], float)).max())
        assert dev <= FK_BODY_SELFCHECK_TOL, f"R15 moved {n} by {dev} -- it must not"
    assert min(d for d, _ in clav_moved) > 0.1, (
        f"R15 changed the right clavicle's rest but its chain barely moved: {clav_moved}")

    # --- assemble the 74 v1 bones: copy verbatim, add scale/is_group --------
    # The 34 body bones keep v1's rest verbatim apart from the R15 clavicle
    # correction above. The 40 finger/thumb bones take rest_local_q from the
    # base-pose probe (R14); their rest_world_p is a placeholder here and is
    # recomputed by FK once the groups exist below.
    out_bones = [{
        "name": b["name"],
        "parent": b["parent"],
        "rest_local_q": (base_finger_q[b["name"]] if b["name"] in base_finger_q
                         else b["rest_local_q"]),
        "rest_local_p": b["rest_local_p"],
        "rest_world_p": b["rest_world_p"],
        "scale": 1.0,
        "is_group": False,
        "solve": b["solve"],
    } for b in v1_bones]
    assert sum(1 for b in out_bones if b["name"] in base_finger_q) == 40

    # reconnect the ten finger islands through their group's (soon-to-exist) index
    group_index = {name: 74 + k for k, name in enumerate(GROUP_NAMES)}
    for b in out_bones:
        if b["name"] in finger_to_group:
            b["parent"] = group_index[finger_to_group[b["name"]]]

    # flip the 30 phalanges (chain root + 2 interior joints; tip stays
    # solve:false). Found structurally by walking parent pointers -- this
    # asset's naming is not reliably unique (e.g. "joint7" names two
    # unrelated bones, see posegoblin_rig.py's module docstring), so a
    # structural walk is safer than a name-suffix match.
    children = {i: [] for i in range(74)}
    for i, b in enumerate(v1_bones):
        if b["parent"] is not None:
            children[b["parent"]].append(i)
    n_flipped = 0
    for root_name in finger_to_group:
        chain = [name_to_i_v1[root_name]]
        while children[chain[-1]]:
            assert len(children[chain[-1]]) == 1, f"finger chain branches at bone {chain[-1]}"
            chain.append(children[chain[-1]][0])
        assert len(chain) == 4, f"{root_name} chain length {len(chain)} != 4 (_1,_2,_3,_tip)"
        for i in chain[:-1]:  # every bone but the tip
            assert out_bones[i]["solve"] is False
            out_bones[i]["solve"] = True
            n_flipped += 1
        assert out_bones[chain[-1]]["solve"] is False  # tip untouched
    assert n_flipped == 30

    # --- append the ten groups ------------------------------------------------
    wrist_world = _fk_world_matrices(v1_bones)
    for k, gname in enumerate(GROUP_NAMES):
        wrist_name = "left_wrist" if k < 5 else "right_wrist"
        wrist_i = name_to_i_v1[wrist_name]
        gl = locals_by_probe[0][gname]  # both probes agree (asserted above); probe 0 arbitrarily

        # self-check (C2): group local pos is ~0, so its world position
        # must equal its wrist's rest_world_p. Verify independently via FK
        # rather than assuming it.
        wrist_stored_p = np.array(v1_bones[wrist_i]["rest_world_p"], float)
        fk_p = (wrist_world[wrist_i] @ _mat_local(gl["pos"], gl["quat_xyzw"], gl["scale"]))[:3, 3]
        dev = float(np.abs(fk_p - wrist_stored_p).max())
        assert dev <= GROUP_WORLD_SELFCHECK_TOL, (
            f"{gname} FK world pos {dev} away from {wrist_name}.rest_world_p (self-check)")

        out_bones.append({
            "name": gname,
            "parent": wrist_i,
            "rest_local_q": _xyzw_to_wxyz(gl["quat_xyzw"]),  # probe is [x,y,z,w] -> asset [w,x,y,z]
            "rest_local_p": gl["pos"],
            "rest_world_p": wrist_stored_p.tolist(),  # == wrist's; local pos ~0, self-checked above
            "scale": gl["scale"],
            "is_group": True,
            "solve": False,
        })

    assert len(out_bones) == 84
    solved = sum(1 for b in out_bones if b["solve"])
    assert solved == 64

    # --- R14: recompute the 40 finger bones' rest_world_p --------------------
    # Their world geometry moves with the corrected locals, so v1's stored value
    # is no longer true for them. Positive control FIRST: the same FK must
    # reproduce the 34 BODY bones' stored rest_world_p, which this build does not
    # touch -- if it cannot, the recomputed finger values are not trustworthy
    # either, and a clean-looking rebuild would be meaningless.
    fk_world_p = _fk_world_positions(out_bones)
    body_dev, body_name = 0.0, None
    for i, b in enumerate(out_bones[:74]):
        if b["name"] in base_finger_q:
            continue
        d = float(np.abs(fk_world_p[i] - np.array(b["rest_world_p"], float)).max())
        if d > body_dev:
            body_dev, body_name = d, b["name"]
    assert body_dev <= FK_BODY_SELFCHECK_TOL, (
        f"_fk_world_positions does not reproduce v1's own body rest_world_p "
        f"({body_name} off by {body_dev}) -- do not trust its finger output")

    finger_moved = []
    for i, b in enumerate(out_bones[:74]):
        if b["name"] not in base_finger_q:
            continue
        was = np.array(b["rest_world_p"], float)
        b["rest_world_p"] = [round(float(v), 6) for v in fk_world_p[i]]
        finger_moved.append((float(np.linalg.norm(np.array(b["rest_world_p"]) - was)), b["name"]))
    assert len(finger_moved) == 40

    doc = {
        "version": "posegoblin_rig_v2",
        "provenance": {
            "captured": "2026-08-22",
            "source": ("posegoblin_rig_v1.json (34 body bones) + capture-rigbase2-11.json "
                       "(40 finger/thumb bones, R14) + capture-rigprobe-09.json + "
                       "capture-rigrestprobe-10.json (topology and the ten group constants)"),
            "note": ("v1's 74 bones at the same indices, plus ten Group nodes "
                      "(transform4..13, indices 74..83) appended: the constant wrist-side "
                      "containers the ten *_thumb_1/*_finger_1 chains actually hang off on "
                      "the live rig (fixed rotation + 0.1 uniform scale). Their `parent` "
                      "flips from null to their group's index; the 30 finger phalanges "
                      "(_1,_2,_3) flip solve:false -> true (tips stay false). Built by "
                      "tools_build_rig_v2.py, which re-runs and prints the reproduction-"
                      "gate numbers on every rebuild."),
            "finger_rest_note": (
                "R14 (2026-08-22): the 40 finger/thumb bones' rest_local_q comes from "
                "capture-rigbase2-11.json -- the rig reset to its TRUE base pose -- NOT "
                "from v1. v1's finger rest descends from capture-rigrest-02.json, which "
                "was captured with the RIGHT hand curled into a partial fist: 33.87 deg "
                "mean / 44.72 deg max from base on all 15 right phalanges, against 5.72 / "
                "7.56 on the left. Harmless while nothing posed the fingers; v16 task 5 "
                "applies rest_local o R(axis, angle), so a posed rest over-curls every "
                "right hand by that amount. In the base pose the two hands carry the SAME "
                "local quaternion on all 20 bones per side to 1.79e-4 deg. Those 40 bones' "
                "rest_world_p is recomputed here by scale-aware FK (the _1 bones do not "
                "move; _2/_3/_tip do). The other 32 BODY bones keep v1's values verbatim "
                "-- the base probe's body is at whatever pose the live app was showing "
                "and must not be read for anything but the hands and the two clavicles "
                "(see clavicle_rest_note). posegoblin_rig_v1.json is untouched and "
                "remains the historical asset."),
            "clavicle_rest_note": (
                "R15 (2026-08-22): the two clavicles' rest_local_q also comes from "
                "capture-rigbase2-11.json. v1's LEFT clavicle already equals the base "
                "probe's to v1's own rounding; its RIGHT is 6.077 deg off -- Scott's "
                "incidental posing recorded as rest, the same defect R14 removed from "
                "the right hand. Reading the probe's BODY is otherwise forbidden above, "
                "so this exception is earned STRUCTURALLY rather than by trusting it: "
                "the two clavicle bone offsets are exact negatives of each other, which "
                "forces their rest frames to differ by 180 deg about X, and the probe's "
                "pair satisfies that exactly ([0.707107, -0.707107, 0, 0] and "
                "[0.707107, +0.707107, 0, 0], mirror deviation 0.0 deg) while v1's does "
                "not. A posed capture cannot land on that by luck. It matters because "
                "the solver emits `rest_local . delta`: a rest frame 6 deg off the app's "
                "own splits a symmetric clavicle rotation asymmetrically between "
                "elevation and protraction, which Scott reported as a dislocated-looking "
                "right shoulder ball. right_shoulder/right_elbow/right_wrist rest_world_p "
                "are recomputed by FK to follow (the clavicle's own does not move; its "
                "rotation cannot move it), the right hand's twenty bones follow on the "
                "R14 path, and the five right-hand groups inherit right_wrist's value. "
                "CAVEAT, measured and deliberately not acted on here: v1's body rest "
                "descends from capture-rigrest-02.json, which is a POSED capture -- 43 of "
                "its 73 bones differ from the base probe, up to 105.9 deg at "
                "right_shoulder. R14 migrated the hands, R15 the clavicles; the rest of "
                "the body is still v1's A-pose and is what the solver solves against."),
        },
        "bones": out_bones,
    }

    # --- build-time reproduction gate: refuse to write a broken asset -------
    gate_results = {}
    for path, probe in zip(PROBE_PATHS, probes):
        worst = _reproduce(doc, probe)
        gate_results[path.name] = worst
        assert worst <= GATE_TOL, (
            f"reproduction gate failed against {path.name}: worst error {worst} > {GATE_TOL}")

    V2.write_text(json.dumps(doc, indent=1))
    finger_moved.sort(reverse=True)
    print(f"{len(out_bones)} nodes, {solved} solved, groups 74..83")
    print(f"  R14 base-pose finger rest: hands symmetric to {worst_sym:.2e} deg "
          f"({worst_sym_name}); offsets match v1 to {worst_offset:.2e}")
    print(f"  R15 clavicle rest from base probe: mirror dev {mirror_dev:.2e} deg, "
          f"left agrees with v1 to {dev_l:.2e} deg, right corrected by {dev_r:.3f} deg")
    print("     rest_world_p followed: "
          + ", ".join(f"{n} {d:.4f}" for d, n in clav_moved))
    print(f"  R14 rest_world_p recomputed for 40 finger bones; FK reproduces the 34 body "
          f"bones to {body_dev:.2e} ({body_name})")
    print(f"     largest shift {finger_moved[0][0]:.4f} ({finger_moved[0][1]}), "
          f"smallest {finger_moved[-1][0]:.2e} ({finger_moved[-1][1]})")
    print(f"wrote {V2} ({V2.stat().st_size/1024:.0f} KB)")
    for name, worst in gate_results.items():
        print(f"  gate vs {name}: worst |world - probe.world| = {worst:.3e}")


if __name__ == "__main__":
    main()
