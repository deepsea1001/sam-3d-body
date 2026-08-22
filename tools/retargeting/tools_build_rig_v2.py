#!/usr/bin/env python
"""Build posegoblin_rig_v2.json: reconnect the ten orphaned finger islands
(`*_thumb_1`/`*_finger_1`, `parent: null` in v1) through the ten constant
Group nodes they actually hang off in the live three.js scene
(`transform4`..`transform13`, each a fixed rotation + 0.1 uniform scale).

No torch: pure numpy/json, reading v1 plus both rig probes
(bind_poses/capture-rigprobe-09.json, capture-rigrestprobe-10.json).

Quaternion conventions (do not mix these up -- see the same note in
tests/test_rig_v2_asset.py):
  - the ASSET (`rest_local_q` on every node, in both v1 and v2) is
    [w,x,y,z] -- fed straight into this repo's QuaternionMath.
  - the PROBES (everything under `nodes[].quat`) are raw three.js dumps,
    [x,y,z,w].
v1 bones are copied verbatim (already [w,x,y,z], untouched). The ten new
groups' `rest_local_q` is read from a probe in [x,y,z,w] and converted to
[w,x,y,z] before it is written to the asset.

Both probes are POSED STATES, not rest poses -- never read rest geometry off
their world matrices. They are used only to (a) learn topology (which group
each `_1` bone hangs from), (b) read the ten groups' constant local
transforms, and (c) validate, by FK-reproducing each probe's OWN locals, that
the topology + group constants this script wrote are actually correct (the
gate at the bottom of main()).

The 40 finger bones' `rest_world_p` is v1's verbatim value -- already true
world geometry (e.g. left_index_finger_1 sits 0.99 units from left_wrist,
real hand scale, not island-local coordinates) -- and is NOT recomputed.
Only the ten new groups need a `rest_world_p`; each group's local position
reads as ~0 (three.js float noise on what is conceptually an identity
offset), so it equals its wrist's rest_world_p -- computed via FK from v1
rest locals and self-checked against the wrist's own stored value below.
"""
import json
from pathlib import Path

import numpy as np

BP = Path(__file__).resolve().parent / "bind_poses"
V1 = BP / "posegoblin_rig_v1.json"
V2 = BP / "posegoblin_rig_v2.json"
PROBE_PATHS = [BP / "capture-rigprobe-09.json", BP / "capture-rigrestprobe-10.json"]

GROUP_NAMES = [f"transform{k}" for k in range(4, 14)]  # transform4..8 left, 9..13 right (binding, from probes)

# Tolerances -- see task-2-report.md for the measured values behind each of these.
BONE_SCALE_TOL = 1e-6              # probe Bone nodes must read scale ~= 1.0
CROSS_PROBE_TOL = 1e-9             # a given group's local must be pose-independent (measured: exactly 0.0)
CROSS_GROUP_TOL = 1e-6             # the ten groups share one fixed local transform (measured: ~5.3e-15)
GROUP_WORLD_SELFCHECK_TOL = 1e-3   # FK group world pos vs its wrist's stored rest_world_p (generous: v1's
                                    # own 6-decimal rounding alone accounts for ~1.7e-5 of chain drift)
GATE_TOL = 1e-6                    # build-time reproduction gate (mirrors tests/test_rig_v2_asset.py)


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
    out = {n["name"]: n for n in probe["nodes"] if n["type"] == "Group" and n["name"] in GROUP_NAMES}
    missing = set(GROUP_NAMES) - set(out)
    assert not missing, f"probe missing group nodes: {sorted(missing)}"
    return out


def _finger1_to_group(probe):
    """{finger/thumb `_1` bone name -> group name}, from probe topology: a
    Bone whose parent_uuid resolves to one of the ten transform Groups."""
    group_uuid_to_name = {g["uuid"]: name for name, g in _probe_groups(probe).items()}
    return {n["name"]: group_uuid_to_name[n["parent_uuid"]]
            for n in probe["nodes"] if n["type"] == "Bone" and n["parent_uuid"] in group_uuid_to_name}


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

    # --- probe sanity: every Bone node reads scale ~= 1.0 -------------------
    worst_bone_scale = max(
        float(np.max(np.abs(np.array(n["scale"], float) - 1.0)))
        for probe in probes for n in probe["nodes"] if n["type"] == "Bone"
    )
    assert worst_bone_scale <= BONE_SCALE_TOL, f"probe Bone scale deviates from 1.0 by {worst_bone_scale}"

    # --- extract each probe's ten group locals (pos, quat_xyzw, scale) ------
    def group_local(node):
        s = np.array(node["scale"], float)
        assert (s.max() - s.min()) <= CROSS_GROUP_TOL, f"{node['name']} scale not uniform: {node['scale']}"
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
    assert worst_cross_probe <= CROSS_PROBE_TOL, f"group locals differ between probes: worst {worst_cross_probe}"

    # cross-group: within one probe, all ten groups share the same fixed
    # rotation + scale (see task-2-report.md for the measured spread).
    worst_cross_group = 0.0
    for gl in locals_by_probe:
        vecs = [_vec(gl[name]) for name in GROUP_NAMES]
        for a in vecs:
            for b in vecs:
                worst_cross_group = max(worst_cross_group, float(np.abs(a - b).max()))
    assert worst_cross_group <= CROSS_GROUP_TOL, f"group locals differ across the ten groups: worst {worst_cross_group}"

    # --- topology: which group each finger/thumb `_1` bone hangs from -------
    f2g = [_finger1_to_group(probe) for probe in probes]
    assert f2g[0] == f2g[1], "finger->group wiring disagrees between probes"
    finger_to_group = f2g[0]
    assert len(finger_to_group) == len(GROUP_NAMES) == 10
    v1_orphans = {b["name"] for i, b in enumerate(v1_bones) if b["parent"] is None and b["name"] != "pelvis"}
    assert set(finger_to_group) == v1_orphans, (set(finger_to_group), v1_orphans)

    # --- assemble the 74 v1 bones: copy verbatim, add scale/is_group --------
    out_bones = [{
        "name": b["name"], "parent": b["parent"],
        "rest_local_q": b["rest_local_q"], "rest_local_p": b["rest_local_p"], "rest_world_p": b["rest_world_p"],
        "scale": 1.0, "is_group": False, "solve": b["solve"],
    } for b in v1_bones]

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

    doc = {
        "version": "posegoblin_rig_v2",
        "provenance": {
            "captured": "2026-08-22",
            "source": "posegoblin_rig_v1.json + capture-rigprobe-09.json + capture-rigrestprobe-10.json",
            "note": ("v1's 74 bones at the same indices, unchanged, plus ten Group nodes "
                      "(transform4..13, indices 74..83) appended: the constant wrist-side "
                      "containers the ten *_thumb_1/*_finger_1 chains actually hang off on "
                      "the live rig (fixed rotation + 0.1 uniform scale). Their `parent` "
                      "flips from null to their group's index; the 30 finger phalanges "
                      "(_1,_2,_3) flip solve:false -> true (tips stay false). Built by "
                      "tools_build_rig_v2.py, which re-runs and prints the reproduction-"
                      "gate numbers on every rebuild."),
        },
        "bones": out_bones,
    }

    # --- build-time reproduction gate: refuse to write a broken asset -------
    gate_results = {}
    for path, probe in zip(PROBE_PATHS, probes):
        worst = _reproduce(doc, probe)
        gate_results[path.name] = worst
        assert worst <= GATE_TOL, f"reproduction gate failed against {path.name}: worst error {worst} > {GATE_TOL}"

    V2.write_text(json.dumps(doc, indent=1))
    print(f"{len(out_bones)} nodes, {solved} solved, groups 74..83")
    print(f"wrote {V2} ({V2.stat().st_size/1024:.0f} KB)")
    for name, worst in gate_results.items():
        print(f"  gate vs {name}: worst |world - probe.world| = {worst:.3e}")


if __name__ == "__main__":
    main()
