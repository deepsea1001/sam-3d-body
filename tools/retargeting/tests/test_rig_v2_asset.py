"""v2 reconnects the ten orphaned finger islands (`*_thumb_1`/`*_finger_1`,
`parent: null` in v1) through the ten constant Group nodes they actually hang
off in the live three.js scene (`transform4`..`transform13`). v1 stays
untouched; v2 is a superset: the same 74 bones at the same indices, plus the
ten groups appended at 74..83.

Quaternion conventions (do not mix these up -- see also the same note in
tools_build_rig_v2.py):
  - the ASSET (`rest_local_q` on every node, in both v1 and v2) is
    [w,x,y,z] -- fed straight into this repo's QuaternionMath.
  - the PROBES (`capture-rigprobe-09.json`, `capture-rigrestprobe-10.json`,
    everything under `nodes[].quat`) are raw three.js dumps, [x,y,z,w].
`_mat_local` below takes [x,y,z,w] (probe convention); `_reproduce` converts
the asset's [w,x,y,z] group quat before calling it -- see the comment at that
call site.

Both probes are POSED STATES, not rest poses (one a review-grid pose, one the
app's own base pose; they differ from each other on 43 of 74 bones). They are
used only to (a) learn topology, (b) read the ten groups' constant local
transforms, and (c) validate by FK-reproducing each probe's OWN locals --
never to read rest geometry off their world matrices.
"""
import json
from pathlib import Path

import numpy as np
import pytest

BP = Path(__file__).resolve().parent.parent / "bind_poses"
V2 = BP / "posegoblin_rig_v2.json"
PROBES = [BP / "capture-rigprobe-09.json", BP / "capture-rigrestprobe-10.json"]


def _mat_local(pos, quat_xyzw, scale):
    x, y, z, w = quat_xyzw
    R = np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                  [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                  [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])
    M = np.eye(4); M[:3, :3] = R * scale; M[:3, 3] = pos
    return M


def test_v2_shape_and_topology():
    d = json.loads(V2.read_text())
    bones = d["bones"]
    assert len(bones) == 84
    v1 = json.loads((BP / "posegoblin_rig_v1.json").read_text())["bones"]
    for i in range(74):  # v1 indices preserved
        assert bones[i]["name"] == v1[i]["name"]
        assert bones[i]["rest_local_q"] == v1[i]["rest_local_q"]
    groups = bones[74:]
    assert [g["name"] for g in groups] == [f"transform{k}" for k in range(4, 14)]
    for g in groups:
        assert g["is_group"] and not g["solve"] and abs(g["scale"] - 0.1) < 1e-9
    name_to_i = {}
    for i, b in enumerate(bones):
        name_to_i.setdefault(b["name"], i)
    # finger islands reconnected: every *_1 finger bone's parent is a group
    for i, b in enumerate(bones[:74]):
        if b["name"].endswith(("_thumb_1", "_finger_1")):
            assert b["parent"] is not None and bones[b["parent"]]["is_group"]
    # groups parent to wrists
    lw, rw = name_to_i["left_wrist"], name_to_i["right_wrist"]
    assert all(g["parent"] == lw for g in groups[:5])
    assert all(g["parent"] == rw for g in groups[5:])
    # solve flags: 34 original + 30 phalanges
    assert sum(1 for b in bones if b.get("solve")) == 64
    for b in bones[:74]:
        if b["name"].endswith("_tip"):
            assert not b["solve"]


def _rest_world_p_mismatches(d, v1):
    """C2: the 74 v1 bones' rest_world_p is copied verbatim into v2, never
    recomputed (the 40 finger bones' values are already true world geometry);
    each of the ten groups' rest_world_p equals its own wrist's (`parent` IS
    the wrist index, by schema -- the group's local position is ~0). Returns
    the offending names; [] means clean.

    Neither half is covered by _reproduce below -- it FKs each node's local
    TRS and never reads rest_world_p at all -- so this is C2's only
    regression coverage under pytest."""
    bones = d["bones"]
    bad = [bones[i]["name"] for i in range(74)
           if bones[i]["rest_world_p"] != v1[i]["rest_world_p"]]
    for g in bones[74:]:
        if g["rest_world_p"] != bones[g["parent"]]["rest_world_p"]:
            bad.append(g["name"])
    return bad


def test_v2_rest_world_p_matches_v1_and_own_wrist():
    d = json.loads(V2.read_text())
    v1 = json.loads((BP / "posegoblin_rig_v1.json").read_text())["bones"]
    assert _rest_world_p_mismatches(d, v1) == []


def test_rest_world_p_check_can_fail_on_a_mutated_copy():
    """Positive control (CLAUDE.md rule 1): _rest_world_p_mismatches must be
    proven capable of a non-empty result for EACH half of C2, not just
    trusted to pass because the real asset happens to be correct."""
    v1 = json.loads((BP / "posegoblin_rig_v1.json").read_text())["bones"]

    d = json.loads(V2.read_text())
    d["bones"][14]["rest_world_p"] = [0.0, 0.0, 0.0]  # left_thumb_1, wrong on purpose
    assert _rest_world_p_mismatches(d, v1) == ["left_thumb_1"]

    d = json.loads(V2.read_text())
    d["bones"][74]["rest_world_p"] = [0.0, 0.0, 0.0]  # transform4, wrong on purpose
    assert _rest_world_p_mismatches(d, v1) == ["transform4"]


def _reproduce(d, probe):
    """FK each v2 node using the PROBE's own local TRS but the V2 asset's
    topology (parent indices, and the asset's stored group locals for group
    nodes) -> worst |world - probe.world| over all 84 nodes.

    Probe pelvis's parent is a nameless container ('' name, identity); v2
    pelvis parent is None -- the lookup falls back from None to ''. The
    first node anchors away any container transform so the comparison is
    container-independent.

    World matrices are resolved by memoized recursion, not a single forward
    pass over `bones`: the ten reconnected finger `_1` bones (indices in
    14..57) now parent to a group at 74..83, ABOVE their own index, which a
    plain `for i in range(84): worlds[i] = worlds[parent] @ local` pass
    would hit before `worlds[parent]` exists (KeyError). Recursing to the
    parent on demand handles a parent index on either side of the child's."""
    bones = d["bones"]
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
            # group nodes take the ASSET's stored local (that is what is
            # under test); bones take the probe's own posed local
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
        Wp = np.asarray(n["world"], float).reshape(4, 4).T   # column-major
        if anchor is None:
            anchor = Wp @ np.linalg.inv(W)
        worst = max(worst, float(np.abs(anchor @ W - Wp).max()))
    return worst


@pytest.mark.parametrize("probe_path", PROBES, ids=["posed", "basepose"])
def test_v2_topology_reproduces_each_probe(probe_path):
    """The probes are states, not rests; this validates topology + the group
    constants independent of pose (spec 4.2)."""
    d = json.loads(V2.read_text())
    probe = json.loads(probe_path.read_text())
    assert _reproduce(d, probe) <= 1e-6


def test_gate_can_fail_on_perturbed_group():
    d = json.loads(V2.read_text())
    q = list(d["bones"][74]["rest_local_q"]); q[0] += 0.05
    d["bones"][74]["rest_local_q"] = q
    probe = json.loads(PROBES[0].read_text())
    assert _reproduce(d, probe) > 1e-3
