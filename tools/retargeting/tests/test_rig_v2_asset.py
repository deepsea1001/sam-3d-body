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
# R14: the rig reset to its TRUE base pose -- the source of the 40 finger/thumb
# bones' rest, and of nothing else (its BODY is at whatever pose the live app was
# showing; right_shoulder alone reads 105.9 deg from v1's rest).
BASE_PROBE = BP / "capture-rigbase2-11.json"

# The 40 finger/thumb bones (3 phalanges + tip on each of five digits, per side).
FINGER_BONE_NAMES = [f"{side}_{stem}_{k}"
                     for side in ("left", "right")
                     for stem in ("thumb", "index_finger", "middle_finger",
                                  "ring_finger", "pinky_finger")
                     for k in ("1", "2", "3", "tip")]


def _mat_local(pos, quat_xyzw, scale):
    x, y, z, w = quat_xyzw
    R = np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                  [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                  [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])
    M = np.eye(4); M[:3, :3] = R * scale; M[:3, 3] = pos
    return M


def _base_probe_finger_locals():
    """{finger bone name -> rest_local_q [w,x,y,z]} straight from the base-pose
    probe. The probe is a raw three.js dump ([x,y,z,w]); the asset is [w,x,y,z].

    Ambiguous names are skipped ("joint7" names two unrelated bones), then every
    one of the 40 is asserted present -- a silently short dict here would turn
    every caller's finger branch into a body branch and prove nothing."""
    nodes = json.loads(BASE_PROBE.read_text())["nodes"]
    counts = {}
    for n in nodes:
        if n["type"] == "Bone":
            counts[n["name"]] = counts.get(n["name"], 0) + 1
    out = {}
    for n in nodes:
        if n["type"] == "Bone" and counts[n["name"]] == 1 and n["name"] in FINGER_BONE_NAMES:
            x, y, z, w = n["quat"]
            out[n["name"]] = [w, x, y, z]
    assert set(out) == set(FINGER_BONE_NAMES) and len(out) == 40, sorted(
        set(FINGER_BONE_NAMES) - set(out))
    return out


def test_v2_shape_and_topology():
    d = json.loads(V2.read_text())
    bones = d["bones"]
    assert len(bones) == 84
    v1 = json.loads((BP / "posegoblin_rig_v1.json").read_text())["bones"]
    base = _base_probe_finger_locals()
    for i in range(74):  # v1 indices preserved
        assert bones[i]["name"] == v1[i]["name"]
        if bones[i]["name"] in base:
            # R14: the 40 finger/thumb bones' rest comes from the TRUE base pose,
            # not from v1 -- whose right hand was captured curled into a fist.
            assert bones[i]["rest_local_q"] == pytest.approx(base[bones[i]["name"]], abs=1e-12)
        else:
            assert bones[i]["rest_local_q"] == v1[i]["rest_local_q"]   # 34 body bones verbatim
    assert sum(1 for b in bones[:74] if b["name"] in base) == 40       # positive control
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


def _fk_world_positions(bones):
    """{index -> (3,) world position} from the asset's OWN stored locals, by
    scale-aware FK. Independent of the runtime package on purpose -- this file
    validates the asset, not the loader. Memoized recursion, because the ten
    reconnected finger `_1` bones (14..57) parent to a group at 74..83."""
    worlds = {}

    def world_of(i):
        if i not in worlds:
            b = bones[i]
            w, x, y, z = b["rest_local_q"]
            M = _mat_local(b["rest_local_p"], [x, y, z, w], b.get("scale", 1.0))
            worlds[i] = M if b["parent"] is None else world_of(b["parent"]) @ M
        return worlds[i]

    return {i: world_of(i)[:3, 3] for i in range(len(bones))}


# 6 decimals is how rest_world_p is written; FK from those same rounded locals
# drifts a little further down a chain (measured worst: 1.7e-5 on the fingers).
_REST_WORLD_P_FK_TOL = 1e-4


def _rest_world_p_mismatches(d, v1):
    """C2, in three branches -- returns the offending names; [] means clean.

      - the 34 BODY bones: rest_world_p is copied verbatim from v1;
      - the 40 FINGER/thumb bones (R14): their locals now come from the base
        pose, so their world geometry moved with them and is RECOMPUTED. It
        must agree with scale-aware FK over the asset's own stored locals;
      - the ten GROUPS: rest_world_p equals their own wrist's (`parent` IS the
        wrist index, by schema -- the group's local position is ~0).

    None of it is covered by _reproduce below -- that FKs each node's local
    TRS and never reads rest_world_p at all -- so this is C2's only
    regression coverage under pytest."""
    bones = d["bones"]
    base = _base_probe_finger_locals()
    fk = _fk_world_positions(bones)
    bad = []
    for i in range(74):
        b = bones[i]
        if b["name"] in base:
            if not np.allclose(b["rest_world_p"], fk[i], atol=_REST_WORLD_P_FK_TOL):
                bad.append(b["name"])
        elif b["rest_world_p"] != v1[i]["rest_world_p"]:
            bad.append(b["name"])
    for g in bones[74:]:
        if g["rest_world_p"] != bones[g["parent"]]["rest_world_p"]:
            bad.append(g["name"])
    return bad


def test_v2_rest_world_p_matches_v1_own_fk_and_own_wrist():
    d = json.loads(V2.read_text())
    v1 = json.loads((BP / "posegoblin_rig_v1.json").read_text())["bones"]
    assert _rest_world_p_mismatches(d, v1) == []


def test_v2_finger_rest_world_p_actually_moved_off_v1():
    """The FK branch above would also pass if nothing had changed, so pin the
    change itself: R14 re-based the finger rest, and the distal bones' world
    positions had to follow. The `_1` bones must NOT move (their world position
    depends on the wrist chain and their own offset, never on their own
    rotation) -- which is what makes this a two-sided check rather than a
    one-sided "something changed"."""
    bones = json.loads(V2.read_text())["bones"]
    v1 = json.loads((BP / "posegoblin_rig_v1.json").read_text())["bones"]
    moved = {}
    for i in range(74):
        if bones[i]["name"] in _base_probe_finger_locals():
            moved[bones[i]["name"]] = float(np.linalg.norm(
                np.array(bones[i]["rest_world_p"]) - np.array(v1[i]["rest_world_p"])))
    assert len(moved) == 40
    roots = {n: v for n, v in moved.items() if n.endswith(("_thumb_1", "_finger_1"))}
    assert len(roots) == 10 and max(roots.values()) < 1e-4, roots

    # v1's right hand carried 33.87 deg mean of baked-in curl against the left's
    # 5.72 -- a ratio of 5.92 -- so every non-root RIGHT bone must have moved
    # several times further than its left twin. Measured per bone: 4.93x-5.89x,
    # tightly clustered, which is the fingerprint of one constant miscapture
    # rather than of scattered noise. Bound at 3x for float headroom.
    pairs = [(n, moved[n], moved[n.replace("left", "right", 1)])
             for n in moved if n.startswith("left_") and not n.endswith("_1")]
    assert len(pairs) == 15
    for n, l, r in pairs:
        assert r > 3.0 * l, (n, l, r)
    assert max(r for _, _, r in pairs) == pytest.approx(0.9949, abs=1e-3)   # right_middle_tip


def test_rest_world_p_check_can_fail_on_a_mutated_copy():
    """Positive control (CLAUDE.md rule 1): _rest_world_p_mismatches must be
    proven capable of a non-empty result for EACH of C2's three branches, not
    just trusted to pass because the real asset happens to be correct."""
    v1 = json.loads((BP / "posegoblin_rig_v1.json").read_text())["bones"]

    d = json.loads(V2.read_text())
    d["bones"][14]["rest_world_p"] = [0.0, 0.0, 0.0]  # left_thumb_1 (finger/FK branch)
    assert _rest_world_p_mismatches(d, v1) == ["left_thumb_1"]

    d = json.loads(V2.read_text())
    i_body = next(i for i in range(74)
                  if d["bones"][i]["name"] == "left_wrist")
    d["bones"][i_body]["rest_world_p"] = [0.0, 0.0, 0.0]  # body branch
    # the ten groups inherit their wrist's value, so five of them go with it
    assert _rest_world_p_mismatches(d, v1)[0] == "left_wrist"

    d = json.loads(V2.read_text())
    d["bones"][74]["rest_world_p"] = [0.0, 0.0, 0.0]  # transform4 (group branch)
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


# --------------------------------------------------------------------------
# R14: the finger rest must be the rig's TRUE base pose, not a posed hand
# --------------------------------------------------------------------------

# The 20 LEFT finger/thumb bones (3 phalanges + tip on each of five digits).
# Their right-hand counterparts are the same names with the side swapped.
_LEFT_FINGER_BONES = [f"left_{stem}_{k}"
                      for stem in ("thumb", "index_finger", "middle_finger",
                                   "ring_finger", "pinky_finger")
                      for k in ("1", "2", "3", "tip")]

# Measured, not assumed. PoseGoblin authors a hand pose as ONE set of absolute
# LOCAL quaternions and applies it to both hands verbatim -- so at the rig's
# true base pose the two hands' finger locals are the SAME quaternion, not
# mirrored ones. Confirmed against `capture-rigbase2-11.json` (Scott reset the
# live rig to base on 2026-08-22): worst left-vs-right disagreement over all 20
# bones is 1.79e-4 deg, i.e. three.js float noise. The 0.01 deg bound below is
# ~56x that noise floor and ~2e5 times smaller than the 37.2 deg asymmetry the
# v1-derived rest actually carried.
_FINGER_SYMMETRY_TOL_DEG = 0.01


def _quat_deg(a, b):
    """Angle between two [w,x,y,z] rotations, sign-normalised."""
    a = np.asarray(a, float); a = a / np.linalg.norm(a)
    b = np.asarray(b, float); b = b / np.linalg.norm(b)
    return float(np.degrees(2 * np.arccos(min(1.0, abs(float(a @ b))))))


def _asymmetric_finger_rests(bones, tol_deg=_FINGER_SYMMETRY_TOL_DEG):
    """[(left bone name, degrees)] for every finger/thumb bone whose LEFT and
    RIGHT rest locals disagree by more than *tol_deg*. [] means symmetric.

    This is R14's regression guard: `capture-rigrest-02.json`, the capture v1's
    finger rest came from, was taken with the RIGHT hand curled ~34 deg into a
    fist, and that curl sat in the asset as "rest" for four months. Because the
    v16 finger transfer applies `rest_local o R(axis, angle)`, a posed rest
    over-curls every right hand by exactly that amount -- a visible left/right
    asymmetry that no test could see. A future re-capture taken with a hand
    posed would reintroduce it silently; this is what stops that."""
    q = {}
    for b in bones:
        q.setdefault(b["name"], b["rest_local_q"])
    out = []
    for n in _LEFT_FINGER_BONES:
        d = _quat_deg(q[n], q[n.replace("left", "right", 1)])
        if d > tol_deg:
            out.append((n, d))
    return out


def test_left_and_right_finger_rests_are_the_same_local_quaternion():
    d = json.loads(V2.read_text())
    assert len(_LEFT_FINGER_BONES) == 20                   # positive control on the name list
    assert {b["name"] for b in d["bones"]} >= set(_LEFT_FINGER_BONES)
    assert _asymmetric_finger_rests(d["bones"]) == []


def test_finger_symmetry_check_fires_on_the_posed_hand_it_exists_to_catch():
    """Two positive controls (CLAUDE.md rule 1), because a clean [] is
    indistinguishable from a check that never ran.

    1. REAL data: v1 is kept in the repo as the historical asset and it carries
       the actual defect -- the right hand curled into a partial fist. The
       detector must name all 15 phalanges (the ten tips are identity on both
       sides even in v1, so they stay symmetric).
    2. SYNTHETIC: a single right-hand bone rotated on a copy of v2 must be
       named, and only it."""
    v1 = json.loads((BP / "posegoblin_rig_v1.json").read_text())["bones"]
    fired = _asymmetric_finger_rests(v1)
    assert [n for n, _ in fired] == [n for n in _LEFT_FINGER_BONES
                                     if not n.endswith("_tip")]
    assert max(deg for _, deg in fired) > 30.0, fired

    d = json.loads(V2.read_text())
    victim = next(b for b in d["bones"] if b["name"] == "right_middle_finger_2")
    w, x, y, z = victim["rest_local_q"]
    half = np.radians(20.0) / 2.0                     # 20 deg extra about local +X
    cw, sw = np.cos(half), np.sin(half)
    victim["rest_local_q"] = [w * cw - x * sw, w * sw + x * cw,
                              y * cw + z * sw, z * cw - y * sw]
    got = _asymmetric_finger_rests(d["bones"])
    assert [n for n, _ in got] == ["left_middle_finger_2"], got
    # 1e-3, not 1e-6: this bone carries the largest residual asymmetry in the
    # asset (3.7e-5 deg of three.js float noise), which rides on the 20 deg.
    assert got[0][1] == pytest.approx(20.0, abs=1e-3)
