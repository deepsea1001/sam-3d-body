"""The rig asset must be internally consistent: FK of its rest locals must
reproduce its captured rest world positions. A re-exported (re-rigged) asset
that breaks this fails HERE, loudly -- never as silently garbled poses.

Bones are keyed by INDEX, not name (ruling 5): PoseGoblin's own
captureCurrentState/RecallPoseCommand key bone state by child.name, and one
pair of bones legitimately shares the literal name "joint7" (left_heel's and
right_heel's toe joint). Matching that -- not diverging from it -- is correct,
so the asset and Rig must survive a duplicate name without collision.

The finger/thumb chains are excluded from the solve (ruling 4): on the live
rig they're parented under a Group with its own non-identity rotation and a
0.1 scale, which FK does not model, and modelling it was ruled out of scope
(not a search input, invisible at display scale). Nothing is deleted from the
asset -- the hierarchy test below still sees all 74 bones -- but the FK test
only checks the solved set, and a separate test pins that solved set to
exactly what's reachable from pelvis, so a future re-rig that reconnects the
fingers shows up as THAT assertion failing, not as silence."""
import os, sys
import numpy as np
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from retargeting.retargeters.posegoblin_rig import load_rig, fk_world_positions


def test_asset_loads_with_full_hierarchy():
    rig = load_rig()
    assert len(rig.order) == 74           # nothing is deleted, see module docstring
    assert rig.version == "posegoblin_rig_v1"
    for i in rig.order:                   # parent-first ordering is load-bearing for FK
        p = rig.parent[i]
        assert p is None or rig.order.index(p) < rig.order.index(i)
    solved_roots = [rig.name[i] for i in rig.order if rig.solve[i] and rig.parent[i] is None]
    assert solved_roots == ["pelvis"]


def test_solved_set_is_exactly_reachable_from_pelvis_and_is_pinned():
    """Cross-checks rig.solve against an INDEPENDENT graph walk (not the flag
    itself) so a future re-rig that reconnects the fingers -- without anyone
    remembering to update `solve` -- fails this assertion instead of passing
    silently with a stale flag."""
    rig = load_rig()
    pelvis_idx = rig.index_of_name["pelvis"]
    reachable = set()
    stack = [pelvis_idx]
    while stack:
        i = stack.pop()
        if i in reachable:
            continue
        reachable.add(i)
        stack.extend(rig.children[i])

    solved = {i for i in rig.order if rig.solve[i]}
    assert solved == reachable, (
        "solve flags disagree with true pelvis-reachability: "
        f"solved-not-reachable={sorted(rig.name[i] for i in solved - reachable)}, "
        f"reachable-not-solved={sorted(rig.name[i] for i in reachable - solved)}"
    )
    assert len(solved) == 34, f"expected 34 solved bones, got {len(solved)}"
    excluded = [i for i in rig.order if not rig.solve[i]]
    assert len(excluded) == 40, f"expected 40 excluded bones, got {len(excluded)}"
    for i in excluded:
        assert i not in reachable, f"{rig.name[i]!r} is excluded but reachable from pelvis"


def test_fk_of_rest_locals_reproduces_captured_world_positions():
    rig = load_rig()
    world = fk_world_positions(rig, rig.rest_local_q)
    solved = [i for i in rig.order if rig.solve[i]]
    assert set(world.keys()) == set(solved)
    err = {i: float(np.linalg.norm(world[i] - rig.rest_world_p[i])) for i in solved}
    bad = {rig.name[i]: round(e, 4) for i, e in err.items() if e > 1e-3}
    assert not bad, f"FK does not reproduce the captured rest: {bad}"


def test_duplicate_bone_name_is_not_silently_resolved():
    """The two 'joint7' bones (ruling 5) must keep independent data, and a
    name lookup for the ambiguous name must raise rather than guess."""
    rig = load_rig()
    joint7_indices = [i for i in rig.order if rig.name[i] == "joint7"]
    assert len(joint7_indices) == 2, "expected the two legitimately-duplicated joint7 bones"
    a, b = joint7_indices
    assert rig.parent[a] != rig.parent[b]              # one under left_heel, one under right_heel
    assert not np.allclose(rig.rest_world_p[a], rig.rest_world_p[b])  # distinct bones, not collided

    assert "joint7" not in rig.index_of_name
    with pytest.raises(KeyError):
        _ = rig.index_of_name["joint7"]
