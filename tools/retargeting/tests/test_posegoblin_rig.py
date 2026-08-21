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


def test_solving_for_the_rest_itself_returns_the_rest_locals():
    """targets == rest world positions => every D is identity => locals == rest.
    The zero of the solver; if this fails nothing else is interpretable.

    Iterates the SOLVED set only (not rig.order): solve_rig_locals returns
    locals for the 34 solved bones alone, matching fk_world_orientations/
    fk_world_positions (module docstring) -- the 40 unsolved finger/thumb
    bones are Task 3's concern."""
    from retargeting.retargeters.posegoblin_rig import load_rig, solve_rig_locals
    rig = load_rig()
    out = solve_rig_locals(rig, dict(rig.rest_world_p))
    solved = [i for i in rig.order if rig.solve[i]]
    worst = max(_qang(out[i], rig.rest_local_q[i]) for i in solved)
    assert worst < 0.5, f"worst bone off rest by {worst:.2f} deg"


def test_a_global_yaw_of_the_targets_is_absorbed_by_the_root_alone():
    """Rotate every target 90 deg about +Y: the solved WORLD directions must
    follow, and every bone's world delta relative to the root must be ~0 —
    i.e. the yaw lands in the root, not smeared down the chain."""
    import numpy as np
    from retargeting.retargeters.posegoblin_rig import (
        load_rig, solve_rig_locals, fk_world_positions)
    rig = load_rig()
    c, s = np.cos(np.pi / 2), np.sin(np.pi / 2)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    targets = {i: R @ p for i, p in rig.rest_world_p.items()}
    out = solve_rig_locals(rig, targets)
    got = fk_world_positions(rig, out)
    solved = [i for i in rig.order if rig.solve[i]]
    # positions can differ by bone-length scaling? no: same skeleton, same
    # lengths, directions solved exactly -> world positions reproduce targets
    err = max(float(np.linalg.norm(got[i] - targets[i])) for i in solved)
    assert err < 1e-2, f"max position error {err}"


def _qang(a, b):
    import numpy as np
    d = abs(float(np.dot(a, b)))
    return float(np.degrees(2 * np.arccos(min(1.0, d))))


_FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "mhr70_row.json")


def test_rig_state_covers_every_rig_bone_and_serializes():
    import json
    from retargeting.retargeters.posegoblin_rig import load_rig, rig_state_from_mhr70
    row = json.load(open(_FIXTURE))
    kp = np.asarray(row["mhr70_xyz"], np.float32).reshape(70, 3)
    st = rig_state_from_mhr70(kp)
    rig = load_rig()
    # ruling 5: pose is NAME-keyed (PoseGoblin reads state.pose[child.name]);
    # the two "joint7" bones collapse to a single entry, last-one-wins in
    # rig.order. 73 unique NAMES out of 74 bones -- not set(rig.order),
    # which is 74 INDICES and is not what the brief's literal text claims.
    unique_names = set(rig.name.values())
    assert len(unique_names) == 73
    assert set(st["pose"]) == unique_names
    q = st["pose"]["pelvis"]
    assert set(q) >= {"isQuaternion", "_x", "_y", "_z", "_w"}
    # Key PRESENCE alone can't catch a transposed component (e.g. w written
    # into the _x slot) -- the global constraint is that [w,x,y,z] -> three.js
    # {_x,_y,_z,_w} conversion happens ONLY at serialization, so verify it did
    # that correctly for a bone whose value is independently recomputable.
    from retargeting.retargeters.posegoblin_rig import rig_targets_from_mhr70, solve_rig_locals
    expected_pelvis_q = solve_rig_locals(rig, rig_targets_from_mhr70(kp))[rig.index_of_name["pelvis"]]
    assert q["_w"] == pytest.approx(float(expected_pelvis_q[0]), abs=1e-9)
    assert q["_x"] == pytest.approx(float(expected_pelvis_q[1]), abs=1e-9)
    assert q["_y"] == pytest.approx(float(expected_pelvis_q[2]), abs=1e-9)
    assert q["_z"] == pytest.approx(float(expected_pelvis_q[3]), abs=1e-9)
    assert st["rigVersion"] == "posegoblin_rig_v1" and st["retargetVersion"] == 2
    json.dumps(st)                                   # wire-safe, no numpy leaks
    assert st["pelvisPosition"]["y"] > 2.0           # rig units (~10), not meters (~1)


def test_solved_directions_land_on_targets_for_a_real_row():
    """Machine gate in miniature (ruling 1). The all-children Kabsch fit
    trades exactness across children, so a multi-child joint cannot be exact
    for any single child -- demanding cosine >=0.99 everywhere asserts the
    impossible. The floor is instead split by STRUCTURE, never by which
    bones happen to fail:

      - a bone whose PARENT has exactly ONE usable child target -> EXACT, 0.99
      - every other bone (parent has 0 or >=2 usable child targets) -> FITTED, 0.90

    "usable child target" mirrors solve_rig_locals's own `pairs` gate
    exactly (the parent itself must have a target, the child must have a
    target, and both the rest-bone and target-bone vectors must be
    non-degenerate) -- so the split is read off rig structure + target
    availability, never off a measured cosine."""
    import json
    from retargeting.retargeters.posegoblin_rig import (
        load_rig, rig_targets_from_mhr70, solve_rig_locals, fk_world_positions)
    row = json.load(open(_FIXTURE))
    kp = np.asarray(row["mhr70_xyz"], np.float32).reshape(70, 3)
    rig = load_rig()
    targets = rig_targets_from_mhr70(kp)
    got = fk_world_positions(rig, solve_rig_locals(rig, targets))
    solved = [i for i in rig.order if rig.solve[i]]

    def usable_child_count(n):
        """Exactly solve_rig_locals's own `pairs`-building gate, count only."""
        if n not in targets:
            return 0
        count = 0
        for c in rig.children[n]:
            if c not in targets:
                continue
            rb = rig.rest_world_p[c] - rig.rest_world_p[n]
            tb = targets[c] - targets[n]
            if np.linalg.norm(rb) > 1e-6 and np.linalg.norm(tb) > 1e-6:
                count += 1
        return count

    rows = []
    bad = {}
    for n in solved:
        if n not in targets:
            continue
        for c in rig.children[n]:
            if c not in targets:
                continue
            tb = targets[c] - targets[n]
            gb = got[c] - got[n]
            if np.linalg.norm(tb) < 1e-6 or np.linalg.norm(gb) < 1e-6:
                continue
            cos = float(tb @ gb / (np.linalg.norm(tb) * np.linalg.norm(gb)))
            kind = "EXACT" if usable_child_count(n) == 1 else "FITTED"
            floor = 0.99 if kind == "EXACT" else 0.90
            label = f"{rig.name[n]}->{rig.name[c]}"
            rows.append((label, kind, round(cos, 4), floor))
            if cos < floor:
                bad[f"{label} ({kind})"] = round(cos, 4)

    print(f"\n{'bone':<28}{'kind':<8}{'cosine':>8}{'floor':>8}")
    for label, kind, cos, floor in rows:
        print(f"{label:<28}{kind:<8}{cos:>8}{floor:>8}")

    # Positive control (CLAUDE.md rule 1): "0 bad" is indistinguishable from
    # "the loop never matched anything" -- e.g. if targets were keyed wrong
    # (name instead of index), `n not in targets` would be vacuously true for
    # every n and this would trivially "pass" having tested nothing. Prove
    # the gate actually ran over a real, non-trivial set of bones.
    assert len(rows) >= 20, f"only {len(rows)} bone-directions were testable -- the gate did not run"
    assert not bad, f"bones off their floor: {bad}"
