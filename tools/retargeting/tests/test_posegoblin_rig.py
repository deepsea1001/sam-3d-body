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
    from retargeting.retargeters.posegoblin_rig import (
        load_rig, rig_state_from_mhr70, rig_targets_from_mhr70, solve_rig_locals, fk_world_positions)
    row = json.load(open(_FIXTURE))
    kp = np.asarray(row["mhr70_xyz"], np.float32).reshape(70, 3)
    st = rig_state_from_mhr70(kp)
    rig = load_rig()
    li = rig.index_of_name
    raw_targets = rig_targets_from_mhr70(kp)
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
    expected_pelvis_q = solve_rig_locals(rig, raw_targets)[li["pelvis"]]
    assert q["_w"] == pytest.approx(float(expected_pelvis_q[0]), abs=1e-9)
    assert q["_x"] == pytest.approx(float(expected_pelvis_q[1]), abs=1e-9)
    assert q["_y"] == pytest.approx(float(expected_pelvis_q[2]), abs=1e-9)
    assert q["_z"] == pytest.approx(float(expected_pelvis_q[3]), abs=1e-9)
    assert st["rigVersion"] == "posegoblin_rig_v1" and st["retargetVersion"] == 8
    # wire-safe: json.dumps ALONE is insufficient -- Python happily emits a
    # bare `NaN`/`Infinity` token (invalid JSON; JavaScript's JSON.parse
    # rejects it), so round-trip through parse_constant and make IT raise.
    # (rig_state_from_mhr70 also guards this internally via
    # _assert_all_finite, but the test independently verifies wire-safety
    # rather than trusting that guard was actually exercised.)
    def _reject_non_finite_token(token):
        raise ValueError(f"non-finite value would reach the wire: {token}")
    json.loads(json.dumps(st), parse_constant=_reject_non_finite_token)

    # ruling 10: pelvisPosition is the rig's REST pelvis position -- a pose
    # viewer shows the POSE, not the subject's translation through camera
    # space. Ground truth from every human-posed capture:
    # {x: 0, y: 10.478578602247456, z: 0}, identical to the rig's own rest.
    rest_pelvis = rig.rest_world_p[li["pelvis"]]
    assert st["pelvisPosition"]["x"] == pytest.approx(float(rest_pelvis[0]), abs=1e-9)
    assert st["pelvisPosition"]["y"] == pytest.approx(float(rest_pelvis[1]), abs=1e-9)
    assert st["pelvisPosition"]["z"] == pytest.approx(float(rest_pelvis[2]), abs=1e-9)

    def _chain_len(points, hip, knee, ankle):
        return (float(np.linalg.norm(points[knee] - points[hip]))
                + float(np.linalg.norm(points[ankle] - points[knee])))

    # ruling 10 note: `s` (leg-length scale) is no longer used for
    # pelvisPosition, but it is still ruling 10's explicitly-kept sole input
    # to groundY -- independently recompute both so that's still checked
    # (it previously had no committed assertion at all).
    rig_leg = np.mean([
        _chain_len(rig.rest_world_p, li["left_hip"], li["left_knee"], li["left_ankle"]),
        _chain_len(rig.rest_world_p, li["right_hip"], li["right_knee"], li["right_ankle"]),
    ])
    tgt_leg = np.mean([
        _chain_len(raw_targets, li["left_hip"], li["left_knee"], li["left_ankle"]),
        _chain_len(raw_targets, li["right_hip"], li["right_knee"], li["right_ankle"]),
    ])
    s = rig_leg / tgt_leg
    feet_idx = [li[n] for n in ("left_heel", "right_heel", "left_big_toe", "right_big_toe")]
    expected_ground_y = min(raw_targets[i][1] for i in feet_idx) * s
    assert st["groundY"] == pytest.approx(expected_ground_y, rel=1e-6)

    # ruling 8: pose-invariant "rig units, not metres" check. The original
    # `pelvisPosition.y > 2.0` heuristic silently assumed an upright subject
    # and broke on this fixture's Cyr-wheel (inverted-in-a-hoop) capture --
    # a low camera-relative pelvis there was CORRECT, not a units bug. FK
    # preserves the rig's own bone lengths under ANY pose (pure rotation of
    # fixed-length rest offsets), so check the SOLVED skeleton's
    # hip->knee->ankle chain length against the rig's rest leg length
    # instead of any position component -- true regardless of which way up
    # the subject is.
    got = fk_world_positions(rig, solve_rig_locals(rig, raw_targets))
    solved_leg = np.mean([
        _chain_len(got, li["left_hip"], li["left_knee"], li["left_ankle"]),
        _chain_len(got, li["right_hip"], li["right_knee"], li["right_ankle"]),
    ])
    assert abs(solved_leg - rig_leg) / rig_leg < 0.05, (
        f"solved leg length {solved_leg:.4f} vs rig rest leg length {rig_leg:.4f} "
        "-- not in rig units")


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
    availability, never off a measured cosine.

    Ruling 9 (amended): pelvis->left_hip / pelvis->right_hip are NOT
    excluded. rig_targets_from_mhr70 now synthesises the pelvis target
    above the hip line (matching the rig's own rest topology, ruling 9)
    instead of leaving it at MHR-70's raw hip MIDPOINT, so the pelvis's
    Kabsch fit is no longer fighting an unsatisfiable topology mismatch.
    Both bones go through the SAME gate as everything else -- no special
    case. A diagnostic (not exclusionary) structural-mismatch check is kept
    to CONFIRM the "structurally unsatisfiable" category is empty; if it
    ever isn't, that bone is still reported here, never silently dropped.

    Ruling 11: bones whose PARENT has no MHR-70 mapping at all (e.g. the
    toe tips: their parent left_toes/right_toes has none) can't be gated
    against a real target -- listed under an explicit UNGATED heading
    instead of silently vanishing from the table, alongside two more
    UNGATED categories for the same underlying reason (absence must never
    be the only evidence): a bone itself lacking a target (its parent has
    one, but there's nothing to point IT at), and a degenerate rest/target
    vector (checked for completeness; empty in practice)."""
    import json
    from retargeting.retargeters.posegoblin_rig import (
        load_rig, rig_targets_from_mhr70, solve_rig_locals, fk_world_positions)
    row = json.load(open(_FIXTURE))
    kp = np.asarray(row["mhr70_xyz"], np.float32).reshape(70, 3)
    rig = load_rig()
    li = rig.index_of_name
    targets = rig_targets_from_mhr70(kp)
    got = fk_world_positions(rig, solve_rig_locals(rig, targets))
    solved = [i for i in rig.order if rig.solve[i]]

    def _ang(a, b):
        return float(np.degrees(np.arccos(np.clip(
            (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))))

    # Ruling 9 evidence: the raw hip MIDPOINT (what the pelvis target used
    # to be -- always exactly 180 deg between the two hip directions, by
    # construction, for any hip positions whatsoever) vs. the synthesised
    # pelvis actually used for solving.
    lhip, rhip = li["left_hip"], li["right_hip"]
    raw_mid = (targets[lhip] + targets[rhip]) / 2
    angle_before = _ang(targets[lhip] - raw_mid, targets[rhip] - raw_mid)
    angle_after = _ang(targets[lhip] - targets[li["pelvis"]], targets[rhip] - targets[li["pelvis"]])
    rest_angle = _ang(rig.rest_world_p[lhip] - rig.rest_world_p[li["pelvis"]],
                       rig.rest_world_p[rhip] - rig.rest_world_p[li["pelvis"]])
    print(f"\nangle(pelvis->left_hip, pelvis->right_hip): before={angle_before:.2f} deg "
          f"(raw hip midpoint), after={angle_after:.2f} deg (synthesised), "
          f"rig rest={rest_angle:.2f} deg")
    assert angle_before > 179.0, "sanity: the raw hip-midpoint target is ~180 deg by construction"
    assert angle_after < 150.0, "pelvis target synthesis did not meaningfully move off the hip midpoint"

    FITTED_FLOOR = 0.90
    MISMATCH_LIMIT_DEG = 2 * np.degrees(np.arccos(FITTED_FLOOR))   # ~51.68 deg; see module docstring math

    def _structural_mismatch_deg(n, c, siblings):
        """Diagnostic only, not an exclusion (ruling 9 amended): the worst
        rest-vs-target pairwise-angle mismatch between bone c and any OTHER
        usable sibling under n. By the spherical triangle inequality,
        error(c)+error(sibling) >= this mismatch for ANY rotation, so a
        value over MISMATCH_LIMIT_DEG would mean no rotation could put both
        inside the FITTED floor -- kept only to CONFIRM that no longer
        happens, not to remove a bone from the gate."""
        rb_c = rig.rest_world_p[c] - rig.rest_world_p[n]
        tb_c = targets[c] - targets[n]
        worst = 0.0
        for cp in siblings:
            if cp == c:
                continue
            rb_cp = rig.rest_world_p[cp] - rig.rest_world_p[n]
            tb_cp = targets[cp] - targets[n]
            worst = max(worst, abs(_ang(rb_c, rb_cp) - _ang(tb_c, tb_cp)))
        return worst

    rows = []
    bad = {}
    structurally_flagged = []       # ruling 9 (amended): expected EMPTY
    no_parent_target = []           # ruling 11
    no_own_target = []              # additional: bone itself has no target
    degenerate = []                 # both targeted, but a vector is ~0 length

    for n in solved:
        siblings = []
        if n in targets:
            for c in rig.children[n]:
                if c not in targets:
                    continue
                rb = rig.rest_world_p[c] - rig.rest_world_p[n]
                tb = targets[c] - targets[n]
                if np.linalg.norm(rb) > 1e-6 and np.linalg.norm(tb) > 1e-6:
                    siblings.append(c)

        for c in rig.children[n]:
            label = f"{rig.name[n]}->{rig.name[c]}"
            if n not in targets:
                no_parent_target.append(label)                 # ruling 11
                continue
            if c not in targets:
                no_own_target.append(label)
                continue
            if c not in siblings:
                degenerate.append(label)
                continue

            tb = targets[c] - targets[n]
            gb = got[c] - got[n]
            cos = float(tb @ gb / (np.linalg.norm(tb) * np.linalg.norm(gb)))
            # Ankles are HINGE-ANCHORED (Scott 2026-08-21: ankle axis is
            # +/-Z; solve is Rz*Rx only, aiming the heel->toe foot axis), so
            # their children are DERIVED, not fit -- the heel ray absorbs
            # whatever the forbidden Y rotation would have taken. Measured on
            # this fixture row: left 0.8819 / right 0.9227; the floor is a
            # regression bound on that anatomy trade, not an aspiration.
            # Same rationale for the Y-constrained neck (Scott: neck nods
            # about Y only, sharing the nod with the head): neck->head is
            # derived, measured 0.9101 on this fixture row.
            # Spine targets are SYNTHETIC (linear interpolation root->neck,
            # mhr70_retargeter.py) -- there is no real keypoint to gate
            # against, so these edges are printed but never scored. Absence
            # must never be silent; a synthetic pass must never count either.
            if (rig.name[n], rig.name[c]) in (("pelvis", "spine_1"),
                                              ("spine_1", "spine_2")):
                kind, floor = "SYNTH", None
            elif rig.name[n] in ("left_ankle", "right_ankle", "neck"):
                kind, floor = "ANCHORED", 0.85
            else:
                kind = "EXACT" if len(siblings) == 1 else "FITTED"
                floor = 0.99 if kind == "EXACT" else FITTED_FLOOR

            if kind == "FITTED":
                mismatch = _structural_mismatch_deg(n, c, siblings)
                if mismatch > MISMATCH_LIMIT_DEG:
                    structurally_flagged.append((label, round(cos, 4), round(mismatch, 1)))

            rows.append((label, kind, round(cos, 4), floor))
            if floor is not None and cos < floor:
                bad[f"{label} ({kind})"] = round(cos, 4)

    print(f"\n{'bone':<28}{'kind':<8}{'cosine':>8}{'floor':>8}")
    for label, kind, cos, floor in rows:
        print(f"{label:<28}{kind:<8}{cos:>8}{'-' if floor is None else floor:>8}")

    print("\nUNGATED (structurally unsatisfiable) -- expected EMPTY, ruling 9 amended:")
    for label, cos, mismatch in structurally_flagged:
        print(f"  {label:<28}cosine={cos}  rest/target pairwise-angle mismatch={mismatch} deg")

    print("\nUNGATED (no target for parent):")
    for label in no_parent_target:
        print(f"  {label}")

    print("\nUNGATED (bone itself has no target):")
    for label in no_own_target:
        print(f"  {label}")

    if degenerate:
        print("\nUNGATED (degenerate rest or target vector):")
        for label in degenerate:
            print(f"  {label}")

    # Positive control (CLAUDE.md rule 1): "0 bad" is indistinguishable from
    # "the loop never matched anything" -- e.g. if targets were keyed wrong
    # (name instead of index), `n not in targets` would be vacuously true for
    # every n and this would trivially "pass" having tested nothing. Prove
    # the gate actually ran over a real, non-trivial set of bones.
    assert len(rows) >= 20, f"only {len(rows)} bone-directions were testable -- the gate did not run"
    assert not structurally_flagged, (
        "ruling 9 (amended) expected no structurally unsatisfiable bones, "
        f"found: {structurally_flagged} -- report, do not exclude or tune"
    )
    assert not bad, f"bones off their floor: {bad}"


def test_synthesized_pelvis_target_reproduces_rig_rest_topology():
    """Ruling 9 (amended) evidence, pinned against the RIG rather than a
    measurement: feed the rig's own rest world positions in AS the targets
    (the self-consistent case -- no pose, no camera, nothing to reconstruct
    except the rig's own fixed geometry) and confirm the synthesised pelvis
    lands back on the rig's actual rest pelvis position, within a small
    tolerance. (Not exactly zero: the target-side orthonormal frame is
    built from spine_1's direction, which has a small forward lean of its
    own in the rig's rest pose, so re-deriving right/up/forward from it
    doesn't perfectly reproduce the world axes bit-for-bit.)"""
    from retargeting.retargeters.posegoblin_rig import load_rig, synthesize_pelvis_target, _leg_len
    rig = load_rig()
    li = rig.index_of_name
    synthesized = synthesize_pelvis_target(rig, dict(rig.rest_world_p))
    rest_pelvis = rig.rest_world_p[li["pelvis"]]
    rig_leg_len = np.mean([
        _leg_len(rig.rest_world_p, li["left_hip"], li["left_knee"], li["left_ankle"]),
        _leg_len(rig.rest_world_p, li["right_hip"], li["right_knee"], li["right_ankle"]),
    ])
    err = float(np.linalg.norm(synthesized - rest_pelvis))
    assert err / rig_leg_len < 0.02, (
        f"synthesized pelvis {synthesized} vs actual rest pelvis {rest_pelvis} "
        f"-- error {err:.4f} rig units ({err / rig_leg_len:.2%} of leg length), "
        "expected near-zero at self-consistency (targets == rig's own rest)"
    )


def test_synthesize_pelvis_target_raises_on_coincident_hips():
    """Review finding: no silent NaN. Coincident hip targets make the hip
    separation vector's norm zero; a naive normalize divides by that zero
    (only a numpy RuntimeWarning, no exception) and produces NaN, which
    would then flow into pelvisPosition/pose and reach json.dumps as a
    literal (invalid) `NaN`. The frame construction must raise instead --
    and never silently fall back to world axes, which would produce a
    plausible-looking but wrong pose for an inverted or supine subject."""
    from retargeting.retargeters.posegoblin_rig import load_rig, synthesize_pelvis_target
    rig = load_rig()
    li = rig.index_of_name
    targets = dict(rig.rest_world_p)          # a full, otherwise-valid target set
    targets[li["right_hip"]] = targets[li["left_hip"]].copy()   # coincident hips
    with pytest.raises(ValueError, match="coincident"):
        synthesize_pelvis_target(rig, targets)


def test_synthesize_pelvis_target_raises_on_collinear_hip_and_up():
    """The other degenerate scenario the review named: hip line and up
    direction collinear (spine_1 lying on the same line as the hip
    separation, through the hip midpoint) -- the cross product defining
    `fwd` is then near-zero. Must raise, not silently produce NaN."""
    from retargeting.retargeters.posegoblin_rig import load_rig, synthesize_pelvis_target
    rig = load_rig()
    li = rig.index_of_name
    targets = dict(rig.rest_world_p)
    lhip, rhip = li["left_hip"], li["right_hip"]
    hip_mid = (targets[lhip] + targets[rhip]) / 2
    right_dir = targets[lhip] - targets[rhip]
    targets[li["spine_1"]] = hip_mid + right_dir   # spine_1 forced onto the hip line
    with pytest.raises(ValueError, match="collinear"):
        synthesize_pelvis_target(rig, targets)


def test_synthesize_pelvis_target_raises_when_up_reference_coincides_with_hip_midpoint():
    """Third guarded degeneracy (not explicitly named by the review, but
    the same underlying divide-by-zero-norm risk): the up reference
    (spine_1) landing exactly on the hip midpoint, so there is no direction
    to build `up` from at all."""
    from retargeting.retargeters.posegoblin_rig import load_rig, synthesize_pelvis_target
    rig = load_rig()
    li = rig.index_of_name
    targets = dict(rig.rest_world_p)
    lhip, rhip = li["left_hip"], li["right_hip"]
    targets[li["spine_1"]] = (targets[lhip] + targets[rhip]) / 2   # spine_1 AT the hip midpoint
    with pytest.raises(ValueError, match="hip midpoint"):
        synthesize_pelvis_target(rig, targets)


def test_assert_all_finite_catches_an_injected_nan():
    """Evidence for the belt-and-braces finiteness guard (item 2 of the
    review fix): construct an otherwise-valid state with a NaN hand-injected
    into one bone's quaternion and confirm _assert_all_finite raises, naming
    that bone -- so a future edit that introduces a non-finite value by some
    OTHER route than the frame-degeneracy guard still fails loudly here,
    before json.dumps ever gets a chance to emit invalid JSON."""
    import json
    from retargeting.retargeters.posegoblin_rig import rig_state_from_mhr70, _assert_all_finite
    row = json.load(open(_FIXTURE))
    kp = np.asarray(row["mhr70_xyz"], np.float32).reshape(70, 3)
    st = rig_state_from_mhr70(kp)          # a real, valid state
    st["pose"]["left_elbow"]["_y"] = float("nan")   # inject
    with pytest.raises(ValueError, match="left_elbow"):
        _assert_all_finite(st)
