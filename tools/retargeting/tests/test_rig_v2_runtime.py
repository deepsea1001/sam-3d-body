"""v16 task 3: the runtime loads posegoblin_rig_v2.json by default and walks
it with scale-aware FK. v2 is a strict superset of v1 (see
tools_build_rig_v2.py / test_rig_v2_asset.py) -- the same 74 bones at the
same indices 0..73, plus ten constant Group nodes appended at 74..83
(`transform4`..`transform13`) that reconnect the ten finger/thumb chains to
their wrists. This file covers the RUNTIME consequences of that asset shape:

  - R6: the asset's index layout is deliberately non-topological (a group at
    e.g. index 74 parents finger bone 14, a LOWER index), so FK must walk an
    explicit `Rig.topo_order`, never raw index/asset order.
  - R2: FK must cover the full 84-node topology, including nodes nobody
    solved, defaulting any missing local to that node's own rest local --
    otherwise a wrist's group child has no world orientation to look up the
    moment anything downstream of it is ever computed.
  - R3: the 30 finger phalanges carry solve:True in the v2 asset (their
    chains are reachable now), but MHR-70 supplies no finger ROTATION data --
    only raw keypoints -- so the generic aim-based solve must never touch
    them; they stay at their rest locals until a later task adds the real
    finger path.
  - The emitted mannequinState pose dict must skip the ten group nodes (the
    viewer already owns them as a fixed container transform) and still
    collapse the two "joint7" bones to one entry, same as v1: 73 keys.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from retargeting.retargeters.posegoblin_rig import (
    load_rig, fk_world_positions, fk_world_orientations, solve_rig_locals,
    rig_targets_from_mhr70, rig_state_from_mhr70)

_BP = Path(__file__).resolve().parent.parent / "bind_poses"
_V1_PATH = _BP / "posegoblin_rig_v1.json"
_NPZ_ROWS = Path(__file__).resolve().parent / "fixtures" / "mhr_npz_rows.json"
_DEV_ROW_KEY = "a6566802a6c9ddd63340ccb4520e0001"

# The ten v1 orphans (`parent: None`) that v2 reconnects through a group --
# the only 10 of the original 74 bones whose `parent` is expected to differ
# between v1 and v2 (test 1 below).
_FINGER_THUMB_ROOTS = {
    "left_thumb_1", "left_index_finger_1", "left_middle_finger_1",
    "left_ring_finger_1", "left_pinky_finger_1",
    "right_thumb_1", "right_index_finger_1", "right_middle_finger_1",
    "right_ring_finger_1", "right_pinky_finger_1",
}


def _dev_row_kp70():
    rows = json.loads(_NPZ_ROWS.read_text())["rows"]
    return np.asarray(rows[_DEV_ROW_KEY]["kp70"], np.float32).reshape(70, 3)


def test_load_rig_v2_default_has_84_nodes_and_matches_v1_geometry():
    """Brief test 1: load_rig() returns 84 nodes, rig.version reflects v2,
    and every original test-relevant field for indices 0..73 equals v1's."""
    rig = load_rig()
    assert len(rig.order) == 84
    assert rig.version == "posegoblin_rig_v2"

    v1 = load_rig(_V1_PATH)
    assert len(v1.order) == 74

    for i in range(74):
        assert rig.name[i] == v1.name[i]
        assert np.array_equal(rig.rest_local_q[i], v1.rest_local_q[i])
        assert np.array_equal(rig.rest_local_p[i], v1.rest_local_p[i])
        assert np.array_equal(rig.rest_world_p[i], v1.rest_world_p[i])
        assert float(rig.scale[i]) == 1.0    # only the ten NEW groups scale down
        assert not rig.is_group[i]           # groups are only ever appended at 74..83
        if rig.name[i] in _FINGER_THUMB_ROOTS:
            assert v1.parent[i] is None                    # v1: disconnected island root
            assert rig.parent[i] is not None
            assert rig.is_group[rig.parent[i]]              # v2: reconnected to a group
        else:
            assert rig.parent[i] == v1.parent[i]             # everyone else: unchanged

    # solve: the 34 originally-solved bones are untouched; the 30 finger
    # phalanges flip false->true (their chains are reachable now); the ten
    # tips stay false. (Whether the SOLVER actually uses that flag for
    # phalanges is ruling 3's concern, tested separately below.)
    v1_solved = {i for i in range(74) if v1.solve[i]}
    v2_solved_among_original = {i for i in range(74) if rig.solve[i]}
    assert len(v1_solved) == 34
    assert v1_solved <= v2_solved_among_original
    assert len(v2_solved_among_original) == 64


def test_fk_world_positions_of_rest_locals_runs_and_finger_tips_in_envelope():
    """Brief test 2: fk_world_positions(rig, rig.rest_local_q) runs (no
    KeyError from the finger islands, now reconnected through unsolved
    groups) and finger-tip world positions are finite and within 3x of the
    pelvis->neck span from the pelvis -- catches the classic
    missed-0.1-scale error, which lands tips ~10x out."""
    rig = load_rig()
    world = fk_world_positions(rig, rig.rest_local_q)   # must not KeyError (R2/R6)

    li = rig.index_of_name
    pelvis = world[li["pelvis"]]
    span = float(np.linalg.norm(world[li["neck"]] - pelvis))
    assert span > 0

    tip_names = [n for n in rig.name.values() if n.endswith("_tip")
                 and ("_thumb_" in n or "_finger_" in n)]
    assert len(tip_names) == 10, tip_names   # positive control: pattern actually matches
    for n in tip_names:
        p = world[li[n]]
        assert np.all(np.isfinite(p)), f"{n} world position is not finite: {p}"
        dist = float(np.linalg.norm(p - pelvis))
        assert dist <= 3 * span, (
            f"{n} is {dist:.2f} rig units from pelvis, more than 3x the "
            f"pelvis->neck span ({span:.2f}) -- looks like the classic "
            "missed-0.1-group-scale bug (finger tips land ~10x too far out)")


def test_scale_aware_fk_matches_verified_finger_distances():
    """Ruling 2's own numbers, pinned tighter than the envelope test above:
    dropping the group's 0.1 scale multiplies these by exactly 10x (checked
    independently against the raw asset before writing this test)."""
    rig = load_rig()
    world = fk_world_positions(rig, rig.rest_local_q)
    li = rig.index_of_name
    wrist_to_finger1 = float(np.linalg.norm(
        world[li["left_index_finger_1"]] - world[li["left_wrist"]]))
    finger1_to_finger2 = float(np.linalg.norm(
        world[li["left_index_finger_2"]] - world[li["left_index_finger_1"]]))
    assert wrist_to_finger1 == pytest.approx(0.994, abs=0.01)
    assert finger1_to_finger2 == pytest.approx(0.37, abs=0.01)


def test_topo_order_visits_every_parent_before_its_child():
    """Brief test 3: iterating FK in topo_order visits every parent before
    its child (explicit assertion over all 84) -- v2's ten groups sit at
    74..83 while their finger children sit at 14..57, so this is NOT true
    of raw index/asset order (rig.order) and must not be assumed."""
    rig = load_rig()
    assert len(rig.topo_order) == 84
    assert set(rig.topo_order) == set(range(84))     # every node visited exactly once

    position = {node: pos for pos, node in enumerate(rig.topo_order)}
    for i in range(84):
        p = rig.parent[i]
        if p is not None:
            assert position[p] < position[i], (
                f"{rig.name[i]!r} (index {i}) precedes its own parent "
                f"{rig.name[p]!r} (index {p}) in topo_order")

    # positive control: rig.order (asset order) does NOT have this property
    # for at least one of the ten reconnected finger roots -- if it did,
    # topo_order's existence would be untested by this file.
    order_position = {node: pos for pos, node in enumerate(rig.order)}
    violations = [i for i in range(84)
                  if rig.parent[i] is not None
                  and order_position[rig.parent[i]] > order_position[i]]
    assert violations, "expected rig.order to violate parent-before-child somewhere"


def test_rig_state_from_mhr70_pose_has_73_keys_and_no_group_entries():
    """Brief test 4: rig_state_from_mhr70(kp70) on the fixture dev row
    yields a pose dict with exactly 73 keys and NO transform* keys -- the
    viewer already owns the ten group container transforms and must not be
    sent them."""
    st = rig_state_from_mhr70(_dev_row_kp70())
    assert len(st["pose"]) == 73
    assert not any(name.startswith("transform") for name in st["pose"])


def test_finger_and_thumb_bones_stay_at_rest_after_a_real_solve():
    """Ruling 3's own required test: solving a real pose leaves all 40
    finger/thumb bones (30 phalanges the v2 asset flags solve:True, plus
    the 10 tips) at their rest locals -- MHR-70 gives no finger rotation
    data, so the generic aim-based solve must never touch them, even though
    rig_targets_from_mhr70 does supply raw keypoint targets for them."""
    rig = load_rig()
    kp = _dev_row_kp70()
    targets = rig_targets_from_mhr70(kp)

    finger_thumb = [i for i in range(74)
                    if ("_thumb_" in rig.name[i]) or ("_finger_" in rig.name[i])]
    assert len(finger_thumb) == 40                            # positive control
    assert any(i in targets for i in finger_thumb), (
        "positive control: rig_targets_from_mhr70 must actually supply "
        "finger keypoint targets, or this test proves nothing")

    solved = solve_rig_locals(rig, targets)
    for i in finger_thumb:
        assert i not in solved, (
            f"{rig.name[i]!r} (index {i}) was solved by the generic path -- "
            "finger/thumb bones must stay unposed (ruling 3) until a later "
            "task adds the real finger path")

    st = rig_state_from_mhr70(kp)
    for i in finger_thumb:
        name = rig.name[i]
        want = rig.rest_local_q[i]
        got = st["pose"][name]
        assert got["_w"] == pytest.approx(float(want[0]), abs=1e-9)
        assert got["_x"] == pytest.approx(float(want[1]), abs=1e-9)
        assert got["_y"] == pytest.approx(float(want[2]), abs=1e-9)
        assert got["_z"] == pytest.approx(float(want[3]), abs=1e-9)
