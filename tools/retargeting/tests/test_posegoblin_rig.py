"""The rig asset must be internally consistent: FK of its rest locals must
reproduce its captured rest world positions. A re-exported (re-rigged) asset
that breaks this fails HERE, loudly — never as silently garbled poses.

Contingency this test also covers: if the model container node carries its own
transform, FK-from-locals will NOT match captured world positions. In that
case the fix is to add the container transform to the asset, not to loosen
this test."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from retargeting.retargeters.posegoblin_rig import load_rig, fk_world_positions


def test_asset_loads_with_full_hierarchy():
    rig = load_rig()
    assert len(rig.order) == 74
    assert rig.version == "posegoblin_rig_v1"
    roots = [b for b in rig.order if rig.parent[b] is None]
    assert roots == ["pelvis"]
    for b in rig.order:               # parent-first ordering is load-bearing for FK
        p = rig.parent[b]
        assert p is None or rig.order.index(p) < rig.order.index(b)


def test_fk_of_rest_locals_reproduces_captured_world_positions():
    rig = load_rig()
    world = fk_world_positions(rig, rig.rest_local_q)
    err = {b: float(np.linalg.norm(world[b] - rig.rest_world_p[b])) for b in rig.order}
    bad = {b: round(e, 4) for b, e in err.items() if e > 1e-3}
    assert not bad, f"FK does not reproduce the captured rest: {bad}"
