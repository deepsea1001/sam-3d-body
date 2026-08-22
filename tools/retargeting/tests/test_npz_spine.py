"""The mannequin's spine driven by the MHR model's own joint rotations.

MHR-70 carries NO spine keypoints: spine_1/spine_2 targets are points
interpolated along the straight pelvis->neck chord, so until v16 the column
could only be posed by SPLITTING the pelvis->chest rotation between two
anchors (v15's swing-split; Scott: "stiff as a board"). Every corpus row
secretly carries the full parametric solve in its `mhr_params_npz` blob --
`joint_global_rots`, per-joint global rotations for the 127-joint MHR
kinematic skeleton, four real spine joints among them. v16 consumes them.

The frame contract these tests rest on: `Delta(j) = R_pose(j) @ R_rest(j)^T`
is a RIG-frame world rotation delta with NO coordinate conversion, because
the SAM head's camera flip diag(1,-1,-1) is applied to coordinates but not
to `joint_global_rots`, and the solver's own cv->rig POSITION map is that
same matrix -- the two cancel. Positions still need the map (see
`_CAM_TO_RIG_POS` below, used only on `pred_joint_coords`); rotations never
do. The two facts are not in conflict; they are different quantities.

Quaternions here are the solver's [w,x,y,z] throughout (three.js and the
PoseGoblin captures use [x,y,z,w]; the only conversion happens at
serialization, in `QuaternionMath.to_threejs_dict`).
"""
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from retargeting import mhr_rots_from_npz
from retargeting.core.math_utils import QuaternionMath as QM
from retargeting.retargeters.posegoblin_rig import (
    fk_world_orientations, fk_world_positions, load_mhr_rest, load_rig,
    rig_state_from_mhr70, rig_targets_from_mhr70, solve_rig_locals)

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mhr_npz_rows.json"
ROWS = json.loads(_FIXTURE.read_text())["rows"]
DEV_ROW = "a6566802a6c9ddd63340ccb4520e0001"

RIG = load_rig()
I = RIG.index_of_name

# Rows of the 127-joint MHR kinematic skeleton (names from
# bind_poses/mhr_skeleton_rest.json), NOT MHR-70 keypoint indices.
MHR_SPINE1 = 35    # c_spine1 -> mannequin spine_1
MHR_SPINE3 = 37    # c_spine3 -> mannequin spine_2
MHR_NECK = 110     # c_neck

# Camera -> rig POSITION map: cv_to_yup's -Y composed with the solver's own
# _CV_YUP_TO_RIG -Z. `pred_joint_coords` is camera-frame, so the reference
# directions below need it. `joint_global_rots` does NOT -- see the module
# docstring.
_CAM_TO_RIG_POS = np.array([1.0, -1.0, -1.0])

# Acceptance floors (ruling R11) = the measured rest-vs-rest CEILING minus a
# 0.03 tolerance fixed in advance. A delta-driven joint can never beat its
# ceiling: the mannequin's and the MHR skeleton's rest geometries genuinely
# differ, and a world delta cannot close that gap.
SPINE1_SPINE2_FLOOR = 0.9612   # rest-vs-rest ceiling 0.9912 (7.6 deg)
SPINE2_NECK_FLOOR = 0.8216     # rest-vs-rest ceiling 0.8516 (31.6 deg)
# pelvis->spine_1 is deliberately NOT scored: its direction is fixed by
# W_R(pelvis) and the rig's rest offset, so spine_1's own rotation cannot
# affect it. It is bit-identical in v15 and v16 and measures the pelvis
# anchor, not the spine.


def _wxyz_to_mat(q):
    """[w,x,y,z] quaternion -> 3x3 rotation matrix.

    Normalises first, deliberately: the rig asset's `rest_local_q` are unit
    only to 6.2e-07 (captured from the live scene, written to JSON), and
    this formula scales a matrix by |q|^2 -- which showed up as a 2.5e-06
    residual in the anchor-row test below until the normalise was added.
    The solver itself is unaffected; it composes quaternions and never
    takes this route."""
    q = np.asarray(q, float)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _quat_deg(a, b):
    """Angle between two [w,x,y,z] rotations, in degrees. Sign-agnostic: q
    and -q are the same rotation, and `from_matrix` may return either."""
    d = abs(float(np.dot(np.asarray(a, float), np.asarray(b, float))))
    return float(np.degrees(2.0 * np.arccos(np.clip(d, 0.0, 1.0))))


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _rots(row_id):
    return np.asarray(ROWS[row_id]["joint_global_rots"], float)


def _kp(row_id):
    return np.asarray(ROWS[row_id]["kp70"], np.float32).reshape(70, 3)


def _solve(row_id, mhr_rots):
    return solve_rig_locals(RIG, rig_targets_from_mhr70(_kp(row_id)), mhr_rots=mhr_rots)


def _world(local_q):
    return fk_world_orientations(RIG, {**RIG.rest_local_q, **local_q})


def test_decode_helper_roundtrip_and_none():
    """A real blob decodes to (127,3,3) float64; empty bytes and a blob
    without the key give None.

    The roundtrip half is the positive control for the two None assertions:
    both `None` results are built by the same `savez_compressed` path that
    demonstrably produces a non-None decode here, so "None" means "the key
    was absent", never "the decoder never ran"."""
    rots = _rots(DEV_ROW).astype(np.float32)   # shards store float32 (runner._serialize_mhr_params)
    buf = io.BytesIO()
    np.savez_compressed(buf, joint_global_rots=rots, shape=np.zeros(45, np.float32))
    got = mhr_rots_from_npz(buf.getvalue())
    assert got is not None, "positive control: a well-formed blob must decode"
    assert got.shape == (127, 3, 3) and got.dtype == np.float64
    assert np.allclose(got, rots, atol=1e-6)

    assert mhr_rots_from_npz(b"") is None

    without = io.BytesIO()
    np.savez_compressed(without, shape=np.zeros(45, np.float32))
    assert mhr_rots_from_npz(without.getvalue()) is None

    # A wrong-shaped array must fail LOUDLY at the boundary, not surface as
    # an IndexError inside quaternion math several frames later.
    flat = io.BytesIO()
    np.savez_compressed(flat, joint_global_rots=rots.reshape(127, 9))
    try:
        mhr_rots_from_npz(flat.getvalue())
    except ValueError as exc:
        assert "(127, 9)" in str(exc)
    else:
        raise AssertionError("a (127,9) joint_global_rots decoded silently")


def test_load_mhr_rest_is_a_parsed_once_singleton():
    a = load_mhr_rest()
    assert load_mhr_rest() is a, "not a singleton: the fixture is re-parsed per call"
    assert a["q_wxyz"].shape == (127, 4)
    assert len(a["names"]) == 127 and len(a["parents"]) == 127
    assert a["names"][MHR_SPINE1] == "c_spine1"
    assert a["names"][MHR_SPINE3] == "c_spine3"
    assert a["names"][MHR_NECK] == "c_neck"
    assert np.allclose(np.linalg.norm(a["q_wxyz"], axis=1), 1.0, atol=1e-6)


def test_runtime_spine_path_never_imports_torch():
    """The rest skeleton comes from a committed JSON fixture, never from
    checkpoints/mhr_model.pt -- so no runtime path may pull torch in.

    Not a "torch not in sys.modules" check (that reads the same whether the
    guard works or the probe never ran): the child process ARMS a meta_path
    finder that raises on any torch import, solves a real row through the
    v16 spine, and then imports torch itself to prove the guard is live."""
    probe = r'''
import json, sys
import numpy as np

class _NoTorch:
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("runtime imported torch: " + name)
        return None

assert "torch" not in sys.modules, "torch was already imported before the guard armed"
sys.meta_path.insert(0, _NoTorch())

from retargeting.retargeters.posegoblin_rig import (
    load_mhr_rest, rig_state_from_mhr70)

rows = json.loads(open(sys.argv[1]).read())["rows"]
row = rows[sys.argv[2]]
kp = np.asarray(row["kp70"], np.float32).reshape(70, 3)
rots = np.asarray(row["joint_global_rots"], float)
st = rig_state_from_mhr70(kp, mhr_rots=rots)
assert load_mhr_rest()["q_wxyz"].shape == (127, 4)

try:                       # positive control: the guard must actually bite
    import torch
except AssertionError:
    print("SOLVED_%d_GUARD_ARMED" % st["retargetVersion"])
else:
    raise SystemExit("guard never fired: it would not have caught a real torch import")
'''
    tools_dir = Path(__file__).resolve().parents[2]
    out = subprocess.run(
        [sys.executable, "-c", probe, str(_FIXTURE), DEV_ROW],
        capture_output=True, text=True, cwd=str(tools_dir),
        env={**os.environ, "PYTHONPATH": str(tools_dir)})
    assert out.returncode == 0, f"probe failed:\n{out.stdout}\n{out.stderr}"
    assert "SOLVED_16_GUARD_ARMED" in out.stdout, out.stdout


def test_identity_deltas_leave_the_spine_at_rest():
    """mhr_rots == the rest globals => every Delta is the identity => both
    spine bones sit at their REST world orientations.

    spine_2's LOCAL is its rest local exactly (its parent spine_1 is also at
    rest). spine_1's is not, and must not be: its parent, the pelvis, still
    carries the real pelvis anchor, so spine_1's local absorbs D(pelvis)^-1
    precisely to hold its world at rest."""
    R_rest = np.stack([_wxyz_to_mat(q) for q in load_mhr_rest()["q_wxyz"]])
    Wr = fk_world_orientations(RIG, RIG.rest_local_q)
    L = _solve(DEV_ROW, R_rest)
    W = _world(L)
    for name in ("spine_1", "spine_2"):
        assert _quat_deg(W[I[name]], Wr[I[name]]) < 1e-4, name
    assert _quat_deg(L[I["spine_2"]], RIG.rest_local_q[I["spine_2"]]) < 1e-4

    # Positive control: this row's spine is nowhere near rest without the
    # identity rots, so "at rest" above is a result, not a vacuous default.
    W15 = _world(_solve(DEV_ROW, None))
    for name in ("spine_1", "spine_2"):
        assert _quat_deg(W15[I[name]], Wr[I[name]]) > 5.0, name


def test_global_arch_transfers_to_spine():
    """Left-multiply every MHR row by a fixed G and G must reappear, exactly,
    as a world rotation on both spine bones.

    The strongest check available here: it separates a correct transfer from
    a plausible-looking wrong one (a conjugated delta, a reversed multiply
    order, or a spurious frame conversion all move the spine somewhere
    believable, but none of them reproduce G).

    G's AXIS is load-bearing, and is deliberately not a coordinate axis.
    `diag(1,-1,-1)` -- the camera flip that must never be applied to
    rotations here -- has det +1: it IS a 180 deg rotation about X, so it
    COMMUTES with every rotation about X. An arch about +X therefore passes
    unchanged against a Delta wrongly conjugated by that flip (measured:
    0.0000 deg deviation), leaving this test blind to precisely the error
    class it exists to catch. A normalised [1,2,3] tilt is symmetric under
    NONE of the 23 non-identity signed coordinate-axis maps -- worst case
    16.25 deg, and 48.15 deg for `diag(1,-1,-1)` itself -- so any of them
    lands far outside the 0.5 deg tolerance below. Do not "simplify" this
    back to a coordinate axis: each of the three is blind to 3 of the 23."""
    t = np.radians(25.0)
    ax = np.array([1.0, 2.0, 3.0])
    ax = ax / np.linalg.norm(ax)
    qG = np.array([np.cos(t / 2), *(np.sin(t / 2) * ax)])   # 25 deg about [1,2,3], [w,x,y,z]
    G = _wxyz_to_mat(qG)
    Wr = fk_world_orientations(RIG, RIG.rest_local_q)

    # (a) rest rotations arched by G: every Delta IS G, so each spine bone's
    #     world orientation must sit exactly G off its rest.
    R_rest = np.stack([_wxyz_to_mat(q) for q in load_mhr_rest()["q_wxyz"]])
    W = _world(_solve(DEV_ROW, G @ R_rest))
    for name in ("spine_1", "spine_2"):
        d = QM.multiply(W[I[name]], QM.conjugate(Wr[I[name]]))
        assert _quat_deg(d, qG) < 0.5, f"{name}: {_quat_deg(d, qG):.3f} deg off G"

    # (b) the same arch on top of the REAL row: each spine bone's world
    #     orientation must move by exactly G relative to the un-arched solve.
    real = _rots(DEV_ROW)
    W0 = _world(_solve(DEV_ROW, real))
    W1 = _world(_solve(DEV_ROW, G @ real))
    for name in ("spine_1", "spine_2"):
        d = QM.multiply(W1[I[name]], QM.conjugate(W0[I[name]]))
        assert _quat_deg(d, qG) < 0.5, f"{name}: {_quat_deg(d, qG):.3f} deg off G"


def test_spine_anchors_are_c_spine1_and_c_spine3():
    """Which MHR rows drive the two spine bones, pinned as a contract.

    The scored-direction test below CANNOT see this: swapping spine_2 from
    c_spine3 (37) to the positionally closer c_spine2 (36) leaves
    spine_1->spine_2 bit-identical and keeps spine_2->neck inside its floor
    on all six rows (measured 0.8265..0.9152 against a 0.8216 floor). The
    choice matters anyway -- both clavicles hang off c_spine3 in the MHR
    skeleton exactly as they hang off spine_2 on the mannequin, so c_spine2
    would leave the long c_spine2->c_spine3 bend to surface at the neck
    instead of the mid-back -- so it is pinned here rather than left to a
    metric that is blind to it.

    Expectation built as MATRICES, independent of the solver's quaternion
    order and sign conventions: W(bone) must be Delta(row) @ W_rest(bone)
    with Delta(row) = R_pose(row) @ R_rest(row)^T."""
    R_pose = _rots(DEV_ROW)
    R_rest = np.stack([_wxyz_to_mat(q) for q in load_mhr_rest()["q_wxyz"]])
    Wr = fk_world_orientations(RIG, RIG.rest_local_q)
    W = _world(_solve(DEV_ROW, R_pose))
    for bone, row in (("spine_1", MHR_SPINE1), ("spine_2", MHR_SPINE3)):
        got = _wxyz_to_mat(W[I[bone]])
        want = R_pose[row] @ R_rest[row].T @ _wxyz_to_mat(Wr[I[bone]])
        # 1e-06: the fixture's rotations come from float32 shard data
        # written to JSON at 7 significant digits, which alone puts the
        # residual at 6.2e-08. Neighbouring rows below sit at 9.3e-02.
        assert np.abs(got - want).max() < 1e-6, f"{bone} is not driven by row {row}"
        # Positive control: the check is row-SPECIFIC, so the neighbouring
        # spine joints must be visibly wrong for the same bone.
        for other in (row - 1, row + 1):
            alt = R_pose[other] @ R_rest[other].T @ _wxyz_to_mat(Wr[I[bone]])
            assert np.abs(got - alt).max() > 1e-3, \
                f"{bone}: rows {row} and {other} are indistinguishable here"


def test_real_row_spine_directions_meet_anchored_floor():
    """Real rots on all six fixture rows: the two spine edges the solve can
    actually influence must land on the MHR skeleton's own directions.

    Reference directions come from `pred_joint_coords`, which is CAMERA
    frame, so they take `_CAM_TO_RIG_POS`. Floors are the measured
    rest-vs-rest ceilings minus 0.03 -- see the constants above."""
    edges = (("spine_1", "spine_2", MHR_SPINE1, MHR_SPINE3, SPINE1_SPINE2_FLOOR),
             ("spine_2", "neck", MHR_SPINE3, MHR_NECK, SPINE2_NECK_FLOOR))
    scored, failures = 0, []
    for row_id in ROWS:
        P_mhr = np.asarray(ROWS[row_id]["pred_joint_coords"], float)
        P = fk_world_positions(RIG, {**RIG.rest_local_q, **_solve(row_id, _rots(row_id))})
        for parent, child, m_parent, m_child, floor in edges:
            got = _unit(P[I[child]] - P[I[parent]])
            want = _unit((P_mhr[m_child] - P_mhr[m_parent]) * _CAM_TO_RIG_POS)
            cos = float(np.dot(got, want))
            scored += 1
            if cos < floor:
                failures.append(f"{row_id[:8]} {parent}->{child}: cos {cos:.4f} < {floor}")
    assert scored == 2 * len(ROWS), f"positive control: scored {scored} edges, expected 12"
    assert not failures, "\n".join(failures)


# Every solved bone whose LOCAL legitimately differs between the two spine
# branches. The two spine bones are the intended change; the other four
# follow from the rig's parent chain, not from any extra reach:
# left_clavicle, right_clavicle and neck are all children of spine_2, so
# their locals absorb its new world frame, and head's local absorbs neck's
# (head's own WORLD anchor, from the nose + eye line, is untouched).
_V16_AFFECTED = ("spine_1", "spine_2", "neck", "head",
                 "left_clavicle", "right_clavicle")


def test_fallback_none_switches_only_the_spine_branch():
    """`mhr_rots=None` really takes the v15 branch, and the switch reaches
    exactly the spine and its descendants -- nothing else.

    v15's own behavior is pinned by the 44 pre-existing tests, every one of
    which calls the solver without `mhr_rots` -- so what is left to prove
    here is that the branch switches at all, and that it does not leak.

    Deliberately NOT the claim that only spine_1 and spine_2 move: a bone's
    local is expressed in its parent's world frame, so spine_2's four
    descendants (`_V16_AFFECTED` above) must change with it, and a test
    demanding otherwise would be demanding a bug. The exact set is asserted
    rather than sampled, so a future edit that reaches one bone further
    fails here instead of passing quietly."""
    targets = rig_targets_from_mhr70(_kp(DEV_ROW))
    v15 = solve_rig_locals(RIG, targets)                          # untouched call site
    v16 = solve_rig_locals(RIG, targets, mhr_rots=_rots(DEV_ROW))
    assert set(v15) == set(v16), "the branch changed WHICH bones are solved"

    for name in ("spine_1", "spine_2"):
        moved = _quat_deg(v15[I[name]], v16[I[name]])
        assert moved > 1.0, f"{name} moved only {moved:.4f} deg -- the branch did not switch"

    # Index-keyed, never name-keyed: two rig bones share the name "joint7".
    expected = {I[n] for n in _V16_AFFECTED}
    changed = {i for i in v15 if not np.allclose(v15[i], v16[i], atol=1e-9)}
    assert changed == expected, (
        f"unexpected: {sorted(RIG.name[i] for i in changed - expected)}, "
        f"missing: {sorted(RIG.name[i] for i in expected - changed)}")


def test_version_pinned_16():
    st15 = rig_state_from_mhr70(_kp(DEV_ROW))
    st16 = rig_state_from_mhr70(_kp(DEV_ROW), mhr_rots=_rots(DEV_ROW))
    assert st15["retargetVersion"] == 16 and st16["retargetVersion"] == 16
    # ...and mhr_rots actually reaches the solve through this entry point,
    # not just through solve_rig_locals.
    assert st16["pose"]["spine_1"] != st15["pose"]["spine_1"]
