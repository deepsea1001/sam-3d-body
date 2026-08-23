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
import pytest

from retargeting import mhr_rots_from_npz
from retargeting.core.math_utils import QuaternionMath as QM
from retargeting.retargeters import posegoblin_rig as PG
from retargeting.retargeters.posegoblin_rig import (
    _mhr_delta_q, fk_world_orientations, fk_world_positions, load_mhr_rest,
    load_rig, rig_state_from_mhr70, rig_targets_from_mhr70, solve_rig_locals)

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mhr_npz_rows.json"
ROWS = json.loads(_FIXTURE.read_text())["rows"]
DEV_ROW = "a6566802a6c9ddd63340ccb4520e0001"

RIG = load_rig()
I = RIG.index_of_name

# Rows of the 127-joint MHR kinematic skeleton (names from
# bind_poses/mhr_skeleton_rest.json), NOT MHR-70 keypoint indices.
MHR_ROOT = 1       # root -> mannequin pelvis
MHR_SPINE1 = 35    # c_spine1 -> mannequin spine_1 under SPINE_SOURCE_MHR
MHR_SPINE2 = 36    # c_spine2 -> mannequin spine_1 under the DEFAULT mapping
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
SPINE1_SPINE2_FLOOR = 0.9612   # rest-vs-rest ceiling 0.9912 (7.6 deg), c_spine1->c_spine3
SPINE1_SPINE2_36_FLOOR = 0.9696  # ceiling 0.9996 (1.6 deg), c_spine2->c_spine3 -- the
                                 # SEGMENT that actually corresponds to the mannequin's
                                 # spine_1->spine_2, and the reference for every mapping
                                 # that drives spine_1 from row 36
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
    assert "SOLVED_17_GUARD_ARMED" in out.stdout, out.stdout


def test_identity_deltas_leave_the_spine_straight_on_the_pelvis():
    """Under the DEFAULT (relative) mapping: mhr_rots == the rest globals =>
    every Delta is the identity => both spine bones sit at their REST LOCALS
    exactly -- a straight column carried by whatever the pelvis anchor did.

    That is the relative mapping's whole meaning, and it is a different
    claim from the absolute mapping's (below): a straight model spine leaves
    the mannequin's spine straight ON THE PELVIS WE PLACED, not straight in
    world."""
    R_rest = np.stack([_wxyz_to_mat(q) for q in load_mhr_rest()["q_wxyz"]])
    L = _solve(DEV_ROW, R_rest)
    for name in ("spine_1", "spine_2"):
        assert _quat_deg(L[I[name]], RIG.rest_local_q[I[name]]) < 1e-4, name

    # Positive control: this row's spine is nowhere near its rest locals
    # without the identity rots, so "at rest" above is a result, not a
    # vacuous default.
    L15 = _solve(DEV_ROW, None)
    for name in ("spine_1", "spine_2"):
        assert _quat_deg(L15[I[name]], RIG.rest_local_q[I[name]]) > 5.0, name


def test_identity_deltas_leave_the_absolute_spine_at_rest_in_world(monkeypatch):
    """The same probe under SPINE_SOURCE_MHR, where it means something else:
    the two spine bones sit at their REST WORLD orientations.

    Kept, not deleted, when the default moved -- it is the statement that
    distinguishes the two mappings, and the reason the absolute one loses
    bend whenever our pelvis and the model's root disagree.

    spine_2's LOCAL is its rest local exactly (its parent spine_1 is also at
    rest). spine_1's is not, and must not be: its parent, the pelvis, still
    carries the real pelvis anchor, so spine_1's local absorbs D(pelvis)^-1
    precisely to hold its world at rest."""
    monkeypatch.setattr(PG, "SPINE_SOURCE", PG.SPINE_SOURCE_MHR)
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


def test_a_global_arch_cancels_but_a_spine_arch_does_not(monkeypatch):
    """Under the DEFAULT (relative) mapping, the two halves of the frame
    contract, and they point opposite ways on purpose.

    (a) Left-multiplying EVERY MHR row by a fixed G rotates the model's root
        along with its chest, so the chest-vs-root rotation is unchanged and
        the mannequin's spine must not move at all. A relative transfer that
        let a global arch through would be an absolute transfer wearing a
        disguise.
    (b) Arching ONLY the two spine rows the mapping reads must move the
        spine -- otherwise (a) would be satisfied by a solve that ignores
        `mhr_rots` entirely, which is exactly the vacuous pass CLAUDE.md
        rule 1 warns about.
    """
    monkeypatch.setattr(PG, "SPINE_SOURCE", PG.SPINE_SOURCE_REL_PERJOINT)
    t = np.radians(25.0)
    ax = np.array([1.0, 2.0, 3.0])
    ax = ax / np.linalg.norm(ax)
    qG = np.array([np.cos(t / 2), *(np.sin(t / 2) * ax)])
    G = _wxyz_to_mat(qG)
    real = _rots(DEV_ROW)

    L0 = _solve(DEV_ROW, real)
    L_all = _solve(DEV_ROW, G @ real)
    for name in ("spine_1", "spine_2"):
        moved = _quat_deg(L_all[I[name]], L0[I[name]])
        assert moved < 1e-4, f"{name} moved {moved:.4f} deg under a GLOBAL arch"

    # ONE row at a time. Arching BOTH spine rows together would leave
    # spine_2's LOCAL untouched -- G cancels between a bone and its parent --
    # so a both-rows probe is blind to exactly half of what it claims to
    # cover (measured: spine_2 moves 0.0000 deg).
    for name, row in (("spine_1", PG.SPINE_PERJOINT_SPINE1_ROW),
                      ("spine_2", MHR_SPINE3)):
        one = np.array(real, copy=True)
        one[row] = G @ real[row]
        moved = _quat_deg(_solve(DEV_ROW, one)[I[name]], L0[I[name]])
        assert moved > 1.0, \
            f"{name} moved only {moved:.4f} deg when its own row {row} was arched"


def test_global_arch_transfers_to_spine(monkeypatch):
    """Left-multiply every MHR row by a fixed G and G must reappear, exactly,
    as a world rotation on both spine bones -- under SPINE_SOURCE_MHR, whose
    absolute world-delta contract this is. Kept when the default moved: it
    is what pins `_mhr_delta_q`'s no-coordinate-conversion frame claim, and
    the test above is the relative mapping's counterpart, not a replacement.

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
    monkeypatch.setattr(PG, "SPINE_SOURCE", PG.SPINE_SOURCE_MHR)
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


def test_rel_perjoint_spine_anchors_are_c_spine2_and_c_spine3_relative_to_the_pelvis(
        monkeypatch):
    """Which MHR rows drive the two spine bones under SPINE_SOURCE_REL_PERJOINT,
    and in which frame -- pinned as a contract.

    This WAS the default's contract until 2026-08-23, when the default moved
    to SPINE_SOURCE_REL_TOTAL. The spine_2 half of it still is the default's
    -- see the test below, which asserts exactly that and takes spine_1 apart
    differently. Kept, monkeypatched, rather than shrunk: the mapping is
    still reachable and still the one-constant revert.

    W(bone) must be D(pelvis) . Delta(root)^-1 . Delta(row) . W_rest(bone):
    the model's rotation RELATIVE TO ITS OWN ROOT, carried by the pelvis we
    actually solved. Built as matrices, independent of the solver's
    quaternion order and sign conventions.

    Two positive controls, because two things could be wrong independently:
    the ROW (checked against its neighbours) and the FRAME (checked against
    the absolute form the same row would give, which is what v16 first
    shipped)."""
    monkeypatch.setattr(PG, "SPINE_SOURCE", PG.SPINE_SOURCE_REL_PERJOINT)
    R_pose = _rots(DEV_ROW)
    R_rest = np.stack([_wxyz_to_mat(q) for q in load_mhr_rest()["q_wxyz"]])
    Wr = fk_world_orientations(RIG, RIG.rest_local_q)
    L = _solve(DEV_ROW, R_pose)
    W = _world(L)
    D_pelvis = _wxyz_to_mat(L[I["pelvis"]]) @ _wxyz_to_mat(Wr[I["pelvis"]]).T
    root = R_pose[1] @ R_rest[1].T

    for bone, row in (("spine_1", PG.SPINE_PERJOINT_SPINE1_ROW), ("spine_2", MHR_SPINE3)):
        got = _wxyz_to_mat(W[I[bone]])
        delta = R_pose[row] @ R_rest[row].T
        want = D_pelvis @ root.T @ delta @ _wxyz_to_mat(Wr[I[bone]])
        assert np.abs(got - want).max() < 1e-6, f"{bone} is not row {row}, pelvis-relative"
        for other in (row - 1, row + 1):
            alt = D_pelvis @ root.T @ (R_pose[other] @ R_rest[other].T) @ _wxyz_to_mat(Wr[I[bone]])
            assert np.abs(got - alt).max() > 1e-3, \
                f"{bone}: rows {row} and {other} are indistinguishable here"
        absolute = delta @ _wxyz_to_mat(Wr[I[bone]])
        assert np.abs(got - absolute).max() > 1e-3, \
            f"{bone}: the relative and absolute forms are indistinguishable on this row"


def test_default_chest_is_c_spine3_relative_and_spine_1_is_its_65_percent():
    """The DEFAULT's contract, in matrices, independent of the solver's
    quaternion order and sign conventions.

    spine_2: D(pelvis) . Delta(root)^-1 . Delta(37) . W_rest(spine_2) --
    identical to the test above, because REL_TOTAL and REL_PERJOINT share
    this chest exactly, and that identity is what carries the 28.0 deg
    total-error result across.

    spine_1: NOT an MHR row. Its world delta is the geodesic 65% of the way
    from the pelvis's delta to the chest's, which is what makes the two
    bones' local angles come out 65/35 on every axis.

    Positive controls: rows 36 and 37 must both be visibly WRONG for
    spine_1 here (row 36 is what the previous default used, and what a
    careless revert would leave behind), and the interpolation must be
    non-trivial on this row -- neither endpoint."""
    R_pose = _rots(DEV_ROW)
    R_rest = np.stack([_wxyz_to_mat(q) for q in load_mhr_rest()["q_wxyz"]])
    Wr = fk_world_orientations(RIG, RIG.rest_local_q)
    L = _solve(DEV_ROW, R_pose)
    W = _world(L)
    D_pelvis = _wxyz_to_mat(L[I["pelvis"]]) @ _wxyz_to_mat(Wr[I["pelvis"]]).T
    root = R_pose[1] @ R_rest[1].T

    got2 = _wxyz_to_mat(W[I["spine_2"]])
    want2 = (D_pelvis @ root.T @ (R_pose[MHR_SPINE3] @ R_rest[MHR_SPINE3].T)
             @ _wxyz_to_mat(Wr[I["spine_2"]]))
    assert np.abs(got2 - want2).max() < 1e-6, "the chest is not row 37, pelvis-relative"

    D_chest = got2 @ _wxyz_to_mat(Wr[I["spine_2"]]).T
    want1 = _wxyz_to_mat(PG._slerp(QM.from_matrix(D_pelvis), QM.from_matrix(D_chest),
                                   PG._SPINE1_SHARE)) @ _wxyz_to_mat(Wr[I["spine_1"]])
    got1 = _wxyz_to_mat(W[I["spine_1"]])
    assert np.abs(got1 - want1).max() < 1e-6, "spine_1 is not the 65% geodesic"
    for other in (MHR_SPINE2, MHR_SPINE3):
        alt = (D_pelvis @ root.T @ (R_pose[other] @ R_rest[other].T)
               @ _wxyz_to_mat(Wr[I["spine_1"]]))
        assert np.abs(got1 - alt).max() > 1e-3, \
            f"spine_1 is indistinguishable from MHR row {other}"
    assert np.abs(got1 - got2).max() > 1e-3                              # not t = 1
    assert np.abs(got1 - D_pelvis @ _wxyz_to_mat(Wr[I["spine_1"]])).max() > 1e-3   # not t = 0


def test_spine_anchors_are_c_spine1_and_c_spine3(monkeypatch):
    """Which MHR rows drive the two spine bones under SPINE_SOURCE_MHR,
    pinned as a contract. Kept when the default moved -- the mapping is
    still reachable and still measured.

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
    monkeypatch.setattr(PG, "SPINE_SOURCE", PG.SPINE_SOURCE_MHR)
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


# The spine-edge check, per mapping. Two things vary with the mapping and
# BOTH must, or the check measures something other than the spine:
#
#   the reference ROW for spine_1->spine_2 -- the segment must be the one the
#   mapping actually drives (c_spine1->c_spine3 for the mapping v16 first
#   shipped, c_spine2->c_spine3 for the two that drive spine_1 from row 36),
#   with the floor re-derived from that pairing's own rest-vs-rest ceiling;
#
#   the reference FRAME -- an absolute transfer is checked in WORLD, and a
#   relative one in the PELVIS's frame, by carrying the model's direction
#   through D(pelvis) . Delta(root)^-1 first. Checking a relative transfer
#   against a world reference measures how far our pelvis sits from the
#   model's root (up to 43.8 deg on these captures) and blames the spine for
#   it -- it would score the default at 0.8984 while the pelvis-relative
#   check, which is the honest one for that mapping, scores it at 0.9996.
#
# `rel_total` is scored against c_spine2->c_spine3 in the relative frame like
# `rel_perjoint`, and it is NOT expected to reach that pairing's ceiling: it
# does not drive spine_1 from row 36 at all, it interpolates 65% of the way
# to the chest. Scoring it against the row it declines to use is the point --
# the gap IS the cost of the distribution ruling, and pinning it here is how
# that cost stays visible instead of being argued away in a report.
_SPINE_EDGE_SPECS = {
    #  source          spine_1 ref row, floor,                 relative frame
    "mhr":          (MHR_SPINE1, SPINE1_SPINE2_FLOOR, False),
    "hybrid":       (MHR_SPINE2, SPINE1_SPINE2_36_FLOOR, False),
    "v15":          (MHR_SPINE1, SPINE1_SPINE2_FLOOR, False),
    "rel_perjoint": (MHR_SPINE2, SPINE1_SPINE2_36_FLOOR, True),
    "rel_total":    (MHR_SPINE2, SPINE1_SPINE2_36_FLOOR, True),
}


def _spine_edge_cosines(row_id, source):
    """{edge label -> (cos, floor)} for the two spine edges the solve can
    actually influence, on one row, under *source*'s own reference rows and
    frame (see _SPINE_EDGE_SPECS).

    Reference directions come from `pred_joint_coords`, which is CAMERA
    frame, so they take `_CAM_TO_RIG_POS`. Floors are the measured
    rest-vs-rest ceilings minus 0.03 -- see the constants above."""
    row1, floor1, relative = _SPINE_EDGE_SPECS[source]
    edges = (("spine_1", "spine_2", row1, MHR_SPINE3, floor1),
             ("spine_2", "neck", MHR_SPINE3, MHR_NECK, SPINE2_NECK_FLOOR))
    rots = _rots(row_id)
    P_mhr = np.asarray(ROWS[row_id]["pred_joint_coords"], float)
    L = _solve(row_id, rots)
    P = fk_world_positions(RIG, {**RIG.rest_local_q, **L})
    carry = None
    if relative:
        Wr = fk_world_orientations(RIG, RIG.rest_local_q)
        d_pelvis = QM.multiply(L[I["pelvis"]], QM.conjugate(Wr[I["pelvis"]]))
        carry = QM.multiply(d_pelvis, QM.conjugate(_mhr_delta_q(rots, MHR_ROOT)))
    out = {}
    for parent, child, m_parent, m_child, floor in edges:
        got = _unit(P[I[child]] - P[I[parent]])
        want = _unit((P_mhr[m_child] - P_mhr[m_parent]) * _CAM_TO_RIG_POS)
        if carry is not None:
            want = QM.rotate_vector(carry, want)
        out[f"{parent}->{child}"] = (float(np.dot(got, want)), floor)
    return out


# Which (row, edge) pairs sit BELOW their floor, per spine mapping, measured
# 2026-08-22 over all sixteen fixture rows. Pinned as an exact set, in the
# house style of `_KNOWN_INVERTED` in test_npz_fingers.py: a new violation
# fails here, and so does a violation that quietly disappears -- either way
# somebody must come back and re-measure.
#
# This was `assert not failures` while the fixture held only the six rows
# that predate the spine captures. It is NOT weakened here: the floors are
# untouched (R11 forbids moving them to make a run pass) and every violation
# is named. What changed is the population -- ten hand-posed captures that
# genuinely bend, arch and laterally flex the spine, directions the original
# six never covered at all.
#
# The counts are the summary, over 36 edges (18 rows x 2) since the crawl row
# 1c3ba88d joined the fixture on 2026-08-23: the default (rel_total) violates
# 9 times (8 over the previous 17 rows), rel_perjoint
# once, the mapping v16 first shipped 5, the hybrid 5, v15 7. The default is
# NOT the leader on this machine-side metric and was never chosen on it --
# see test_where_the_default_leads... below, which asserts both halves.
_SPINE_FLOOR_VIOLATIONS = {
    # THE DEFAULT since 2026-08-23. Its spine_2->neck row is rel_perjoint's,
    # bit for bit -- same chest. The eight spine_1->spine_2 rows are the
    # PRICE of the 65/35 distribution ruling, and every one of them is a row
    # where the model bends its own mid-back well past 65% of its total
    # (0693dd37 is the extreme: |rel(36)| 87.8 against a 66.5 total, so a
    # 65% interpolation lands 0.7336 against a 0.9696 floor). A two-bone
    # spine cannot both reproduce the model's intermediate joint and split
    # the total the way Scott's rig does; he chose the split.
    "rel_total": {
        ("0693dd37", "spine_1->spine_2"),   # deep pike fold, 0.7336
        ("1c3ba88d", "spine_1->spine_2"),   # crawl, 0.8289 -- the row the
                                            # npz pelvis anchor was measured
                                            # on; its MHR column is
                                            # non-monotonic like the pike's
        ("38608eb8", "spine_1->spine_2"),   # robert-crouch, 0.8992
        ("3b66ffdf", "spine_1->spine_2"),   # deep-forward-fold, 0.9258
        ("4fe66c92", "spine_1->spine_2"),   # forward-fold, 0.9687
        ("8ea93cbf", "spine_2->neck"),      # scorpion-handstand, 0.8062 -- the
                                            # chest is rel_perjoint's, so this
                                            # one is rel_perjoint's too
        ("a5a0e4f1", "spine_1->spine_2"),   # inverted-tuck, 0.9323
        ("a75968b1", "spine_1->spine_2"),   # seated-forward-fold, 0.9437
        ("b7c95336", "spine_1->spine_2"),   # hoop-inverted, 0.9565
    },
    "rel_perjoint": {
        ("8ea93cbf", "spine_2->neck"),      # scorpion-handstand, 0.8062
    },
    "mhr": {
        ("1e6a7a60", "spine_1->spine_2"),   # 0.9504
        ("8ea93cbf", "spine_1->spine_2"),   # 0.8911
        ("8ea93cbf", "spine_2->neck"),      # 0.8062
        ("9029c8a8", "spine_1->spine_2"),   # 0.9499
        ("a9099833", "spine_1->spine_2"),   # 0.9371
    },
    "hybrid": {
        ("3b66ffdf", "spine_2->neck"),      # 0.7943 -- spine_2 is v15's
        ("4fe66c92", "spine_2->neck"),      # 0.7980    landmark anchor under
        ("a5a0e4f1", "spine_2->neck"),      # 0.8171    the hybrid, so these
        ("a75968b1", "spine_2->neck"),      # 0.7793    five are v15's too
        ("b7c95336", "spine_2->neck"),      # 0.8039
    },
    "v15": {
        ("1e6a7a60", "spine_1->spine_2"),   # 0.9581
        ("3b66ffdf", "spine_2->neck"),      # 0.7943
        ("4fe66c92", "spine_2->neck"),      # 0.7980
        ("9029c8a8", "spine_1->spine_2"),   # 0.9572
        ("a5a0e4f1", "spine_2->neck"),      # 0.8171
        ("a75968b1", "spine_2->neck"),      # 0.7793
        ("b7c95336", "spine_2->neck"),      # 0.8039
    },
}


@pytest.mark.parametrize("source", sorted(_SPINE_EDGE_SPECS))
def test_real_row_spine_directions_against_the_anchored_floors(source, monkeypatch):
    """Real rots on every fixture row, under each spine mapping: exactly the
    pinned (row, edge) pairs may sit below their floor, and no others."""
    monkeypatch.setattr(PG, "SPINE_SOURCE", source)
    scored, below = 0, set()
    for row_id in ROWS:
        for label, (cos, floor) in _spine_edge_cosines(row_id, source).items():
            scored += 1
            if cos < floor:
                below.add((row_id[:8], label))
    assert scored == 2 * len(ROWS), f"positive control: scored {scored} edges"
    want = _SPINE_FLOOR_VIOLATIONS[source]
    assert below == want, (
        f"{source}: new violations {sorted(below - want)}, "
        f"repaired violations {sorted(want - below)}")


def test_where_the_default_leads_on_the_machine_side_and_where_it_does_not(monkeypatch):
    """The machine-side comparison as an assertion rather than a number in a
    report, INCLUDING the two edges where the default does not win.

    spine_1->spine_2: `rel_perjoint` leads outright and sits at its ceiling,
    because it drives spine_1 from the very row this edge is scored against.
    The DEFAULT does not: it interpolates 65% of the way to the chest, and
    lands 0.957 against that 0.9996. That gap is the price of the
    distribution ruling and it is asserted, not mentioned.

    spine_2->neck: v15's landmark anchor (which the hybrid shares) has the
    better MEAN, 0.8746 against 0.8515 -- and the worse tail, min 0.7793
    against 0.8062, which is why it violates the floor five times and the
    two relative mappings once each.

    The default was chosen on the total chest-vs-pelvis error against the
    spine-zeroed captures (28.0 deg vs 39.3) and then on Scott's own 65/35
    rig convention -- never on this metric. Hiding either loss would
    misrepresent that.

    Not a tautology -- `mhr` and `rel_perjoint` transfer the SAME row 37 to
    spine_2 and score identically here once each is measured in its own
    frame, and `rel_total` joins them."""
    stats = {}
    for source in sorted(_SPINE_EDGE_SPECS):
        monkeypatch.setattr(PG, "SPINE_SOURCE", source)
        per: dict = {}
        for row_id in ROWS:
            for label, (cos, _) in _spine_edge_cosines(row_id, source).items():
                per.setdefault(label, []).append(cos)
        stats[source] = {k: (float(np.mean(v)), float(np.min(v))) for k, v in per.items()}

    lead = max(stats, key=lambda s: stats[s]["spine_1->spine_2"][0])
    assert lead == "rel_perjoint", f"{lead} leads spine_1->spine_2: {stats}"
    assert stats["rel_perjoint"]["spine_1->spine_2"][0] > 0.999      # measured 0.9996
    # The default's cost on that edge, pinned. It is not the leader here.
    assert stats["rel_total"]["spine_1->spine_2"][0] == pytest.approx(0.950, abs=0.005)
    assert stats["rel_total"]["spine_1->spine_2"][1] == pytest.approx(0.734, abs=0.005)

    # spine_2->neck: better mean for the landmark anchor, worse worst case.
    assert stats["v15"]["spine_2->neck"][0] > stats["rel_perjoint"]["spine_2->neck"][0]
    assert stats["v15"]["spine_2->neck"][1] < stats["rel_perjoint"]["spine_2->neck"][1]
    # `mhr` and `rel_perjoint` put spine_2 in the same place relative to
    # their own reference frames, so this edge must agree between them --
    # to 1e-5 rather than exactly, since the two reach it through different
    # quaternion composition orders (measured 1.05e-06 apart).
    assert stats["mhr"]["spine_2->neck"] == pytest.approx(
        stats["rel_perjoint"]["spine_2->neck"], abs=1e-5)
    # rel_total's chest IS rel_perjoint's, from one expression in the solver,
    # so this edge is the same quantity -- read off FK POSITIONS whose
    # accumulation differs upstream at spine_1, hence 1e-12 and not exact.
    assert stats["rel_total"]["spine_2->neck"] == pytest.approx(
        stats["rel_perjoint"]["spine_2->neck"], abs=1e-12)
    # v15 and hybrid share spine_2's anchor by construction, so this edge is
    # the same quantity in both -- but it is read off FK POSITIONS whose
    # accumulation differs upstream at spine_1, which lands one ULP apart.
    assert stats["v15"]["spine_2->neck"] == pytest.approx(
        stats["hybrid"]["spine_2->neck"], abs=1e-12)


def test_the_five_mappings_are_not_the_same_measurement(monkeypatch):
    """Positive control for the parametrised test above: the pinned
    violation sets must come from five genuinely different solves, not from
    a monkeypatch that silently failed to take. On one row, the underlying
    cosines must all differ."""
    row = DEV_ROW
    got = {}
    for source in sorted(_SPINE_EDGE_SPECS):
        monkeypatch.setattr(PG, "SPINE_SOURCE", source)
        got[source] = {k: round(v[0], 4) for k, v in _spine_edge_cosines(row, source).items()}
    pairs = [(a, b) for a in got for b in got if a < b]
    assert len(pairs) == 10                                       # positive control
    for a, b in pairs:
        assert got[a] != got[b], f"{a} and {b} measure identically: {got[a]}"
    # TWO pairs share spine_2's anchor by construction -- hybrid with v15
    # (both keep v15's landmark chest) and rel_total with rel_perjoint (both
    # take the same root-relative row 37) -- so that edge must MATCH within
    # each pair while spine_1's must not. The sharpest statement available
    # that the switch moves exactly what it claims to, and nothing further.
    for a, b in (("hybrid", "v15"), ("rel_perjoint", "rel_total")):
        assert got[a]["spine_2->neck"] == got[b]["spine_2->neck"], got
        assert got[a]["spine_1->spine_2"] != got[b]["spine_1->spine_2"], got


# Every solved bone whose LOCAL legitimately differs between the two spine
# branches. The two spine bones are the intended change; the rest follow
# from the rig's parent chain, not from any extra reach:
# left_clavicle, right_clavicle and neck are all children of spine_2, so
# their locals absorb its new world frame, and head's local absorbs neck's
# (head's own WORLD anchor, from the nose + eye line, is untouched).
#
# The two SHOULDERS joined this list when `_aim_delta` landed (2026-08-22).
# Before it, the clavicle's world delta was the world-minimal rotation onto
# the shoulder target and so did not depend on the spine AT ALL -- the
# clavicle's local absorbed the whole difference and the chain went quiet
# there. A roll-free local puts that roll in the clavicle's WORLD delta
# instead, which the shoulder's local is then expressed against. It is a
# re-parameterisation and the test asserts it as one: measured on the dev
# row, the shoulder LOCALS move 0.32 and 15.50 deg while their WORLD
# orientations move 2e-14 deg, and the elbows and wrists do not move at all.
_V16_AFFECTED = ("spine_1", "spine_2", "neck", "head",
                 "left_clavicle", "right_clavicle",
                 "left_shoulder", "right_shoulder")


def test_fallback_none_switches_only_the_spine_branch():
    """`mhr_rots=None` really takes the v15 branch, and among the bones BOTH
    branches solve the switch reaches exactly the spine and its descendants
    -- nothing else.

    v15's own behavior is pinned by the 44 pre-existing tests, every one of
    which calls the solver without `mhr_rots` -- so what is left to prove
    here is that the branch switches at all, and that it does not leak.

    Deliberately NOT the claim that only spine_1 and spine_2 move: a bone's
    local is expressed in its parent's world frame, so spine_2's four
    descendants (`_V16_AFFECTED` above) must change with it, and a test
    demanding otherwise would be demanding a bug. The exact set is asserted
    rather than sampled, so a future edit that reaches one bone further
    fails here instead of passing quietly.

    Until v16 task 5 the two branches also solved the SAME 34 bones, and this
    test asserted exactly that. They no longer do, by design: `mhr_rots` now
    drives the ten finger chains as well as the spine, so the v16 set is the
    v15 34 plus the 30 phalanges of every digit that passes the integrity
    gate (all ten digits, on this row). That claim is kept here as an exact
    set DIFFERENCE rather than deleted -- a branch that started solving
    anything else extra, or stopped solving a v15 bone, still fails here."""
    targets = rig_targets_from_mhr70(_kp(DEV_ROW))
    v15 = solve_rig_locals(RIG, targets)                          # untouched call site
    v16 = solve_rig_locals(RIG, targets, mhr_rots=_rots(DEV_ROW))
    phalanges = {i for i in RIG.order
                 if ("_thumb_" in RIG.name[i] or "_finger_" in RIG.name[i])
                 and not RIG.name[i].endswith("_tip")}
    assert len(phalanges) == 30                                   # positive control
    assert set(v16) - set(v15) == phalanges, "the branch solved something unexpected"
    assert set(v15) - set(v16) == set(), "the branch stopped solving a v15 bone"

    for name in ("spine_1", "spine_2"):
        moved = _quat_deg(v15[I[name]], v16[I[name]])
        assert moved > 1.0, f"{name} moved only {moved:.4f} deg -- the branch did not switch"

    # Index-keyed, never name-keyed: two rig bones share the name "joint7".
    expected = {I[n] for n in _V16_AFFECTED}
    changed = {i for i in v15 if not np.allclose(v15[i], v16[i], atol=1e-9)}
    assert changed == expected, (
        f"unexpected: {sorted(RIG.name[i] for i in changed - expected)}, "
        f"missing: {sorted(RIG.name[i] for i in expected - changed)}")

    # The shoulders are on that list as a re-parameterisation (see
    # _V16_AFFECTED): their locals move because the clavicle's world frame
    # moved under them, and their own WORLD orientation must not budge --
    # it comes from the arm keypoints, which no spine branch touches.
    W15 = fk_world_orientations(RIG, {**RIG.rest_local_q, **v15})
    W16 = fk_world_orientations(RIG, {**RIG.rest_local_q, **v16})
    for name in ("left_shoulder", "right_shoulder"):
        assert _quat_deg(v15[I[name]], v16[I[name]]) > 1e-6      # positive control
        spun = _quat_deg(W15[I[name]], W16[I[name]])
        assert spun < 1e-6, f"{name}'s WORLD orientation moved {spun:.3e} deg"


def test_version_pinned_17():
    st15 = rig_state_from_mhr70(_kp(DEV_ROW))
    st16 = rig_state_from_mhr70(_kp(DEV_ROW), mhr_rots=_rots(DEV_ROW))
    assert st15["retargetVersion"] == 17 and st16["retargetVersion"] == 17
    # ...and mhr_rots actually reaches the solve through this entry point,
    # not just through solve_rig_locals.
    assert st16["pose"]["spine_1"] != st15["pose"]["spine_1"]
