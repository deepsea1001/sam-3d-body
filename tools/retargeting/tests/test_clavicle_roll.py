"""The clavicle must not roll about its own long axis.

Scott, 2026-08-22: the solver "rolls the clavicle about its own long axis
instead of elevating the arm", and his IK then snaps the clavicle to its
joint limits and throws the arm behind the body. The specification is his
own hand-corrected pose.

TWO captures of the same arms-overhead pose, both committed below as
literals so this file is hermetic:

  captures/capture-07299b5a00c6bd984d223716c5510bda-25.json   the solver's output
  captures/capture-07299b5a00c6bd984d223716c5510bda-26.json   Scott's correction

(in poseforge3d; that repo's `captures/` is not importable from here, and the
two poses are the whole of the evidence, so they are pinned rather than read.)

Decomposing each clavicle's LOCAL delta-from-rest into twist about the
clavicle's own long axis versus the remaining swing -- the long axis being
the rest direction from the clavicle to its child shoulder, carried into the
clavicle's rest local frame, [0.875, 0.325, -0.360] on the left and its
negation on the right:

                      clavicle total   TWIST(long axis)   swing   shoulder total
    LEFT  solver           101.4            100.9          12.3        128.8
    LEFT  Scott             18.6              6.1          17.6        114.2
    RIGHT solver           135.2            135.2           4.6         81.2
    RIGHT Scott             12.5              4.1          11.8         75.3

The two poses AGREE about where the arm goes -- clavicle->shoulder differs by
5.6 deg on the left and 13.1 deg on the right -- and disagree by 96 and 131
deg about the roll around that near-identical axis.

Why that roll is invented rather than measured: the clavicle has exactly one
child with a target, and a single direction constraint cannot observe
rotation about the axis pointing at that child.
`test_clavicle_roll_is_unobservable_from_the_shoulder` proves it on the rig
rather than asserting it -- an arbitrary roll injected into the clavicle's
local leaves the shoulder's world position bit-identical. So zeroing it is
not a preference between two readings of the data; there is no reading of
the data that prefers any value at all, and the rig's joint limits (which
constrain the LOCAL) are the only thing left with an opinion.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import retargeting.retargeters.posegoblin_rig as PG                # noqa: E402
from retargeting.retargeters.posegoblin_rig import (               # noqa: E402
    load_rig, rig_targets_from_mhr70, solve_rig_locals, solved_indices,
    fk_world_orientations, fk_world_positions)
from retargeting.core.math_utils import QuaternionMath as QM       # noqa: E402

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mhr_npz_rows.json"
ROWS = json.loads(_FIXTURE.read_text())["rows"]
RIG = load_rig()
I = RIG.index_of_name
WR = fk_world_orientations(RIG, RIG.rest_local_q)

# Both captures, [x,y,z,w] exactly as three.js wrote them (the solver works in
# [w,x,y,z]; `_wxyz` converts, and no coordinate conversion is ever applied to
# a rotation). Only the bones this file reasons about; the rest of each pose
# is irrelevant to the clavicle and would be noise here.
_CAP25 = {
    "pelvis": [0.027844528167256247, 0.7089971527576647, 0.7025053373980635, -0.055087107910113606],
    "spine_1": [0.7424791282604752, 0.0550507759091155, 0.10252038466505642, 0.6596852274785521],
    "spine_2": [0.04539531268931225, -0.13840324368550955, 0.001050802895443802, 0.9893349665069422],
    "left_clavicle": [0.06275158273914597, -0.024832196989176075, -0.2775163564503858, 0.9583487514345264],
    "left_shoulder": [0.23616794446827621, 0.05835773833997135, -0.6472497537172637, -0.7224195453724872],
    "right_clavicle": [-0.3349648051732884, -0.3992892973526516, -0.0021090004463359263, 0.8534429472448904],
    "right_shoulder": [-0.8359789341265156, 0.13532166714964994, 0.5074831608562864, -0.1590336066714438],
}
_CAP26 = {
    "pelvis": [0.027844528167256247, 0.7089971527576647, 0.7025053373980635, -0.055087107910113606],
    "spine_1": [0.7424791282604752, 0.0550507759091155, 0.10252038466505642, 0.6596852274785521],
    "spine_2": [0.04539531268931225, -0.13840324368550955, 0.001050802895443802, 0.9893349665069422],
    "left_clavicle": [-0.6978438539508077, -0.11407872502392193, 0.11407872502392191, 0.6978438539508078],
    "left_shoulder": [0.7118617043539234, -0.013808372137763841, 0.39408947167159114, 0.581167558498269],
    "right_clavicle": [0.6978452352599511, -0.11407027494042259, -0.1140702749404226, 0.697845235259951],
    "right_shoulder": [0.7118617043539234, -0.013808372137763841, 0.39408947167159114, 0.581167558498269],
}

# Scott's worst hand-posed clavicle roll on this pose, measured from _CAP26
# by `test_the_captures_state_the_defect` below: 6.14 deg (left), 4.14 deg
# (right). The ceiling this file holds the solver to is that worst value
# rounded up -- it is a CEILING on invented roll, not a target: the solver
# lands at 2e-14 deg, because the quantity is unobservable and the honest
# answer is "none".
SCOTT_WORST_ROLL_DEG = 7.0

# Residual of the roll-free construction, measured 2026-08-22 over every
# fixture row x both sides x both spine mappings: worst 2.0e-14 deg. Pinned
# five orders of magnitude above that so float noise never fails this and a
# genuine degree never passes.
ROLL_FREE_TOL_DEG = 1e-9


def _wxyz(q):
    """three.js [x,y,z,w] -> solver [w,x,y,z]."""
    return np.array([q[3], q[0], q[1], q[2]], float)


def _pose_locals(cap):
    return {I[k]: _wxyz(v) for k, v in cap.items()}


def _deg(q):
    q = np.asarray(q, float)
    return float(np.degrees(2 * np.arctan2(np.linalg.norm(q[1:]), abs(q[0]))))


def _twist_deg(q, axis):
    """Angle of q's twist component about *axis* (swing-twist, projection
    form -- the same decomposition VectorMath.swing_twist_decompose uses)."""
    q = np.asarray(q, float)
    if q[0] < 0:
        q = -q
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    tw = np.array([q[0], *(np.dot(q[1:], axis) * axis)])
    n = np.linalg.norm(tw)
    return 0.0 if n < 1e-12 else _deg(tw / n)


def _swing_deg(q, axis):
    q = np.asarray(q, float)
    if q[0] < 0:
        q = -q
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    tw = np.array([q[0], *(np.dot(q[1:], axis) * axis)])
    n = np.linalg.norm(tw)
    tw = np.array([1.0, 0.0, 0.0, 0.0]) if n < 1e-12 else tw / n
    return _deg(QM.multiply(q, QM.conjugate(tw)))


def _aim_axis(n, c):
    """The rest direction from bone *n* to its child *c*, in world and in
    n's own rest LOCAL frame. The local one is what the rig's joint limits
    and Scott's IK read."""
    d = RIG.rest_world_p[c] - RIG.rest_world_p[n]
    d = d / np.linalg.norm(d)
    return d, QM.rotate_vector(QM.conjugate(WR[n]), d)


def _local_delta(local_q, n):
    """Local rotation of bone *n* measured from its rest local."""
    return QM.multiply(QM.conjugate(RIG.rest_local_q[n]), local_q[n])


def _rots(rid):
    return np.asarray(ROWS[rid]["joint_global_rots"], float)


def _kp(rid):
    return np.asarray(ROWS[rid]["kp70"], np.float32).reshape(70, 3)


def _single_child_aims(targets, solved, rots):
    """{bone -> child} for every bone the solver aims with EXACTLY ONE usable
    child pair -- mirroring solve_rig_locals's own precedence (anchors first,
    then the ankle hinge, then `pairs`) rather than naming bones, so a rig
    change that moves a bone into or out of this branch shows up here."""
    anchors = PG._anchor_deltas(RIG, targets, WR, rots)
    out = {}
    for n in solved:
        if (RIG.name[n] in PG._FINGER_PHALANGE_NAMES or n not in targets
                or n in anchors or RIG.name[n] in PG._ANKLE_FOOT):
            continue
        usable = []
        for c in RIG.children[n]:
            if c not in targets:
                continue
            rb = RIG.rest_world_p[c] - RIG.rest_world_p[n]
            tb = np.asarray(targets[c], float) - np.asarray(targets[n], float)
            if np.linalg.norm(rb) > 1e-6 and np.linalg.norm(tb) > 1e-6:
                usable.append(c)
        if len(usable) == 1:
            out[n] = usable[0]
    return out


# --------------------------------------------------------------------------- #
# The specification: the two captures
# --------------------------------------------------------------------------- #

def test_the_captures_state_the_defect():
    """Reproduce the table in the module docstring from the pinned poses.

    Positive control for every twist number in this file (CLAUDE.md rule 1):
    the same measurement, on the same axis, must be able to return both a
    huge number and a tiny one -- otherwise "the twist is zero" would be
    indistinguishable from "the metric returns zero"."""
    rows = {}
    for side in ("left", "right"):
        ci, si = I[f"{side}_clavicle"], I[f"{side}_shoulder"]
        _, ax = _aim_axis(ci, si)
        for tag, cap in (("solver", _CAP25), ("scott", _CAP26)):
            L = _pose_locals(cap)
            dl, ds = _local_delta(L, ci), _local_delta(L, si)
            rows[(side, tag)] = (_deg(dl), _twist_deg(dl, ax), _swing_deg(dl, ax), _deg(ds))
    print(f"\n{'':22s}{'total':>9s}{'TWIST':>9s}{'swing':>9s}{'shoulder':>10s}")
    for k, v in rows.items():
        print(f"  {k[0]:6s}{k[1]:15s}" + "".join(f"{x:9.1f}" for x in v[:3]) + f"{v[3]:10.1f}")

    # The defect: the solver rolls, Scott does not.
    assert rows[("left", "solver")][1] > 80.0
    assert rows[("right", "solver")][1] > 80.0
    assert rows[("left", "scott")][1] < SCOTT_WORST_ROLL_DEG
    assert rows[("right", "scott")][1] < SCOTT_WORST_ROLL_DEG

    # ...and the arm is in broadly the same place in both, which is what
    # makes the roll gratuitous rather than the price of reaching.
    for side in ("left", "right"):
        assert abs(rows[(side, "solver")][3] - rows[(side, "scott")][3]) < 20.0

    # The aim itself barely differs: 5.6 deg left, 13.1 deg right.
    P25 = fk_world_positions(RIG, {**RIG.rest_local_q, **_pose_locals(_CAP25)})
    P26 = fk_world_positions(RIG, {**RIG.rest_local_q, **_pose_locals(_CAP26)})
    for side, want in (("left", 5.6), ("right", 13.1)):
        ci, si = I[f"{side}_clavicle"], I[f"{side}_shoulder"]
        a = P25[si] - P25[ci]
        b = P26[si] - P26[ci]
        got = np.degrees(np.arccos(np.clip(
            a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1)))
        assert got == pytest.approx(want, abs=0.15), f"{side} aim differs by {got:.2f} deg"


def test_clavicle_roll_is_unobservable_from_the_shoulder():
    """Why zero and not "some small amount": nothing in the data has an
    opinion. Inject an arbitrary roll about the clavicle's own long axis
    into Scott's pose -- the shoulder's world POSITION does not move at all,
    so no shoulder target can ever prefer one value over another.

    Positive control: the same magnitude of rotation about a PERPENDICULAR
    axis moves the shoulder by centimetres, so "nothing moved" is a property
    of the axis and not of the test."""
    L = _pose_locals(_CAP26)
    base = fk_world_positions(RIG, {**RIG.rest_local_q, **L})
    for side in (("left"), ("right")):
        ci, si = I[f"{side}_clavicle"], I[f"{side}_shoulder"]
        # FK carries the child on rest_local_p, so THAT is the exact axis a
        # roll cannot move the shoulder about. It is the same axis the rest
        # of this file reads off rest_world_p -- 0.000000 deg apart, both
        # [0.875, 0.325, -0.360] -- but rest_world_p was captured from the
        # live scene and is stored rounded, which shows up here as ~8e-07 of
        # spurious motion and nowhere else. Asserted, not assumed.
        _, from_world = _aim_axis(ci, si)
        ax_local = np.asarray(RIG.rest_local_p[si], float)
        ax_local = ax_local / np.linalg.norm(ax_local)
        assert float(np.dot(ax_local, from_world)) > 1 - 1e-11
        perp = np.cross(ax_local, [0.0, 0.0, 1.0])
        for angle in (np.radians(37.0), np.radians(-113.0)):
            rolled = dict(L)
            rolled[ci] = QM.multiply(L[ci], PG._axis_angle_q(ax_local, angle))
            got = fk_world_positions(RIG, {**RIG.rest_local_q, **rolled})
            moved = float(np.linalg.norm(got[si] - base[si]))
            assert moved < 1e-12, f"{side}: roll moved the shoulder {moved:.3e}"

            off = dict(L)
            off[ci] = QM.multiply(L[ci], PG._axis_angle_q(perp, angle))
            got = fk_world_positions(RIG, {**RIG.rest_local_q, **off})
            moved_perp = float(np.linalg.norm(got[si] - base[si]))
            assert moved_perp > 0.5, \
                f"{side}: positive control -- off-axis rotation moved only {moved_perp:.3e}"


# --------------------------------------------------------------------------- #
# The fix, against the captures and against every fixture row
# --------------------------------------------------------------------------- #

def test_the_aim_carries_no_roll_relative_to_the_parent():
    """The single-child aim, run on Scott's own pose.

    The row behind these captures is not in this repo's fixtures, so the
    solver cannot be re-run on it end to end -- but everything the
    single-child branch consumes IS recoverable from capture -25, which is
    the solver's own output: the parent's world delta and the direction the
    solver actually aimed the clavicle at. Feed those two back through the
    production aim and the clavicle's LOCAL must come out roll-free while
    landing on the same direction.

    And the sharpest statement available that this is the RIGHT roll-free
    value and not merely a roll-free one: the clavicle local it produces
    lands 8.1 / 13.6 deg from the local Scott posed by hand, where the
    shipped solver's is 106.8 / 131.9 deg away. What is left is his aim
    disagreeing with ours by 5.6 / 13.1 deg plus the 6.1 / 4.1 deg of roll
    he happened to leave in -- not roll this solve invented."""
    L25 = _pose_locals(_CAP25)
    L26 = _pose_locals(_CAP26)
    W25 = fk_world_orientations(RIG, {**RIG.rest_local_q, **L25})
    for side in ("left", "right"):
        ci, si = I[f"{side}_clavicle"], I[f"{side}_shoulder"]
        p = RIG.parent[ci]
        rb, ax_local = _aim_axis(ci, si)
        d_parent = QM.multiply(W25[p], QM.conjugate(WR[p]))
        d_solver = QM.multiply(W25[ci], QM.conjugate(WR[ci]))
        tb = QM.rotate_vector(d_solver, rb)          # where the solver aimed it

        d_new = PG._aim_delta(rb, tb, d_parent)

        landed = QM.rotate_vector(d_new, rb)
        assert float(np.dot(landed, tb)) > 1 - 1e-12, f"{side}: the aim moved"

        rel = QM.multiply(QM.conjugate(d_parent), d_new)
        local = QM.multiply(QM.conjugate(WR[ci]), QM.multiply(rel, WR[ci]))
        roll = _twist_deg(local, ax_local)
        assert roll < ROLL_FREE_TOL_DEG, f"{side}: invented {roll:.3f} deg of roll"

        # ...and it agrees with Scott's own clavicle, which the shipped one
        # does not. Both distances measured, both asserted -- an improvement
        # claimed against only one of them would be half a measurement.
        l_fix = QM.normalize(QM.multiply(QM.conjugate(W25[p]),
                                         QM.multiply(d_new, WR[ci])))
        near = _deg(QM.multiply(QM.conjugate(l_fix), L26[ci]))
        far = _deg(QM.multiply(QM.conjugate(L25[ci]), L26[ci]))
        print(f"\n{side}: local total {_deg(local):.2f} deg, roll {roll:.3e} deg "
              f"(solver captured {_twist_deg(_local_delta(L25, ci), ax_local):.1f}, "
              f"Scott {_twist_deg(_local_delta(L26, ci), ax_local):.1f}); "
              f"distance to Scott's local {far:.1f} -> {near:.1f} deg")
        assert far > 100.0, f"{side}: the shipped clavicle is only {far:.1f} deg from Scott's"
        assert near < 15.0, f"{side}: the roll-free clavicle is {near:.1f} deg from Scott's"


@pytest.mark.parametrize("source", ["mhr", "rel_perjoint"])
def test_no_invented_roll_on_any_fixture_row(source, monkeypatch):
    """End to end through solve_rig_locals, every fixture row, under BOTH
    spine mappings: no bone the solver aims at a single child may carry any
    roll about that aim axis in its LOCAL.

    Parametrised over the spine because the clavicles hang off spine_2 and
    the roll they were inheriting was spine_2's -- so a fix that only held
    under one spine mapping would be a coincidence, not a fix.

    Since R15 the two CLAVICLES no longer reach this branch when `mhr_rots`
    is present: they are anchored from the model's own chest-relative
    rotation instead (`_anchor_deltas`), and their roll-freeness is pinned by
    tests/test_clavicle_source.py, which owns that path. What is left here is
    the two EYES -- so the per-row count drops from 4 to 2, and the clavicles
    are asserted ABSENT-because-anchored rather than allowed to vanish
    quietly. The fallback path, where the clavicles do still take this
    branch, is covered by test_the_aim_path_clavicle_is_still_roll_free."""
    monkeypatch.setattr(PG, "SPINE_SOURCE", source)
    scored, worst = 0, (0.0, None)
    swings, names = [], set()
    for rid in ROWS:
        rots = _rots(rid)
        targets = rig_targets_from_mhr70(_kp(rid))
        L = solve_rig_locals(RIG, targets, mhr_rots=rots)
        for n, c in _single_child_aims(targets, solved_indices(RIG, rots), rots).items():
            _, ax = _aim_axis(n, c)
            d = _local_delta(L, n)
            roll = _twist_deg(d, ax)
            swings.append(_swing_deg(d, ax))
            names.add(RIG.name[n])
            scored += 1
            if roll > worst[0]:
                worst = (roll, f"{rid[:8]} {RIG.name[n]}")
            assert roll < ROLL_FREE_TOL_DEG, \
                f"{rid[:8]} {RIG.name[n]}: {roll:.3f} deg of roll about its own aim axis"
    # Positive controls: the branch must actually have been exercised, and
    # the locals must not be identity (which would make "roll == 0" vacuous).
    assert scored == 2 * len(ROWS), f"expected 2 single-child bones per row, scored {scored}"
    assert names == {"left_eye", "right_eye"}, f"single-child aims are now {sorted(names)}"
    # The clavicles left this branch for a REASON, and it must be the anchor
    # -- not a target that quietly went missing.
    rots = _rots(next(iter(ROWS)))
    targets = rig_targets_from_mhr70(_kp(next(iter(ROWS))))
    anchors = PG._anchor_deltas(RIG, targets, WR, rots)
    for side in ("left", "right"):
        assert I[f"{side}_clavicle"] in anchors, f"{side}_clavicle is neither aimed nor anchored"
    # Vacuity guard, not a quality bar: "roll == 0" would be trivially true of
    # an identity local. Re-calibrated for R15's population -- with the
    # clavicles gone to the anchor path the two eyes are what is left, and
    # their worst swing over these rows measures 9.33 deg (it was 56 deg while
    # the clavicles were here). The roll assertion above is untouched.
    assert max(swings) > 5.0, f"every local is near-rest -- worst swing {max(swings):.2f} deg"
    print(f"\n{source}: {scored} single-child locals, worst roll {worst[0]:.2e} deg ({worst[1]})")


def test_the_aim_path_clavicle_is_still_roll_free():
    """fb42e71's contract on the path the clavicle still takes: no `mhr_rots`,
    so no rotations to transfer, so the single-child aim -- and it must still
    put no roll in the local.

    Kept separate from the parametrised test above because that one now finds
    only the eyes: with rotations present the clavicles are anchored (R15).
    Positive control: the locals must not be near-rest, or "roll == 0" is
    vacuous here too."""
    worst, swings = 0.0, []
    for rid in ROWS:
        L = solve_rig_locals(RIG, rig_targets_from_mhr70(_kp(rid)))
        for side in ("left", "right"):
            ci, si = I[f"{side}_clavicle"], I[f"{side}_shoulder"]
            _, ax = _aim_axis(ci, si)
            d = _local_delta(L, ci)
            worst = max(worst, _twist_deg(d, ax))
            swings.append(_swing_deg(d, ax))
    assert len(swings) == 2 * len(ROWS)                      # positive control
    assert max(swings) > 20.0, f"every clavicle is near-rest, worst {max(swings):.2f} deg"
    assert worst < ROLL_FREE_TOL_DEG, f"the aim path invented {worst:.4f} deg of roll"


def test_the_clavicle_fix_moves_no_joint_in_the_world():
    """Proof the arm still reaches: the roll-free clavicle and the world-
    minimal rotation it replaces put the clavicle, the shoulder, the elbow
    and the wrist in exactly the same place, and leave the whole arm's WORLD
    orientation identical. Only the parameterisation changed -- the roll
    moved out of the clavicle's local and into a world delta nothing else
    reads.

    The old branch is reconstructed from the shipped solve rather than
    remembered, so this stays honest if the aim itself is ever changed: the
    old clavicle delta is the world-minimal rotation onto the same target,
    and the shoulder's local is rebuilt underneath it (the shoulder is
    world-ANCHORED by _anchor_deltas, so its world orientation is the same
    in both parameterisations and its local absorbs the difference).

    Run WITHOUT `mhr_rots` since R15: with rotations present the clavicle no
    longer takes the aim path at all (it is anchored from the model's own
    chest-relative rotation), so there would be no aim to re-parameterise.
    This is the fallback path -- rows with no `mhr_params_npz` blob -- and
    `_aim_delta`'s contract has to keep holding there.

    The displacement bound is DERIVED, not chosen. The two deltas differ by
    a roll about `rest_world_p`'s clavicle->shoulder direction, while FK
    carries the shoulder on `rest_local_p` -- and the asset stores those
    rounded, so the two disagree by 3.5e-05 / 8.7e-05 deg. A roll of at most
    180 deg about an axis alpha off the offset displaces it by at most
    2|offset|sin(alpha), and everything below the shoulder rides along
    rigidly. That is the bound asserted; if the rig is ever re-exported at
    full precision it tightens by itself."""
    bound = 0.0
    for side in ("left", "right"):
        ci, si = I[f"{side}_clavicle"], I[f"{side}_shoulder"]
        off = np.asarray(RIG.rest_local_p[si], float)
        _, from_world = _aim_axis(ci, si)
        sin_a = float(np.linalg.norm(np.cross(off / np.linalg.norm(off), from_world)))
        bound = max(bound, 2.0 * float(np.linalg.norm(off)) * sin_a)
    assert bound < 1e-5, f"the rig's own rest rounding grew to {bound:.3e}"

    worst_pos, worst_rot, checked = 0.0, 0.0, 0
    for rid in ROWS:
        rots = _rots(rid)
        targets = rig_targets_from_mhr70(_kp(rid))
        # Normalised on BOTH sides of the comparison. QuaternionMath.multiply
        # does not renormalise, so the solve's own locals come back up to
        # 1.1e-05 off unit (and the shipped pose 7.5e-06) -- fk_world_positions
        # scales by |q|^2, which puts 5e-05 of pure numerical drift on the
        # wrist and would be read here as the arm having moved. Pre-existing
        # and unrelated to the clavicle; removed from both poses rather than
        # absorbed into a tolerance.
        L = {i: QM.normalize(q)
             for i, q in solve_rig_locals(RIG, targets).items()}
        W = fk_world_orientations(RIG, {**RIG.rest_local_q, **L})
        new = fk_world_positions(RIG, {**RIG.rest_local_q, **L})
        for side in ("left", "right"):
            ci, si = I[f"{side}_clavicle"], I[f"{side}_shoulder"]
            p = RIG.parent[ci]
            rb, _ = _aim_axis(ci, si)
            tb = np.asarray(targets[si], float) - np.asarray(targets[ci], float)
            tb = tb / np.linalg.norm(tb)
            w_old = QM.multiply(QM.from_two_vectors(rb, tb), WR[ci])   # pre-fix world
            old = dict(L)
            # Normalised because this reconstruction stacks four extra
            # quaternion products on top of the solve's own, and
            # QuaternionMath.multiply does not renormalise: unchecked, the
            # locals reach |q| = 1.00001, which fk_world_positions turns into
            # 5e-05 of SCALING at the wrist and would be read here as the arm
            # having moved. A local quaternion is a rotation; this is the
            # reconstruction paying its own numerical bill, not a tolerance.
            old[ci] = QM.normalize(QM.multiply(QM.conjugate(W[p]), w_old))
            old[si] = QM.normalize(QM.multiply(QM.conjugate(w_old), W[si]))
            P_old = fk_world_positions(RIG, {**RIG.rest_local_q, **old})
            W_old = fk_world_orientations(RIG, {**RIG.rest_local_q, **old})
            for nm in (f"{side}_clavicle", f"{side}_shoulder",
                       f"{side}_elbow", f"{side}_wrist"):
                worst_pos = max(worst_pos,
                                float(np.linalg.norm(P_old[I[nm]] - new[I[nm]])))
                checked += 1
            # The arm's WORLD orientation is untouched from the shoulder
            # down -- only the clavicle's changes, and only by a roll.
            for nm in (f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist"):
                worst_rot = max(worst_rot, _deg(QM.multiply(
                    QM.conjugate(W_old[I[nm]]), W[I[nm]])))
            # spun = Wr^-1 (D_old^-1 D_new) Wr, and both deltas map rb onto
            # the same target, so this is a rotation about rb carried into
            # the clavicle's rest local frame -- exactly `_aim_axis`'s local.
            spun = QM.multiply(QM.conjugate(W_old[ci]), W[ci])
            _, ax_bone = _aim_axis(ci, si)
            # Positive control in one: the two parameterisations really do
            # differ (so "the positions agree" is not a pose compared with
            # itself), and what they differ BY is roll and nothing else.
            assert _deg(spun) > 1.0, \
                f"{rid[:8]} {side}: the two clavicle deltas are the same rotation"
            assert _swing_deg(spun, ax_bone) < 1e-6, (
                f"{rid[:8]} {side}: the clavicle's world orientation changed by "
                f"{_swing_deg(spun, ax_bone):.3e} deg of SWING, not just roll")
    assert checked == 8 * len(ROWS), f"positive control: checked {checked} joints"
    print(f"\nold vs new, {checked} joints: worst displacement {worst_pos:.3e} rig units "
          f"(rest-rounding bound {bound:.3e}), worst world-orientation change "
          f"{worst_rot:.3e} deg (shoulder..wrist)")
    assert worst_pos <= bound, \
        f"the arm moved by {worst_pos:.3e}, past the {bound:.3e} the rig's rounding allows"
    assert worst_rot < 1e-6, f"the arm's world orientation moved by {worst_rot:.3e} deg"


def test_the_aim_is_exact_even_when_the_parent_has_already_landed_it():
    """`_aim_delta` composes its swing on top of the parent's delta, so the
    swing it needs is usually the small residual -- and
    QuaternionMath.from_two_vectors returns the IDENTITY for any pair within
    0.81 deg, which would silently drop that residual entirely.

    Measured on the motion corpus before `_min_swing_q`: the shortcut fired
    on 6 of 1200 clavicle aims and cost up to 0.74 deg of aim (0.02 rig units
    at the shoulder). Pinned here on the exact geometry that triggers it --
    a parent that has already brought the bone to within 0.1 deg of its
    target -- rather than by hoping a fixture row happens to.

    Positive control: from_two_vectors, on the same pair, really does return
    the identity, so this is testing a difference that exists."""
    rest = np.array([0.6, 0.48, -0.64])
    rest = rest / np.linalg.norm(rest)
    axis = np.array([0.2, 0.9, -0.3])
    # _axis_angle_q does not normalise its axis, and a non-unit quaternion
    # would SCALE what rotate_vector returns -- which this test would then
    # read as an aim error.
    d_parent = PG._axis_angle_q(axis / np.linalg.norm(axis), np.radians(97.0))
    landed = QM.rotate_vector(d_parent, rest)
    # target 0.1 deg off where the parent already puts the bone
    off = np.cross(landed, [0.0, 0.0, 1.0])
    off = off / np.linalg.norm(off)
    target = landed * np.cos(np.radians(0.1)) + off * np.sin(np.radians(0.1))

    assert _deg(QM.from_two_vectors(landed, target)) == 0.0, \
        "positive control: from_two_vectors no longer takes its shortcut here"

    got = QM.rotate_vector(PG._aim_delta(rest, target, d_parent), rest)
    got = got / np.linalg.norm(got)
    # arcsin|cross|, not arccos(dot): arccos is ill-conditioned at dot = 1 and
    # reads 8.5e-07 deg of pure float noise where the true error is 3.6e-15.
    err = np.degrees(np.arcsin(np.clip(float(np.linalg.norm(np.cross(got, target))), 0, 1)))
    assert err < 1e-9, f"the aim landed {err:.4e} deg off its target"


def test_the_arm_still_lands_on_its_targets():
    """Machine-gate style direction check over every fixture row, SPLIT BY
    PATH -- because R15 changed what the clavicle promises.

    FALLBACK (`mhr_rots=None`): the clavicle is still a single-child aim and
    still satisfies it outright, so clavicle->shoulder keeps the EXACT floor.

    TRANSFER (`mhr_rots` present): it does NOT. The clavicle now takes the
    model's own chest-relative rotation, which lands the shoulder where the
    MODEL puts it rather than exactly on MHR-70's shoulder keypoint, and the
    two disagree -- the mannequin's rest shoulder sits 26.5 deg off MHR's.
    That residual was the price of not manufacturing 20-30 deg of invented
    protraction to close a rest gap (tests/test_clavicle_source.py), and
    Scott accepted it knowingly -- until the ball being off its keypoint
    turned up as arm OVERSHOOT in his renders.

    task-clavcorrect then bounded it: the transferred clavicle is aimed back
    at the keypoint by at most `_CLAV_AIM_CORRECTION_MAX_DEG` = 15. So this
    edge is much closer to exact than it was, and the bound TIGHTENS here
    rather than loosening. Measured over the fixture rows (21 since
    task-relpeak, whose peak chest also moved the corrected tail 0.9470 ->
    0.9641 -- nearer the keypoints, same direction as every relpeak
    position number):

        f07064f (transfer only)   median cosine 0.9464, worst 0.8315 (33.7 deg)
        with the bounded aim       median cosine 0.9988, worst 0.9641 (15.4 deg)

    Pinned as a REGRESSION BOUND at 0.90, named -- not as an aspiration, and
    never applied to the fallback path where the old guarantee still holds.
    The bound moved 0.80 -> 0.90 because the measurement did; do not move it
    back to make a run pass.

    What does NOT move either way is everything below the shoulder: the
    shoulder, elbow and wrist are world-anchored from the arm keypoints, so
    shoulder->elbow and elbow->wrist stay EXACT on both paths. Measured over
    1800 corpus rows: cosine 1.000000, all 3600."""
    seen = {}
    for label, kw, clav_floor in (("fallback", {}, 0.9999),
                                  ("transfer", {"mhr_rots": True}, 0.90)):
        scored, bad, clav = 0, {}, []
        for rid in ROWS:
            targets = rig_targets_from_mhr70(_kp(rid))
            L = solve_rig_locals(RIG, targets,
                                 **({"mhr_rots": _rots(rid)} if kw else {}))
            P = fk_world_positions(RIG, {**RIG.rest_local_q, **L})
            for side in ("left", "right"):
                for a, b, floor in ((f"{side}_clavicle", f"{side}_shoulder", clav_floor),
                                    (f"{side}_shoulder", f"{side}_elbow", 0.90),
                                    (f"{side}_elbow", f"{side}_wrist", 0.90)):
                    ai, bi = I[a], I[b]
                    got = P[bi] - P[ai]
                    want = np.asarray(targets[bi], float) - np.asarray(targets[ai], float)
                    cos = float(got @ want / (np.linalg.norm(got) * np.linalg.norm(want)))
                    scored += 1
                    if a.endswith("_clavicle"):
                        clav.append(cos)
                    if cos < floor:
                        bad[f"{rid[:8]} {a}->{b}"] = round(cos, 4)
        assert scored == 6 * len(ROWS), f"positive control: scored {scored} edges"
        assert not bad, f"{label}: arm directions below floor: {bad}"
        seen[label] = np.array(clav)
        print(f"\n{label}: clavicle->shoulder cosine median {np.median(seen[label]):.4f} "
              f"worst {seen[label].min():.4f}")

    # The split is the point, so assert the two paths really are different --
    # otherwise the transfer's looser floor would be measuring the fallback.
    assert seen["fallback"].min() > 0.9999
    assert seen["transfer"].min() == pytest.approx(0.9641, abs=0.01)
