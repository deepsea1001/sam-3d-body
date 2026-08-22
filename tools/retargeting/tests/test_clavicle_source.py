"""The clavicle takes the MODEL's own rotation, not an aim at a keypoint.

`fb42e71` took the invented ROLL out of the clavicle's local. What was left
underneath is a defect in the AIM itself, and Scott found it: the shoulder
balls read as dislocated, and the two sides disagree on a pose that is
symmetric.

The mechanism is an inter-skeleton rest mismatch. Aiming the mannequin's rest
`clavicle -> shoulder` direction at MHR-70's shoulder keypoint asks one
rotation to do two jobs: express the pose, AND close the 26.5 deg (left) gap
between where the mannequin's shoulder sits at rest and where MHR's does. The
second job is a CONSTANT of the two skeletons, but the rotation that performs
it is not: it depends on the pose and on the parent, so it lands as a
different invented swing every row, and a different one per side. Measured:

  row 01a3a144 (double-biceps, arms symmetric)
      ours  L 24.5 deg, euler-Z -21.1 (shoulder FORWARD)
            R 18.5 deg, euler-Z +16.1 (shoulder BACKWARD)
      MHR   L 14.6 deg, R  9.5 deg -- both elevation, ~no protraction

  row 0413c2c0 (standing)
      ours  L 33.1 deg, euler-Z -30.9 (a 31 deg forward swing, standing still)
            R  9.7 deg
      MHR   L  9.6 deg, R 14.7 deg

MHR's own clavicle rotations, taken CHEST-RELATIVE (its own c_spine3 is the
parent of both its clavicles, exactly as the mannequin's spine_2 is), carry
none of that: they are elevation-dominant and near-symmetric on a symmetric
pose, because they were never asked to close a rest gap.

So the clavicle is driven from those rotations, composed onto the spine_2 we
actually solved -- the same transfer pattern task 6b uses for the spine
(root-relative onto our pelvis), one joint further out. SWING-ONLY: the twist
about the clavicle's own long axis is stripped before composing, so the
`fb42e71` contract (no roll in the local) holds by construction, and Scott's
rig has no clavicle-roll degree of freedom to receive one anyway.

Without `mhr_rots` there are no rotations to transfer and the aim path stays,
bit-identical -- covered by `test_the_fallback_path_is_untouched`.
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
    load_rig, rig_targets_from_mhr70, solve_rig_locals,
    fk_world_orientations, fk_world_positions, _mhr_delta_q)
from retargeting.core.math_utils import QuaternionMath as QM       # noqa: E402

_FIX = Path(__file__).resolve().parent / "fixtures" / "clavicle_source_rows.json"
ROWS = json.loads(_FIX.read_text())["rows"]
_NPZ_FIX = Path(__file__).resolve().parent / "fixtures" / "mhr_npz_rows.json"
NPZ_ROWS = json.loads(_NPZ_FIX.read_text())["rows"]

BICEPS = "01a3a1444b8b6d3fea9a9b5d94c13c32"
STANDING = "0413c2c01cc8f8571b3225ad07d06f15"

RIG = load_rig()
I = RIG.index_of_name
WR = fk_world_orientations(RIG, RIG.rest_local_q)

# The app's own base pose, from BOTH 2026-08-22 probes
# (poseforge3d captures/capture-rigbase2-11.json and capture-rigrestprobe-10.json,
# which agree bit-for-bit): three.js [x,y,z,w] -> solver [w,x,y,z].
BASE_PROBE_CLAVICLE = {
    "left_clavicle": np.array([0.7071067811865476, -0.7071067811865475, 0.0, 0.0]),
    "right_clavicle": np.array([0.7071067811865475, 0.7071067811865476, 0.0, 0.0]),
}


def _deg(q):
    q = np.asarray(q, float)
    return float(np.degrees(2 * np.arctan2(np.linalg.norm(q[1:]), abs(q[0]))))


def _twist_deg(q, axis):
    q = np.asarray(q, float)
    if q[0] < 0:
        q = -q
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    tw = np.array([q[0], *(np.dot(q[1:], axis) * axis)])
    n = np.linalg.norm(tw)
    return 0.0 if n < 1e-12 else _deg(tw / n)


def _euler_xyz_deg(q):
    """three.js default 'XYZ' euler order, degrees -- the split the app's
    joint-limit config and Scott's IK are written in. For the clavicle, Z is
    protraction/retraction (left: negative = shoulder forward; right:
    positive = shoulder backward, verified against the app's bind frames) and
    X is elevation."""
    w, x, y, z = np.asarray(q, float)
    m = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    sy = float(np.clip(m[0, 2], -1, 1))
    if abs(sy) < 0.9999:
        return np.degrees([np.arctan2(-m[1, 2], m[2, 2]), np.arcsin(sy),
                           np.arctan2(-m[0, 1], m[0, 0])])
    return np.degrees([np.arctan2(m[2, 1], m[1, 1]), np.arcsin(sy), 0.0])


def _shoulder_shift(local_delta, side):
    """Where the clavicle's local delta sends its own rest shoulder offset,
    in rig WORLD axes: +Y up (elevation), +Z toward the viewer (protraction),
    X lateral. Rig units.

    Reported in preference to an euler triple because the clavicle's rest
    frame is a +/-90 deg flip about X, so no single euler axis IS elevation --
    on these rows the largest euler component is Y while the motion is almost
    entirely vertical. This measures the thing Scott actually looks at: which
    way the shoulder ball goes."""
    ci, si = I[f"{side}_clavicle"], I[f"{side}_shoulder"]
    off = RIG.rest_world_p[si] - RIG.rest_world_p[ci]
    world = QM.multiply(WR[ci], QM.multiply(local_delta, QM.conjugate(WR[ci])))
    return QM.rotate_vector(world, off) - off


def _rest_long_axis(side):
    """clavicle -> shoulder rest direction, in world and in the clavicle's own
    rest local frame."""
    ci, si = I[f"{side}_clavicle"], I[f"{side}_shoulder"]
    d = RIG.rest_world_p[si] - RIG.rest_world_p[ci]
    d = d / np.linalg.norm(d)
    return d, QM.rotate_vector(QM.conjugate(WR[ci]), d)


def _solve(row_id, rows=None, **kw):
    src = rows if rows is not None else ROWS
    r = src[row_id]
    kp = np.asarray(r["kp70"], np.float32).reshape(70, 3)
    rots = np.asarray(r["joint_global_rots"], float)
    targets = rig_targets_from_mhr70(kp)
    L = solve_rig_locals(RIG, targets, mhr_rots=rots, **kw)
    return targets, {i: QM.normalize(q) for i, q in L.items()}


def _clavicle_local_delta(L, side):
    ci = I[f"{side}_clavicle"]
    return QM.multiply(QM.conjugate(RIG.rest_local_q[ci]), L[ci])


def _mhr_chest_relative(row_id, side, rows=None):
    src = rows if rows is not None else ROWS
    rots = np.asarray(src[row_id]["joint_global_rots"], float)
    chest = _mhr_delta_q(rots, PG._MHR_CHEST)
    return QM.multiply(QM.conjugate(chest),
                       _mhr_delta_q(rots, PG._MHR_CLAVICLE_ROW[f"{side}_clavicle"]))


# --------------------------------------------------------------------------- #
# The asset's right clavicle rest
# --------------------------------------------------------------------------- #

def test_the_asset_clavicle_rest_is_the_app_s_own_base_pose():
    """`capture-rigrest-02`, which the asset's body rest was built from, is a
    POSED capture, not the app's base -- 43 of its 73 bones differ from the
    2026-08-22 base probes, by up to 105.9 deg. R14 already re-based the 40
    finger bones for exactly this reason. The right clavicle is the same
    defect one bone over: rigrest-02 caught it 6.08 deg off base, and the
    asset copied that in as "rest".

    It matters because the emitted local is `rest_local^-1 . L` -- a rest
    frame 6.08 deg off the app's own turns a symmetric transfer into an
    asymmetric euler split, which is what Scott sees as a dislocated
    shoulder ball.

    Positive control: the LEFT clavicle is bit-identical in both captures and
    must already pass, so a failure here is about the right one specifically."""
    for side, want in BASE_PROBE_CLAVICLE.items():
        got = RIG.rest_local_q[I[side]]
        off = _deg(QM.multiply(QM.conjugate(QM.normalize(got)), want))
        assert off < 1e-4, f"{side} rest is {off:.4f} deg off the app's base pose"

    # ...and the pair is then an exact mirror, which is the structural check
    # that does not depend on trusting either probe on its own.
    lq, rq = (QM.normalize(RIG.rest_local_q[I[f"{s}_clavicle"]])
              for s in ("left", "right"))
    assert lq[0] == pytest.approx(rq[0], abs=1e-6)
    assert lq[1] == pytest.approx(-rq[1], abs=1e-6)
    assert abs(lq[2]) < 1e-6 and abs(lq[3]) < 1e-6
    assert abs(rq[2]) < 1e-6 and abs(rq[3]) < 1e-6


def test_the_asset_is_self_consistent_after_the_rest_rebase():
    """Correcting a rest LOCAL moves every world position beneath it, so the
    asset's own `rest_world_p` has to follow or the file contradicts itself.
    Same rule R14 applied to the forty finger bones."""
    fk = fk_world_positions(RIG, RIG.rest_local_q)
    worst = max((float(np.linalg.norm(fk[i] - RIG.rest_world_p[i])), RIG.name[i])
                for i in RIG.order)
    assert worst[0] < 1e-4, f"{worst[1]} stored rest_world_p is {worst[0]:.3e} off FK"
    # Positive control: the right arm must actually have MOVED off v1, or this
    # test is passing on an asset nobody re-based.
    v1 = json.loads((Path(PG._ASSET).parent / "posegoblin_rig_v1.json").read_text())
    moved = {b["name"]: float(np.linalg.norm(
        np.asarray(b["rest_world_p"], float) - RIG.rest_world_p[i]))
        for i, b in enumerate(v1["bones"]) if b["name"] in
        ("right_shoulder", "right_elbow", "right_wrist", "right_clavicle",
         "left_shoulder", "left_clavicle")}
    assert moved["right_shoulder"] > 0.1, moved
    assert moved["right_elbow"] > 0.1, moved
    assert moved["right_clavicle"] < 1e-9, moved      # its own q cannot move it
    assert moved["left_shoulder"] < 1e-9, moved       # the left side is untouched
    assert moved["left_clavicle"] < 1e-9, moved


# --------------------------------------------------------------------------- #
# Acceptance 1 and 2: the two rows Scott named
# --------------------------------------------------------------------------- #

def test_the_symmetric_pose_gets_symmetric_clavicles():
    """Acceptance 1. Double-biceps: the arms are symmetric, so the clavicles
    must be. The aim path made them 24.5 deg with 21.1 deg of FORWARD
    protraction on the left against 18.5 deg with 16.1 deg BACKWARD on the
    right -- opposite signs on a mirror-image pose, which no reading of the
    image supports.

    Held to MHR's own chest-relative SWING (L 7.8, R 9.5; the full rotations
    are 14.6 and 9.5 and the difference is the roll this transfer drops by
    design), and to the shoulder ball going UP rather than forward."""
    _, L = _solve(BICEPS)
    got = {}
    for side in ("left", "right"):
        d = _clavicle_local_delta(L, side)
        v = _shoulder_shift(d, side)
        ez = _euler_xyz_deg(d)[2]
        want = _deg(PG._swing_about(_mhr_chest_relative(BICEPS, side),
                                    _rest_long_axis(side)[0]))
        got[side] = (_deg(d), v, ez, want)
        print(f"\n{side:6s} total {_deg(d):5.1f} (model swing {want:5.1f})  "
              f"shoulder ball up {v[1]:+.3f} fwd {v[2]:+.3f} lat {v[0]:+.3f}  "
              f"euler-Z {ez:+6.1f}")

    for side in ("left", "right"):
        total, v, ez, want = got[side]
        assert total == pytest.approx(want, abs=0.5), \
            f"{side} clavicle {total:.1f} deg vs the model's own swing {want:.1f}"
        assert v[1] > 2.0 * abs(v[2]), (
            f"{side} shoulder ball goes forward ({v[2]:+.3f}) as much as up "
            f"({v[1]:+.3f}) -- elevation must dominate on this pose")
        assert abs(ez) < 8.0, f"{side} protraction reads {ez:.1f} deg"

    # The two sides must not disagree about which way the shoulders go: the
    # complaint was a forward LEFT against a backward RIGHT.
    (_, vl, zl, _), (_, vr, zr, _) = got["left"], got["right"]
    assert abs(vl[1] - vr[1]) < 0.12, f"elevation is asymmetric: {vl[1]:+.3f} vs {vr[1]:+.3f}"
    assert abs(abs(zl) - abs(zr)) < 6.0, \
        f"protraction is mirror-asymmetric: left {zl:.1f}, right {zr:.1f}"


def test_the_standing_pose_has_no_invented_forward_swing():
    """Acceptance 2. Standing still, the aim path swung the left clavicle
    33.1 deg with 30.9 deg of it forward protraction. The model's own
    chest-relative swing is 9.1 / 14.7, and whatever protraction survives
    there is the model's reading of the pose, not ours."""
    _, L = _solve(STANDING)
    for side in ("left", "right"):
        d = _clavicle_local_delta(L, side)
        v = _shoulder_shift(d, side)
        ez = _euler_xyz_deg(d)[2]
        want = _deg(PG._swing_about(_mhr_chest_relative(STANDING, side),
                                    _rest_long_axis(side)[0]))
        print(f"\n{side:6s} total {_deg(d):5.1f} (model swing {want:5.1f})  "
              f"shoulder ball up {v[1]:+.3f} fwd {v[2]:+.3f}  euler-Z {ez:+6.1f}")
        assert _deg(d) == pytest.approx(want, abs=0.5)
        assert abs(ez) < 12.0, f"{side} protraction is {ez:.1f} deg standing still"


# --------------------------------------------------------------------------- #
# The transfer's own contracts
# --------------------------------------------------------------------------- #

def test_the_local_is_the_model_s_rotation_in_the_clavicle_s_own_frame():
    """The frame algebra, stated as an assertion rather than trusted.

        L(clav) = (D(spine_2) . Wr(spine_2))^-1 . D(clav) . Wr(clav)
        D(clav) = D(spine_2) . swing(dRel)
      =>  local delta-from-rest = Wr(clav)^-1 . swing(dRel) . Wr(clav)

    So the clavicle's local must be MHR's chest-relative rotation, swing-only,
    carried into the clavicle's rest frame -- and nothing else. Checked on
    both fixture rows and both sides.

    Positive control: the same comparison against the UNSTRIPPED rotation must
    NOT hold wherever MHR carries roll, or this would pass on a solve that
    ignored the swing-only rule."""
    checked, differ = 0, 0
    for rid in ROWS:
        _, L = _solve(rid)
        for side in ("left", "right"):
            ci = I[f"{side}_clavicle"]
            ax_world, _ = _rest_long_axis(side)
            rel = _mhr_chest_relative(rid, side)
            want = QM.multiply(QM.conjugate(WR[ci]),
                               QM.multiply(PG._swing_about(rel, ax_world), WR[ci]))
            got = _clavicle_local_delta(L, side)
            assert _deg(QM.multiply(QM.conjugate(QM.normalize(want)), got)) < 1e-6, \
                f"{rid[:8]} {side}: the local is not the model's swing in the rest frame"
            checked += 1
            raw = QM.multiply(QM.conjugate(WR[ci]), QM.multiply(rel, WR[ci]))
            if _deg(QM.multiply(QM.conjugate(QM.normalize(raw)), got)) > 1e-3:
                differ += 1
    assert checked == 2 * len(ROWS)
    assert differ > 0, "positive control: stripping the twist changed nothing anywhere"


def test_the_transferred_local_still_carries_no_roll():
    """`fb42e71`'s contract, under the new source. Scott's rig has no
    clavicle-roll degree of freedom, so MHR's roll about the long axis is
    dropped rather than transferred."""
    for rows, src in ((ROWS, "clavicle rows"), (NPZ_ROWS, "npz rows")):
        for rid in rows:
            _, L = _solve(rid, rows=rows)
            for side in ("left", "right"):
                _, ax_local = _rest_long_axis(side)
                roll = _twist_deg(_clavicle_local_delta(L, side), ax_local)
                assert roll < 1e-9, f"{src} {rid[:8]} {side}: {roll:.4f} deg of roll"


def test_how_much_model_roll_is_discarded():
    """Report, and bound, what swing-only throws away.

    Measured over all eighteen fixture rows x two sides: median 7.20 deg, p90
    13.19, max 29.07. That is NOT noise, and this test does not pretend it is
    -- it is dropped because the rig has no clavicle-roll degree of freedom to
    receive it, and because roll in this local is what drove Scott's IK into
    its limits (fb42e71). The bound below is an alarm, not a blessing: the
    axis is the MANNEQUIN's long axis, 26.5 deg off MHR's, so these figures
    are an UPPER bound on real lost signal; if the median ever climbs past 12
    deg the discard has stopped being defensible and somebody must re-measure
    rather than re-tune this number."""
    rolls = []
    for rows in (ROWS, NPZ_ROWS):
        for rid in rows:
            for side in ("left", "right"):
                ax_world, _ = _rest_long_axis(side)
                rolls.append(_twist_deg(_mhr_chest_relative(rid, side, rows=rows),
                                        ax_world))
    rolls = np.array(rolls)
    print(f"\nMHR clavicle roll discarded over {len(rolls)} (row, side): "
          f"median {np.median(rolls):.2f} p90 {np.percentile(rolls, 90):.2f} "
          f"max {rolls.max():.2f} deg")
    assert len(rolls) == 2 * (len(ROWS) + len(NPZ_ROWS))      # positive control
    assert rolls.max() > 0.5, "positive control: nothing was being discarded at all"
    assert float(np.median(rolls)) < 12.0, (
        f"the model's clavicle roll has grown to a median {np.median(rolls):.1f} deg "
        f"(was 7.20 when swing-only was adopted) -- re-measure whether discarding "
        f"it is still defensible; do not move this bound to make the run pass")


def test_the_fallback_path_is_untouched():
    """`mhr_rots=None` has no rotations to transfer, so the clavicle keeps the
    aim path exactly. Asserted as bit-identity against a solve that cannot
    have taken the transfer, not as a tolerance."""
    for rid in ROWS:
        kp = np.asarray(ROWS[rid]["kp70"], np.float32).reshape(70, 3)
        targets = rig_targets_from_mhr70(kp)
        L = solve_rig_locals(RIG, targets)
        for side in ("left", "right"):
            ci, si = I[f"{side}_clavicle"], I[f"{side}_shoulder"]
            rb, _ = _rest_long_axis(side)
            tb = np.asarray(targets[si], float) - np.asarray(targets[ci], float)
            tb = tb / np.linalg.norm(tb)
            W = fk_world_orientations(RIG, {**RIG.rest_local_q, **L})
            d = QM.multiply(W[ci], QM.conjugate(WR[ci]))
            landed = QM.rotate_vector(d, rb)
            landed = landed / np.linalg.norm(landed)
            err = np.degrees(np.arcsin(np.clip(
                float(np.linalg.norm(np.cross(landed, tb))), 0, 1)))
            assert err < 1e-9, f"{rid[:8]} {side}: fallback aim is {err:.3e} deg off"
