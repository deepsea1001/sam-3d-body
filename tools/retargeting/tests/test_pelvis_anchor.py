"""PELVIS_SOURCE_NPZ_ROOT: the pelvis takes the model's own root rotation.

THE DEFECT. The pelvis is the last anchor still solved purely from keypoints.
`_orthonormal_frame_from_hips_and_up` fixes ONE axis from the hip line -- which
MHR-70 localises well -- and takes the pitch about it from the hipmid->spine_1
chord, which MHR-70 does not localise at all: there are no mid-spine keypoints,
`spine_1` is INTERPOLATED by MHR70Retargeter, and on a prone, piked or crawling
body that chord is both short and nearly parallel to the camera ray. The frame
is then perfectly orthonormal and pitched wrong, and because every other bone
composes onto the pelvis, the whole assembled body inherits the error: correct
relative bends at a wrong global attitude.

Measured against MHR's own root rotation -- `joint_global_rots` row 1,
delta-from-rest via the committed bind_poses/mhr_skeleton_rest.json, the same
`_mhr_delta_q` the spine and the clavicles already transfer through:

    0693dd37 (deep pike fold)  65.90 deg out   <- the 1800-row corpus MAXIMUM
    1c3ba88d (crawl)           53.24 deg out
    corpus (1800 rows)         median 8.75, p90 33.48, 45.2% over 10 deg

THE FIX, one expression. When `mhr_rots` is present the pelvis's WORLD
ORIENTATION is `Delta(root) . pelvis_rest_world`, i.e. its world DELTA is
`Delta(root)` outright. No frame conversion: MHR model space and rig space are
the same ROTATIONAL frame (see `_mhr_delta_q`'s docstring -- positions differ
by the camera map diag(1,-1,-1) and rotations never do, verified two ways).

The pelvis POSITION is untouched. `pelvisPosition` is the rig's own REST pelvis
position (ruling 10) and `groundY` is read off the TARGETS; neither reads
`A[pelvis]`, so this change moves attitude only.

WHAT MUST NOT MOVE, and why it structurally cannot. Both transfers above are
RELATIVE and compose onto the pelvis:

    chest    = A[pelvis] . conj(Delta(root)) . Delta(37)
    clavicle = A[spine_2] . swing(conj(Delta(37)) . Delta(row))

so `conj(pelvis_world) . chest_world` and `conj(chest_world) . clavicle_world`
both cancel `A[pelvis]` algebraically. Those two relatives -- the curved back
and the shoulder girdle the eye actually reads -- are bit-identical under both
anchors. That is the whole reason those transfers were built relative, and this
file asserts it rather than assuming it.

WHAT DOES MOVE, stated up front:
  - the clavicle AIM CORRECTION (b2d58c1) is the one exception. It re-aims the
    transferred clavicle at a WORLD keypoint, by at most 15 deg, so it reads
    the pelvis through the chest. Fixing the pelvis moves the residual it
    corrects -- downwards, on both named rows. Pinned below, not hidden.
  - every world POSITION below the pelvis, since the hips hang off it.
    Directions are unchanged (the limb anchors are built from targets alone),
    so the arm and leg AIMS re-solve exactly; the bones translate.
  - `pelvis->hip` direction against kp70 DEGRADES wherever MHR's root
    disagrees with the observed hip line. That is the accepted trade, the same
    shape as the clavicle's: reported, pinned, not gated.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from retargeting.core.math_utils import QuaternionMath as QM
from retargeting.retargeters import posegoblin_rig as PG
from retargeting.retargeters.posegoblin_rig import (
    fk_world_orientations, fk_world_positions, load_rig, rig_state_from_mhr70,
    rig_targets_from_mhr70, solve_rig_locals)

ROWS = json.loads(
    (Path(__file__).parent / "fixtures" / "mhr_npz_rows.json").read_text())["rows"]

PIKE_ROW = "0693dd37755e0d6eb9012857f5e3b405"    # deep pike fold, 65.90 deg out
CRAWL_ROW = "1c3ba88d8b32b3c20a458eb5512ee3f8"   # crawl, 53.24 deg out
DEV_ROW = "a6566802a6c9ddd63340ccb4520e0001"
MHR_ROOT = 1

# Camera -> rig POSITION map: cv_to_yup's -Y composed with the solver's own
# _CV_YUP_TO_RIG -Z, exactly as test_npz_spine.py defines it. Needed for
# `pred_joint_coords`, which is camera-frame; `joint_global_rots` never needs
# it (see _mhr_delta_q).
_CAM_TO_RIG_POS = np.array([1.0, -1.0, -1.0])

RIG = load_rig()
I = RIG.index_of_name
WREST = fk_world_orientations(RIG, RIG.rest_local_q)

_conj = QM.conjugate
_unit = lambda v: np.asarray(v, float) / np.linalg.norm(v)


def _ang(q):
    q = np.asarray(q, float)
    if q[0] < 0:
        q = -q
    return float(np.degrees(2 * np.arctan2(np.linalg.norm(q[1:]), q[0])))


def _between(qa, qb):
    """Angle from *qa* to *qb*, degrees."""
    return _ang(QM.multiply(_conj(qa), qb))


def _rots(row_id):
    return np.asarray(ROWS[row_id]["joint_global_rots"], float)


def _kp(row_id):
    return np.asarray(ROWS[row_id]["kp70"], np.float32).reshape(70, 3)


def _at(source, monkeypatch, row_id):
    """Full solve of *row_id* under PELVIS_SOURCE = *source*, WITH rotations."""
    monkeypatch.setattr(PG, "PELVIS_SOURCE", source)
    return solve_rig_locals(RIG, rig_targets_from_mhr70(_kp(row_id)),
                            mhr_rots=_rots(row_id))


def _world(L):
    return fk_world_orientations(RIG, {**RIG.rest_local_q, **L})


def _positions(L):
    return fk_world_positions(RIG, {**RIG.rest_local_q, **L})


def _model_root(row_id):
    """MHR's own root rotation as a rig-frame world delta."""
    return PG._mhr_delta_q(_rots(row_id), MHR_ROOT)


def _pelvis_delta(L):
    """The solved pelvis's world DELTA. The pelvis is the rig root, so its
    local IS its world; strip the rest local to get the delta."""
    return QM.multiply(np.asarray(L[I["pelvis"]], float),
                       _conj(RIG.rest_local_q[I["pelvis"]]))


def _disagreement(source, monkeypatch, row_id):
    """The headline metric: how far the SOLVED pelvis is from the model's own
    root rotation, in degrees."""
    return _between(_pelvis_delta(_at(source, monkeypatch, row_id)),
                    _model_root(row_id))


def _relative(W, parent, child):
    return QM.multiply(_conj(W[I[parent]]), W[I[child]])


def _dir_cos(P, T, a, b):
    """Cosine between the SOLVED a->b bone direction and the kp70 target's."""
    return float(np.dot(_unit(P[I[b]] - P[I[a]]),
                        _unit(np.asarray(T[I[b]], float) - np.asarray(T[I[a]], float))))


# --------------------------------------------------------------------------
# 1. The defect, and the fix, on the two rows it was reported from.
# --------------------------------------------------------------------------

def test_the_hip_line_anchor_is_the_thing_that_is_66_and_53_degrees_out(monkeypatch):
    """THE DEFECT, kept as a live measurement rather than a number in a report.

    These two are the evidence the change was approved on and they are read off
    the shipped hip-line construction, not off a reconstruction of it -- so if
    that construction is ever repaired at source, this test says so loudly
    instead of quietly agreeing with a stale figure.

    0693dd37 is the WORST row in the 1800-row motion corpus."""
    pike = _disagreement(PG.PELVIS_SOURCE_HIPS, monkeypatch, PIKE_ROW)
    crawl = _disagreement(PG.PELVIS_SOURCE_HIPS, monkeypatch, CRAWL_ROW)
    assert pike == pytest.approx(65.90, abs=0.05), pike
    assert crawl == pytest.approx(53.24, abs=0.05), crawl
    # Positive control: the metric is not saturated -- a row where the hip-line
    # construction and the model's root already agree reads near zero.
    quiet = _disagreement(PG.PELVIS_SOURCE_HIPS, monkeypatch,
                          "1e6a7a60b4539a8fd09867d7e2fa3db6")
    assert quiet < 1.5, quiet


def test_the_npz_anchor_lands_on_the_models_root_on_the_two_named_rows(monkeypatch):
    """ACCEPTANCE 1. 65.90 -> 0 and 53.24 -> 0.

    Exactly zero, not approximately: `A[pelvis]` IS `Delta(root)`, the same
    quaternion the metric compares against. Anything above 1e-9 means a
    conversion crept in between the two."""
    for row_id, before in ((PIKE_ROW, 65.90), (CRAWL_ROW, 53.24)):
        after = _disagreement(PG.PELVIS_SOURCE_NPZ_ROOT, monkeypatch, row_id)
        assert after < 1e-9, f"{row_id[:8]}: {before} deg -> {after:.6f} deg, expected ~0"


def test_the_npz_anchor_lands_on_the_models_root_on_every_fixture_row(monkeypatch):
    """ACCEPTANCE 1, over the whole fixture -- the two named rows are the worst
    two, not the only two.

    Positive control in the same test: the construction this replaces
    disagrees by a median 24.9 deg over the same twenty-one rows (19.7 over
    the eighteen before task-relpeak's flexion rows joined -- deep folds are
    where the hip-line heuristic is weakest), so a "0.0 everywhere" that came
    from a solve that never ran cannot pass."""
    after, before = [], []
    for row_id in sorted(ROWS):
        after.append(_disagreement(PG.PELVIS_SOURCE_NPZ_ROOT, monkeypatch, row_id))
        before.append(_disagreement(PG.PELVIS_SOURCE_HIPS, monkeypatch, row_id))
    assert len(after) == 21                                  # positive control
    assert max(after) < 1e-9, f"worst {max(after):.3e} deg"
    assert float(np.median(before)) == pytest.approx(24.9, abs=0.5), sorted(before)
    assert max(before) == pytest.approx(65.90, abs=0.05)


# --------------------------------------------------------------------------
# 2. The central invariant: the relative transfers do not move.
# --------------------------------------------------------------------------

def test_the_chest_relative_to_the_pelvis_is_bit_identical(monkeypatch):
    """ACCEPTANCE 3, the design's central claim.

    SPINE_SOURCE_REL_TOTAL composes the model's root-relative chest ONTO the
    solved pelvis, so `conj(pelvis_world) . spine_2_world` cancels `A[pelvis]`
    exactly. The curved back the eye reads is therefore untouched by re-
    orienting the pelvis -- which is the entire reason that transfer was built
    relative (task-reltotal) rather than absolute.

    Positive control in the same test: the pelvis's own WORLD orientation
    really did move on these rows, so "unchanged" cannot mean "nothing was
    re-solved"."""
    moved = []
    for row_id in sorted(ROWS):
        Wh = _world(_at(PG.PELVIS_SOURCE_HIPS, monkeypatch, row_id))
        Wn = _world(_at(PG.PELVIS_SOURCE_NPZ_ROOT, monkeypatch, row_id))
        d = _between(_relative(Wh, "pelvis", "spine_2"),
                     _relative(Wn, "pelvis", "spine_2"))
        assert d < 1e-9, f"{row_id[:8]}: chest-vs-pelvis moved {d:.3e} deg"
        # spine_1 is a 65% geodesic slerp from the pelvis to that chest, so it
        # is relative to the pelvis too, and equally invariant.
        d1 = _between(_relative(Wh, "pelvis", "spine_1"),
                      _relative(Wn, "pelvis", "spine_1"))
        assert d1 < 1e-9, f"{row_id[:8]}: spine_1-vs-pelvis moved {d1:.3e} deg"
        moved.append(_between(Wh[I["pelvis"]], Wn[I["pelvis"]]))
    assert len(moved) == 21                                  # positive control
    assert max(moved) == pytest.approx(65.90, abs=0.05), max(moved)
    assert float(np.median(moved)) > 5.0, float(np.median(moved))


def test_the_clavicle_relative_to_the_chest_moves_only_through_the_aim_cap(monkeypatch):
    """ACCEPTANCE 3, the half that is NOT bit-identical, and the mechanism.

    The clavicle TRANSFER is chest-relative and cancels `A[pelvis]` exactly
    like the chest does. The AIM CORRECTION on top of it (b2d58c1) does not:
    it re-aims the transferred clavicle at the kp70 shoulder keypoint, a WORLD
    direction, by at most `_CLAV_AIM_CORRECTION_MAX_DEG`. Re-orienting the
    pelvis moves the residual that correction sees, so the corrected clavicle
    moves with it -- bounded by twice the cap plus the sphere's own holonomy,
    and in practice DOWNWARDS, because a pelvis that agrees with the model
    leaves the chest nearer the keypoints it is being aimed at.

    Both halves are asserted: cap 0 is bit-identical, the production cap is
    not, and the corrected clavicle is measured to land CLOSER to its keypoint
    under the npz anchor than under the hip-line one."""
    monkeypatch.setattr(PG, "_CLAV_AIM_CORRECTION_MAX_DEG", 0.0)
    for row_id in sorted(ROWS):
        Wh = _world(_at(PG.PELVIS_SOURCE_HIPS, monkeypatch, row_id))
        Wn = _world(_at(PG.PELVIS_SOURCE_NPZ_ROOT, monkeypatch, row_id))
        for side in ("left", "right"):
            d = _between(_relative(Wh, "spine_2", f"{side}_clavicle"),
                         _relative(Wn, "spine_2", f"{side}_clavicle"))
            assert d < 1e-9, f"{row_id[:8]} {side}: transfer moved {d:.3e} deg"

    monkeypatch.setattr(PG, "_CLAV_AIM_CORRECTION_MAX_DEG", 15.0)
    changed, err_h, err_n = [], [], []
    for row_id in sorted(ROWS):
        T = rig_targets_from_mhr70(_kp(row_id))
        Lh = _at(PG.PELVIS_SOURCE_HIPS, monkeypatch, row_id)
        Ln = _at(PG.PELVIS_SOURCE_NPZ_ROOT, monkeypatch, row_id)
        Wh, Wn = _world(Lh), _world(Ln)
        Ph, Pn = _positions(Lh), _positions(Ln)
        for side in ("left", "right"):
            changed.append(_between(_relative(Wh, "spine_2", f"{side}_clavicle"),
                                    _relative(Wn, "spine_2", f"{side}_clavicle")))
            err_h.append(_dir_cos(Ph, T, f"{side}_clavicle", f"{side}_shoulder"))
            err_n.append(_dir_cos(Pn, T, f"{side}_clavicle", f"{side}_shoulder"))
    assert len(changed) == 42                                # positive control
    assert max(changed) > 1.0, "the aim correction did not move with the pelvis"
    assert max(changed) < 2 * PG._CLAV_AIM_CORRECTION_MAX_DEG + 4.1, max(changed)
    # ...and it moves the right way: the girdle lands NEARER its keypoints.
    assert float(np.median(err_n)) > float(np.median(err_h))


# --------------------------------------------------------------------------
# 3. The limbs: directions re-solve exactly, positions translate.
# --------------------------------------------------------------------------

def test_every_limb_direction_is_bit_identical(monkeypatch):
    """ACCEPTANCE 4. The four limb chains are anchored from the TARGETS alone
    (`_LIMB_CHAINS` in `_anchor_deltas` never reads `A[pelvis]`), so their
    world deltas -- and therefore every bone DIRECTION under them -- are
    untouched by the pelvis. The bones translate; they do not turn.

    Asserted on FK positions, the way the poseforge3d harness scores them, so
    this is the harness's own metric and not a proxy for it."""
    n = 0
    for row_id in sorted(ROWS):
        T = rig_targets_from_mhr70(_kp(row_id))
        Ph = _positions(_at(PG.PELVIS_SOURCE_HIPS, monkeypatch, row_id))
        Pn = _positions(_at(PG.PELVIS_SOURCE_NPZ_ROOT, monkeypatch, row_id))
        for side in ("left", "right"):
            for a, b in ((f"{side}_hip", f"{side}_knee"),
                         (f"{side}_knee", f"{side}_ankle"),
                         (f"{side}_shoulder", f"{side}_elbow"),
                         (f"{side}_elbow", f"{side}_wrist")):
                ch, cn = _dir_cos(Ph, T, a, b), _dir_cos(Pn, T, a, b)
                assert abs(ch - cn) < 1e-12, f"{row_id[:8]} {a}->{b}: {ch} != {cn}"
                n += 1
        # Positive control: the legs really did MOVE, they just did not turn.
        assert np.linalg.norm(Pn[I["left_knee"]] - Ph[I["left_knee"]]) > 1e-6
    assert n == 8 * 21                                       # positive control


def test_the_pelvis_to_hip_edge_is_the_accepted_trade(monkeypatch):
    """ACCEPTANCE 4's cost, pinned rather than gated.

    `pelvis->hip` is the one edge built from the hip line itself, so the old
    anchor reproduces it by construction (cosine ~1.000 everywhere) and the new
    one cannot: it takes the model's opinion of the root instead. On rows where
    MHR's root and the observed hip line disagree, this edge degrades by
    exactly that disagreement. Same shape as the clavicle transfer's trade
    (b2d58c1 §3): a world-position edge is given up for a rotation the eye
    reads as the body's global attitude.

    This is REPORTED, not gated -- there is no floor here on purpose. The
    poseforge3d harness scores `pelvis->{left,right}_hip` against a WORLD
    reference and will need the same relative/report-only reclassification the
    spine edges got (task-pelvis report, concerns)."""
    before, after = [], []
    for row_id in sorted(ROWS):
        T = rig_targets_from_mhr70(_kp(row_id))
        Ph = _positions(_at(PG.PELVIS_SOURCE_HIPS, monkeypatch, row_id))
        Pn = _positions(_at(PG.PELVIS_SOURCE_NPZ_ROOT, monkeypatch, row_id))
        for side in ("left", "right"):
            before.append(_dir_cos(Ph, T, "pelvis", f"{side}_hip"))
            after.append(_dir_cos(Pn, T, "pelvis", f"{side}_hip"))
    assert len(before) == 42                                 # positive control
    # The old anchor holds this edge in a NARROW BAND -- 0.9665..0.9940, median
    # 0.9789 -- rather than at 1.0: the frame construction reproduces the hip
    # LINE exactly, but `pelvis->hip` is a different vector from any of its
    # three axes and the target hips do not sit on the reconstructed offset.
    # The npz anchor gives that band up: median 0.9400 (0.9508 before
    # task-relpeak's three flexion rows joined), worst 0.6620 on the pike
    # row, which is that row's 65.9 deg of root-vs-hip-line disagreement
    # showing up in the one edge that can see it.
    assert min(before) == pytest.approx(0.9665, abs=0.002), min(before)
    assert float(np.median(before)) == pytest.approx(0.9789, abs=0.002)
    assert min(after) == pytest.approx(0.6620, abs=0.002), min(after)
    assert float(np.median(after)) == pytest.approx(0.9400, abs=0.002)
    assert float(np.median(after)) < float(np.median(before))


# --------------------------------------------------------------------------
# 4. The fallback, and the position machinery.
# --------------------------------------------------------------------------

def test_a_row_without_rotations_keeps_the_hip_line_anchor_bit_identically():
    """ACCEPTANCE 6. `mhr_rots=None` has no root rotation to read, so the v15
    hip-line construction stays -- and stays BIT-identical, not merely close.

    Read off `_anchor_deltas` itself, where the bit-identity claim actually
    lives: the None path must evaluate the SAME expression, not an equivalent
    one. (`solve_rig_locals`'s emitted local is that delta composed with the
    rest local and then decomposed again, so it round-trips at ~1e-16 -- real,
    and not what this test is about. It is asserted separately below, in
    degrees.)

    Compared against the frame construction computed here from the rig and the
    targets, which is what `test_pelvis_facing.py` pins the shipped behaviour
    of. Positive control: the SAME row solved WITH its rotations lands
    somewhere else entirely, so this equality cannot be an artefact of a solve
    that ignores the switch."""
    for row_id in (PIKE_ROW, CRAWL_ROW, DEV_ROW):
        T = rig_targets_from_mhr70(_kp(row_id))
        got = PG._anchor_deltas(RIG, T, WREST, None)[I["pelvis"]]
        fr = PG._orthonormal_frame_from_hips_and_up(
            RIG.rest_world_p, I["left_hip"], I["right_hip"], I["spine_1"])
        ft = PG._orthonormal_frame_from_hips_and_up(
            T, I["left_hip"], I["right_hip"], I["spine_1"])
        want = QM.from_matrix(np.column_stack(ft[:3]) @ np.column_stack(fr[:3]).T)
        assert np.asarray(got, float).tobytes() == np.asarray(want, float).tobytes(), (
            f"{row_id[:8]}: the None path is no longer the hip-line construction "
            f"({_between(got, want):.3e} deg apart)")
        emitted = _pelvis_delta(solve_rig_locals(RIG, T))
        assert _between(emitted, want) < 1e-12, _between(emitted, want)
        with_rots = _pelvis_delta(solve_rig_locals(RIG, T, mhr_rots=_rots(row_id)))
        assert _between(emitted, with_rots) > 1.0            # positive control


def test_the_pelvis_position_machinery_is_untouched(monkeypatch):
    """The design's other half: ATTITUDE only. `pelvisPosition` is the rig's
    own rest position (ruling 10) and `groundY` is read off the targets; this
    change may not perturb either.

    Positive control: the pelvis QUATERNION in the same emitted state differs
    between the two sources, so "unchanged" is not "the state never varied"."""
    for row_id in (PIKE_ROW, CRAWL_ROW):
        kp, rots = _kp(row_id), _rots(row_id)
        monkeypatch.setattr(PG, "PELVIS_SOURCE", PG.PELVIS_SOURCE_HIPS)
        sh = rig_state_from_mhr70(kp, rots)
        monkeypatch.setattr(PG, "PELVIS_SOURCE", PG.PELVIS_SOURCE_NPZ_ROOT)
        sn = rig_state_from_mhr70(kp, rots)
        assert sh["pelvisPosition"] == sn["pelvisPosition"]
        assert sh["groundY"] == sn["groundY"]
        assert sh["retargetVersion"] == sn["retargetVersion"] == 17
        assert sh["pose"]["pelvis"] != sn["pose"]["pelvis"]   # positive control


def test_an_unrecognised_pelvis_source_raises(monkeypatch):
    """The same guard `SPINE_SOURCE` carries, for the same reason: falling
    through to the hip line on a typo would ship a different pelvis than the
    constant names, undetectably."""
    monkeypatch.setattr(PG, "PELVIS_SOURCE", "npz-root")      # hyphen, not underscore
    with pytest.raises(ValueError, match="PELVIS_SOURCE"):
        solve_rig_locals(RIG, rig_targets_from_mhr70(_kp(DEV_ROW)),
                         mhr_rots=_rots(DEV_ROW))


# --------------------------------------------------------------------------
# 5. What the pelvis->hip edge is actually measuring, and what the whole
#    assembled body does. Both found by LOOKING at the render (CLAUDE.md
#    rule 2) and then measuring, because the first draft of that picture
#    made the change look like a regression and it is not.
# --------------------------------------------------------------------------

def test_the_hip_LINE_survives_and_the_pelvis_to_hip_offset_is_a_rest_gap(monkeypatch):
    """Why `test_the_pelvis_to_hip_edge_is_the_accepted_trade` is NOT the hip
    line going wrong.

    Three rest-geometry facts, read off the two committed assets:

      MHR rest `r_upleg->l_upleg` vs the rig's `right_hip->left_hip`:  0.00 deg
      MHR rest `root->l_upleg`   vs the rig's `pelvis->left_hip`:     25.00 deg

    The two skeletons agree EXACTLY on where the hip line points at rest and
    disagree by 25 deg on where the pelvis NODE sits relative to it -- the rig
    puts its pelvis well above the hip line (105.2 deg between pelvis->left_hip
    and pelvis->right_hip in rest; ruling 9's own note), MHR's root does not.

    So composing MHR's root delta onto the rig's pelvis reproduces the hip LINE
    and re-places the pelvis NODE. `pelvis->hip` measures the 25 deg asset gap;
    the hip line measures the pose. Here the hip line lands within 0.17 deg of
    kp70 on every fixture row -- against 0.00 for the old anchor, which built
    itself from that line and so reproduces it by construction.

    Positive control: `Delta(root)` explains MHR's OWN posed hip line, taken
    from `pred_joint_coords`, to under 0.01 deg -- so the 0.17 above is the
    kp70/model gap and not a transfer error."""
    L_UPLEG, R_UPLEG, MHR_ROOT_J = 2, 18, 1
    rest = PG.load_mhr_rest()
    mhr_hip = _unit(rest["rest_p_cm"][L_UPLEG] - rest["rest_p_cm"][R_UPLEG])
    rig_hip = _unit(RIG.rest_world_p[I["left_hip"]] - RIG.rest_world_p[I["right_hip"]])
    assert float(mhr_hip @ rig_hip) == pytest.approx(1.0, abs=1e-4), float(mhr_hip @ rig_hip)

    mhr_ph = _unit(rest["rest_p_cm"][L_UPLEG] - rest["rest_p_cm"][MHR_ROOT_J])
    rig_ph = _unit(RIG.rest_world_p[I["left_hip"]] - RIG.rest_world_p[I["pelvis"]])
    gap = float(np.degrees(np.arccos(np.clip(float(mhr_ph @ rig_ph), -1, 1))))
    assert gap == pytest.approx(25.00, abs=0.05), gap

    worst_line, worst_own = 0.0, 0.0
    for row_id in sorted(ROWS):
        T = rig_targets_from_mhr70(_kp(row_id))
        P = _positions(_at(PG.PELVIS_SOURCE_NPZ_ROOT, monkeypatch, row_id))
        worst_line = max(worst_line, float(np.degrees(np.arccos(np.clip(
            _dir_cos(P, T, "right_hip", "left_hip"), -1, 1)))))
        # ...and the model's own root against the model's own posed hips.
        P_mhr = np.asarray(ROWS[row_id]["pred_joint_coords"], float) * _CAM_TO_RIG_POS
        posed = _unit(P_mhr[L_UPLEG] - P_mhr[R_UPLEG])
        got = _unit(QM.rotate_vector(_model_root(row_id), mhr_hip))
        worst_own = max(worst_own, float(np.degrees(np.arccos(
            np.clip(float(got @ posed), -1, 1)))))
    assert worst_line < 0.2, worst_line
    assert worst_own < 0.01, worst_own                       # positive control


def test_the_assembled_body_lands_closer_to_its_keypoints(monkeypatch):
    """ACCEPTANCE 4, the metric the render is a picture OF: whole-body
    position error against kp70.

    Eighteen body joints, scale-free (RMS distance as a fraction of the rig's
    own leg length) with the TARGETS aligned at the pelvis -- the rig root,
    whose position this change does not touch, so both solves are scored in
    one frame. This is the honest positional summary that
    `pelvis->hip` alone is not.

    Fixture (21 rows, spine default rel_peak): median 0.1893 -> 0.0731,
    worst 0.3789 -> 0.1049, better on 18 of 21 and never worse by more than
    0.014. The relpeak flip moved BOTH columns -- the deeper chest curl
    costs the hip-line pelvis on its worst rows (before-max 0.2988 ->
    0.3789) and pays the shipped anchor back double (after-max 0.1847 ->
    0.1049: the pike body finally lands on its keypoints). 18-row
    task-pelvis figures under rel_total were 0.1489 -> 0.0733.
    Corpus (1800, task-pelvis, under rel_total): median 0.0871 -> 0.0574,
    p90 0.2163 -> 0.0850, better on 1562 rows."""
    body = ["left_hip", "right_hip", "left_knee", "right_knee", "left_ankle",
            "right_ankle", "spine_1", "spine_2", "neck", "head",
            "left_clavicle", "right_clavicle", "left_shoulder",
            "right_shoulder", "left_elbow", "right_elbow", "left_wrist",
            "right_wrist"]

    def _leg(P):
        return float(np.mean([
            np.linalg.norm(P[I["left_knee"]] - P[I["left_hip"]])
            + np.linalg.norm(P[I["left_ankle"]] - P[I["left_knee"]]),
            np.linalg.norm(P[I["right_knee"]] - P[I["right_hip"]])
            + np.linalg.norm(P[I["right_ankle"]] - P[I["right_knee"]])]))

    def _rms(P, T):
        s = _leg(P) / _leg({k: np.asarray(v, float) for k, v in T.items()})
        off = P[I["pelvis"]] - np.asarray(T[I["pelvis"]], float) * s
        d = [np.linalg.norm(P[I[n]] - (np.asarray(T[I[n]], float) * s + off))
             for n in body]
        return float(np.sqrt(np.mean(np.square(d)))) / _leg(P)

    before, after = [], []
    for row_id in sorted(ROWS):
        T = rig_targets_from_mhr70(_kp(row_id))
        before.append(_rms(_positions(_at(PG.PELVIS_SOURCE_HIPS, monkeypatch, row_id)), T))
        after.append(_rms(_positions(_at(PG.PELVIS_SOURCE_NPZ_ROOT, monkeypatch, row_id)), T))
    assert len(before) == 21                                 # positive control
    assert float(np.median(before)) == pytest.approx(0.1893, abs=0.002)
    assert float(np.median(after)) == pytest.approx(0.0731, abs=0.002)
    assert max(before) == pytest.approx(0.3789, abs=0.002)   # the pike row
    assert max(after) == pytest.approx(0.1049, abs=0.002)
    improved = sum(a < b for a, b in zip(after, before))
    assert improved == 18, improved
    assert max(a - b for a, b in zip(after, before)) < 0.015
