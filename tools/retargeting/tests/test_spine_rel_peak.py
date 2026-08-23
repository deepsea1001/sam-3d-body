"""SPINE_SOURCE_REL_PEAK: the total is the column's PEAK, not its end.

The defect this mapping exists for, measured on row 0693dd37 (the deep pike
fold): MHR's cumulative root-relative rotation along its own spine column --
root, c_spine0, c_spine1, c_spine2, c_spine3 -- is 0, 41.9, 79.0, 87.8,
66.5 deg. The column is NON-MONOTONIC: it curls to 87.8 deg at c_spine2 and
counter-rotates 21 deg back by c_spine3. REL_TOTAL reads ONLY the end, so
exactly on flexion rows (extension is monotonic -- end == peak -- which is
why extension looked right) it drops the column's real curl, and the
mannequin's long chest segment (spine_2->neck is 53% of its column) amplifies
the loss into the too-upright look Scott reported. 490 of the 1800-row
motion corpus (27.2%) are non-monotonic like it.

REL_PEAK keeps EVERYTHING else about REL_TOTAL -- the root-relative frame,
the composition onto the solved pelvis, the 65/35 geodesic split -- and
changes one thing: the total is the LARGEST rotation along the sampled
column (33 uniform arc samples, slerped between the cumulative deltas at the
column joints; ties go to the largest t). By construction peak >= end, and a
monotonic column lands within sampling resolution of rel_total's end:
fixture peak-vs-end is 87.4/66.5 on the pike, 61.4/50.1 on the crouch
(e622f027), 67.0/66.9 on the fold (d3359029), 33.1/31.4 on the bridge
(b657df59).

Arc positions come from `load_mhr_rest()["rest_p_cm"]` at runtime -- never
hardcoded -- so they track the committed rest fixture: root 0.0, c_spine0
0.070, c_spine1 0.272, c_spine2 0.476, c_spine3 0.812 of the root->c_neck
arc.

Quaternions are the solver's [w,x,y,z]; conversions happen only at
serialization.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from retargeting.core.math_utils import QuaternionMath as QM
from retargeting.retargeters import posegoblin_rig as PG
from retargeting.retargeters.posegoblin_rig import (
    fk_world_orientations, load_mhr_rest, load_rig, rig_targets_from_mhr70,
    solve_rig_locals)

ROWS = json.loads(
    (Path(__file__).parent / "fixtures" / "mhr_npz_rows.json").read_text())["rows"]

PIKE_ROW = "0693dd37755e0d6eb9012857f5e3b405"     # peak 87.4 vs end 66.5
CROUCH_ROW = "e622f0279e7c43e4a01fe0663decbea4"   # peak 61.4 vs end 50.1
FOLD_ROW = "d33590290bd4cb7f082c33213d8c56c9"     # peak 67.0 vs end 66.9
BRIDGE_ROW = "b657df59fed7c9b1adc9afc70937ae8b"   # peak 33.1 vs end 31.4
MONO_ROW = "1e6a7a60b4539a8fd09867d7e2fa3db6"     # strictly monotonic column
MHR_ROOT, MHR_SPINE3 = 1, 37

RIG = load_rig()
I = RIG.index_of_name
WREST = fk_world_orientations(RIG, RIG.rest_local_q)

_conj = QM.conjugate


def _ang(q):
    q = np.asarray(q, float)
    if q[0] < 0:
        q = -q
    return float(np.degrees(2 * np.arctan2(np.linalg.norm(q[1:]), q[0])))


def _rots(row_id):
    return np.asarray(ROWS[row_id]["joint_global_rots"], float)


def _solve(row_id):
    return solve_rig_locals(RIG, rig_targets_from_mhr70(
        np.asarray(ROWS[row_id]["kp70"], np.float32).reshape(70, 3)),
        mhr_rots=_rots(row_id))


def _at(source, monkeypatch, row_id):
    monkeypatch.setattr(PG, "SPINE_SOURCE", source)
    return _solve(row_id)


def _world(L):
    return fk_world_orientations(RIG, {**RIG.rest_local_q, **L})


def _rel_delta(rots, row):
    """MHR's rotation at *row* RELATIVE TO ITS OWN ROOT."""
    return QM.multiply(_conj(PG._mhr_delta_q(rots, MHR_ROOT)),
                       PG._mhr_delta_q(rots, row))


def _chest_total(L):
    """The solved chest-vs-pelvis rotation angle -- what the eye reads as a
    curved back, and the quantity every spine ruling was measured on."""
    W = _world(L)
    d_pelvis = QM.multiply(W[I["pelvis"]], _conj(WREST[I["pelvis"]]))
    d_chest = QM.multiply(W[I["spine_2"]], _conj(WREST[I["spine_2"]]))
    return _ang(QM.multiply(_conj(d_pelvis), d_chest))


def _local_delta(L, name):
    return QM.multiply(np.asarray(L[I[name]], float),
                       _conj(RIG.rest_local_q[I[name]]))


# --------------------------------------------------------------------------
# 1. The defect and the fix, on the row it was reported from.
# --------------------------------------------------------------------------

def test_the_default_chest_total_is_the_interior_peak_on_the_pike_row(
        monkeypatch):
    """THE POINT, under the shipped default: the solved chest total on the
    pike row is the column's 87.6 deg peak, not the 66.5 deg end.

    Written to FAIL under SPINE_SOURCE_REL_TOTAL, which reads the end and
    renders this fold 21 deg too upright.

    The end is asserted as the positive control, from the same fixture
    arithmetic AND from a rel_total solve -- so 'the peak' cannot be an
    artifact of the measurement and rel_total's own behavior is pinned
    unchanged beside it."""
    rots = _rots(PIKE_ROW)
    end = _ang(_rel_delta(rots, MHR_SPINE3))
    assert end == pytest.approx(66.5, abs=0.1)           # the fixture premise

    total = _chest_total(_solve(PIKE_ROW))               # the DEFAULT
    assert total == pytest.approx(87.6, abs=1.0), total
    assert total > end + 15.0, (
        f"chest total {total:.1f} deg is not the interior peak -- the end is "
        f"{end:.1f} and the column curls 21 deg past it")

    # Positive control: rel_total, selected by name, still reads the end.
    t_end = _chest_total(_at(PG.SPINE_SOURCE_REL_TOTAL, monkeypatch, PIKE_ROW))
    assert t_end == pytest.approx(end, abs=1e-6), t_end


def test_the_crouch_row_reads_its_peak_too(monkeypatch):
    """The second interior-peak acceptance row: peak 61.5, end 50.1."""
    rots = _rots(CROUCH_ROW)
    end = _ang(_rel_delta(rots, MHR_SPINE3))
    assert end == pytest.approx(50.1, abs=0.1)           # the fixture premise
    total = _chest_total(_at(PG.SPINE_SOURCE_REL_PEAK, monkeypatch, CROUCH_ROW))
    assert total == pytest.approx(61.5, abs=1.0), total
    assert total > end + 8.0, total


# --------------------------------------------------------------------------
# 2. Monotonic columns land on rel_total, within sampling resolution.
# --------------------------------------------------------------------------

def test_monotonic_columns_land_within_sampling_resolution_of_rel_total(
        monkeypatch):
    """Where the column does NOT counter-rotate, the peak IS (near) the end,
    so rel_peak must reproduce rel_total's total -- the fold within 2 deg,
    the bridge (an EXTENSION row: arches are monotonic, which is why
    extension always looked right) within 3.

    Positive control: the pike row differs by >15 deg under the same
    measurement, so agreement here is a property of these columns and not of
    a comparison that cannot see a difference."""
    for row_id, tol in ((FOLD_ROW, 2.0), (BRIDGE_ROW, 3.0)):
        t_peak = _chest_total(_at(PG.SPINE_SOURCE_REL_PEAK, monkeypatch, row_id))
        t_end = _chest_total(_at(PG.SPINE_SOURCE_REL_TOTAL, monkeypatch, row_id))
        assert abs(t_peak - t_end) < tol, \
            f"{row_id[:8]}: peak total {t_peak:.1f} vs end total {t_end:.1f}"
    t_peak = _chest_total(_at(PG.SPINE_SOURCE_REL_PEAK, monkeypatch, PIKE_ROW))
    t_end = _chest_total(_at(PG.SPINE_SOURCE_REL_TOTAL, monkeypatch, PIKE_ROW))
    assert t_peak - t_end > 15.0                          # positive control


def test_a_strictly_monotonic_column_reproduces_the_end_sample(monkeypatch):
    """Tie-break contract, on data: angle ties go to the LARGEST t, so a
    column whose angle never decreases returns the END sample -- which sits
    exactly on rel_total's root-relative row 37.

    Two cases: a real monotonic fixture row (the peak is the endpoint sample
    itself, so the match is exact to slerp-endpoint float noise, far inside
    sampling resolution), and a synthetic CONSTANT column (every joint
    carries the same delta -- every sample past c_spine0 ties, and the last
    must win)."""
    rots = _rots(MONO_ROW)
    peak = PG._rel_peak_q(rots)
    end = _rel_delta(rots, MHR_SPINE3)
    assert _ang(QM.multiply(peak, _conj(end))) < 1e-6, \
        "the monotonic column's peak is not the end sample"

    # Synthetic constant column: rows 34..37 all carry c_spine3's rotation.
    flat = np.array(rots, copy=True)
    for row in (34, 35, 36):
        flat[row] = rots[MHR_SPINE3]
    peak_flat = PG._rel_peak_q(flat)
    end_flat = _rel_delta(flat, MHR_SPINE3)
    assert _ang(QM.multiply(peak_flat, _conj(end_flat))) < 1e-6, \
        "a constant column did not reproduce its end"

    # Positive control: the pike's peak is nowhere near its end.
    assert _ang(QM.multiply(PG._rel_peak_q(_rots(PIKE_ROW)),
                            _conj(_rel_delta(_rots(PIKE_ROW), MHR_SPINE3)))) > 15.0


# --------------------------------------------------------------------------
# 3. The distribution is untouched: 65/35 on every row, same as rel_total.
# --------------------------------------------------------------------------

def test_spine_1_takes_exactly_65_percent_under_rel_peak(monkeypatch):
    """The peak changes WHAT total is spent, never HOW: spine_1's share of
    the local bend is `_SPINE1_SHARE` to machine precision on every fixture
    row, exactly as under rel_total -- the geodesic split is untouched.

    Positive control: rel_perjoint's share still ranges freely over the same
    rows, so the 0.65 is a property of the split and not of the fixture."""
    shares, ref = [], []
    for row_id in ROWS:
        for src, sink in ((PG.SPINE_SOURCE_REL_PEAK, shares),
                          (PG.SPINE_SOURCE_REL_PERJOINT, ref)):
            L = _at(src, monkeypatch, row_id)
            a1, a2 = _ang(_local_delta(L, "spine_1")), _ang(_local_delta(L, "spine_2"))
            assert a1 + a2 > 5.0, f"{row_id[:8]}: {src} bend too small to divide"
            sink.append(a1 / (a1 + a2))
    assert len(shares) == 21                                  # positive control
    assert max(abs(s - PG._SPINE1_SHARE) for s in shares) < 1e-9, \
        f"share range {min(shares):.9f}..{max(shares):.9f}"
    assert max(abs(s - PG._SPINE1_SHARE) for s in ref) > 0.1


# --------------------------------------------------------------------------
# 4. The sampled column: resolved by name, arc from the rest fixture, cached.
# --------------------------------------------------------------------------

def test_the_column_is_resolved_by_name_with_arc_from_the_rest_fixture():
    """The five column rows and c_neck come from the rest fixture's `names`
    -- never hardcoded indices -- and the arc fractions are the cumulative
    rest segment lengths re-derived here independently from `rest_p_cm`.
    Cached like `load_mhr_rest` itself: one dict per process.

    The approx pins (0.070/0.272/0.476/0.812) are the measured geometry of
    the committed fixture; if a re-extracted rest ever moves them, this
    fails loudly instead of the peak silently sampling a different spine."""
    col = PG._spine_column()
    assert PG._spine_column() is col, "not cached: the column is re-derived per call"

    rest = load_mhr_rest()
    names = rest["names"]
    want_names = ("root", "c_spine0", "c_spine1", "c_spine2", "c_spine3")
    assert tuple(names[j] for j in col["rows"]) == want_names
    assert names[col["neck_row"]] == "c_neck"

    p = rest["rest_p_cm"]
    chain = list(col["rows"]) + [col["neck_row"]]
    cum = [0.0]
    for a, b in zip(chain, chain[1:]):
        cum.append(cum[-1] + float(np.linalg.norm(p[b] - p[a])))
    want = np.asarray(cum[:-1], float) / cum[-1]
    assert np.allclose(col["arc"], want, atol=1e-12), (col["arc"], want)
    assert np.allclose(col["arc"], [0.0, 0.070, 0.272, 0.476, 0.812], atol=0.001), \
        col["arc"]
    assert PG._SPINE_PEAK_SAMPLES == 33


# --------------------------------------------------------------------------
# 5. Blast radius against rel_total: the chest itself moves -- on purpose.
# --------------------------------------------------------------------------

def test_rel_peak_moves_the_chest_and_the_column_moves_with_it(monkeypatch):
    """What flipping rel_total -> rel_peak changes on a non-monotonic row, as
    an exact set of bone indices -- and what it deliberately does NOT.

    Unlike the rel_perjoint -> rel_total flip (which redistributed a fixed
    chest), the peak MOVES THE CHEST: spine_2's world carries the extra curl,
    so the neck, head and both clavicles ride with it in WORLD terms, and
    the locals that are expressed against moved frames absorb the change --
    spine_2's own, the head's (its world anchor is fixed while its parent
    moved), both clavicles' (through the world-aimed correction) and both
    shoulders'. The NECK's local is deliberately absent from that set: the
    npz neck is chest-relative (test_npz_neck.py), so the chest cancels out
    of it algebraically -- same invariant as the cap-0 clavicle transfer.
    The shoulders' WORLD orientations must not move -- they come from the
    arm keypoints. The pelvis and every limb are untouched entirely.

    On a monotonic row the two mappings agree to sampling resolution
    (asserted above), so this set is asserted on the pike row, where the
    difference is the point."""
    tot = _at(PG.SPINE_SOURCE_REL_TOTAL, monkeypatch, PIKE_ROW)
    pk = _at(PG.SPINE_SOURCE_REL_PEAK, monkeypatch, PIKE_ROW)
    assert set(tot) == set(pk)
    changed = {i for i in tot if not np.allclose(tot[i], pk[i], atol=1e-9)}
    expected = {I[n] for n in ("spine_1", "spine_2", "head",
                               "left_clavicle", "right_clavicle",
                               "left_shoulder", "right_shoulder")}
    assert changed == expected, (
        f"unexpected: {sorted(RIG.name[i] for i in changed - expected)}, "
        f"missing: {sorted(RIG.name[i] for i in expected - changed)}")
    wt, wp = _world(tot), _world(pk)
    for name in ("spine_1", "spine_2"):
        assert _ang(QM.multiply(wp[I[name]], _conj(wt[I[name]]))) > 5.0, name
    for name in ("left_shoulder", "right_shoulder"):
        spun = _ang(QM.multiply(wp[I[name]], _conj(wt[I[name]])))
        assert spun < 1e-6, f"{name}'s WORLD orientation moved {spun:.3e} deg"
    assert _ang(QM.multiply(wp[I["pelvis"]], _conj(wt[I["pelvis"]]))) < 1e-12
