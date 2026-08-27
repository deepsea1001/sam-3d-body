"""`modelDriven`: which solver gates actually FIRED, not which were offered.

v18 moved four parts of the solve onto the model's own rotations: the pelvis
takes Delta(root), the spine takes the root-relative PEAK, the clavicles take
the model's chest-relative rotation, and the neck takes c_neck chest-relative.
All four are gated on `mhr_rots`, which comes from the row's `mhr_params_npz`
blob. With no blob every one of them silently drops to its v15 keypoint
construction and the payload still looks well-formed -- that is the defect
this field exists for: a run whose npz never arrived is indistinguishable, at
the wire, from one where the model drove the pose.

CONTRACT
    `modelDriven` is a list, a SUBSET of ("pelvis", "spine", "clavicles",
    "neck"), naming the gates whose model-driven branch actually executed.
    The empty list means full keypoint fallback. It is additive to the
    payload; `retargetVersion` stays 18 because it tracks the ALGORITHM, not
    the payload shape.

THE TRAP THIS FILE EXISTS TO CATCH
    The tempting implementation is

        modelDriven = list(GATES) if mhr_rots is not None else []

    and it is wrong, because two of the four gates are COMPOUND. `clavicles`
    additionally needs `s2 in A`; `neck` additionally needs `s2 in A` and the
    bone actually being named "neck"; and the spine's relative sources need
    `pv0 in A`. Recording what was SUPPLIED rather than what FIRED yields a
    field that is always full and therefore says nothing -- it would
    reintroduce exactly the blindness it was added to remove. So the tests
    below are not satisfied by the presence of the key: controls drive real
    partial fires and pin that the field NARROWS.

    Worth knowing before writing more of them: THE GATES ARE CHAINED. On the
    default path the spine gate is itself what puts spine_2 into `A`, so
    withholding spine_2's TARGET does not clear the compound half of
    `clavicles`/`neck` -- it stays satisfied, and a control built that way
    passes against a hardcoded implementation. Reaching a real partial fire
    means taking the spine's landmark branch (`SPINE_SOURCE_V15`, a supported
    setting) and, for the compound gates, starving that branch of the targets
    it needs too. Both are done below.

    Per-side clavicle skips are legitimate (a re-rigged bone with no single
    child has no defined long axis), so "clavicles" reports that at least one
    side took the model's rotation. That is a deliberate choice of resolution:
    the contract is a flat four-element vocabulary and cannot say "left only".
"""
import json
from pathlib import Path

import numpy as np
import pytest

from retargeting.retargeters import posegoblin_rig as PG
from retargeting.retargeters.posegoblin_rig import (
    MODEL_DRIVEN_GATES, load_rig, rig_state_from_mhr70, rig_targets_from_mhr70)

FIX = Path(__file__).parent / "fixtures"
ROWS = json.loads((FIX / "mhr_npz_rows.json").read_text())["rows"]
DEV_ROW = "a6566802a6c9ddd63340ccb4520e0001"

RIG = load_rig()


def _kp(row_id):
    return np.asarray(ROWS[row_id]["kp70"], np.float32).reshape(70, 3)


def _rots(row_id):
    return np.asarray(ROWS[row_id]["joint_global_rots"], float)


ALL_ROW_IDS = [r for r in ROWS if "kp70" in ROWS[r] and "joint_global_rots" in ROWS[r]]


def test_the_gate_vocabulary_is_exactly_the_four_v18_gates():
    assert tuple(MODEL_DRIVEN_GATES) == ("pelvis", "spine", "clavicles", "neck")


@pytest.mark.parametrize("row_id", ALL_ROW_IDS)
def test_no_npz_means_full_keypoint_fallback(row_id):
    """The whole point: a row with no blob must SAY it drove nothing."""
    state = rig_state_from_mhr70(_kp(row_id), None)
    assert state["modelDriven"] == []


@pytest.mark.parametrize("row_id", ALL_ROW_IDS)
def test_every_reported_gate_is_from_the_vocabulary(row_id):
    state = rig_state_from_mhr70(_kp(row_id), _rots(row_id))
    assert set(state["modelDriven"]) <= set(MODEL_DRIVEN_GATES)


@pytest.mark.parametrize("row_id", ALL_ROW_IDS)
def test_the_report_is_ordered_and_carries_no_duplicates(row_id):
    """A wire field is easier to diff and cache when its order is fixed."""
    got = rig_state_from_mhr70(_kp(row_id), _rots(row_id))["modelDriven"]
    assert len(got) == len(set(got))
    assert got == [g for g in MODEL_DRIVEN_GATES if g in got]


@pytest.mark.parametrize("row_id", ALL_ROW_IDS)
def test_a_complete_npz_row_drives_all_four_gates(row_id):
    """These fixture rows all carry a full blob and a stock rig."""
    state = rig_state_from_mhr70(_kp(row_id), _rots(row_id))
    assert state["modelDriven"] == list(MODEL_DRIVEN_GATES)


def test_retarget_version_is_unchanged_by_an_additive_field():
    state = rig_state_from_mhr70(_kp(DEV_ROW), _rots(DEV_ROW))
    assert state["retargetVersion"] == 18


def _fire(targets, rots):
    fired = set()
    PG._anchor_deltas(RIG, targets, PG.fk_world_orientations(RIG, {}), rots,
                      fired=fired)
    return fired


def test_control_a_gate_that_does_not_run_is_not_reported(monkeypatch):
    """Falsifiability: drive a real partial fire and pin that the field NARROWS.

    `SPINE_SOURCE_V15` is a supported setting of the spine switch -- the whole
    point of that switch is that the alternatives stay reachable -- and it
    takes the spine's LANDMARK anchor while the npz is still fully present.
    So the model does not drive the spine, and an implementation reporting
    what was SUPPLIED rather than what RAN still claims all four here.
    """
    monkeypatch.setattr(PG, "SPINE_SOURCE", PG.SPINE_SOURCE_V15)
    fired = _fire(rig_targets_from_mhr70(_kp(DEV_ROW)), _rots(DEV_ROW))

    assert "spine" not in fired
    assert "pelvis" in fired
    # Still fire: the v15 landmark anchor sets the spine_2 delta these two
    # depend on, so their compound half is satisfied without the spine gate.
    assert "clavicles" in fired
    assert "neck" in fired


def test_control_the_compound_gates_narrow_when_the_spine_2_anchor_is_absent(
        monkeypatch):
    """The other half: `clavicles` and `neck` really are compound.

    Their extra condition is `s2 in A`, which on the default path is satisfied
    by the SPINE gate itself -- the gates are chained, which is why withholding
    spine_2's target alone proves nothing. Take the landmark branch AND starve
    it of the targets it needs, and nothing sets the spine_2 delta, so both
    compound gates drop out while `mhr_rots` is untouched.
    """
    monkeypatch.setattr(PG, "SPINE_SOURCE", PG.SPINE_SOURCE_V15)
    targets = rig_targets_from_mhr70(_kp(DEV_ROW))
    s2 = RIG.index_of_name["spine_2"]
    fired = _fire({i: p for i, p in targets.items() if i != s2}, _rots(DEV_ROW))

    assert "clavicles" not in fired
    assert "neck" not in fired
    assert "pelvis" in fired, "the pelvis gate reads neither spine_2 nor targets"


def test_control_the_field_is_not_a_constant():
    """The three controls above must disagree with the full-set row."""
    full = set(rig_state_from_mhr70(_kp(DEV_ROW), _rots(DEV_ROW))["modelDriven"])
    none_ = set(rig_state_from_mhr70(_kp(DEV_ROW), None)["modelDriven"])
    assert full == set(MODEL_DRIVEN_GATES)
    assert none_ == set()
