"""Byte-identity re-extraction control for the non-monotonic pike row.

77dbfd2 ("feat(rig): commit the non-monotonic pike row to the npz fixture")
appended row 0693dd37 to mhr_npz_rows.json on the strength of a control
described only in that commit's message: "all sixteen already-committed rows
re-extract byte-for-byte first ... and nothing is appended unless they do."
No extraction script and no fixture-integrity test were committed alongside
it -- `grep -rn "mhr_npz_rows"` across both repos finds consumers only -- so
the one control that licensed appending row 17 was unreproducible from a
checkout (task-reltotal-review.md SHOULD-FIX 2). This file is that control,
committed: it re-derives the pike row's four fixture fields from the SAME
corpus (review-rerun/motion/shards, motion-diverse-1k -- see the fixture's
own "provenance" key) the row was originally pulled from, under the rounding
reverse-engineered from the committed values, and asserts the result is
IDENTICAL to what 77dbfd2 committed.

The rounding was not guessed: kp70 and pred_joint_coords round to 6dp,
joint_global_rots to 7dp, cam_t is unrounded (only the float32->float64
round-trip read_shards itself already does) -- each verified independently
against all seventeen fixture rows, maxabsdiff exactly 0.0 per field, before
this file was written.

SKIPPED, not failed, when the corpus is not mounted (CLAUDE.md rule 1: a
skip with a stated reason is not the same failure mode as a silent pass) --
most checkouts will not have this scratch corpus, and that must read as "not
run" rather than as either red or green. Point MHR_NPZ_FIXTURE_CORPUS_DIR at
a directory of *.parquet motion shards to run it for real.
"""
from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mhr_npz_rows.json"
PIKE_ROW_ID = "0693dd37755e0d6eb9012857f5e3b405"

# Where to find the corpus (unset -> skip, the common case for any checkout
# that isn't holding this scratch corpus) and the poseforge3d checkout that
# owns read_shards (hardcoded default mirrors this repo's own retarget.py /
# validate_rig_retarget.py _sam3d_path() convention: a real path, overridable).
_CORPUS_ENV = "MHR_NPZ_FIXTURE_CORPUS_DIR"
_WORKTREE_ENV = "POSEFORGE3D_WORKTREE_PATH"
_DEFAULT_WORKTREE = "/Users/scotteaton/Dropbox/CODE/poseforge3d/.worktrees/mannequin-query"


def _poseforge3d_path() -> None:
    worktree = os.environ.get(_WORKTREE_ENV, _DEFAULT_WORKTREE)
    if worktree not in sys.path:
        sys.path.insert(0, worktree)


def _find_read_shards():
    """Returns poseforge3d.store.shards.read_shards, or None if the
    poseforge3d checkout at _WORKTREE_ENV/_DEFAULT_WORKTREE isn't importable."""
    _poseforge3d_path()
    try:
        from poseforge3d.store.shards import read_shards
    except ImportError:
        return None
    return read_shards


def _corpus_dir() -> Path | None:
    raw = os.environ.get(_CORPUS_ENV, "")
    if not raw:
        return None
    p = Path(raw)
    return p if p.is_dir() else None


_CORPUS = _corpus_dir()
_READ_SHARDS = _find_read_shards() if _CORPUS is not None else None
_SKIP = _CORPUS is None or _READ_SHARDS is None
if _CORPUS is None:
    _SKIP_REASON = (
        f"corpus absent -- set {_CORPUS_ENV} to a directory of *.parquet "
        f"motion shards (the fixture's own provenance names 'rerun-20260821 "
        f"motion shards (motion-diverse-1k)') to run this control; got "
        f"{os.environ.get(_CORPUS_ENV) or '<unset>'!r}"
    )
elif _READ_SHARDS is None:
    _SKIP_REASON = (
        f"poseforge3d.store.shards not importable from "
        f"{os.environ.get(_WORKTREE_ENV, _DEFAULT_WORKTREE)!r} -- set "
        f"{_WORKTREE_ENV} to a poseforge3d checkout"
    )
else:
    _SKIP_REASON = ""


def _extract_pike_row() -> dict[str, np.ndarray]:
    """Re-extracts PIKE_ROW_ID's four fixture fields from the corpus.

    kp70 <- mhr70_xyz reshaped (70,3), round 6dp. cam_t <- unrounded.
    joint_global_rots/pred_joint_coords <- decoded from the row's
    mhr_params_npz blob (np.savez_compressed), round 7dp / 6dp respectively.
    """
    rows = _READ_SHARDS(str(_CORPUS))
    by_id = {r["point_id"]: r for r in rows}
    assert PIKE_ROW_ID in by_id, (
        f"{PIKE_ROW_ID} not among {len(rows)} rows read from {_CORPUS} -- "
        f"wrong corpus (expected motion-diverse-1k's review-rerun shards)."
    )
    row = by_id[PIKE_ROW_ID]

    kp70 = np.round(
        np.asarray(row["mhr70_xyz"], dtype=np.float64).reshape(70, 3), 6)
    cam_t = np.asarray(row["cam_t"], dtype=np.float64)
    npz = np.load(io.BytesIO(row["mhr_params_npz"]))
    joint_global_rots = np.round(
        np.asarray(npz["joint_global_rots"], dtype=np.float64), 7)
    pred_joint_coords = np.round(
        np.asarray(npz["pred_joint_coords"], dtype=np.float64), 6)

    return {
        "kp70": kp70,
        "cam_t": cam_t,
        "joint_global_rots": joint_global_rots,
        "pred_joint_coords": pred_joint_coords,
    }


def _fixture_pike_row() -> dict[str, np.ndarray]:
    fixture = json.loads(FIXTURE.read_text())
    row = fixture["rows"][PIKE_ROW_ID]
    return {k: np.asarray(row[k], dtype=np.float64)
            for k in ("kp70", "cam_t", "joint_global_rots", "pred_joint_coords")}


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
def test_pike_row_reextracts_byte_identical_to_the_committed_fixture():
    """The control 77dbfd2's commit message described but never committed
    (task-reltotal-review.md SHOULD-FIX 2): re-derive row 0693dd37 from the
    corpus it was extracted from and require EXACT (np.array_equal, not
    np.isclose) agreement with what is on disk in mhr_npz_rows.json, field
    by field."""
    extracted = _extract_pike_row()
    fixture = _fixture_pike_row()
    for key, fixture_val in fixture.items():
        extracted_val = extracted[key]
        assert extracted_val.shape == fixture_val.shape, (
            f"{key}: shape {extracted_val.shape} != fixture {fixture_val.shape}")
        assert np.array_equal(extracted_val, fixture_val), (
            f"{key} NOT byte-identical: maxabsdiff="
            f"{np.max(np.abs(extracted_val - fixture_val)):.3e}")


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
def test_the_reextraction_control_catches_a_corrupted_byte():
    """Positive control for the test above (CLAUDE.md rule 1): an equality
    check that can never see a mismatch would read as a byte-identity
    control while actually being unable to fail. Flips every bit of one
    byte of one freshly re-extracted (not fixture) float64's in-memory
    representation -- a real corruption shape, not an epsilon nudge -- and
    requires the SAME np.array_equal the test above uses to report it."""
    extracted = _extract_pike_row()
    corrupted = extracted["pred_joint_coords"].copy()
    scalar_bytes = bytearray(corrupted[5, 1].tobytes())
    scalar_bytes[-1] ^= 0xFF                      # flips sign + top exponent bits
    corrupted[5, 1] = np.frombuffer(bytes(scalar_bytes), dtype=np.float64)[0]
    assert not np.array_equal(corrupted, extracted["pred_joint_coords"])  # sanity: corruption took

    fixture = _fixture_pike_row()
    assert not np.array_equal(corrupted, fixture["pred_joint_coords"]), (
        "corrupted a byte and the byte-identity comparison still reported a match")
