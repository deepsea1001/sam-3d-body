"""Byte-identity re-extraction control for EVERY row in mhr_npz_rows.json.

77dbfd2 ("feat(rig): commit the non-monotonic pike row to the npz fixture")
appended row 0693dd37 to mhr_npz_rows.json on the strength of a control
described only in that commit's message: "all sixteen already-committed rows
re-extract byte-for-byte first ... and nothing is appended unless they do."
No extraction script and no fixture-integrity test were committed alongside
it -- `grep -rn "mhr_npz_rows"` across both repos finds consumers only -- so
the one control that licensed appending row 17 was unreproducible from a
checkout (task-reltotal-review.md SHOULD-FIX 2). This file is that control,
committed: it re-derives each fixture row's four fields from the SAME corpus
(review-rerun/motion/shards, motion-diverse-1k -- see the fixture's own
"provenance" key) the rows were originally pulled from, under the rounding
reverse-engineered from the committed values, and asserts the result is
IDENTICAL to what is on disk.

It covers ALL rows, not just the appended one, because that is precisely the
claim each append is licensed by: row 18 (1c3ba88d, the crawl row behind the
npz pelvis anchor, task-pelvis 2026-08-23) was appended only after all
seventeen prior rows re-extracted byte-for-byte, exactly as row 17 was. A
control that only ever re-checks the newest row could not have caught the
append reformatting or perturbing an older one.

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
_FIXTURE_ROWS = json.loads(FIXTURE.read_text())["rows"]
ROW_IDS = sorted(_FIXTURE_ROWS)
PIKE_ROW_ID = "0693dd37755e0d6eb9012857f5e3b405"
CRAWL_ROW_ID = "1c3ba88d8b32b3c20a458eb5512ee3f8"

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


_CORPUS_ROWS: dict | None = None


def _corpus_rows() -> dict:
    """The corpus, read once per session and keyed by point_id -- 1800 rows
    off five parquet shards is seconds, and this file asks for eighteen of
    them."""
    global _CORPUS_ROWS
    if _CORPUS_ROWS is None:
        rows = _READ_SHARDS(str(_CORPUS))
        assert rows, f"read_shards returned nothing from {_CORPUS}"
        _CORPUS_ROWS = {r["point_id"]: r for r in rows}
    return _CORPUS_ROWS


def _extract_row(row_id: str) -> dict[str, np.ndarray]:
    """Re-extracts *row_id*'s four fixture fields from the corpus.

    kp70 <- mhr70_xyz reshaped (70,3), round 6dp. cam_t <- unrounded.
    joint_global_rots/pred_joint_coords <- decoded from the row's
    mhr_params_npz blob (np.savez_compressed), round 7dp / 6dp respectively.
    """
    by_id = _corpus_rows()
    assert row_id in by_id, (
        f"{row_id} not among {len(by_id)} rows read from {_CORPUS} -- "
        f"wrong corpus (expected motion-diverse-1k's review-rerun shards)."
    )
    row = by_id[row_id]

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


def _fixture_row(row_id: str) -> dict[str, np.ndarray]:
    row = _FIXTURE_ROWS[row_id]
    return {k: np.asarray(row[k], dtype=np.float64)
            for k in ("kp70", "cam_t", "joint_global_rots", "pred_joint_coords")}


def test_the_two_appended_rows_are_actually_in_the_fixture():
    """Runs with or without the corpus, so a fixture that silently lost the
    pike or the crawl row cannot read as a clean skip. Eighteen rows as of
    2026-08-23."""
    assert PIKE_ROW_ID in _FIXTURE_ROWS
    assert CRAWL_ROW_ID in _FIXTURE_ROWS
    assert len(ROW_IDS) == 18, sorted(r[:8] for r in ROW_IDS)


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
def test_every_fixture_row_reextracts_byte_identical():
    """The control 77dbfd2's commit message described but never committed
    (task-reltotal-review.md SHOULD-FIX 2), generalised to every row: re-derive
    the row from the corpus it was extracted from and require EXACT
    (np.array_equal, not np.isclose) agreement with what is on disk in
    mhr_npz_rows.json, field by field.

    This is the control each append is licensed by. 1c3ba88d (the crawl row,
    task-pelvis) was added only after the seventeen rows preceding it passed
    this, exactly as 0693dd37 was added after sixteen."""
    checked = 0
    for row_id in ROW_IDS:
        extracted = _extract_row(row_id)
        fixture = _fixture_row(row_id)
        for key, fixture_val in fixture.items():
            extracted_val = extracted[key]
            assert extracted_val.shape == fixture_val.shape, (
                f"{row_id[:8]} {key}: shape {extracted_val.shape} != "
                f"fixture {fixture_val.shape}")
            assert np.array_equal(extracted_val, fixture_val), (
                f"{row_id[:8]} {key} NOT byte-identical: maxabsdiff="
                f"{np.max(np.abs(extracted_val - fixture_val)):.3e}")
            checked += 1
    assert checked == 4 * 18, checked        # positive control: the loop ran


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
def test_the_reextraction_control_catches_a_corrupted_byte():
    """Positive control for the test above (CLAUDE.md rule 1): an equality
    check that can never see a mismatch would read as a byte-identity
    control while actually being unable to fail. Flips every bit of one
    byte of one freshly re-extracted (not fixture) float64's in-memory
    representation -- a real corruption shape, not an epsilon nudge -- and
    requires the SAME np.array_equal the test above uses to report it."""
    extracted = _extract_row(PIKE_ROW_ID)
    corrupted = extracted["pred_joint_coords"].copy()
    scalar_bytes = bytearray(corrupted[5, 1].tobytes())
    scalar_bytes[-1] ^= 0xFF                      # flips sign + top exponent bits
    corrupted[5, 1] = np.frombuffer(bytes(scalar_bytes), dtype=np.float64)[0]
    assert not np.array_equal(corrupted, extracted["pred_joint_coords"])  # sanity: corruption took

    fixture = _fixture_row(PIKE_ROW_ID)
    assert not np.array_equal(corrupted, fixture["pred_joint_coords"]), (
        "corrupted a byte and the byte-identity comparison still reported a match")
