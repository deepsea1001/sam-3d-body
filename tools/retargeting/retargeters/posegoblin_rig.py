"""The PoseGoblin rig as a versioned asset, and the rig-native solve.

Why this exists: the shipped rig rests in an A-pose, authored Z-up, and its
rest directions agree with default_human.json at mean cosine +0.065. No fixed
per-bone correction can bridge two different rest GEOMETRIES (measured spread
61 deg across five hand-matched poses), so poses must be solved against the
rig's own rest. See the design doc named in the plan header.

Bones are keyed by INDEX, not name. PoseGoblin's own captureCurrentState/
RecallPoseCommand key bone state by `state.pose[child.name]`, and one pair of
bones -- the toe-adjacent joint under each heel -- legitimately shares the
literal name "joint7" (both collapse to the same state.pose entry on the live
rig; matching that behaviour, not diverging from it, is correct). Index
identity is therefore load-bearing for FK: `Rig.parent`, `rest_local_q`,
`rest_local_p`, `rest_world_p` and `children` are all index-keyed so the two
`joint7` bones keep independent data instead of one silently overwriting the
other. `index_of_name` resolves the 72 genuinely-unique names to an index and
deliberately omits ambiguous ones -- looking up "joint7" by name raises
KeyError rather than silently returning one of the two.

Ten finger/thumb chains (40 bones counting the tip) were, in v1, `parent:
None` orphans -- disconnected from the rig they are actually rendered on.
v2 (the default since v16 task 3) reconnects them through the ten Group
nodes they hang off in the live scene (`transform4`..`transform13`,
appended at indices 74..83): a fixed rotation plus a 0.1 uniform scale
each, applied to every finger/thumb bone beneath it. FK walks the full
84-node topology now (`Rig.topo_order`, NOT `Rig.order` -- the groups sit
at HIGHER indices than their finger children, so a pass over plain index
order hits an unprocessed parent) and threads that scale through node
POSITION (`fk_world_positions`; orientation is scale-invariant, so
`fk_world_orientations` carries no scale term). The 30 finger PHALANGES
(chain root + two interior joints; the ten tips stay `solve: False`) carry
`solve: True` in the v2 asset -- their chains are reachable from pelvis
now -- but MHR-70 supplies no finger ROTATION data, only raw keypoints that
`rig_targets_from_mhr70` maps onto them like any other bone, so they are
never posed by the generic aim-based path. Nothing is deleted from the
asset, and both versions stay loadable (`load_rig(path=...)`) -- the FK
self-consistency tests pin exactly what's reachable from pelvis and what's
actually solved, so a future re-rig that changes either shows up as a
failing assertion, never as silence.

Those 30 phalanges ARE posed since v16 task 5, when the caller supplies
`mhr_rots` -- from the MHR model's own hand rotations, never from
keypoints. Not by the world-DELTA transfer the spine uses: that is correct
only while the two rigs' rests nearly agree (7.6-31.6 deg for the spine)
and their rest FINGER directions are far apart -- 3-49 deg, median 27, in
each skeleton's own anatomical hand frame, and 3-161 deg before R14 -- which
made it bend fingers sideways and twist them. Instead the joint's BEND
MAGNITUDE -- frame-independent, and so immune to that mismatch -- is
transferred as a signed angle about the rig's OWN measured flexion axis
(`bind_poses/mannequin_finger_axes.json`, extracted from PoseGoblin's
shipped hand presets by `tools_extract_finger_axes.py`; not +X everywhere,
the thumb disproves that). Its sign comes from an anatomical hand frame
each skeleton builds from its own landmarks, so chirality is derived rather
than hardcoded. A per-digit DATA-INTEGRITY gate (`_finger_digits_passing`
-- non-finite, improper rotation, absurd bend; deliberately NOT a
pose-quality judgement) drops a digit back to its rest locals when its
source rotations are unusable, and `solved_indices` answers that gated
question so "which bones does the solver solve" stays one function's
answer. Without `mhr_rots` all 40 finger/thumb bones stay at rest exactly
as in v15.

The spine is posed from the MHR model's OWN joint rotations since v16 task
4, when the caller supplies them (`mhr_rots`); MHR-70 has no spine
keypoints, so without them the column can only be posed by splitting the
pelvis->chest rotation between two anchors, and pelvis-vs-chest pitch is
structurally unobservable ("stiff as a board"). Every corpus row carries
the full parametric solve in its `mhr_params_npz` blob -- see
`mhr_rots_from_npz`, `load_mhr_rest` and `_mhr_delta_q` below. Rows without
the blob still take the v15 path, which is kept intact under `mhr_rots is
None`.

WHICH of the model's rotations, and in what frame, is a switch --
`SPINE_SOURCE`, seven named mappings, so every comparison behind the three
rulings stays reproducible instead of living in a scratch branch. The
default takes the LARGEST rotation RELATIVE TO THE MODEL'S OWN ROOT found
along its sampled spine column -- not the rotation at the column's end,
which under-reports the curl wherever the column counter-rotates back, and
it does so exactly on flexion -- composes that peak onto the pelvis we
actually solved, and splits it 65/35 between the mannequin's two spine
bones. See the block above `SPINE_SOURCE` for the rulings and what each
costs.
"""
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import io
import json
import numpy as np

from ..core.math_utils import QuaternionMath

_ASSET = Path(__file__).resolve().parent.parent / "bind_poses" / "posegoblin_rig_v2.json"
_MHR_REST_ASSET = Path(__file__).resolve().parent.parent / "bind_poses" / "mhr_skeleton_rest.json"
_FINGER_AXES_ASSET = (Path(__file__).resolve().parent.parent / "bind_poses"
                      / "mannequin_finger_axes.json")


@dataclass(frozen=True)
class Rig:
    version: str
    order: list           # asset-order bone INDICES (int), ALL nodes. NOT guaranteed
                          # parent-first under v2 (see topo_order) -- kept as asset
                          # order because state serialization depends on it (ruling 6)
    topo_order: list      # parent-first bone INDICES (int), ALL nodes, groups included.
                          # v2's groups sit at 74..83 while their finger children sit at
                          # 14..57, so this is the order every FK walk must use instead
                          # of `order` (ruling 6)
    name: dict            # index -> bone name (str); NOT unique, see module docstring
    parent: dict          # index -> parent index | None
    rest_local_q: dict    # index -> (4,) [w,x,y,z]
    rest_local_p: dict    # index -> (3,)
    rest_world_p: dict    # index -> (3,)
    scale: np.ndarray     # (N,) float; per-node LOCAL uniform scale. 1.0 everywhere
                          # except the ten v2 groups, which carry 0.1
    is_group: np.ndarray  # (N,) bool; True for the ten constant group nodes (74..83).
                          # solve:False; excluded from the emitted pose -- the viewer
                          # already owns them as a fixed container transform
    children: dict        # index -> [child indices], asset order
    solve: dict           # index -> bool; False for the ten finger/thumb TIPS (and, in
                          # v1, the whole finger/thumb islands)
    index_of_name: dict   # unique name -> index; ambiguous names (e.g. "joint7") omitted


def _topological_order(order: list, parent: dict, children: dict) -> list:
    """Parent-before-child visitation order over every node in *order*
    (ruling 6): v2's index layout is deliberately non-topological -- the
    ten group nodes sit at 74..83 while their finger children sit at
    14..57 -- so a single forward pass over `range(len(order))` raises
    KeyError on exactly those ten chains the moment a parent lookup lands
    on an unprocessed group. Pre-order DFS from each root (`parent is
    None`), visiting children in ascending index order for a deterministic
    result."""
    visited: set = set()
    result: list = []

    def visit(i: int) -> None:
        if i in visited:
            return
        visited.add(i)
        result.append(i)
        for c in children[i]:
            visit(c)

    for i in order:
        if parent[i] is None:
            visit(i)
    # Review finding (v16 task 3): a dangling parent (points at an index
    # never visited as a root or reached as a child) or a cycle would
    # silently yield a SHORT order here -- every downstream FK walk would
    # just quietly cover fewer nodes, caught previously only by a
    # hardcoded set(range(84)) in one test. Fail at the source instead.
    assert set(result) == set(order), (
        f"topological order missed {set(order) - set(result)} -- a "
        "dangling or cyclic parent pointer left some node(s) unreached")
    return result


def load_rig(path: Path = _ASSET) -> Rig:
    d = json.loads(Path(path).read_text())
    bones = d["bones"]
    order = list(range(len(bones)))  # asset position IS the index; NOT necessarily
                                      # parent-first under v2 -- see topo_order
    name = {i: b["name"] for i, b in enumerate(bones)}
    parent = {i: b["parent"] for i, b in enumerate(bones)}
    children: dict = {i: [] for i in order}
    for i, b in enumerate(bones):
        if b["parent"] is not None:
            children[b["parent"]].append(i)
    name_counts = Counter(name.values())
    index_of_name = {b["name"]: i for i, b in enumerate(bones) if name_counts[b["name"]] == 1}
    return Rig(
        version=d["version"], order=order, topo_order=_topological_order(order, parent, children),
        name=name, parent=parent,
        rest_local_q={i: np.asarray(b["rest_local_q"], float) for i, b in enumerate(bones)},
        rest_local_p={i: np.asarray(b["rest_local_p"], float) for i, b in enumerate(bones)},
        rest_world_p={i: np.asarray(b["rest_world_p"], float) for i, b in enumerate(bones)},
        # v1.json predates scale/is_group (every bone implicitly scale 1.0,
        # not a group); .get(..., default) loads both versions uniformly.
        scale=np.array([float(b.get("scale", 1.0)) for b in bones], dtype=float),
        is_group=np.array([bool(b.get("is_group", False)) for b in bones], dtype=bool),
        children=children,
        solve={i: bool(b["solve"]) for i, b in enumerate(bones)},
        index_of_name=index_of_name,
    )


_MHR_REST: dict | None = None


def load_mhr_rest() -> dict:
    """The MHR kinematic skeleton's REST pose, parsed once and cached.

    `bind_poses/mhr_skeleton_rest.json` (v16 task 1) is a committed fixture
    extracted from checkpoints/mhr_model.pt ONCE, offline. Nothing on the
    runtime path may re-derive it: reading the checkpoint means importing
    torch, and this module is imported by the poseforge3d harness and the
    review server, neither of which has any other reason to load a
    multi-gigabyte model. Numpy and json only, therefore, and a
    module-level singleton so a per-row solve does not re-read the file.

    Returns `{"q_wxyz": (127,4) ndarray, "names": [...], "parents": [...],
    "rest_p_cm": (127,3) ndarray}`. `q_wxyz[j]` is joint j's rotation to MHR
    MODEL world -- already accumulated down the chain, in the solver's
    [w,x,y,z] convention (the fixture is authored that way; three.js and the
    PoseGoblin captures use [x,y,z,w]). `rest_p_cm[j]` is that skeleton's
    rest POSITION, accumulated from the fixture's `template_offsets_cm` with
    each joint's PARENT's global rest rotation (`p[j] = p[parent] +
    R_rest[parent] @ offset[j]` -- the convention test_mhr_rest_fixture.py
    pins against measured geometry). Only the v16 finger transfer reads it,
    to build MHR's anatomical hand frame.
    """
    global _MHR_REST
    if _MHR_REST is None:
        d = json.loads(_MHR_REST_ASSET.read_text())
        q = np.asarray(d["rest_global_q_wxyz"], float)
        parents = list(d["parents"])
        offsets = np.asarray(d["template_offsets_cm"], float)
        p = np.zeros_like(offsets)
        for j, par in enumerate(parents):
            p[j] = (offsets[j] if par < 0
                    else p[par] + QuaternionMath.rotate_vector(q[par], offsets[j]))
        _MHR_REST = {
            "q_wxyz": q,
            "names": list(d["names"]),
            "parents": parents,
            "rest_p_cm": p,
        }
    return _MHR_REST


def mhr_rots_from_npz(blob: bytes) -> np.ndarray | None:
    """Decode `joint_global_rots` from a shard's `mhr_params_npz` blob.

    Numpy only -- the blob is a plain `np.savez_compressed` archive of the
    SAM-3D parametric output (poseforge3d `runner._serialize_mhr_params`),
    stored float32; this returns float64 because every consumer here is
    quaternion math against the float64 rest fixture.

    Returns None for empty bytes (a row whose blob was never written) and
    for an archive without the key (the mock estimator writes a valid but
    empty npz) -- both mean "no rotations for this row", and the caller
    falls back to the v15 spine. A PRESENT but wrong-shaped array is a
    different thing entirely and raises, rather than surfacing later as an
    IndexError inside quaternion math.
    """
    if not blob:
        return None
    with np.load(io.BytesIO(blob)) as z:
        if "joint_global_rots" not in z:
            return None
        rots = np.asarray(z["joint_global_rots"], float)
    if rots.shape != (127, 3, 3):
        raise ValueError(
            f"joint_global_rots has shape {rots.shape}, expected (127, 3, 3) -- "
            f"the 127-joint MHR kinematic skeleton this module's rest fixture "
            f"and spine rows are both indexed against")
    return rots


# Rows of the 127-joint MHR kinematic skeleton (names from
# bind_poses/mhr_skeleton_rest.json) that drive the mannequin's two spine
# bones. NOT MHR-70 keypoint indices -- a different ordering entirely.
_MHR_SPINE_1 = 35   # c_spine1
_MHR_SPINE_2 = 37   # c_spine3, NOT the positionally closer c_spine2 (36): the
                    # mannequin's neck and both clavicles hang off spine_2
                    # exactly as the human's hang off c_spine3, so driving
                    # spine_2 from c_spine2 would leave the long
                    # c_spine2->c_spine3 segment's bend to surface at the neck
                    # instead of the mid-back.
_MHR_SPINE_1_HYBRID = 36   # c_spine2 -- spine_1 under SPINE_SOURCE_HYBRID.
                    # The mannequin has TWO spine bones where the human has
                    # four, so the mannequin's spine_1 spans roughly
                    # c_spine0..c_spine2. Rest-vs-rest, the mannequin's
                    # spine_1->spine_2 segment sits 1.6 deg off MHR's
                    # c_spine2->c_spine3 and 7.6 deg off c_spine1->c_spine3.
_MHR_ROOT = 1       # `root`, the MHR joint the mannequin's pelvis corresponds
                    # to; the parent end of the real pelvis->chest rotation.

# The two clavicles, and the MHR joint that is the parent of BOTH of them.
# _MHR_CHEST is c_spine3 -- the same row that drives spine_2, and for the same
# reason: MHR's l_clavicle and r_clavicle hang off c_spine3 exactly as the
# mannequin's two clavicles hang off spine_2. That one-to-one correspondence
# is what makes a CHEST-RELATIVE transfer well posed; see the clavicle block
# in _anchor_deltas.
_MHR_CHEST = _MHR_SPINE_2
_MHR_CLAVICLE_ROW = {"left_clavicle": 74, "right_clavicle": 38}

# task-clavcorrect (2026-08-23): how far the transferred clavicle may then be
# aimed BACK at the kp70 shoulder keypoint.
#
# The transfer below buys an exact clavicle ROTATION at the price of the
# shoulder ROOT: the ball drifts a corpus median 0.516 rig units off its
# keypoint (`clavicle->shoulder` cosine median 0.949). Everything under the
# shoulder is world-anchored from the arm keypoints and stays exact, so the
# whole arm becomes a right direction from a wrong origin -- which is what
# Scott reports as hand/arm OVERSHOOT on crouch and ground-contact poses (rows
# 2ba1b3e0, 38608eb8, 4fe66c92; his captures -39/-40/-41).
#
# WHAT THE RESIDUAL ACTUALLY IS -- measured, because the design premise this
# was written from ("correcting from the transferred pose carries no
# rest-geometry bias; it is chest-position error plus noise") turned out to be
# FALSE. Over 500 corpus rows the correction's rotation axis, taken in the
# posed clavicle's own frame, scatters a median 11.8 deg (left) / 22.1 deg
# (right) about a single fixed axis. That axis is a minimal swing, so it is
# provably confined to the great circle perpendicular to the clavicle's rest
# long axis -- axes uniform on THAT circle scatter a median 79.1 deg (85.5 is
# the null for a full sphere, which this axis never explores). It is a
# near-CONSTANT per-side bias, not noise. Its source is the rig asset:
# the mannequin's two rest `clavicle->shoulder` directions are 36.72 deg from
# being mirror images of each other, while MHR-70's own keypoint pair is 3.09
# deg from mirrored. So this correction is closing an asset asymmetry, and
# every degree of it lands in the LOCAL as protraction -- the exact signature
# f07064f removed. That trade is linear and has no sweet spot: each degree of
# cap buys one degree of aim and costs ~0.95 deg of local protraction.
#
# It is capped and kept anyway, because in the WORLD -- what Scott looks at --
# it is a large improvement, measured on the 16 corpus rows whose kp70 girdle
# is itself within 5 deg of mirrored: the two shoulder balls sit a median
# 24.97 deg from mirrored under the pure transfer (the rig's own rest is
# 26.14, i.e. the transfer inherits the asset defect untouched), 8.66 deg
# after this correction, against 4.12 deg for the keypoints themselves. The
# cap is what stops it going all the way (4.71 deg) and dragging the whole
# rest gap into the local with it.
#
# 15 deg is measured against the residual, not picked round: over 3600 corpus
# clavicles the pre-correction error is median 18.3, p90 27.0, max 41.0 deg.
# THE COSTS ARE REAL and belong next to the number: |clavicle local| rises
# again (fixture row 0693dd37 left 41.9 -> 52.7 deg), and on a row whose
# residual FITS under the cap the clavicle becomes bit-identical to the
# pre-f07064f aim on the same spine (test_spine_source pins one). If the
# local is ever judged to matter more than the ball's position, this constant
# is the whole dial -- 0.0 restores the pure transfer exactly.
#
# Roll-freeness is NOT a property of this number: the correction re-aims
# through `_aim_delta`, so the LOCAL is twist-free about the clavicle's long
# axis at any cap -- tests/test_clavicle_aim_correction.py checks 0, 15, 180.
_CLAV_AIM_CORRECTION_MAX_DEG = 15.0

# Where the mannequin's spine comes from when the caller supplies
# `mhr_rots`. Seven named mappings, one switch, so every comparison in the
# task-6b report stays reproducible instead of living in a scratch branch:
SPINE_SOURCE_MHR = "mhr"        # v16 task 4: spine_1 <- Delta(35), spine_2 <- Delta(37)
SPINE_SOURCE_HYBRID = "hybrid"  # spine_1 <- Delta(36); spine_2 keeps v15's
                                # neck+clavicle construction (its world anchor
                                # is then bit-identical to v15's)
SPINE_SOURCE_V15 = "v15"        # both bones from v15 even with mhr_rots present
                                # -- the fingers and the plumbing still ship
SPINE_SOURCE_REAL_TOTAL = "real_total"
                                # v15's DISTRIBUTION driven by v16's real
                                # TOTAL: the pelvis->chest rotation comes from
                                # the npz, the 65/35 swing split and the
                                # twist-stays-at-the-chest rule are v15's,
                                # untouched. See _anchor_deltas.
SPINE_SOURCE_REL_PERJOINT = "rel_perjoint"
                                # per-joint like SPINE_SOURCE_MHR, but each
                                # bone takes the model's rotation RELATIVE TO
                                # ITS OWN ROOT, composed onto the pelvis we
                                # actually solved -- see _anchor_deltas.
SPINE_SOURCE_REL_TOTAL = "rel_total"
                                # REL_PERJOINT's chest exactly --
                                # the same root-relative Delta(37) composed
                                # onto our pelvis -- with that TOTAL split
                                # 65/35 across the two bones instead of
                                # spine_1 taking a second MHR row of its own.
                                # See _anchor_deltas and the block below.
SPINE_SOURCE_REL_PEAK = "rel_peak"
                                # THE DEFAULT. REL_TOTAL's construction with
                                # one change: the total is the LARGEST
                                # root-relative rotation along the sampled
                                # spine column (`_rel_peak_q`), not the
                                # rotation at its end. The end under-reports
                                # the column's curl wherever it
                                # counter-rotates -- 27% of the motion
                                # corpus, all of it flexion. See the block
                                # below.

# Which MHR row drives spine_1 under SPINE_SOURCE_REL_PERJOINT. 36
# (c_spine2), not the 35 (c_spine1) v16 first shipped: the mannequin has two
# spine bones where the human has four, and it is the SEGMENT c_spine2->
# c_spine3 that corresponds to spine_1->spine_2 -- 1.6 deg apart at rest,
# against 7.6 deg for c_spine1->c_spine3. Driving spine_1 from the joint at
# the base of that segment reproduces its direction to 0.9996 of the 0.9996
# rest ceiling across all fifteen captures; row 35 reaches only 0.9593.
SPINE_PERJOINT_SPINE1_ROW = 36

# THE DEFAULT is SPINE_SOURCE_REL_PEAK. Three separate rulings sit behind
# it: WHICH FRAME the total is read in (task 6b, measured), HOW IT IS
# DISTRIBUTED (Scott, 2026-08-23, from a pose he corrected by hand), and
# WHERE ALONG THE COLUMN it is read (task-relpeak, 2026-08-24, measured and
# render-verified). Reverting any one is one constant -- REL_TOTAL and
# REL_PERJOINT share their chest exactly, so flipping between them changes
# the DISTRIBUTION and nothing else, and REL_PEAK is REL_TOTAL with the
# total read at the column's peak instead of its end, so flipping between
# THOSE changes where the total is read and nothing else.
#
# --- the distribution, and the defect that settled it -------------------
#
# Row 0693dd37, a deep pike fold (captures -37 solver, -38 his correction).
# MHR's root-relative delta at c_spine2 (row 36) is 87.8 deg while at
# c_spine3 (row 37) it is only 66.5: the human's spine curvature is NOT
# monotonic along its own column, and 490 of the 1800 motion corpus rows
# are like it (27.2%; median overshoot 2.3 deg, max 21.3). Per-joint
# transfer must then put all 87.8 deg into spine_1 and drive spine_2
# BACKWARDS to land the chest on row 37 -- measured -20.9 deg of EXTENSION
# on a forward fold, where Scott hand-posed +21.5 of flexion. The chest was
# right; the column read broken.
#
# The TOTAL was right too: 66.5 deg against Scott's own 61.3. So REL_TOTAL
# keeps that total and spends it the way the rig itself does. `_SPINE1_SHARE`
# is not a fudge here -- his six spine-zeroed captures land spine_1 at 0.650
# +/- 0.002 of the total local bend across -37 deg of extension, +76 of
# flexion and 51 of lateral bend. A geodesic split reproduces that exactly,
# on every axis, by construction. Under REL_PERJOINT the same quantity ranges
# 0.598 to 0.967 over the fixture rows.
#
# What that costs, stated plainly: the mannequin's spine_1->spine_2 SEGMENT
# no longer reproduces MHR's c_spine2->c_spine3 direction (it cannot -- a
# 65% interpolation is not an 87.8 deg joint), and test_npz_spine.py pins
# the resulting floor violations by name. The chest end, which carries the
# neck and both clavicles, is untouched.
#
# --- the total (task-6b report) ----------------------------------------
#
# The metric is the TOTAL chest-versus-pelvis rotation -- what the eye reads
# as a curved back -- scored against six captures Scott hand-posed with the
# spine joints ZEROED first, the only spine ground truth in this project
# with no solver output underneath it. Mean error over those six:
#
#     v15 39.3 | mhr(35,37) 38.0 | hybrid 39.3 | real_total 39.3 | REL 28.0
#
# REL_TOTAL scores 28.0 as well, and necessarily: it is a redistribution of
# REL_PERJOINT's chest, and a redistribution cannot move a chest-vs-pelvis
# total. It is the PER-JOINT error that moves, on the clean six: spine_1
# 19.90 -> 18.15, spine_2 18.38 -> 10.67, all-19-bones 17.25 -> 16.76, and
# the realised spine_1 share 0.784 -> 0.650 against Scott's own 0.650.
#
# `real_total` in that table is the same total in the ABSOLUTE frame: it
# drives v15's split from the model's real pelvis->chest rotation but leaves
# spine_2 on v15's landmark anchor, so it inherits v15's chest error exactly.
# Relative-total was the untested cell of that sweep, not a re-run of it.
#
# `hybrid` and `real_total` tie v15 exactly, and necessarily: both leave
# spine_2's world anchor at v15's landmark construction and only
# redistribute bend between the two bones, which cannot move a chest-vs-
# pelvis total at all.
#
# What was actually wrong with SPINE_SOURCE_MHR is a FRAME error, not a
# correspondence error. It applies Delta(row) as an ABSOLUTE world rotation
# while the pelvis is anchored separately from the hip keypoints, so the
# chest lands right in world terms and wrong relative to the pelvis we
# placed -- and the relative bend absorbs the difference. Measured over the
# six: the bend lost against what the model reports correlates with our
# pelvis's disagreement with the model's root at r = +0.973 (0.9 deg
# disagreement -> 0.0 deg lost; 43.8 deg -> 38.1 deg lost). Composing the
# model's ROOT-RELATIVE rotation onto our own pelvis removes that by
# construction, and reproduces the model's total bend magnitude exactly on
# every capture.
#
# Residual error is dominated by something this module cannot fix: the MHR
# model itself estimates about 81% of the bend Scott judges from the same
# image.
#
# Beware the OTHER captures when re-deriving any of this. Everything posed
# before 2026-08-22 16:00 was authored on top of a pose the review server
# had already applied, and that server caches `retargetVersion 15` at
# import -- so those captures are v15's own output plus a partial
# correction, and v15's error against them is a LOWER BOUND, not an
# estimate. Per-joint LOCAL error was reported as confounded in 6b, and the
# confound is real -- the 65/35 split is Scott's posing control, not anatomy,
# so it penalises any method that distributes differently. What 6b could not
# know is that he WANTS that control's distribution in the solve as well; he
# ruled so on 2026-08-23, which is what promotes REL_TOTAL over REL_PERJOINT
# above. `provenance` on every entry in
# tests/fixtures/ground_truth_captures.json says which set a capture is in.
#
# --- the peak (task-relpeak, 2026-08-24) --------------------------------
#
# REL_TOTAL fixed extension and left forward FLEXION rendering too upright.
# Measured cause: MHR's spine column is NON-MONOTONIC in flexion. On the
# pike fixture row 0693dd37 the cumulative root-relative rotation along
# root, c_spine0, c_spine1, c_spine2, c_spine3 is 0, 41.9, 79.0, 87.8,
# 66.5 deg -- the column curls to 87.8 and counter-rotates 21 deg back by
# its end, and the END is all REL_TOTAL reads. 490 of the 1800-row motion
# corpus (27.2%) are like it, every one a flexion row; extension is
# monotonic (end == peak), which is why arches looked right all along. The
# mannequin amplifies what the end drops: its chest segment (spine_2->neck)
# is 53% of its whole column, so a chest that stops at 66.5 deg reads as an
# upright back no matter what spine_1 does below it.
#
# The peak reads the column's real curl where the end orientation
# under-reports it: `_rel_peak_q` samples the root-relative rotation at 33
# uniform arc positions along the column (slerped between the cumulative
# deltas at the five column joints, arc positions from the committed rest
# fixture -- see _spine_column) and takes the sample with the LARGEST
# angle, ties to the largest t. By construction peak angle >= end angle,
# and a monotonic column lands within sampling resolution of REL_TOTAL
# (fixture: fold 67.0 vs 66.9 deg, bridge 33.1 vs 31.4). Render-verified on
# the pike, crouch and fold rows before this shipped.
#
# What it costs, stated plainly: the chest is no longer the model's own
# c_spine3 orientation -- spine_2's world deliberately overshoots it by
# exactly the counter-rotation the end threw away, so the spine_2->neck
# machine edge degrades on non-monotonic rows. That trade is the point: the
# eye reads the curl, and the curl is what the end was dropping.
SPINE_SOURCE = SPINE_SOURCE_REL_PEAK

_SPINE_SOURCES = frozenset({SPINE_SOURCE_MHR, SPINE_SOURCE_HYBRID,
                            SPINE_SOURCE_V15, SPINE_SOURCE_REAL_TOTAL,
                            SPINE_SOURCE_REL_PERJOINT, SPINE_SOURCE_REL_TOTAL,
                            SPINE_SOURCE_REL_PEAK})


# Where the pelvis's WORLD ORIENTATION comes from. Same switch shape as
# SPINE_SOURCE above, and for the same reason: two constructions of one
# anchor, one of them the shipped default, both live and both tested.
#
# THE DEFECT the default exists for. The pelvis was the last anchor solved
# purely from keypoints. `_orthonormal_frame_from_hips_and_up` fixes ONE axis
# from the hip line -- which MHR-70 localises well -- and takes the PITCH
# about it from the hipmid->spine_1 chord, which MHR-70 does not localise at
# all: there are no mid-spine keypoints, `spine_1` is INTERPOLATED by
# MHR70Retargeter.compute_joint_positions, and on a prone, piked or crawling
# body that chord is both short and nearly along the camera ray. The frame
# comes out perfectly orthonormal and pitched wrong, and since every other
# bone composes onto the pelvis, the whole assembled body inherits the error:
# correct relative bends at a wrong global attitude, which is what Scott
# reads off the viewer.
#
# Measured against MHR's own root rotation (`_mhr_delta_q(m, _MHR_ROOT)`),
# over the 1800-row motion corpus: median 8.75 deg, p90 33.48, max 65.90 --
# 45.2% of rows over 10 deg. The two worst named rows are 0693dd37 (deep
# pike, 65.90 -- the corpus maximum) and 1c3ba88d (crawl, 53.24), both in
# tests/fixtures/mhr_npz_rows.json.
#
# THE FIX is one expression: the pelvis's world DELTA is Delta(root)
# outright, so its world ORIENTATION is Delta(root) . pelvis_rest_world. No
# frame conversion -- MHR model space and rig space are the same ROTATIONAL
# frame; see `_mhr_delta_q`'s docstring for the two independent measurements
# of that, and do not insert a flip here for the same reasons stated there.
#
# WHAT IT DOES NOT TOUCH. The pelvis POSITION: `pelvisPosition` is the rig's
# own REST pelvis position (ruling 10) and `groundY` is read off the targets;
# neither reads this delta. And the two RELATIVE transfers downstream --
# SPINE_SOURCE_REL_TOTAL's chest and the clavicles' chest-relative rotation
# -- compose ONTO the pelvis, so `conj(pelvis_world) . chest_world` and
# `conj(chest_world) . clavicle_world` cancel this delta algebraically. The
# curved back and the shoulder girdle are bit-identical under both sources.
# That is exactly what those transfers were built relative FOR (task-reltotal:
# "to survive pelvis error"), and tests/test_pelvis_anchor.py asserts it.
#
# WHAT IT COSTS, stated plainly. `pelvis->hip` as a world DIRECTION was
# reproduced by the old anchor almost by construction and is not any more: it
# now degrades by exactly however far MHR's root sits from the observed hip
# line. Same shape as the clavicle transfer's trade (_CLAV_AIM_CORRECTION_
# MAX_DEG above) -- a world-position edge given up for a rotation the eye
# reads as the body's global attitude. It is reported, not gated.
PELVIS_SOURCE_HIPS = "hips"          # v15..v17: the hip-line frame above
PELVIS_SOURCE_NPZ_ROOT = "npz_root"  # THE DEFAULT: Delta(root), the model's own

# THE DEFAULT since 2026-08-23 (task-pelvis). Flipping this back to
# PELVIS_SOURCE_HIPS restores the previous anchor exactly -- one line, and
# every test for that construction is still live.
PELVIS_SOURCE = PELVIS_SOURCE_NPZ_ROOT

_PELVIS_SOURCES = frozenset({PELVIS_SOURCE_HIPS, PELVIS_SOURCE_NPZ_ROOT})


def _mhr_delta_q(mhr_rots: np.ndarray, row: int) -> np.ndarray:
    """World rotation delta for MHR skeleton *row*, as [w,x,y,z].

        Delta(j) = R_pose(j) @ R_rest(j)^T

    -- directly usable as a RIG-frame world delta, with NO coordinate
    conversion. That is the crux of the v16 spine and it is verified, not
    assumed: the SAM head applies its camera flip diag(1,-1,-1) to
    COORDINATES but not to `joint_global_rots`, and the solver's own cv->rig
    position map (cv_to_yup's -Y composed with `_CV_YUP_TO_RIG`'s -Z) is
    that same matrix -- the two cancel, so MHR model space and rig space are
    the same ROTATIONAL frame. Measured two ways: a bone-direction control
    over 30 corpus rows x 12 body bones lands at min cosine +0.9995 using
    this identity (test_mhr_rest_fixture.py reproduces it on the committed
    fixture), and the global-arch test in test_npz_spine.py recovers a
    planted rotation exactly.

    Do NOT "fix" a mirrored-looking result by inserting a flip here.
    Positions need the map and rotations do not; the two facts are not in
    conflict, they are different quantities. Applying the camera map to
    model-frame rotations produces two upright skeletons pointing 160 deg
    apart -- plausible-looking and wrong.

    A trap for anyone adding an order control (task-pelvis review finding
    1): MHR's rest ROOT quaternion is exactly identity ([1,0,0,0]), so at
    the root Delta(root) == R_pose(root) and this function's composition
    order (R_pose @ R_rest^T, not the reverse) is UNFALSIFIABLE there -- a
    "confirms the order" check run against the root alone is vacuous no
    matter what it reports. Use a joint with non-identity rest instead
    (r_ball, rest 179.22 deg separates the two orders by 162 deg).
    """
    return QuaternionMath.multiply(
        QuaternionMath.from_matrix(np.asarray(mhr_rots, float)[row]),
        QuaternionMath.conjugate(load_mhr_rest()["q_wxyz"][row]))


# 33 samples put neighbours ~0.025 apart in arc fraction -- fine enough that
# a peak landing between two samples costs under half a degree on the worst
# fixture row (87.4 sampled against 87.8 at the c_spine2 node itself), and
# the monotonic-column agreement with SPINE_SOURCE_REL_TOTAL stays inside
# the 2-3 deg acceptance bands.
_SPINE_PEAK_SAMPLES = 33

_SPINE_COLUMN: dict | None = None


def _spine_column() -> dict:
    """The MHR spine column the peak is sampled along, resolved once.

    Rows are looked up by NAME in the rest fixture -- root, c_spine0,
    c_spine1, c_spine2, c_spine3, plus c_neck -- never hardcoded, so a
    re-extracted fixture that reorders joints moves this with it. `arc` is
    each column joint's cumulative rest segment length from `rest_p_cm`,
    normalized by the full root->c_neck arc, so c_spine3 sits at its TRUE
    fraction of the column (0.812 on the committed fixture: the c_spine3->
    c_neck segment is real spine and the peak search must not pretend the
    column ends at 1.0). Module-level singleton like `load_mhr_rest`, whose
    cached dict this derives from.

    Returns `{"rows": [5 ints], "neck_row": int, "arc": (5,) ndarray}`.
    """
    global _SPINE_COLUMN
    if _SPINE_COLUMN is None:
        rest = load_mhr_rest()
        names = rest["names"]
        rows = [names.index(n) for n in
                ("root", "c_spine0", "c_spine1", "c_spine2", "c_spine3")]
        neck = names.index("c_neck")
        p = rest["rest_p_cm"]
        cum = [0.0]
        for a, b in zip(rows, rows[1:] + [neck]):
            cum.append(cum[-1] + float(np.linalg.norm(p[b] - p[a])))
        _SPINE_COLUMN = {
            "rows": rows,
            "neck_row": neck,
            "arc": np.asarray(cum[:-1], float) / cum[-1],
        }
    return _SPINE_COLUMN


def _rel_peak_q(mhr_rots: np.ndarray) -> np.ndarray:
    """The LARGEST root-relative rotation along the spine column, [w,x,y,z].

    MHR's column is non-monotonic on 27% of the motion corpus -- it curls
    past its own end orientation and counter-rotates back, exactly on
    flexion -- so the end (c_spine3, what SPINE_SOURCE_REL_TOTAL reads)
    under-reports the curl there. This reconstructs the column as a
    piecewise geodesic through the cumulative root-relative deltas at the
    five column joints (`cum_j = conj(Delta(root)) . Delta(j)`, identity at
    the root), samples it at `_SPINE_PEAK_SAMPLES` uniform arc positions
    over [0, arc(c_spine3)], and returns the sample with the largest
    rotation angle. Ties go to the LAST sample (largest t), so a flat or
    monotonic column returns the end sample -- which IS rel_total's
    quantity, to slerp-endpoint float noise.
    """
    col = _spine_column()
    arc = col["arc"]
    root_c = QuaternionMath.conjugate(_mhr_delta_q(mhr_rots, col["rows"][0]))
    cums = [np.array([1.0, 0.0, 0.0, 0.0])]
    cums += [QuaternionMath.multiply(root_c, _mhr_delta_q(mhr_rots, j))
             for j in col["rows"][1:]]
    peak, peak_angle = cums[0], -1.0
    for t in np.linspace(0.0, arc[-1], _SPINE_PEAK_SAMPLES):
        k = min(int(np.searchsorted(arc, t, side="right")) - 1, len(arc) - 2)
        q = _slerp(cums[k], cums[k + 1], (t - arc[k]) / (arc[k + 1] - arc[k]))
        angle = 2.0 * float(np.arctan2(np.linalg.norm(q[1:]), abs(q[0])))
        if angle >= peak_angle:
            peak, peak_angle = q, angle
    return peak


def fk_world_orientations(rig: Rig, local_q: dict) -> dict:
    """World orientations for EVERY node in the rig, solved or not (ruling
    2) -- not just the SOLVED set this function used to cover. A node
    absent from *local_q* falls back to its own rest local
    (`local_q.get(i, rig.rest_local_q[i])`), so every existing caller, which
    passes a solved-ONLY dict (`solve_rig_locals`'s own `rig.rest_local_q`
    seed; `rig_state_from_mhr70`'s `{**rig.rest_local_q, **solved}`; the
    poseforge3d harness's `fk_world_positions(rig, solved)`), keeps working
    unchanged -- the unsolved remainder just resolves to rest.

    Traversal follows `rig.topo_order`, never `rig.order`: under v2 the ten
    group nodes sit at indices 74..83 while their finger children sit at
    14..57, so a single pass over `rig.order` would look up an unprocessed
    parent (KeyError) for those ten chains the moment a finger bone is
    reached. Uniform scale never affects orientation, so this function
    carries no scale term -- see `fk_world_positions` for where the v2
    groups' 0.1 scale actually enters the computation.
    """
    W: dict = {}
    for i in rig.topo_order:
        p = rig.parent[i]
        q = np.asarray(local_q.get(i, rig.rest_local_q[i]), float)
        W[i] = q if p is None else QuaternionMath.multiply(W[p], q)
    return W


def fk_world_positions(rig: Rig, local_q: dict) -> dict:
    """World positions for EVERY node in the rig, solved or not (ruling 2)
    -- see `fk_world_orientations` for the local_q fallback-to-rest rule
    and why traversal must follow `rig.topo_order`.

    Scale-aware (v16 task 3): the ten v2 group nodes carry a 0.1 uniform
    scale that shrinks every rest offset beneath them, so position FK
    threads a cumulative scale down the chain --

        cum_scale(i) = cum_scale(parent) * rig.scale[i]
        W_p(i) = W_p(parent) + rotate(W_R(parent), cum_scale(parent) * rest_local_p(i))

    -- rather than the old scale-is-always-1 `P[p] + rotate(Wq[p],
    rest_local_p[i])`. Uniform scale never affects orientation, so
    `fk_world_orientations` (source of W_R above) needs no equivalent term
    -- but dropping this one puts finger tips ten times too far out
    (verified: left_index_finger_1 sits 0.994 rig units from left_wrist;
    without the group's 0.1 scale it lands at 9.94).
    """
    Wq = fk_world_orientations(rig, local_q)
    P: dict = {}
    cum_scale: dict = {}
    for i in rig.topo_order:
        p = rig.parent[i]
        if p is None:
            P[i] = rig.rest_local_p[i].copy()
            cum_scale[i] = float(rig.scale[i])
        else:
            P[i] = P[p] + QuaternionMath.rotate_vector(Wq[p], cum_scale[p] * rig.rest_local_p[i])
            cum_scale[i] = cum_scale[p] * float(rig.scale[i])
    return P


def _kabsch_q(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Least-squares rotation (as [w,x,y,z]) taking unit rows a -> b."""
    u, _s, vt = np.linalg.svd(a.T @ b)
    d = float(np.sign(np.linalg.det(vt.T @ u.T))) or 1.0
    return QuaternionMath.from_matrix(vt.T @ np.diag([1.0, 1.0, d]) @ u.T)


# Limb chains and their hinge sign, per the rig's own contract
# (poseGoblin config/ik-config.js: every chain hingeAxis 'z';
# config/joint-limits.js: elbow z [-148, 20] => flexion is LOCAL -Z,
# knee z [-20.5, 151.5] => flexion is LOCAL +Z; Scott 2026-08-21: "Elbows
# need to rotate around +/-Z. internal/external rotation of shoulder is X").
# The hand-posed captures confirm it: every posed elbow/knee delta from rest
# is Z-dominant to two decimals. sign s below maps the bend-plane normal
# n = unit(u x f) onto the mid-joint's LOCAL +Z world image: rotating the
# upper segment u onto the lower f is +theta about n; a flexion of -theta
# about local +Z is +theta about -(local +Z), so elbows need +Z -> -n.
_LIMB_CHAINS = (
    ("left_shoulder", "left_elbow", "left_wrist", -1.0),
    ("right_shoulder", "right_elbow", "right_wrist", -1.0),
    ("left_hip", "left_knee", "left_ankle", +1.0),
    ("right_hip", "right_knee", "right_ankle", +1.0),
)
_MIN_BEND_SIN = 0.05     # < ~3 deg bend: no stable bend plane; generic solve
_PLANE_TRUST_SIN = 0.26  # < ~15 deg bend: plane too noisy for LEG twist; foot decides
_SPINE1_SHARE = 0.65     # lumbar share of the pelvis->chest rotation (Scott's captures)
# Ankle roll bias: Scott's pick from the roll ladder on row c5cf2013
# (2026-08-22) -- the detector's tiny foot labels carry a sole-roll bias
# only an eye can calibrate. Applied about the solved foot axis, ramped by
# pointedness so flat planted feet (long approved) stay untouched. Lives in
# rig_state_from_mhr70, never in solve_rig_locals: the rest-roundtrip and
# the machine gate stay pure. One-row calibration -- revisit per his eye.
_ANKLE_ROLL_ENABLED = False   # DISABLED per Scott 2026-08-22 ("leave the code in
                              # place") pending review of more pointed-toe poses;
                              # flip to True to restore his calibration below.
_ANKLE_ROLL_BIAS_DEG = 20.0   # Scott dialed down from his initial +30 ladder pick
_ANKLE_ROLL_RAMP = (0.25, 0.60)   # pointedness: 0 bias below, full above


def _unit_or_none(v):
    n = np.linalg.norm(v)
    return None if n < _FRAME_EPS else np.asarray(v, float) / n


def _slerp(q0, q1, t):
    """Geodesic interpolation between [w,x,y,z] quaternions."""
    q0 = np.asarray(q0, float); q1 = np.asarray(q1, float)
    d = float(np.dot(q0, q1))
    if d < 0.0:
        q1, d = -q1, -d
    if d > 0.9995:                      # nearly parallel: lerp + normalise
        out = q0 + t * (q1 - q0)
        return out / np.linalg.norm(out)
    th = np.arccos(np.clip(d, -1.0, 1.0))
    return (np.sin((1 - t) * th) * q0 + np.sin(t * th) * q1) / np.sin(th)


def _axis_angle_q(axis, angle):
    axis = np.asarray(axis, float)
    h = angle / 2.0
    return np.array([np.cos(h), *(np.sin(h) * axis)])


def _twist_to_meet_cone(w, axis, f_t, c, prefer):
    """Angle psi rotating w about *axis* so dot(rotated_w, f_t) == c.

    Rotating the upper segment's swing brings the hinge to w; the remaining
    freedom is twist about the segment (*axis*). A pure local-Z bend keeps
    the hinge/lower-segment angle at its rest value (cos = c), so the twist
    must land the hinge on that cone. Two solutions exist; the one whose
    hinge lies on the anatomical bend side (max dot with *prefer*) wins.
    Unreachable c is clamped to the nearest attainable value. Returns None
    only when the geometry is degenerate."""
    wpar = np.dot(w, axis) * axis
    wperp = w - wpar
    A_ = float(np.dot(wperp, f_t))
    B_ = float(np.dot(np.cross(axis, wperp), f_t))
    C_ = c - float(np.dot(wpar, f_t))
    R = float(np.hypot(A_, B_))
    if R < 1e-9:
        return None
    base = float(np.arctan2(B_, A_))
    if abs(C_) <= R:
        d = float(np.arccos(np.clip(C_ / R, -1.0, 1.0)))
        cands = (base + d, base - d)
    else:
        cands = (base if C_ > 0 else base + np.pi,)
    best, best_score = None, -2.0
    for psi in cands:
        h = QuaternionMath.rotate_vector(_axis_angle_q(axis, psi), w)
        score = float(np.dot(h, prefer))
        if score > best_score:
            best, best_score = psi, score
    return best


def _pair_delta(a1, a2, b1, b2):
    """World rotation mapping direction pair (a1, a2) exactly onto (b1, b2).

    Orthonormal-frame alignment with Gram-Schmidt on the second vector of
    each pair. Returns None when either pair is degenerate (second vector
    parallel to the first) -- callers fall back to the generic solve rather
    than fabricate an axis.
    """
    out = []
    for v1, v2 in ((a1, a2), (b1, b2)):
        e1 = np.asarray(v1, float)
        n1 = np.linalg.norm(e1)
        if n1 < _FRAME_EPS:
            return None
        e1 = e1 / n1
        e2 = np.asarray(v2, float) - np.dot(v2, e1) * e1
        n2 = np.linalg.norm(e2)
        if n2 < 1e-3:
            return None
        e2 = e2 / n2
        out.append(np.column_stack([e1, e2, np.cross(e1, e2)]))
    return QuaternionMath.from_matrix(out[1] @ out[0].T)


def _swing_about(q, axis):
    """*q* with its twist about *axis* removed -- the swing half of a
    swing-twist decomposition, in the projection form
    VectorMath.swing_twist_decompose uses.

    `q = swing . twist`, so the swing takes *axis* exactly where q takes it
    and carries no rotation about it. Used to drop the model's clavicle ROLL
    before transferring the rest of its rotation: the mannequin's clavicle has
    no roll degree of freedom to receive one (Scott's rig, and `_aim_delta`'s
    contract since fb42e71), and rotation about a bone's own long axis is the
    component neither skeleton's shoulder position can see anyway.

    A 180 deg swing leaves the twist genuinely undefined (the twist part
    collapses to zero norm); *q* is returned unchanged there rather than
    fabricating an axis, which is the same rule _pair_delta follows.
    """
    q = np.asarray(q, float)
    if q[0] < 0:
        q = -q
    axis = np.asarray(axis, float)
    n = np.linalg.norm(axis)
    if n < 1e-9:
        return q
    axis = axis / n
    tw = np.array([q[0], *(np.dot(q[1:], axis) * axis)])
    ntw = np.linalg.norm(tw)
    if ntw < 1e-9:
        return q
    return QuaternionMath.multiply(q, QuaternionMath.conjugate(tw / ntw))


def _min_swing_q(a, b):
    """Minimal rotation taking unit-ish *a* onto *b*, EXACTLY.

    QuaternionMath.from_two_vectors returns the identity whenever the two
    vectors are within acos(0.9999) = 0.81 deg of each other. That shortcut
    is harmless where the rotation is large and pays for itself where the
    cross product is degenerate -- but `_aim_delta` composes its swing on
    top of the PARENT's delta, so the swing it needs is usually the SMALL
    residual, and dropping it silently loses the whole aim. Measured over
    1200 clavicle aims on the motion corpus: the shortcut fires on 6 and
    costs up to 0.74 deg of aim error, which is 0.02 rig units at the
    shoulder. The general branch of from_two_vectors -- w = 1 + dot, xyz =
    cross -- is well conditioned all the way down to dot = 1 (it tends to
    the identity smoothly), so it is used unconditionally here; only the
    genuinely singular anti-parallel case still needs its own branch.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return QuaternionMath.identity()
    a, b = a / na, b / nb
    d = float(np.dot(a, b))
    if d < -0.9999:
        perp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, perp)
        axis = axis / np.linalg.norm(axis)
        return np.array([0.0, axis[0], axis[1], axis[2]])
    q = np.array([1.0 + d, *np.cross(a, b)])
    return q / np.linalg.norm(q)


def _capped_q(q, max_deg):
    """*q* with its rotation ANGLE clamped to *max_deg*, same axis, same sign.

    Clamping the angle and keeping the axis is the only reduction that stays
    on the geodesic between the two directions the rotation was built from --
    scaling a quaternion's components, or slerping toward identity, both do
    the same thing for a pure rotation, but only this form makes the applied
    angle readable and therefore assertable to 1e-6 (which is what
    test_a_huge_residual_clamps_at_exactly_the_cap needs).

    Below the cap *q* comes back untouched, so a correction that already fits
    is not perturbed by passing through here.
    """
    q = np.asarray(q, float)
    if q[0] < 0:
        q = -q
    n = float(np.linalg.norm(q[1:]))
    if n < 1e-12:
        return q                       # identity: nothing to clamp
    half = float(np.arctan2(n, q[0]))
    lim = float(np.radians(max_deg)) / 2.0
    if half <= lim:
        return q
    axis = q[1:] / n
    return np.array([np.cos(lim), *(np.sin(lim) * axis)])


def _aim_delta(rest_dir, target_dir, d_parent):
    """World delta aiming *rest_dir* at *target_dir* with NO roll about that
    axis in the bone's LOCAL rotation.

    The single-child case. One direction pair leaves rotation about the aim
    axis unconstrained -- moving a bone about the very axis that points at
    its child leaves the child exactly where it was -- so no reading of the
    data prefers any value, and the only question left is which value is
    least harmful. That is a question about the LOCAL: the local is what the
    rig's joint limits constrain and what PoseGoblin's IK reads.

    The minimal WORLD rotation (`from_two_vectors(rest_dir, target_dir)`,
    what this branch used through v16) answers it badly. Being roll-free in
    WORLD means the local -- `Wr(b)^-1 (D(p)^-1 D(b)) Wr(b)`, see
    solve_rig_locals -- must roll BACK by whatever roll the PARENT carries
    about this axis. An unobservable degree of freedom then ends up fixed
    entirely by a bone the aim has nothing to do with.

    Measured on Scott's arms-overhead capture pair (2026-08-22; poseforge3d
    `captures/capture-07299b5a…-25.json`, the solver's own output, against
    `-26.json`, his hand correction): D(spine_2) is 163.9 deg and carries
    130.6 / 119.0 deg of roll about the left / right clavicle's long axis,
    and the clavicle LOCALS come out at 100.9 / 135.2 deg of almost pure
    roll where Scott poses 6.1 / 4.1 -- while the two poses agree about
    where the arm goes to within 5.6 / 13.1 deg. His IK then drives the
    clavicle into its joint limits and throws the arm behind the body.

    Composing the swing ON TOP of the parent's delta makes the LOCAL
    roll-free instead: with D'(b) = swing(D(p).rest_dir -> target_dir) .
    D(p), the relative rotation D(p)^-1 D'(b) is the minimal rotation taking
    rest_dir to D(p)^-1 target_dir -- axis perpendicular to rest_dir by
    construction, hence zero twist about it. Nothing else moves: the aim is
    exact (D'(b).rest_dir == target_dir), and the two deltas differ only by
    a rotation about rest_dir, which is the direction of the child's own
    rest offset, so every world POSITION below the bone is unchanged.

    *d_parent* None (a root, or a parent the caller holds no delta for)
    falls back to the world-minimal rotation: with no parent frame there is
    nothing to be roll-free relative TO.
    """
    if d_parent is None:
        return _min_swing_q(rest_dir, target_dir)
    swing = _min_swing_q(QuaternionMath.rotate_vector(d_parent, rest_dir),
                         target_dir)
    return QuaternionMath.multiply(swing, d_parent)


_ANKLE_FOOT = {
    "left_ankle": ("left_heel", "left_big_toe", "left_small_toe"),
    "right_ankle": ("right_heel", "right_big_toe", "right_small_toe"),
}


def _try_ankle_delta(rig: Rig, targets: dict, Wr: dict, D: dict, n: int):
    """World delta posing the ankle as Rz(theta)*Rx(psi) ONLY -- never Y.

    Scott (2026-08-21): "ankle is incorrect - axis is +/-Z"; the rig's own
    limits agree (ankle: z [-81, 106] plantar/dorsiflexion, x [-75, 75]
    inversion/eversion, no y entry), and every posed capture ankle is a Z or
    Z+X composition. The free-orientation Kabsch this replaces invented Y
    twist from the heel/toe fit.

    Aims the foot axis (heel -> toe midpoint): psi is solved so the axis's
    local-Z height is reachable (Rz preserves z), theta closes the remaining
    angle in the XY plane. Exact whenever the target pitch lies inside the
    rig's reachable cone; clamped to the nearest pitch otherwise. Returns
    None (caller falls back to the generic solve) if the foot targets are
    absent or the geometry is degenerate."""
    names = _ANKLE_FOOT.get(rig.name[n])
    p = rig.parent[n]
    if names is None or p is None or p not in D:
        return None
    li = rig.index_of_name
    hi, bi, si = (li[x] for x in names)
    if not all(k in targets for k in (hi, bi, si)):
        return None
    rest = rig.rest_world_p
    axis_rest = _unit_or_none(0.5 * (rest[bi] + rest[si]) - rest[hi])
    axis_tgt = _unit_or_none(0.5 * (np.asarray(targets[bi], float) + np.asarray(targets[si], float))
                             - np.asarray(targets[hi], float))
    if axis_rest is None or axis_tgt is None:
        return None
    v_w, u_w = axis_rest, axis_tgt
    pre = QuaternionMath.multiply(D[p], Wr[n])          # ankle frame before its own delta
    v = QuaternionMath.rotate_vector(QuaternionMath.conjugate(Wr[n]), v_w)
    u = QuaternionMath.rotate_vector(QuaternionMath.conjugate(pre), u_w)
    R = float(np.hypot(v[1], v[2]))
    if R < 1e-6:
        return None
    phi0 = float(np.arctan2(v[1], v[2]))
    c = float(np.clip(u[2], -R, R))                     # unreachable pitch: clamp
    d = float(np.arccos(np.clip(c / R, -1.0, 1.0)))
    # Both psi branches aim the foot axis exactly; they are +/- mirrors of
    # the sole's roll, scored by the toe-line + heel evidence. NOTE
    # (2026-08-22): Scott reports the roll reads reversed on real images; a
    # blanket branch inversion was tried and REJECTED -- it breaks the
    # rest-roundtrip sanity (rest evidence is correct evidence), proving
    # the rule cannot be a global flip. Resolution pending his A/B pick on
    # live rows; see the session ledger.
    toe_rest = _unit_or_none(rest[bi] - rest[si])
    toe_tgt = _unit_or_none(np.asarray(targets[bi], float)
                            - np.asarray(targets[si], float))
    heel_rest = _unit_or_none(rest[hi] - rest[n])
    heel_tgt = _unit_or_none(np.asarray(targets[hi], float)
                             - np.asarray(targets[n], float))
    best = None
    for psi in (phi0 + d, phi0 - d):
        psi = float(np.arctan2(np.sin(psi), np.cos(psi)))
        w = np.array([v[0],
                      v[1] * np.cos(psi) - v[2] * np.sin(psi),
                      v[1] * np.sin(psi) + v[2] * np.cos(psi)])
        if np.hypot(w[0], w[1]) < 1e-6 or np.hypot(u[0], u[1]) < 1e-6:
            continue                                    # foot axis along the hinge
        theta = float(np.arctan2(u[1], u[0]) - np.arctan2(w[1], w[0]))
        delta = QuaternionMath.multiply(
            _axis_angle_q(np.array([0.0, 0.0, 1.0]), theta),
            _axis_angle_q(np.array([1.0, 0.0, 0.0]), psi))
        d_world = QuaternionMath.multiply(
            D[p], QuaternionMath.multiply(
                Wr[n], QuaternionMath.multiply(delta, QuaternionMath.conjugate(Wr[n]))))
        score = 0.0
        if toe_rest is not None and toe_tgt is not None:
            score += float(np.dot(
                QuaternionMath.rotate_vector(d_world, toe_rest), toe_tgt))
        if heel_rest is not None and heel_tgt is not None:
            score += 0.25 * float(np.dot(
                QuaternionMath.rotate_vector(d_world, heel_rest), heel_tgt))
        if best is None or score > best[0]:
            best = (score, d_world)
    return None if best is None else best[1]


def _anchor_deltas(rig: Rig, targets: dict[int, np.ndarray], Wr: dict,
                   mhr_rots: np.ndarray | None = None) -> dict[int, np.ndarray]:
    """World deltas for the anchored bones: pelvis (hip frame), the four limb
    chains (bend-plane hinge), the head (nose + eye line) and the two spine
    bones. Facing and twist become constraints here; the generic per-bone
    direction solve cannot see either.

    *mhr_rots*, when given, is a (127,3,3) `joint_global_rots` array
    (`mhr_rots_from_npz`) and drives the spine directly -- see the spine
    block below. None keeps the v15 construction, for rows with no blob."""
    li = rig.index_of_name
    rest = rig.rest_world_p
    A: dict[int, np.ndarray] = {}

    # A row with no `mhr_params_npz` blob has no root rotation to read, so it
    # takes the hip-line construction whatever the switch says -- resolved
    # here, exactly as `source` is for the spine below, so both branches read
    # ONE variable. See PELVIS_SOURCE for the defect this exists for.
    pelvis_source = PELVIS_SOURCE if mhr_rots is not None else PELVIS_SOURCE_HIPS
    if pelvis_source not in _PELVIS_SOURCES:
        raise ValueError(
            f"PELVIS_SOURCE is {pelvis_source!r} -- expected one of "
            f"{sorted(_PELVIS_SOURCES)}. Falling through to the hip line on a "
            f"typo would ship a different pelvis than the constant names, "
            f"undetectably -- and every bone in the rig composes onto it.")
    if pelvis_source == PELVIS_SOURCE_NPZ_ROOT:
        # The model's own root rotation, as a rig-frame world delta. The
        # pelvis ORIENTATION only; its POSITION stays where ruling 10 put it.
        A[li["pelvis"]] = _mhr_delta_q(mhr_rots, _MHR_ROOT)
    elif all(k in targets for k in (li["left_hip"], li["right_hip"], li["spine_1"])):
        fr = _orthonormal_frame_from_hips_and_up(
            rest, li["left_hip"], li["right_hip"], li["spine_1"])
        ft = _orthonormal_frame_from_hips_and_up(
            targets, li["left_hip"], li["right_hip"], li["spine_1"])
        A[li["pelvis"]] = QuaternionMath.from_matrix(
            np.column_stack(ft[:3]) @ np.column_stack(fr[:3]).T)

    for root, mid, end, sgn in _LIMB_CHAINS:
        ri, mi, ei = li[root], li[mid], li[end]
        if not all(k in targets for k in (ri, mi, ei)):
            continue
        u_t = _unit_or_none(np.asarray(targets[mi], float) - np.asarray(targets[ri], float))
        f_t = _unit_or_none(np.asarray(targets[ei], float) - np.asarray(targets[mi], float))
        if u_t is None or f_t is None:
            continue
        u_r = _unit_or_none(rest[mi] - rest[ri])
        f_r = _unit_or_none(rest[ei] - rest[mi])
        z_r = QuaternionMath.rotate_vector(Wr[mi], np.array([0.0, 0.0, 1.0]))
        if u_r is None or f_r is None:
            continue
        n_raw = np.cross(u_t, f_t)
        bend_sin = float(np.linalg.norm(n_raw))
        foot = _ANKLE_FOOT.get(rig.name[ei])
        if bend_sin < _PLANE_TRUST_SIN and foot is not None:
            # Near-straight LEG: the bend plane is noise (measured: knee bends
            # under 5 deg put 24/29 corpus heels below the floor, q10 cosine
            # 0.31), but a straight leg's twist is exactly what orients the
            # foot -- so the foot axis chooses it instead of the plane.
            hi2, bi2, si2 = (li[x] for x in foot)
            if all(k in targets for k in (hi2, bi2, si2)):
                fa_r = _unit_or_none(0.5 * (rest[bi2] + rest[si2]) - rest[hi2])
                fa_t = _unit_or_none(
                    0.5 * (np.asarray(targets[bi2], float) + np.asarray(targets[si2], float))
                    - np.asarray(targets[hi2], float))
                if fa_r is not None and fa_t is not None:
                    swing = QuaternionMath.from_two_vectors(u_r, u_t)
                    a0 = QuaternionMath.rotate_vector(swing, fa_r)
                    pa = a0 - np.dot(a0, u_t) * u_t
                    pb = fa_t - np.dot(fa_t, u_t) * u_t
                    if np.linalg.norm(pa) > 1e-3 and np.linalg.norm(pb) > 1e-3:
                        pa, pb = pa / np.linalg.norm(pa), pb / np.linalg.norm(pb)
                        psi = float(np.arctan2(np.dot(u_t, np.cross(pa, pb)),
                                               np.dot(pa, pb)))
                        d_root = QuaternionMath.multiply(_axis_angle_q(u_t, psi), swing)
                        h = _unit_or_none(QuaternionMath.rotate_vector(d_root, z_r))
                        a = QuaternionMath.rotate_vector(d_root, f_r)
                        if h is not None:
                            a_p = a - np.dot(a, h) * h
                            b_p = f_t - np.dot(f_t, h) * h
                            if np.linalg.norm(a_p) > 1e-6 and np.linalg.norm(b_p) > 1e-6:
                                a_p, b_p = (a_p / np.linalg.norm(a_p),
                                            b_p / np.linalg.norm(b_p))
                                phi = float(np.arctan2(
                                    np.dot(h, np.cross(a_p, b_p)), np.dot(a_p, b_p)))
                                A[ri] = d_root
                                A[mi] = QuaternionMath.multiply(
                                    _axis_angle_q(h, phi), d_root)
            continue
        if bend_sin < _MIN_BEND_SIN:
            continue                      # near-straight arm: hinge undefined
        prefer = sgn * n_raw / bend_sin
        # The rig's own rest cone: the angle between the hinge and the lower
        # segment is preserved by any pure-Z bend, so the shoulder/hip twist
        # must place the hinge where the cone passes through the target.
        c = float(np.dot(z_r, f_r))
        swing = QuaternionMath.from_two_vectors(u_r, u_t)
        w = QuaternionMath.rotate_vector(swing, z_r)
        psi = _twist_to_meet_cone(w, u_t, f_t, c, prefer)
        if psi is None:
            d_root = _pair_delta(u_r, z_r, u_t, prefer)   # degenerate: frame fallback
            d_mid = _pair_delta(f_r, z_r, f_t, prefer)
            if d_root is not None:
                A[ri] = d_root
            if d_mid is not None:
                A[mi] = d_mid
            continue
        d_root = QuaternionMath.multiply(_axis_angle_q(u_t, psi), swing)
        h = QuaternionMath.rotate_vector(d_root, z_r)
        a = QuaternionMath.rotate_vector(d_root, f_r)
        a_p = a - np.dot(a, h) * h
        b_p = f_t - np.dot(f_t, h) * h
        na, nb = np.linalg.norm(a_p), np.linalg.norm(b_p)
        if na < _FRAME_EPS or nb < _FRAME_EPS:
            A[ri] = d_root
            continue                      # forearm along hinge: bend angle undefined
        a_p, b_p = a_p / na, b_p / nb
        phi = float(np.arctan2(np.dot(h, np.cross(a_p, b_p)), np.dot(a_p, b_p)))
        A[ri] = d_root
        A[mi] = QuaternionMath.multiply(_axis_angle_q(h, phi), d_root)

    hi, ni_ = li["head"], li["nose"]
    le, re_ = li["left_eye"], li["right_eye"]
    if all(k in targets for k in (hi, ni_, le, re_)):
        d = _pair_delta(rest[ni_] - rest[hi], rest[le] - rest[re_],
                        np.asarray(targets[ni_], float) - np.asarray(targets[hi], float),
                        np.asarray(targets[le], float) - np.asarray(targets[re_], float))
        if d is not None:
            A[hi] = d

    s2, nk = li["spine_2"], li["neck"]
    # A row with no `mhr_params_npz` blob has no rotations to read, so it
    # takes v15 whatever the switch says. Resolving that here keeps every
    # branch below reading ONE variable.
    source = SPINE_SOURCE if mhr_rots is not None else SPINE_SOURCE_V15
    if source not in _SPINE_SOURCES:
        raise ValueError(
            f"SPINE_SOURCE is {source!r} -- expected one of {sorted(_SPINE_SOURCES)}. "
            f"Falling through to v15 on a typo would ship a different spine than "
            f"the constant names, undetectably.")
    pv0 = li["pelvis"]
    if source in (SPINE_SOURCE_REL_PERJOINT, SPINE_SOURCE_REL_TOTAL,
                  SPINE_SOURCE_REL_PEAK) and pv0 in A:
        # RELATIVE, all three of them.
        #
        # SPINE_SOURCE_MHR applies Delta(row) as an absolute world rotation
        # while our pelvis is anchored independently from the hip keypoints
        # -- so the chest lands right in world terms and wrong relative to
        # the pelvis we actually placed, and the RELATIVE bend (the thing
        # the eye reads as a curved back) absorbs the whole discrepancy.
        # Measured: on the three spine-zeroed captures where our pelvis is
        # furthest from the model's root, the absolute transfer throws away
        # half to two thirds of the bend the model reports. Composing the
        # model's root-relative rotation onto OUR pelvis removes that
        # failure mode by construction.
        #
        # REL_PERJOINT and REL_TOTAL read that composition at c_spine3, from
        # one expression, so the total chest-vs-pelvis rotation -- the
        # metric that chose the relative frame -- and everything hanging off
        # spine_2 (the neck, both clavicles, and the shoulders through them)
        # are bit-identical between THOSE two; only the distribution
        # differs. REL_PEAK reads the column's PEAK instead: the same
        # root-relative frame, sampled along the whole column, because the
        # end alone under-reports the curl wherever the column
        # counter-rotates -- 27% of the corpus, all flexion. On a monotonic
        # column the peak IS the end sample and the three chests agree to
        # sampling resolution. See the SPINE_SOURCE block and _rel_peak_q.
        root_d = QuaternionMath.conjugate(_mhr_delta_q(mhr_rots, _MHR_ROOT))
        if source == SPINE_SOURCE_REL_PEAK:
            chest = QuaternionMath.multiply(A[pv0], _rel_peak_q(mhr_rots))
        else:
            chest = QuaternionMath.multiply(
                A[pv0], QuaternionMath.multiply(root_d,
                                                _mhr_delta_q(mhr_rots, _MHR_SPINE_2)))
        A[s2] = chest
        if source == SPINE_SOURCE_REL_PERJOINT:
            # spine_1 takes its OWN MHR row, c_spine2. Faithful to the
            # model's intermediate joint -- and exactly what breaks when
            # the model's curvature is not monotonic along its column:
            # where |rel(36)| exceeds |rel(37)| (490 of 1800 corpus rows)
            # spine_1 overshoots the total and spine_2 must run BACKWARDS
            # to land on it. See the SPINE_SOURCE block above.
            A[li["spine_1"]] = QuaternionMath.multiply(
                A[pv0], QuaternionMath.multiply(
                    root_d, _mhr_delta_q(mhr_rots, SPINE_PERJOINT_SPINE1_ROW)))
        else:
            # REL_TOTAL and REL_PEAK (the default): spend the total the way
            # the rig itself does. A geodesic split puts both bones on ONE
            # axis, so spine_1 takes `_SPINE1_SHARE` of the angle and
            # spine_2 the rest -- flexion, lateral bend and twist alike,
            # which is what Scott's spine-zeroed captures measure (0.650 +/-
            # 0.002 across every axis). Both bones then bend the same way,
            # always; a non-monotonic column cannot express its
            # counter-rotation in the two bones' SHAPE -- under the peak its
            # curl magnitude survives into the total instead, which is the
            # division of labour on purpose.
            #
            # NOT the swing-only split below: that one deliberately keeps
            # TWIST at the chest because v15's total is a landmark
            # construction dominated by spurious hips-vs-clavicle twist.
            # This total is the model's own rotation, whose twist is real
            # spine twist, and Scott's captures distribute it 65/35 like
            # everything else.
            A[li["spine_1"]] = _slerp(A[pv0], chest, _SPINE1_SHARE)
    elif source == SPINE_SOURCE_MHR:
        # v16 (task 4): the mannequin's spine posed from the MHR model's own
        # joint rotations, replacing the chest construction and the
        # swing-split below. Both are world deltas straight out of
        # `_mhr_delta_q` -- no coordinate conversion, see there. spine_1
        # takes c_spine1 and spine_2 takes c_spine3 (`_MHR_SPINE_2`, not the
        # positionally closer c_spine2). The neck bridge at the end of this
        # function reads `A[s2]` either way and needs no branch of its own.
        A[li["spine_1"]] = _mhr_delta_q(mhr_rots, _MHR_SPINE_1)
        A[s2] = _mhr_delta_q(mhr_rots, _MHR_SPINE_2)
    else:
        # v15's chest anchor, shared by SPINE_SOURCE_V15 and
        # SPINE_SOURCE_HYBRID. Built from OBSERVABLE landmarks -- the neck
        # direction plus the clavicle line -- where a real Delta(37) would
        # transfer a rotation whose reference geometry is the MHR
        # skeleton's, not the mannequin's. spine_2 carries the neck, the
        # head and both arms, so that reference mismatch compounds down
        # four chains; measured, it is where the all-real mapping loses.
        #
        # Rows with no blob reach here too (v15 proper), and MHR-70 has no
        # spine keypoints at all -- which is what makes the swing-split
        # below necessary for them.
        lc, rc = li["left_clavicle"], li["right_clavicle"]
        if all(k in targets for k in (s2, nk, lc, rc)):
            d = _pair_delta(rest[nk] - rest[s2], rest[lc] - rest[rc],
                            np.asarray(targets[nk], float) - np.asarray(targets[s2], float),
                            np.asarray(targets[lc], float) - np.asarray(targets[rc], float))
            if d is not None:
                A[s2] = d

        pv = li["pelvis"]
        if source == SPINE_SOURCE_HYBRID:
            # The hybrid's whole content: spine_1 from the model's own
            # rotation at c_spine2, spine_2 from the landmark anchor above.
            # The swing-split below is skipped entirely -- it exists only to
            # invent a spine_1 for a row with no rotation to read, and its
            # 65% share is a CONSTANT where a real per-pose rotation can
            # express a distribution that varies pose to pose.
            A[li["spine_1"]] = _mhr_delta_q(mhr_rots, _MHR_SPINE_1_HYBRID)
        # Spine flexion (Scott 2026-08-21: "stiff as a board"). Two separable
        # halves, and this code deliberately keeps them separable: WHAT the
        # total pelvis->chest rotation is, and HOW it is distributed across
        # the mannequin's two spine bones.
        #
        # The DISTRIBUTION is not a tunable. Six captures hand-posed with the
        # spine joints ZEROED first -- no solver output underneath them --
        # span -37 deg extension, +76 deg flexion and 49 deg of lateral bend,
        # and every one of them lands spine_1 at 0.650 +/- 0.002 of the total
        # local bend. The mannequin's spine is effectively ONE degree of
        # freedom for bend direction, split 65/35 by the rig itself.
        # `_SPINE1_SHARE` is that behaviour, not a fudge factor: it was
        # originally read off two captures (crouch 60.7/32.8, hoop
        # 37.0/19.9), and the independent six confirm it. Do not touch this
        # FIXED-RATIO construction, and do not "improve" it by driving the
        # two bones independently -- per-joint real rotations cannot
        # reproduce a fixed ratio, which is what SPINE_SOURCE_MHR and
        # SPINE_SOURCE_HYBRID both founder on.
        #
        # That prohibition is scoped to THESE fixed-ratio constructions, not
        # to per-joint transfer in general: the shipped default,
        # SPINE_SOURCE_REL_PERJOINT (above), drives spine_1 and spine_2
        # independently from per-joint MHR rows on purpose. It was measured
        # on TOTAL chest-vs-pelvis rotation error, not on per-joint local
        # error against this 65/35 split -- a metric this split itself
        # confounds, since the split is Scott's own posing habit, not an
        # anatomical constant. On the total-error metric REL_PERJOINT wins
        # (a measured ruling, not an oversight of the paragraph above).
        #
        # The TOTAL is where v15 is genuinely weak, and it is the only thing
        # SPINE_SOURCE_REAL_TOTAL changes.
        elif pv in A and (source == SPINE_SOURCE_REAL_TOTAL or s2 in A):
            if source == SPINE_SOURCE_REAL_TOTAL:
                # v16: the REAL pelvis->chest rotation, straight out of the
                # npz. Same construction as v15's, one joint pair over: the
                # child's world delta expressed relative to the parent's.
                # MHR-70 carries no mid-spine keypoints at all, so v15 has to
                # INFER this total from a straight hipmid->neck chord plus the
                # clavicle line -- which is exactly why the column reads
                # "stiff as a board" (6.5-8.1 deg of bend regardless of pose).
                r_rel = QuaternionMath.multiply(
                    QuaternionMath.conjugate(_mhr_delta_q(mhr_rots, _MHR_ROOT)),
                    _mhr_delta_q(mhr_rots, _MHR_SPINE_2))
                chest = QuaternionMath.multiply(A[pv], r_rel)
            else:
                # v15: inferred from landmarks. `chest` IS A[s2] here, so
                # every fallback below is bit-identical to the pre-refactor
                # code (verified against the fixture rows, not assumed).
                r_rel = QuaternionMath.multiply(QuaternionMath.conjugate(A[pv]), A[s2])
                chest = A[s2]
            if r_rel[0] < 0:
                r_rel = -np.asarray(r_rel, float)
            # Split only the SWING (pitch + lateral) of the pelvis->chest
            # rotation; keep its TWIST about the torso axis concentrated at the
            # chest, as the pre-split solve did. Both anchors take their
            # vertical from the same hipmid->neck chord (no mid-spine keypoints
            # exist), so their relative rotation is dominated by hips-vs-
            # clavicles TWIST -- and distributing twist along the column
            # corkscrews it (Scott's crow-pose report: 59.5 deg about -Y read
            # as an arch bending backwards). Swing alone curves the column the
            # way flexion looks.
            chord = _unit_or_none(np.asarray(targets[nk], float)
                                  - 0.5 * (np.asarray(targets[li["left_hip"]], float)
                                           + np.asarray(targets[li["right_hip"]], float)))
            if chord is not None:
                axis_p = QuaternionMath.rotate_vector(QuaternionMath.conjugate(A[pv]), chord)
                d_par = float(np.dot(r_rel[1:], axis_p))
                tw = np.array([r_rel[0], *(d_par * np.asarray(axis_p, float))])
                ntw = np.linalg.norm(tw)
                if ntw > 1e-9:
                    tw = tw / ntw
                    swing = QuaternionMath.multiply(r_rel, QuaternionMath.conjugate(tw))
                    part = _slerp(np.array([1.0, 0.0, 0.0, 0.0]), swing, _SPINE1_SHARE)
                    A[li["spine_1"]] = QuaternionMath.multiply(A[pv], part)
                else:
                    A[li["spine_1"]] = _slerp(A[pv], chest, _SPINE1_SHARE)
            else:
                A[li["spine_1"]] = _slerp(A[pv], chest, _SPINE1_SHARE)

    # Clavicles: the MODEL's own rotation, CHEST-RELATIVE, composed onto the
    # spine_2 we actually solved. Same transfer pattern as the relative spine
    # above (root-relative onto our pelvis), one joint further out, and for the
    # same reason -- an ABSOLUTE transfer would land the clavicle right in
    # world terms and wrong relative to the chest we placed.
    #
    # What this replaces, and why. Left to the generic path a clavicle is a
    # single-child aim: point the rig's rest clavicle->shoulder direction at
    # MHR-70's shoulder keypoint. That asks one rotation to do two jobs -- to
    # express the pose, AND to close the gap between where the MANNEQUIN's
    # shoulder sits at rest and where MHR's does, 26.5 deg on the left. The
    # gap is a constant of the two skeletons; the rotation that closes it is
    # not, because it depends on the pose and on the parent. So it lands as a
    # different invented swing every row and a different one per side, and the
    # corpus median |clavicle local| of 23 deg IS that bias, not anatomy.
    # Measured on the two rows Scott named (tests/test_clavicle_source.py):
    # on a SYMMETRIC double-biceps pose the aim gave left 24.5 deg mostly
    # forward against right 18.5 deg mostly BACKWARD, where the model's own
    # rotations say 14.6 / 9.5, elevation-dominant and symmetric; on a plain
    # standing pose it swung the left clavicle 33 deg, 31 of it protraction.
    # He reads the result as a dislocated shoulder ball.
    #
    # MHR's clavicles hang off c_spine3, which is also the row driving
    # spine_2, so the correspondence is one-to-one and the relative rotation
    # needs no re-basing between skeletons.
    #
    # SWING-ONLY: the model's twist about the mannequin's own clavicle long
    # axis is stripped first. That axis is unobservable from the shoulder
    # POSITION (see _aim_delta), the rig has no clavicle-roll DOF to receive
    # it, and it is what fb42e71 established must not appear in this local.
    #
    # What that discards is NOT noise, and the size is worth knowing: over all
    # eighteen fixture rows x two sides the model carries a median 7.20 deg of
    # roll about this axis, p90 13.19, max 29.07. It is dropped anyway for two
    # reasons and kept honest by a third:
    #   - there is nowhere for it to go. The rig has no clavicle-roll degree
    #     of freedom, and roll in this local is precisely what drove Scott's
    #     IK into its joint limits (fb42e71).
    #   - the axis is the MANNEQUIN's long axis, which sits 26.5 deg off MHR's,
    #     so part of what is measured here as twist is MHR swing our axis reads
    #     as roll. The figures above are an upper bound on real lost signal,
    #     not an estimate of it.
    #   - test_how_much_model_roll_is_discarded pins the median under 12 deg,
    #     so a future model or re-rig that starts carrying genuinely large
    #     clavicle roll fails loudly instead of being silently thrown away.
    #
    # Without `mhr_rots` there is nothing to transfer and the aim path stays.
    if mhr_rots is not None and s2 in A:
        chest_inv = QuaternionMath.conjugate(_mhr_delta_q(mhr_rots, _MHR_CHEST))
        for bone, row in _MHR_CLAVICLE_ROW.items():
            ci = li[bone]
            child = rig.children[ci]
            if len(child) != 1:
                continue              # re-rigged: the long axis is no longer defined
            axis = _unit_or_none(rest[child[0]] - rest[ci])
            if axis is None:
                continue
            rel = QuaternionMath.multiply(chest_inv, _mhr_delta_q(mhr_rots, row))
            A[ci] = QuaternionMath.multiply(A[s2], _swing_about(rel, axis))
            # ...then aim what that produced back at the shoulder keypoint,
            # by at most _CLAV_AIM_CORRECTION_MAX_DEG -- see that constant for
            # what the residual was measured to BE (a near-constant per-side
            # asset asymmetry, not noise) and what capping it costs.
            #
            # Re-aiming rather than composing, on purpose. `_swing_about(rel,
            # axis)` is already the UNIQUE twist-free rotation that sends
            # `axis` where `rel` sends it (a rotation with a given image of
            # `axis` is twist-free about it only at zero twist), so the delta
            # above IS an aim, at the direction the model's own rotation
            # chose. Feeding a corrected direction back through `_aim_delta`
            # therefore moves the aim and nothing else -- and inherits its
            # contract, so the LOCAL stays roll-free by construction instead
            # of by luck. Composing the correction on top would not: the
            # product of two swings about different axes carries twist.
            #
            # A zero cap reproduces this line's own output exactly, which is
            # what lets the tests measure "before" without a second solver.
            si = child[0]
            if ci in targets and si in targets:
                want = _unit_or_none(np.asarray(targets[si], float)
                                     - np.asarray(targets[ci], float))
                if want is not None:
                    cur = QuaternionMath.rotate_vector(A[ci], axis)
                    corr = _capped_q(_min_swing_q(cur, want),
                                     _CLAV_AIM_CORRECTION_MAX_DEG)
                    A[ci] = _aim_delta(axis,
                                       QuaternionMath.rotate_vector(corr, cur),
                                       A[s2])

    # Neck: two constructions, resolved at the block below the wrists. With
    # `mhr_rots` it is the model's own c_neck, chest-relative; without, the
    # v15 Y-twist half-bridge -- half the chest->head rotation's Y component
    # (Scott 2026-08-21: "head and neck forward tilt in Y"; his captures
    # split a look-down ~half/half -- crouch neck +38 / head +36 -- and
    # every posed neck is pure Y to two decimals). Either way the head
    # anchor above is world-exact, so its local absorbs the remainder
    # exactly.
    # Wrists: palm orientation from the hand keypoints (Scott 2026-08-21:
    # "do we do anything with the wrists? ... or pronation supination?" --
    # before this, the wrist inherited the elbow rigidly and pro/sup was
    # lost). Hand axis (wrist -> middle knuckle) + knuckle line (index_1 <->
    # pinky_1) encode pronation/supination, flexion and deviation together.
    # Rest directions come straight from rest_world_p, which was captured
    # from the live scene and therefore already includes the finger groups'
    # 0.1 scale (fk_world_positions reproduces these too since v16 task 3 --
    # rest_world_p is simply the more direct source, already on hand here).
    # Degenerate/missing knuckles: _pair_delta returns None and the wrist
    # keeps the inherit-elbow fallback.
    for side in ("left", "right"):
        wi = li[f"{side}_wrist"]
        mi2 = li[f"{side}_middle_finger_1"]
        ii2 = li[f"{side}_index_finger_1"]
        pi2 = li[f"{side}_pinky_finger_1"]
        if not all(k in targets for k in (wi, mi2, ii2, pi2)):
            continue
        d = _pair_delta(rest[mi2] - rest[wi], rest[ii2] - rest[pi2],
                        np.asarray(targets[mi2], float) - np.asarray(targets[wi], float),
                        np.asarray(targets[ii2], float) - np.asarray(targets[pi2], float))
        if d is not None:
            A[wi] = d

    hi = li["head"]
    if s2 in A and rig.name[nk] == "neck":
        if mhr_rots is not None:
            # The model's own neck, CHEST-RELATIVE -- the clavicle pattern
            # one joint up, well posed for the same reason: MHR's c_neck
            # hangs off c_spine3 exactly as the mannequin's neck hangs off
            # spine_2. The Y-bridge below was built for rows where twist is
            # all the landmarks can support; the model's c_neck carries
            # 20-50 deg of REAL local bend on deep folds (pike row: 21.9),
            # every degree of which the bridge dropped into the head's
            # local, where it read as a hinged skull on a straight neck.
            # Chest-relative means source-independent: whichever chest the
            # spine switch built, `conj(W(spine_2)) . W(neck)` cancels it,
            # so the neck LOCAL is the model's quantity verbatim
            # (test_npz_neck.py pins both facts). The head anchor stays
            # world-exact above; its local absorbs this exactly as it
            # absorbed the bridge. No `hi in A` requirement -- unlike the
            # bridge, nothing here reads the head.
            A[nk] = QuaternionMath.multiply(A[s2], QuaternionMath.multiply(
                QuaternionMath.conjugate(_mhr_delta_q(mhr_rots, _MHR_SPINE_2)),
                _mhr_delta_q(mhr_rots, _spine_column()["neck_row"])))
        elif hi in A:
            # No rotations: the v15 Y-twist half-bridge, bit-identical.
            r_ln = QuaternionMath.multiply(
                QuaternionMath.conjugate(Wr[nk]),
                QuaternionMath.multiply(
                    QuaternionMath.conjugate(A[s2]),
                    QuaternionMath.multiply(A[hi], Wr[nk])))
            if r_ln[0] < 0:
                r_ln = -np.asarray(r_ln, float)
            kappa = 2.0 * float(np.arctan2(r_ln[2], r_ln[0]))   # Y twist of the total
            dy = _axis_angle_q(np.array([0.0, 1.0, 0.0]), kappa / 2.0)
            A[nk] = QuaternionMath.multiply(
                A[s2], QuaternionMath.multiply(
                    Wr[nk], QuaternionMath.multiply(dy, QuaternionMath.conjugate(Wr[nk]))))

    return A


# Ruling 3 (v16 task 3): the v2 asset flips these 30 finger phalanges (chain
# root + two interior joints per digit; the ten tips stay solve:False) to
# solve:True, since their chains are reachable from pelvis now. But MHR-70
# supplies no finger ROTATION data -- only raw keypoints, which
# rig_targets_from_mhr70 maps onto them like any other bone -- so left in
# `solved` below, the generic aim-based branch would silently start posing
# fingers from noisy, ungated keypoint data purely as a side effect of the
# asset switch. Excluded here by NAME rather than a structural walk: unlike
# "joint7" elsewhere in this rig (module docstring), all 30 of these are
# uniquely named, and solve_rig_locals already resolves everything else by
# name via rig.index_of_name.
#
# v16 task 5 lifts that exclusion CONDITIONALLY: these 30 are solved when
# the caller supplies `mhr_rots` (from the MHR hand rotations, per digit,
# behind the integrity gate below) and stay excluded when it does not. The
# exclusion still lives in exactly one place -- `solved_indices` -- so
# "which bones does the solver solve" remains one function's answer.
_FINGER_PHALANGE_NAMES = frozenset({
    "left_thumb_1", "left_thumb_2", "left_thumb_3",
    "left_index_finger_1", "left_index_finger_2", "left_index_finger_3",
    "left_middle_finger_1", "left_middle_finger_2", "left_middle_finger_3",
    "left_ring_finger_1", "left_ring_finger_2", "left_ring_finger_3",
    "left_pinky_finger_1", "left_pinky_finger_2", "left_pinky_finger_3",
    "right_thumb_1", "right_thumb_2", "right_thumb_3",
    "right_index_finger_1", "right_index_finger_2", "right_index_finger_3",
    "right_middle_finger_1", "right_middle_finger_2", "right_middle_finger_3",
    "right_ring_finger_1", "right_ring_finger_2", "right_ring_finger_3",
    "right_pinky_finger_1", "right_pinky_finger_2", "right_pinky_finger_3",
})


_DIGIT_NAMES = ("thumb", "index", "middle", "ring", "pinky")

# Rows of the 127-joint MHR kinematic skeleton (names from
# bind_poses/mhr_skeleton_rest.json), NOT MHR-70 keypoint indices. All 30
# verified by name against the checkpoint's joint_names, and the thumb
# correspondence two independent ways: mannequin `thumb_1` sits at 0.453 of
# hand span from the wrist vs MHR `l_thumb1` at 0.505 and `l_thumb0` at
# 0.219, and the mannequin's decreasing segment pattern matches the chain
# from `thumb1`. MHR's thumb CMC (`thumb0`) and pinky metacarpal (`pinky0`)
# have no mannequin counterpart; their motion rides inside the `_1` joint,
# whose bend is measured against the WRIST rather than against MHR's own
# parent -- which is also the mannequin's topology (a phalange `_1` hangs
# off its group, and the group off the wrist).
_MHR_WRIST_ROW = {"left": 78, "right": 42}
_MHR_FINGER_ROWS = {
    "left_thumb_1": 97, "left_thumb_2": 98, "left_thumb_3": 99,
    "left_index_finger_1": 92, "left_index_finger_2": 93, "left_index_finger_3": 94,
    "left_middle_finger_1": 88, "left_middle_finger_2": 89, "left_middle_finger_3": 90,
    "left_ring_finger_1": 84, "left_ring_finger_2": 85, "left_ring_finger_3": 86,
    "left_pinky_finger_1": 80, "left_pinky_finger_2": 81, "left_pinky_finger_3": 82,
    "right_thumb_1": 61, "right_thumb_2": 62, "right_thumb_3": 63,
    "right_index_finger_1": 56, "right_index_finger_2": 57, "right_index_finger_3": 58,
    "right_middle_finger_1": 52, "right_middle_finger_2": 53, "right_middle_finger_3": 54,
    "right_ring_finger_1": 48, "right_ring_finger_2": 49, "right_ring_finger_3": 50,
    "right_pinky_finger_1": 44, "right_pinky_finger_2": 45, "right_pinky_finger_3": 46,
}

# Data-integrity gate (ruling R12, as revised). Its ONLY job is to stop
# garbage rotations reaching the rig; it deliberately does not judge pose
# quality. The brief's original per-digit kp70 cosine gate was measured and
# discarded: it dropped 31 of 60 digits, because the cross-rig rest-geometry
# ceiling per digit is 0.258-0.804 (median 0.614) -- that cosine mostly
# measures rest mismatch, so a floor on it drops digits roughly independently
# of whether the transfer is right.
#
# _ROT_ATOL: the corpus stores joint_global_rots float32, so a genuine row's
# worst |R^T R - I| over the six fixture rows is 3.8e-7 and worst |det - 1|
# 4.0e-7. 1e-4 leaves that a ~250x margin while still rejecting anything
# meaningfully non-orthonormal (a 1.5x-scaled row lands at 1.25).
#
# _FINGER_BEND_LIMIT_DEG: bends beyond this are not a hand. Calibrated
# against measured range of motion, not guessed -- the largest full-fist arc
# measured anywhere on this rig is 108.0 deg (left_index_finger_2; the same
# 108.0 from the asset's own rest, mannequin_finger_axes.json, and from
# Relaxed->Fist -- R14 brought those into agreement), and the largest
# per-joint bend anywhere in the six fixture rows is 51.6 deg. 135 is 1.25x
# the former and 2.6x the latter, so it cannot bite real data: measured
# headroom over the whole fixture is 83.4 deg.
#
# It is a bound on GARBAGE, and its power is worth knowing rather than
# assuming. A uniformly random rotation exceeds 135 deg 47.5% of the time,
# so a 3-joint digit of random rotations is rejected 86% of the time, and a
# whole 5-digit hand of them 54% of the time (measured over 200 seeds). The
# sabotage control that exercises it is therefore a FIXED-SEED test, not a
# proof -- and it is backed by three deterministic per-reason controls that
# fire with probability 1. See test_npz_fingers.py.
_ROT_ATOL = 1e-4
_FINGER_BEND_LIMIT_DEG = 135.0

_FINGER_AXES: dict | None = None
_FINGER_REFS: dict = {}          # rig version -> per-bone MHR-frame flexion axes


def load_finger_axes() -> dict:
    """The rig's OWN per-bone finger flexion axes, parsed once and cached.

    `bind_poses/mannequin_finger_axes.json` (v16 task 5) is a committed
    fixture extracted offline by `tools_extract_finger_axes.py` from
    PoseGoblin's shipped hand-pose presets -- a sibling repo, deliberately
    not a runtime dependency.

    Returns `{"axis": {bone name: (3,) unit ndarray}, "fist_angle_deg":
    {bone name: float}}`, 30 bones. `axis` is in the bone's OWN REST-LOCAL
    frame, so that

        posed_local = rest_local_q o quat(axis, theta)      [w,x,y,z]

    is the rig's rest at theta = 0 and PoseGoblin's stored Fist preset at
    theta = fist_angle_deg -- exactly, on both hands, which is what makes
    Fist usable as ground truth for this transfer.

    The axes are MEASURED, never assumed. `+X` is right for the four
    fingers' middle and distal phalanges on both hands and wrong for the
    thumb, whose three joints come out mixed-XY, mixed-XZ and pure -Z.
    """
    global _FINGER_AXES
    if _FINGER_AXES is None:
        d = json.loads(_FINGER_AXES_ASSET.read_text())
        _FINGER_AXES = {
            "axis": {k: np.asarray(v["axis"], float) for k, v in d["bones"].items()},
            "fist_angle_deg": {k: float(v["fist_angle_deg"]) for k, v in d["bones"].items()},
            "rig_version": str(d["provenance"]["rig_version"]),
        }
    return _FINGER_AXES


def _hand_frame(wrist, middle1, index1, pinky1) -> np.ndarray:
    """A right-handed anatomical frame for one hand, columns [along, across,
    a x b], built from four landmarks the MHR skeleton and the mannequin
    both have.

    Built by the SAME formula on both skeletons, which is what makes the
    hands' chirality derived rather than hardcoded: the frame is always
    right-handed, so a LEFT hand's flexion axis comes out along +across and
    a RIGHT hand's along -across, all by itself.
    """
    a = np.asarray(middle1, float) - np.asarray(wrist, float)
    a = a / np.linalg.norm(a)
    b = np.asarray(pinky1, float) - np.asarray(index1, float)
    b = b - float(np.dot(b, a)) * a
    b = b / np.linalg.norm(b)
    return np.column_stack([a, b, np.cross(a, b)])


def _finger_flex_refs(rig: Rig | None = None) -> dict:
    """Per bone, the flexion axis carried across to the MHR REST world frame.

    The rig knows which way each of its own joints bends (`load_finger_axes`,
    measured from PoseGoblin's Fist preset). MHR supplies a bend but no
    opinion about which direction counts as flexion. This transports the
    rig's answer onto MHR's skeleton by expressing it in the anatomical hand
    frame each skeleton builds from its own wrist/middle_1/index_1/pinky_1
    (`_hand_frame`) and re-expanding it in the other's -- per SIDE, so the
    two hands' opposite chirality is carried, not assumed.

    Why per BONE rather than one knuckle-line normal for the whole hand: the
    four fingers' axes really are the knuckle line (|component| 0.94-0.999
    along it) but the thumb's are not -- thumb_1 comes out along the hand and
    thumb_3 out of the palm plane -- so a single normal would give the thumb
    a meaningless sign.

    The MHR-frame result is the right frame to compare against: the relative
    rotation `conj(Delta(parent)) o Delta(row)` equals the joint's own local
    delta conjugated into its REST parent's world frame, so its axis lives in
    MHR rest world.

    Cached per rig VERSION, so passing a non-default *rig* is honoured rather
    than answered from the default rig's cache. `_finger_locals` additionally
    refuses to apply the axes to a rig they were not measured against.
    """
    r = load_rig() if rig is None else rig
    if r.version in _FINGER_REFS:
        return _FINGER_REFS[r.version]
    axes = load_finger_axes()["axis"]
    Wr = fk_world_orientations(r, r.rest_local_q)
    mhr_p = load_mhr_rest()["rest_p_cm"]
    mhr_names = load_mhr_rest()["names"]
    mhr_i = {n: j for j, n in enumerate(mhr_names)}
    refs: dict = {}
    for side, pre in (("left", "l_"), ("right", "r_")):
        rig_f = _hand_frame(r.rest_world_p[r.index_of_name[f"{side}_wrist"]],
                            r.rest_world_p[r.index_of_name[f"{side}_middle_finger_1"]],
                            r.rest_world_p[r.index_of_name[f"{side}_index_finger_1"]],
                            r.rest_world_p[r.index_of_name[f"{side}_pinky_finger_1"]])
        mhr_f = _hand_frame(mhr_p[mhr_i[pre + "wrist"]], mhr_p[mhr_i[pre + "middle1"]],
                            mhr_p[mhr_i[pre + "index1"]], mhr_p[mhr_i[pre + "pinky1"]])
        for name in _MHR_FINGER_ROWS:
            if not name.startswith(side + "_"):
                continue
            # bone-local axis -> rig world (at rest) -> hand-frame coefficients
            # -> MHR's hand frame -> MHR rest world.
            axis_world = QuaternionMath.rotate_vector(Wr[r.index_of_name[name]], axes[name])
            v = mhr_f @ (rig_f.T @ axis_world)
            refs[name] = v / np.linalg.norm(v)
    # Fail at the source rather than as a KeyError deep inside the per-row
    # bend loop: the two side filters above must between them cover every row.
    assert set(refs) == set(_MHR_FINGER_ROWS), (
        f"flexion references missing for {sorted(set(_MHR_FINGER_ROWS) - set(refs))}")
    _FINGER_REFS[r.version] = refs
    return refs


def _is_proper_rotation(m) -> bool:
    """Finite, orthonormal to _ROT_ATOL, determinant +1. The data-integrity
    half of the finger gate -- see _ROT_ATOL above for where the tolerance
    comes from."""
    m = np.asarray(m, float)
    if m.shape != (3, 3) or not np.all(np.isfinite(m)):
        return False
    return (bool(np.allclose(m.T @ m, np.eye(3), atol=_ROT_ATOL))
            and abs(float(np.linalg.det(m)) - 1.0) <= _ROT_ATOL)


def _finger_bends(mhr_rots: np.ndarray, rig: Rig | None = None) -> dict:
    """Per phalange, `(signed flexion angle in RADIANS, |bend| in DEGREES)`.

    The bend is `conj(Delta(parent)) o Delta(row)` -- the joint's rotation
    beyond its own rest, with the parent being the MHR WRIST for a `_1` and
    the previous phalange otherwise. Its MAGNITUDE is a frame-independent
    scalar and therefore immune to the rest-geometry mismatch (the two rigs'
    rest finger directions are 3-49 deg apart, median 27) that made a
    world-DELTA transfer bend fingers sideways.

    The signed angle is that rotation projected onto the rig's own flexion
    axis for the bone (`_finger_flex_refs`): `angle * dot(axis, ref)`. The
    projection, rather than magnitude-times-sign, so a rotation that is
    mostly NOT flexion contributes mostly no flexion. Per-bone median
    |alignment| over the six fixture rows, after R14: 0.665 at the thumb MCP
    on BOTH hands (the two sides agree exactly now; before R14 they read 0.632
    and 0.449), 0.871-0.996 on the other 28 joint/side pairs. Against
    swing-twist about the same axis it differs by at most 1.2 deg in that
    range. The thumb CMC is the one joint whose corpus rotation runs the other
    way (median -0.889 / -0.953); see _KNOWN_INVERTED in test_npz_fingers.py.

    Both values are NaN for a bone whose source rows are not usable
    rotations; the gate reads that as a rejection.

    *rig* selects whose flexion axes the projection uses; None means the
    default. It has to be threaded through rather than defaulted internally
    (review finding, v16 task 5): this function used to call
    `_finger_flex_refs()` with no argument, so every runtime path silently
    projected onto the DEFAULT rig's axes however the caller was invoked. The
    |bend| MAGNITUDE is frame-independent and does not depend on *rig* at all
    -- only the signed angle does.
    """
    m = np.asarray(mhr_rots, float)
    if m.shape != (127, 3, 3):
        raise ValueError(
            f"mhr_rots has shape {m.shape}, expected (127, 3, 3) -- the "
            f"127-joint MHR kinematic skeleton _MHR_FINGER_ROWS indexes into")
    refs = _finger_flex_refs(rig)
    out: dict = {}
    for side in ("left", "right"):
        for digit in _DIGIT_NAMES:
            prev = _MHR_WRIST_ROW[side]
            for name in _digit_bone_names(side, digit):
                row = _MHR_FINGER_ROWS[name]
                if not (_is_proper_rotation(m[prev]) and _is_proper_rotation(m[row])):
                    out[name] = (float("nan"), float("nan"))
                else:
                    rel = QuaternionMath.multiply(
                        QuaternionMath.conjugate(_mhr_delta_q(m, prev)), _mhr_delta_q(m, row))
                    if rel[0] < 0.0:              # shortest arc, so `ang` is <= 180 deg
                        rel = -rel
                    axis, ang = QuaternionMath.to_axis_angle(rel)
                    out[name] = (float(ang * np.dot(axis, refs[name])), float(np.degrees(ang)))
                prev = row
    return out


def _digit_bone_names(side: str, digit: str) -> list:
    """The three mannequin phalange names of one digit, root outwards. The
    thumb is `{side}_thumb_{k}`; the four fingers `{side}_{digit}_finger_{k}`."""
    stem = "thumb" if digit == "thumb" else f"{digit}_finger"
    return [f"{side}_{stem}_{k}" for k in (1, 2, 3)]


def _digit_of(name: str) -> tuple:
    """`"right_index_finger_2"` -> `("right", "index")`."""
    side, rest = name.split("_", 1)
    return side, rest.split("_")[0]


def _finger_digits_passing(mhr_rots: np.ndarray, rig: Rig | None = None) -> set:
    """The `(side, digit)` pairs whose three source bends are usable data.

    Rejection is per DIGIT, not per bone: a digit whose middle phalange is
    garbage cannot be half-posed, and its three bones fall back to their
    rest locals through the ordinary unsolved-bone path. Rejects when any of
    the digit's three bends is non-finite (a non-finite or improper source
    rotation, the wrist row included -- it is the `_1` joint's parent) or
    beyond `_FINGER_BEND_LIMIT_DEG`.

    Deliberately NOT a pose-quality judgement: nothing here compares the
    result against keypoints.

    *rig* is threaded to `_finger_bends` for consistency, not because it
    changes the answer: this reads only the |bend| MAGNITUDE, which is
    frame-independent. Passing it keeps a non-default rig from paying for the
    default rig's references, and keeps one rig in play down the whole path.
    """
    bends = _finger_bends(mhr_rots, rig)
    return {(side, digit) for side in ("left", "right") for digit in _DIGIT_NAMES
            if all(np.isfinite(bends[n][1]) and bends[n][1] <= _FINGER_BEND_LIMIT_DEG
                   for n in _digit_bone_names(side, digit))}


def _finger_locals(rig: Rig, mhr_rots: np.ndarray, indices: list) -> dict:
    """Absolute local quaternions for the phalange *indices*, from MHR's bends.

        L(b) = rest_local(b) o quat(rig axis(b), signed MHR bend(b))

    *indices* comes from `solved_indices`, so this never has to re-decide
    which digits survived the gate.

    The hand as a whole still rides on the solved WRIST -- only the fingers'
    own bending comes from here. Adduction/abduction (PoseGoblin's "Spread")
    is out of scope: one axis per joint, and the measured off-axis component
    at the knuckles is small.
    """
    fixture = load_finger_axes()
    # The axes were measured against ONE rig asset's rest. Applying them to a
    # different one would silently produce a wrong hand -- a re-rig must fail
    # here, loudly, and be answered by re-running tools_extract_finger_axes.py.
    if fixture["rig_version"] != rig.version:
        raise ValueError(
            f"mannequin_finger_axes.json was extracted against "
            f"{fixture['rig_version']!r} but this rig is {rig.version!r} -- "
            f"re-run tools_extract_finger_axes.py")
    axes = fixture["axis"]
    bends = _finger_bends(mhr_rots, rig)
    return {i: QuaternionMath.multiply(
                rig.rest_local_q[i],
                QuaternionMath.from_axis_angle(axes[rig.name[i]], bends[rig.name[i]][0]))
            for i in indices}


def solved_indices(rig: Rig, mhr_rots: np.ndarray | None = None) -> list:
    """The bone indices `solve_rig_locals` actually solves, in `rig.order`.

    The SINGLE authoritative source for "solved" (ruling 3): `rig.solve`
    alone is not enough under v2, since the 30 finger phalanges carry
    solve:True but are posed only from MHR hand ROTATIONS, per digit, behind
    the integrity gate -- not from the generic aim-based path.
    `solve_rig_locals` calls this itself rather than re-deriving the filter
    inline, and so must every OTHER reader of "which bones will
    solve_rig_locals return" -- including outside this module. (Review
    finding, v16 task 3: poseforge3d's validate_rig_retarget.py machine gate
    read `rig.solve` directly and, as a result, silently started gating the
    30 unposed finger phalanges against real keypoint targets the moment v2
    shipped -- every row failing. A direct `rig.solve` read is exactly the
    bug this function exists to make impossible to repeat.)

    *mhr_rots* is optional with a None default so existing callers that pass
    only *rig* -- the poseforge3d harness does today -- keep working and keep
    getting exactly the v15 34. With rotations present the answer grows to
    those 34 plus the phalanges of every digit that passes
    `_finger_digits_passing`: up to 64, fewer when a digit's source rotations
    are unusable. Answering the gated question HERE, rather than letting
    `solve_rig_locals` quietly return less than this function promised, is
    what keeps the two from drifting apart.
    """
    if mhr_rots is None:
        return [i for i in rig.order
                if rig.solve[i] and rig.name[i] not in _FINGER_PHALANGE_NAMES]
    ok = _finger_digits_passing(mhr_rots, rig)
    return [i for i in rig.order
            if rig.solve[i] and (rig.name[i] not in _FINGER_PHALANGE_NAMES
                                 or _digit_of(rig.name[i]) in ok)]


def solve_rig_locals(rig: Rig, targets: dict[int, np.ndarray],
                     mhr_rots: np.ndarray | None = None) -> dict[int, np.ndarray]:
    """Solve absolute local quaternions posing the rig onto *targets*.

    Per bone, a world DELTA D(b) rotates the rig's rest bone directions onto
    the target directions: all-children Kabsch for >=2 usable children,
    `_aim_delta` for exactly 1 (a single pair leaves twist unconstrained; an
    SVD there would invent one, and so -- less obviously -- does a minimal
    WORLD rotation, whose local then carries the PARENT's roll about the aim
    axis; see there), parent's delta when the bone or its children have no
    targets. The consumer REPLACES locals, and most rig rest locals are
    non-identity, so deltas are composed with the rest:

        L(b) = (D(p) * Wr(p))^-1 * D(b) * Wr(b)      (root: D * Wr)

    `targets` is keyed by bone INDEX (subset OK; world positions, Y-up).
    Returns absolute local quaternions keyed by bone INDEX for the SOLVED
    set only -- exactly `solved_indices(rig, mhr_rots)`, never a bone more
    or fewer. Without *mhr_rots* that is the 34 bones the rig has always
    solved; with it, those 34 plus the phalanges of the digits that pass the
    integrity gate, which come from `_finger_locals` and never from the
    aim-based path below. `fk_world_orientations`/`fk_world_positions`
    cover every node now (ruling 2), so Wr below DOES have entries for
    bones this function never solves -- a solved wrist has an unsolved
    GROUP child under v2 (the finger islands are reconnected there now,
    not disconnected at a captured `parent: None` root as in v1), and that
    is fine: this function never looks up Wr, D or L for a group or a
    finger phalange. `targets` may carry entries for both (MHR-70 supplies
    raw finger keypoints, and rig_targets_from_mhr70 maps them onto the
    phalanges like any other bone); they are never read, since neither
    group nor phalange is ever `n` here, nor a solved bone's direct child
    (a solved wrist's only children are its five groups, and a group with
    no target is dropped from `pairs` before it can influence anything).
    Unsolved bones fall back to their rest locals; that fallback is the
    caller's job (`rig_state_from_mhr70`), not this function's.

    *mhr_rots*, when given, is this row's (127,3,3) `joint_global_rots`
    (`mhr_rots_from_npz`); it poses the spine from the model's own
    rotations instead of the v15 construction, and the ten finger chains
    from the model's own hand rotations. Optional with a None default so
    existing call sites -- the poseforge3d harness and bridge -- keep
    working untouched, and so rows whose blob is missing still solve.
    """
    Wr = fk_world_orientations(rig, rig.rest_local_q)
    solved = solved_indices(rig, mhr_rots)
    # The phalanges are posed from MHR joint ANGLES about the rig's own
    # measured axes, not by aiming bones at keypoints: `aim` is the rest of
    # the solved set, and is what the delta machinery below covers. Splitting
    # here rather than filtering later keeps every `D[p]` lookup in that loop
    # on a bone the loop itself solved -- a phalange's parent is a GROUP,
    # which has no delta at all.
    fingers = [n for n in solved if rig.name[n] in _FINGER_PHALANGE_NAMES]
    aim = [n for n in solved if rig.name[n] not in _FINGER_PHALANGE_NAMES]
    # Anchored bones get exact frame deltas (facing and hinge twist are
    # constraints there); everything else falls through to the generic
    # child-direction solve below. See _anchor_deltas.
    anchors = _anchor_deltas(rig, targets, Wr, mhr_rots)
    D: dict = {}
    for n in aim:
        if n in anchors:
            D[n] = anchors[n]
            continue
        d_ankle = _try_ankle_delta(rig, targets, Wr, D, n)
        if d_ankle is not None:
            D[n] = d_ankle
            continue
        pairs = []
        if n in targets:
            for c in rig.children[n]:
                if c not in targets:
                    continue
                rb = rig.rest_world_p[c] - rig.rest_world_p[n]
                tb = np.asarray(targets[c], float) - np.asarray(targets[n], float)
                nr, nt = np.linalg.norm(rb), np.linalg.norm(tb)
                if nr > 1e-6 and nt > 1e-6:
                    pairs.append((rb / nr, tb / nt))
        p = rig.parent[n]
        if len(pairs) >= 2:
            D[n] = _kabsch_q(np.array([x[0] for x in pairs]),
                             np.array([x[1] for x in pairs]))
        elif len(pairs) == 1:
            D[n] = _aim_delta(pairs[0][0], pairs[0][1],
                              D[p] if p is not None and p in D else None)
        else:
            D[n] = D[p] if p is not None else QuaternionMath.identity()

    L: dict = {}
    for n in aim:
        p = rig.parent[n]
        wb = QuaternionMath.multiply(D[n], Wr[n])
        if p is None:
            L[n] = wb
        else:
            wp = QuaternionMath.multiply(D[p], Wr[p])
            L[n] = QuaternionMath.multiply(QuaternionMath.conjugate(wp), wb)
    if fingers:
        L.update(_finger_locals(rig, mhr_rots, fingers))
    return L


ROOT_DISPLAY_YAW_DEG: float = 0.0   # obsolete under the chirality fix below; kept inert at 0

# cv_to_yup negates only Y -- a det(-1) REFLECTION of camera space. The
# labeled anatomy of its output is chirally mirrored: over 118 corpus rows,
# dot(cross(up, right_hip - left_hip), nose - head) came out negative on 117
# (median -0.883) -- labeled left/right disagree with face-and-up on
# essentially every real detection. A proper rotation can never fit a mirror
# (_kabsch_q forces det +1), so the near-coplanar pelvis fit paid the
# reflection off in the one axis it barely constrains: facing. Measured
# before this fix: solved pelvis 174.8 deg from person-forward, IQR 0.2 deg,
# deterministic and corpus-wide. Negating Z as well restores det(+1) -- the
# composite cv->rig map is then a 180-deg rotation of camera space about X --
# and lands camera-facing subjects facing +Z, the viewer: the convention
# every hand-posed ground-truth capture was made in.
_CV_YUP_TO_RIG = np.array([1.0, 1.0, -1.0])


def rig_targets_from_mhr70(kp_cam: np.ndarray) -> dict[int, np.ndarray]:
    """(70,3) CV camera keypoints -> rig-bone-INDEX-keyed world targets, Y-up.

    Reuses MHR70Retargeter.compute_joint_positions for cv_to_yup, the
    keypoint mapping, and the synthesised joints (root = hip midpoint, head =
    ear midpoint, interpolated spine). This loads default_human.json only to
    construct a valid MHR70Retargeter instance -- compute_joint_positions
    never reads self.bind_pose, so nothing here is SOLVED against that
    asset; the solve itself (solve_rig_locals) never touches it either.

    Index-keyed, not name-keyed (ruling 6): solve_rig_locals and
    fk_world_positions are index-keyed throughout because the rig has two
    bones sharing the name "joint7", so rig.index_of_name resolves the 72
    genuinely unique names and deliberately raises KeyError for "joint7".
    MHR-70 names are unique and none of them maps to a mannequin bone named
    "joint7", so this lookup is safe here -- but if it ever DID raise, it
    must raise, not silently drop the joint.
    """
    from .mhr70_retargeter import MHR70Retargeter
    from ..bind_poses.loader import BindPoseLoader
    from ..exporters.mannequin_exporter import MHR70_TO_MANNEQUIN

    rig = load_rig()
    pos = MHR70Retargeter(BindPoseLoader.get_default_bind_pose()) \
        .compute_joint_positions(np.asarray(kp_cam, np.float32))
    targets: dict[int, np.ndarray] = {}
    for j, p in pos.items():
        bone_name = MHR70_TO_MANNEQUIN.get(j)
        if bone_name is None:
            continue                                  # no mannequin equivalent: deliberate skip
        targets[rig.index_of_name[bone_name]] = (
            np.asarray(p, float) * _CV_YUP_TO_RIG)    # KeyError propagates, never caught

    # ruling 9 (amended): MHR-70's `root` is the exact hip MIDPOINT -- level
    # with the hips, so pelvis->left_hip and pelvis->right_hip are always
    # exactly 180 deg apart in this raw target. The rig's own pelvis sits
    # ABOVE the hip line (105.2 deg apart in rest): same topology, different
    # height, and a rotation cannot change the angle between two vectors, so
    # a hip-midpoint pelvis target makes the pelvis's own Kabsch fit
    # unsatisfiable by construction. Replace it with the rig's own
    # above-the-hip-line offset, rebuilt in the target's own frame.
    targets[rig.index_of_name["pelvis"]] = synthesize_pelvis_target(rig, targets)
    return targets


def _leg_len(points: dict, hip: int, knee: int, ankle: int) -> float:
    """hip->knee->ankle chain length. *points* and the three args share one
    index space (rig bone index): both rig.rest_world_p and the targets from
    rig_targets_from_mhr70 are index-keyed (ruling 6)."""
    return (float(np.linalg.norm(points[knee] - points[hip]))
            + float(np.linalg.norm(points[ankle] - points[knee])))


_FRAME_EPS = 1e-6  # matches solve_rig_locals's own "vector is degenerate" tolerance
                    # elsewhere in this module; far below any real bone length in
                    # metres (MHR-70 targets) or rig units (rest positions), but large
                    # enough to catch genuine coincidence/collinearity rather than noise.


def _orthonormal_frame_from_hips_and_up(points: dict, left_hip: int, right_hip: int,
                                         up_ref: int) -> tuple:
    """right = direction from right_hip to left_hip; up = direction from the
    hip midpoint to *up_ref*, re-orthonormalised against right (Gram-Schmidt
    via cross products, so it's exact even when the raw up reference isn't
    perfectly perpendicular to the hip line); fwd completes a right-handed
    orthonormal basis. Returns (right, up, fwd, hip_mid).

    Raises ValueError naming the degenerate condition and the offending
    joint positions if any basis vector's norm is below _FRAME_EPS before
    normalising -- never silently falls back to world axes, which would
    produce a plausible-looking but wrong pose for an inverted or supine
    subject. SAM-3D output is noisy and this runs over thousands of real
    detections; a bad one must fail loudly, not emit a NaN that only a
    numpy RuntimeWarning would hint at.
    """
    hip_mid = (points[left_hip] + points[right_hip]) / 2

    right_raw = points[left_hip] - points[right_hip]
    right_norm = np.linalg.norm(right_raw)
    if right_norm < _FRAME_EPS:
        raise ValueError(
            f"degenerate pelvis frame: left_hip and right_hip are coincident "
            f"(separation {right_norm:.3e} < {_FRAME_EPS:.0e}) -- "
            f"left_hip={points[left_hip]!r}, right_hip={points[right_hip]!r}")
    right = right_raw / right_norm

    up_raw = points[up_ref] - hip_mid
    up_raw_norm = np.linalg.norm(up_raw)
    if up_raw_norm < _FRAME_EPS:
        raise ValueError(
            f"degenerate pelvis frame: the up reference coincides with the hip "
            f"midpoint (separation {up_raw_norm:.3e} < {_FRAME_EPS:.0e}) -- "
            f"up_ref={points[up_ref]!r}, hip_mid={hip_mid!r}")

    fwd_raw = np.cross(right, up_raw)
    fwd_norm = np.linalg.norm(fwd_raw)
    if fwd_norm < _FRAME_EPS:
        raise ValueError(
            f"degenerate pelvis frame: the hip line and the up direction are "
            f"collinear (cross-product norm {fwd_norm:.3e} < {_FRAME_EPS:.0e}) -- "
            f"left_hip={points[left_hip]!r}, right_hip={points[right_hip]!r}, "
            f"up_ref={points[up_ref]!r}")
    fwd = fwd_raw / fwd_norm

    up_result_raw = np.cross(fwd, right)
    up_result_norm = np.linalg.norm(up_result_raw)
    if up_result_norm < _FRAME_EPS:
        # Unreachable if the two guards above hold: fwd and right are then
        # unit vectors that are exactly orthogonal by construction (fwd was
        # built as a cross product involving right), so this cross
        # product's norm is exactly 1. Kept as defense in depth, per
        # "check each basis vector's norm before normalising."
        raise ValueError(
            f"degenerate pelvis frame: fwd x right had near-zero norm "
            f"({up_result_norm:.3e} < {_FRAME_EPS:.0e}), which should be "
            f"unreachable -- fwd={fwd!r}, right={right!r}")
    up = up_result_raw / up_result_norm

    return right, up, fwd, hip_mid


def synthesize_pelvis_target(rig: Rig, targets: dict[int, np.ndarray]) -> np.ndarray:
    """Ruling 9 (amended) -- rebuild the rig's own "pelvis sits above the
    hip line" offset in the TARGET's own frame, instead of leaving the
    pelvis target at the raw hip midpoint MHR70Retargeter computes.

    1. Rig rest: how far above the hip line the pelvis sits, expressed
       scale-free (as a fraction of the rig's own leg length) in the rig's
       own world-aligned right/up/forward axes.
    2. Target: an orthonormal frame built from the TARGETS themselves --
       never the rig's world axes, since the subject may be lying down or
       inverted -- so the reconstructed offset points the anatomically
       correct way regardless of the subject's orientation in camera space.
    3. Re-express the rig's offset in that frame and scale by the target's
       own leg length.

    `targets` must already contain left_hip, right_hip, left/right knee and
    ankle, and spine_1; everything but the pelvis entry is read, not
    written. (No neck fallback: MHR70Retargeter.compute_joint_positions_static
    unconditionally computes spine1, so rig_targets_from_mhr70's output
    always has it -- a fallback for "spine_1 missing" would be a branch
    with no reachable input, not a guard against the degenerate cases
    above, which are handled explicitly in
    _orthonormal_frame_from_hips_and_up instead.)
    """
    li = rig.index_of_name
    lhip, rhip = li["left_hip"], li["right_hip"]
    lknee, rknee = li["left_knee"], li["right_knee"]
    lankle, rankle = li["left_ankle"], li["right_ankle"]
    pelvis_i = li["pelvis"]

    rest = rig.rest_world_p
    hip_mid_rest = (rest[lhip] + rest[rhip]) / 2
    offset_rest = rest[pelvis_i] - hip_mid_rest
    rig_leg_len = np.mean([_leg_len(rest, lhip, lknee, lankle),
                            _leg_len(rest, rhip, rknee, rankle)])
    offset_frac = offset_rest / rig_leg_len

    right_t, up_t, fwd_t, hip_mid_t = _orthonormal_frame_from_hips_and_up(
        targets, lhip, rhip, li["spine_1"])
    tgt_leg_len = np.mean([_leg_len(targets, lhip, lknee, lankle),
                            _leg_len(targets, rhip, rknee, rankle)])

    offset_target = (offset_frac[0] * right_t + offset_frac[1] * up_t
                      + offset_frac[2] * fwd_t) * tgt_leg_len
    return hip_mid_t + offset_target


def _assert_all_finite(state: dict) -> None:
    """Belt to _orthonormal_frame_from_hips_and_up's guard-clause braces: a
    future edit that introduces a non-finite value by some OTHER route must
    still fail loudly here, before json.dumps ever gets a chance to emit a
    bare `NaN`/`Infinity` -- invalid JSON, and something JavaScript's
    JSON.parse rejects on the consuming end. Raises ValueError naming the
    first offending bone/field, not a summary of all of them."""
    for bone, q in state["pose"].items():
        for k in ("_x", "_y", "_z", "_w"):
            if not np.isfinite(q[k]):
                raise ValueError(f"non-finite value in pose[{bone!r}][{k!r}]: {q[k]}")
    for k, v in state["pelvisPosition"].items():
        if not np.isfinite(v):
            raise ValueError(f"non-finite value in pelvisPosition[{k!r}]: {v}")
    if not np.isfinite(state["groundY"]):
        raise ValueError(f"non-finite value in groundY: {state['groundY']}")


def rig_state_from_mhr70(kp_cam: np.ndarray, mhr_rots: np.ndarray | None = None) -> dict:
    """(70,3) CV camera keypoints -> full mannequinState for the PoseGoblin
    viewer: every rig bone posed (rig units), ready to serialize.

    *mhr_rots* is this row's (127,3,3) `joint_global_rots` when its
    `mhr_params_npz` blob is available (`mhr_rots_from_npz`), and poses both
    the spine AND the fingers from the model's own rotations (v16 task 5);
    None keeps the v15 spine and rest fingers."""
    rig = load_rig()
    li = rig.index_of_name
    targets = rig_targets_from_mhr70(kp_cam)
    # index-keyed, the SOLVED bones per solved_indices' own contract -- 34
    # without mhr_rots, up to 64 (34 + the phalanges of every digit that
    # passes its integrity gate) with it (ruling 7)
    solved = solve_rig_locals(rig, targets, mhr_rots)

    if ROOT_DISPLAY_YAW_DEG:
        t = np.radians(ROOT_DISPLAY_YAW_DEG)
        yaw = np.array([np.cos(t / 2), 0.0, np.sin(t / 2), 0.0])
        pelvis_i = li["pelvis"]
        solved[pelvis_i] = QuaternionMath.multiply(yaw, solved[pelvis_i])

    # Ankle roll bias (see _ANKLE_ROLL_BIAS_DEG above): applied here so the
    # pure solve stays bias-free.
    Wr_all = None if not _ANKLE_ROLL_ENABLED else fk_world_orientations(rig, rig.rest_local_q)
    Wq = None if Wr_all is None else fk_world_orientations(rig, {**rig.rest_local_q, **solved})
    for side in ("left", "right") if _ANKLE_ROLL_ENABLED else ():
        ai = li[f"{side}_ankle"]
        ki, hi2 = rig.parent[ai], li[f"{side}_heel"]
        bi2, si2 = li[f"{side}_big_toe"], li[f"{side}_small_toe"]
        if not (ai in solved and ki in solved
                and all(k in targets for k in (ki, ai, hi2, bi2, si2))):
            continue
        fa_t = np.asarray(targets[bi2], float) + np.asarray(targets[si2], float)
        fa_t = fa_t * 0.5 - np.asarray(targets[hi2], float)
        shin = np.asarray(targets[ai], float) - np.asarray(targets[ki], float)
        nf, ns = np.linalg.norm(fa_t), np.linalg.norm(shin)
        if nf < _FRAME_EPS or ns < _FRAME_EPS:
            continue
        pointed = abs(float(np.dot(fa_t / nf, shin / ns)))
        lo, hi_r = _ANKLE_ROLL_RAMP
        scale = min(1.0, max(0.0, (pointed - lo) / (hi_r - lo)))
        if scale <= 0.0:
            continue
        # Anatomically mirrored (Scott: "the left foot needs the opposite
        # sign"): a consistent inversion/eversion bias flips sign across the
        # midline. +30 right, -30 left, about each foot's own heel->toe axis.
        if side == "left":
            scale = -scale
        d_a = QuaternionMath.multiply(Wq[ai], QuaternionMath.conjugate(Wr_all[ai]))
        rest_axis = (0.5 * (rig.rest_world_p[bi2] + rig.rest_world_p[si2])
                     - rig.rest_world_p[hi2])
        na = np.linalg.norm(rest_axis)
        if na < _FRAME_EPS:
            continue
        axis_now = QuaternionMath.rotate_vector(d_a, rest_axis / na)
        bias = _axis_angle_q(np.asarray(axis_now, float) / np.linalg.norm(axis_now),
                             np.radians(_ANKLE_ROLL_BIAS_DEG * scale))
        w_new = QuaternionMath.multiply(bias, Wq[ai])
        solved[ai] = QuaternionMath.multiply(QuaternionMath.conjugate(Wq[ki]), w_new)


    # ruling 7: solve_rig_locals deliberately covers only the solved bones --
    # padding its own output would claim to have solved bones it never
    # touched. Assembly owns the merge: every bone must land in the state, so
    # the 40 unsolved finger/thumb bones (30 phalanges excluded per ruling 3
    # / _FINGER_PHALANGE_NAMES, plus the 10 tips, which are solve:False
    # outright) AND the ten constant group nodes all fall back to their REST
    # locals -- the pose comprehension below then drops the groups entirely
    # (the viewer owns them; see there).
    full_by_index = {**rig.rest_local_q, **solved}

    # ruling 5: PoseGoblin reads state.pose[child.name], so the emitted pose
    # is NAME-keyed, not index-keyed. The two "joint7" bones collapse to a
    # single entry -- last-one-wins by iterating rig.order, which keeps the
    # original 74 bones (indices 0..73) in their v1 parent-first order,
    # matching PoseGoblin's own captureCurrentState/RecallPoseCommand
    # contract instead of diverging from it. The ten v2 group nodes
    # (`transform4`..`transform13`, is_group=True) are excluded here: the
    # viewer already owns them as a fixed container transform it never
    # poses, so sending it ten `transform*` entries would be sending
    # something it never asked for (v16 task 3) -- 73 keys out, same as v1.
    pose = {rig.name[i]: QuaternionMath.to_threejs_dict(full_by_index[i])
            for i in rig.order if not rig.is_group[i]}

    # scale: subject leg length (meters, pose-invariant) -> rig units
    rig_leg = np.mean([
        _leg_len(rig.rest_world_p, li["left_hip"], li["left_knee"], li["left_ankle"]),
        _leg_len(rig.rest_world_p, li["right_hip"], li["right_knee"], li["right_ankle"]),
    ])
    tgt_leg = np.mean([
        _leg_len(targets, li["left_hip"], li["left_knee"], li["left_ankle"]),
        _leg_len(targets, li["right_hip"], li["right_knee"], li["right_ankle"]),
    ])
    s = rig_leg / max(tgt_leg, 1e-6)

    # Feet mannequin bone names verified against MHR70_TO_MANNEQUIN's VALUES
    # (not guessed): left_heel/right_heel/left_big_toe/right_big_toe.
    feet_idx = [li[n] for n in ("left_heel", "right_heel", "left_big_toe", "right_big_toe")]
    feet = [targets[i][1] for i in feet_idx if i in targets]

    # ruling 10: pelvisPosition is the rig's REST pelvis position, not the
    # subject's camera-space position scaled to rig units. A pose viewer
    # shows the POSE, not the subject's translation through space -- the
    # scaled camera-space position could (and for this fixture's Cyr-wheel
    # capture, did) land the figure almost inside the fixed default camera.
    # `s` is still needed for groundY, so it stays computed above.
    pelvis = rig.rest_world_p[li["pelvis"]]

    from ..exporters.mannequin_exporter import MannequinExporter
    state = {
        "pose": pose,
        "pelvisPosition": {"x": float(pelvis[0]), "y": float(pelvis[1]), "z": float(pelvis[2])},
        "groundY": float(min(feet) * s) if feet else 0.0,
        "cameraState": MannequinExporter.get_default_camera_state(),
        "rigVersion": rig.version,
        "retargetVersion": 18,
    }
    _assert_all_finite(state)   # belt: no non-finite value reaches the wire, regardless of cause
    return state
