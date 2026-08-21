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

Ten bones (five fingers/thumb x two hands; 40 bones counting descendants) have
`parent: None` in the capture. On the live rig they are actually parented
under a Group carrying its own non-identity rotation AND a 0.1 scale -- the
container-transform contingency this module's tests anticipate -- but
modelling rotation+scale propagation through FK for them was ruled out of
scope (finger articulation is not a search input; hand pose is invisible at
display scale). Their `solve` flag is False, and fk_world_orientations/
fk_world_positions skip them. Nothing is deleted from the asset -- a future
re-rig may reconnect them -- but the FK self-consistency test pins the solved
set to exactly what's reachable from pelvis, so a re-rig that changes this
shows up as a failing assertion, never as silence.
"""
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np

from ..core.math_utils import QuaternionMath

_ASSET = Path(__file__).resolve().parent.parent / "bind_poses" / "posegoblin_rig_v1.json"


@dataclass(frozen=True)
class Rig:
    version: str
    order: list          # parent-first bone INDICES (int), all 74, solved and unsolved
    name: dict            # index -> bone name (str); NOT unique, see module docstring
    parent: dict          # index -> parent index | None
    rest_local_q: dict    # index -> (4,) [w,x,y,z]
    rest_local_p: dict    # index -> (3,)
    rest_world_p: dict    # index -> (3,)
    children: dict        # index -> [child indices], asset order
    solve: dict           # index -> bool; False for the finger/thumb islands
    index_of_name: dict   # unique name -> index; ambiguous names (e.g. "joint7") omitted


def load_rig(path: Path = _ASSET) -> Rig:
    d = json.loads(Path(path).read_text())
    bones = d["bones"]
    order = list(range(len(bones)))  # asset position IS the index; verified parent-first
    name = {i: b["name"] for i, b in enumerate(bones)}
    parent = {i: b["parent"] for i, b in enumerate(bones)}
    children: dict = {i: [] for i in order}
    for i, b in enumerate(bones):
        if b["parent"] is not None:
            children[b["parent"]].append(i)
    name_counts = Counter(name.values())
    index_of_name = {b["name"]: i for i, b in enumerate(bones) if name_counts[b["name"]] == 1}
    return Rig(
        version=d["version"], order=order, name=name, parent=parent,
        rest_local_q={i: np.asarray(b["rest_local_q"], float) for i, b in enumerate(bones)},
        rest_local_p={i: np.asarray(b["rest_local_p"], float) for i, b in enumerate(bones)},
        rest_world_p={i: np.asarray(b["rest_world_p"], float) for i, b in enumerate(bones)},
        children=children,
        solve={i: bool(b["solve"]) for i, b in enumerate(bones)},
        index_of_name=index_of_name,
    )


def fk_world_orientations(rig: Rig, local_q: dict) -> dict:
    """World orientations for the SOLVED set only (see module docstring)."""
    W: dict = {}
    for i in rig.order:
        if not rig.solve[i]:
            continue
        p = rig.parent[i]
        q = np.asarray(local_q[i], float)
        W[i] = q if p is None else QuaternionMath.multiply(W[p], q)
    return W


def fk_world_positions(rig: Rig, local_q: dict) -> dict:
    """World positions for the SOLVED set only (see module docstring)."""
    Wq = fk_world_orientations(rig, local_q)
    P: dict = {}
    for i in rig.order:
        if not rig.solve[i]:
            continue
        p = rig.parent[i]
        if p is None:
            P[i] = rig.rest_local_p[i].copy()
        else:
            P[i] = P[p] + QuaternionMath.rotate_vector(Wq[p], rig.rest_local_p[i])
    return P
