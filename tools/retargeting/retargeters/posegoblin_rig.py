"""The PoseGoblin rig as a versioned asset, and the rig-native solve.

Why this exists: the shipped rig rests in an A-pose, authored Z-up, and its
rest directions agree with default_human.json at mean cosine +0.065. No fixed
per-bone correction can bridge two different rest GEOMETRIES (measured spread
61 deg across five hand-matched poses), so poses must be solved against the
rig's own rest. See the design doc named in the plan header.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np

from ..core.math_utils import QuaternionMath

_ASSET = Path(__file__).resolve().parent.parent / "bind_poses" / "posegoblin_rig_v1.json"


@dataclass(frozen=True)
class Rig:
    version: str
    order: list          # parent-first bone names
    parent: dict         # name -> parent name | None
    rest_local_q: dict   # name -> (4,) [w,x,y,z]
    rest_local_p: dict   # name -> (3,)
    rest_world_p: dict   # name -> (3,)
    children: dict       # name -> [child names], asset order


def load_rig(path: Path = _ASSET) -> Rig:
    d = json.loads(Path(path).read_text())
    bones = d["bones"]
    order = [b["name"] for b in bones]
    parent = {b["name"]: b["parent"] for b in bones}
    children: dict = {n: [] for n in order}
    for b in bones:
        if b["parent"] is not None:
            children[b["parent"]].append(b["name"])
    return Rig(
        version=d["version"], order=order, parent=parent,
        rest_local_q={b["name"]: np.asarray(b["rest_local_q"], float) for b in bones},
        rest_local_p={b["name"]: np.asarray(b["rest_local_p"], float) for b in bones},
        rest_world_p={b["name"]: np.asarray(b["rest_world_p"], float) for b in bones},
        children=children,
    )


def fk_world_orientations(rig: Rig, local_q: dict) -> dict:
    W: dict = {}
    for n in rig.order:
        p = rig.parent[n]
        q = np.asarray(local_q[n], float)
        W[n] = q if p is None else QuaternionMath.multiply(W[p], q)
    return W


def fk_world_positions(rig: Rig, local_q: dict) -> dict:
    Wq = fk_world_orientations(rig, local_q)
    P: dict = {}
    for n in rig.order:
        p = rig.parent[n]
        if p is None:
            P[n] = rig.rest_local_p[n].copy()
        else:
            P[n] = P[p] + QuaternionMath.rotate_vector(Wq[p], rig.rest_local_p[n])
    return P
