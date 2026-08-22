#!/usr/bin/env python
"""One-time extraction: MHR skeleton rest globals -> committed JSON fixture.

Dev-only (imports torch). Runtime code loads only the JSON.
Checkpoint quats are XYZW (roma); the fixture stores WXYZ (solver convention).
Rest global(j) = rest global(parent) o prerotation(j); shape/scale params
morph translations only, never prerotations, so rest ORIENTATIONS are
subject-independent (spec 2).
"""
import datetime
import json
from pathlib import Path

import numpy as np
import torch

CKPT = Path(__file__).resolve().parent.parent.parent / "checkpoints" / "mhr_model.pt"
OUT = Path(__file__).resolve().parent / "bind_poses" / "mhr_skeleton_rest.json"


def main():
    m = torch.jit.load(str(CKPT), map_location="cpu")
    sk = m.character_torch.skeleton
    names = list(sk.joint_names)
    parents = [int(p) for p in sk.joint_parents.numpy()]
    offsets = sk.joint_translation_offsets.numpy().astype(float)
    prerot_xyzw = sk.joint_prerotations.numpy().astype(float)   # (127,4) xyzw

    def qmul(a, b):  # wxyz
        w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
        return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                         w1*x2 + x1*w2 + y1*z2 - z1*y2,
                         w1*y2 - x1*z2 + y1*w2 + z1*x2,
                         w1*z2 + x1*y2 - y1*x2 + z1*w2])

    pre_wxyz = np.concatenate([prerot_xyzw[:, 3:4], prerot_xyzw[:, :3]], axis=1)
    rest = np.zeros((127, 4))
    for j in range(127):
        p = parents[j]
        rest[j] = pre_wxyz[j] if p < 0 else qmul(rest[p], pre_wxyz[j])
        rest[j] /= np.linalg.norm(rest[j])

    doc = {"version": 1,
           "provenance": {"source": "checkpoints/mhr_model.pt",
                          "extracted": datetime.date.today().isoformat()},
           "names": names, "parents": parents,
           "rest_global_q_wxyz": np.round(rest, 9).tolist(),
           "template_offsets_cm": np.round(offsets, 6).tolist()}
    OUT.write_text(json.dumps(doc))
    print(f"wrote {OUT} ({OUT.stat().st_size/1024:.0f} KB, {len(names)} joints)")


if __name__ == "__main__":
    main()
