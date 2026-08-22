#!/usr/bin/env python
"""One-time extraction: the rig's OWN finger flexion axes -> committed JSON.

Dev-only. Reads PoseGoblin's shipped hand-pose presets (a sibling repo, not
a runtime dependency) and the v2 rig asset; runtime code loads only the
emitted JSON.

Why measure instead of assume: v16 task 5's first design applied MHR's
world rotation DELTA to each phalange. That is right when the two rigs'
rests nearly agree (the spine, 7.6-31.6 deg apart) and wrong here -- the
two rests' finger directions are 37-75 deg apart, and the transferred hand
bent fingers sideways. The fix is to transfer a signed ANGLE about the
rig's own joint axis, which means knowing that axis. `hand_poses.json`
holds four hand-posed presets (Relaxed/Fist/Point/Spread) that exercise the
fingers through their full range, so the axis is measurable rather than
guessed -- and it must be measured, because it is NOT +X everywhere: the
thumb's three joints come out about mixed-XY, mixed-XZ and pure -Z.

Per bone the emitted `axis` is a UNIT vector in that bone's OWN REST-LOCAL
frame and `fist_angle_deg` the angle about it, such that

    posed_local = rest_local_q o quat(axis, theta)          [w,x,y,z]

lands exactly on the stored Fist preset at theta = fist_angle_deg and
exactly on the rig's rest at theta = 0. (PoseGoblin state.pose quaternions
are three.js [x,y,z,w]; everything here is the solver's [w,x,y,z].)

Both hands are measured against their OWN rest, not mirrored from one side.
That is what the data supports: PoseGoblin stores a hand preset as one set
of absolute LOCAL quaternions and applies it to both hands unchanged
(verified -- `default_poses.json`'s "flex" right hand is bit-identical to
`hand_poses.json`'s "Fist"), so each joint's axis is the same vector in
each bone's own local frame, and measuring per side only redistributes the
ANGLE. It does redistribute it: the v2 asset's right-hand rest sits 0.630
of the way from the left-hand rest to the Fist on all 15 bones (a constant
ratio -- the asset was captured with the right hand part-closed), so the
right hand's rest-to-Fist angles are uniformly 63% of the left's.
"""
import datetime
import json
from pathlib import Path

import numpy as np

from retargeting.core.math_utils import QuaternionMath
from retargeting.retargeters.posegoblin_rig import load_rig

HAND_POSES = Path("/Users/scotteaton/Dropbox/CODE/poseGoblin/poses/hand_poses.json")
OUT = Path(__file__).resolve().parent / "bind_poses" / "mannequin_finger_axes.json"
REFERENCE_POSE = "Fist"

_DIGITS = {"thumb": ["thumb_1", "thumb_2", "thumb_3"]}
for _f in ("index", "middle", "ring", "pinky"):
    _DIGITS[_f] = [f"{_f}_finger_{k}" for k in (1, 2, 3)]


def _shortest(q):
    """Same rotation, w >= 0 -- so to_axis_angle returns the <=180 deg arc."""
    q = QuaternionMath.normalize(np.asarray(q, float))
    return -q if q[0] < 0.0 else q


def main(hand_poses: Path = HAND_POSES, out: Path = OUT) -> None:
    presets = {e["name"]: e["pose"] for e in json.loads(Path(hand_poses).read_text())}
    if REFERENCE_POSE not in presets:
        raise SystemExit(f"{hand_poses} has no {REFERENCE_POSE!r} preset "
                         f"(found {sorted(presets)})")
    fist = presets[REFERENCE_POSE]
    rig = load_rig()

    bones = {}
    for names in _DIGITS.values():
        for nm in names:
            # The preset is authored on the LEFT bone names and applied to
            # both hands verbatim; only the rest it is measured against is
            # per-side.
            target = _shortest(QuaternionMath.from_threejs_dict(fist["left_" + nm]))
            for side in ("left", "right"):
                rest = _shortest(rig.rest_local_q[rig.index_of_name[f"{side}_{nm}"]])
                axis, ang = QuaternionMath.to_axis_angle(
                    _shortest(QuaternionMath.multiply(QuaternionMath.conjugate(rest), target)))
                bones[f"{side}_{nm}"] = {
                    "axis": [round(float(v), 9) for v in axis],
                    "fist_angle_deg": round(float(np.degrees(ang)), 6),
                }

    doc = {
        "version": 1,
        "provenance": {
            "hand_poses": str(hand_poses),
            "reference_pose": REFERENCE_POSE,
            "rig_asset": f"bind_poses/{rig.version}.json",
            "rig_version": rig.version,
            "extracted": datetime.date.today().isoformat(),
        },
        "convention": (
            "axis is a UNIT vector in the bone's OWN REST-LOCAL frame; "
            "posed_local = rest_local_q o quat(axis, theta), quaternions [w,x,y,z]. "
            "theta=0 is the rig's rest, theta=fist_angle_deg is the stored Fist preset."),
        "bones": bones,
    }
    Path(out).write_text(json.dumps(doc, indent=1))
    print(f"wrote {out} ({out.stat().st_size / 1024:.1f} KB, {len(bones)} bones)")


if __name__ == "__main__":
    main()
