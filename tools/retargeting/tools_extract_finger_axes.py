#!/usr/bin/env python
"""One-time extraction: the rig's OWN finger flexion axes -> committed JSON.

Dev-only. Reads PoseGoblin's shipped hand-pose presets (a sibling repo, not
a runtime dependency) and the v2 rig asset; runtime code loads only the
emitted JSON.

Why measure instead of assume: v16 task 5's first design applied MHR's
world rotation DELTA to each phalange. That is right when the two rigs'
rests nearly agree (the spine, 7.6-31.6 deg apart) and wrong here -- the
two rests' finger directions are 3-49 deg apart (median 27), and the transferred hand
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
each bone's own local frame.

R14 (2026-08-22): this used to redistribute the ANGLE as well -- the right
hand's rest-to-Fist arcs came out uniformly 0.6300 of the left's, on all 15
bones, because the v2 asset's rest was itself 63% closed on the right (it
descended from `capture-rigrest-02.json`, captured with the right hand posed).
The asset now takes its finger rest from the rig's true base pose, and the two
hands' arcs agree to 1e-4 deg. The AXIS directions are unchanged to 0.0023 deg
by that correction -- which is itself the corroboration that the miscapture was
pure flexion about these very axes: a rest displaced ALONG the axis cannot
change the direction of `conj(rest) o Fist`, only its magnitude.

The left/right agreement is ASSERTED below, not assumed and never averaged: a
disagreement would mean the base pose is not symmetric after all, which is a
finding, not something to smooth over.
"""
import datetime
import json
from pathlib import Path

import numpy as np

from retargeting.core.math_utils import QuaternionMath
from retargeting.retargeters.posegoblin_rig import load_rig

# Sibling repo (see module docstring), not a runtime dependency. Kept
# REPO-RELATIVE, not the absolute /Users/scotteaton/... path this used to
# hardcode -- e0e9426 already ruled out exactly that shape of reference (a
# cross-repo absolute path baked into committed provenance) for the v2 rig
# asset's provenance.note; this is the same fix applied to this asset's
# provenance.hand_poses, which `str(hand_poses)` below writes verbatim.
_SAM3D_BODY_ROOT = Path(__file__).resolve().parents[2]   # .../sam-3d-body
HAND_POSES = Path("../poseGoblin/poses/hand_poses.json")   # relative to the repo root
OUT = Path(__file__).resolve().parent / "bind_poses" / "mannequin_finger_axes.json"
REFERENCE_POSE = "Fist"
# R14 left/right agreement bounds. Measured on the corrected asset: 0.0023 deg
# and 1.0e-4 deg -- these are ~40x and ~10x that, still four orders of magnitude
# below the 34 deg asymmetry the old asset carried.
AXIS_SYMMETRY_TOL_DEG = 0.1
ARC_SYMMETRY_TOL_DEG = 0.001

_DIGITS = {"thumb": ["thumb_1", "thumb_2", "thumb_3"]}
for _f in ("index", "middle", "ring", "pinky"):
    _DIGITS[_f] = [f"{_f}_finger_{k}" for k in (1, 2, 3)]


def _shortest(q):
    """Same rotation, w >= 0 -- so to_axis_angle returns the <=180 deg arc."""
    q = QuaternionMath.normalize(np.asarray(q, float))
    return -q if q[0] < 0.0 else q


def main(hand_poses: Path = HAND_POSES, out: Path = OUT) -> None:
    # *hand_poses* is recorded into provenance below AS GIVEN (relative by
    # default) -- resolved against the repo root only for the actual read,
    # never for what gets written, so the committed asset stays portable.
    hand_poses_path = Path(hand_poses)
    if not hand_poses_path.is_absolute():
        hand_poses_path = (_SAM3D_BODY_ROOT / hand_poses_path).resolve()
    presets = {e["name"]: e["pose"] for e in json.loads(hand_poses_path.read_text())}
    if REFERENCE_POSE not in presets:
        raise SystemExit(f"{hand_poses_path} has no {REFERENCE_POSE!r} preset "
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

    # R14 gate: the two hands must now agree, per bone, on BOTH the axis
    # direction and the arc. Fail loudly rather than average -- see the module
    # docstring. Measured on the corrected asset: 0.0023 deg / 1.0e-4 deg.
    worst_axis, worst_arc, worst_name = 0.0, 0.0, None
    for names in _DIGITS.values():
        for nm in names:
            a = np.asarray(bones["left_" + nm]["axis"], float)
            b = np.asarray(bones["right_" + nm]["axis"], float)
            ang = float(np.degrees(np.arccos(min(1.0, max(-1.0, float(a @ b))))))
            worst_arc = max(worst_arc, abs(bones["left_" + nm]["fist_angle_deg"]
                                           - bones["right_" + nm]["fist_angle_deg"]))
            if ang > worst_axis:
                worst_axis, worst_name = ang, nm
    if worst_axis > AXIS_SYMMETRY_TOL_DEG or worst_arc > ARC_SYMMETRY_TOL_DEG:
        raise SystemExit(
            f"left/right finger axes DISAGREE (worst axis {worst_axis:.4f} deg on "
            f"{worst_name}, worst arc {worst_arc:.4f} deg). The rig asset's two hands "
            f"are not at the same rest -- do not average this away, find out why.")

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
    print(f"  left/right agreement: axis {worst_axis:.2e} deg (worst on {worst_name}), "
          f"arc {worst_arc:.2e} deg")
    arcs = [v["fist_angle_deg"] for v in bones.values()]
    print(f"  rest->Fist arcs: {min(arcs):.2f}..{max(arcs):.2f} deg")


if __name__ == "__main__":
    main()
