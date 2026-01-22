# FBX Export for Maya Design

## Overview

Export SAM 3D Body skeleton data to FBX format for Maya rigging and animation workflows. Uses Blender as an intermediary for FBX generation.

## Architecture

```
demo.py --export_fbx
    ↓
tools/fbx_export.py (serialize data to temp JSON)
    ↓
blender --background --python tools/blender_fbx_export.py -- <args>
    ↓
output/skeleton.fbx
```

**Components:**
- `tools/fbx_export.py` - Main export module, CLI support
- `tools/blender_fbx_export.py` - Blender-side script for armature creation and FBX export

## Joint Hierarchy

```
root (at midHip)
│
├── spine1 (1/3 from midHip to neck)
│   └── spine2 (2/3 from midHip to neck)
│       └── neck (index 69 - pit of neck)
│           │
│           ├── head (midpoint of ears)
│           │   ├── left_ear (3)
│           │   ├── right_ear (4)
│           │   ├── left_eye (1)
│           │   ├── right_eye (2)
│           │   └── nose (0)
│           │
│           ├── left_clavicle (5% from neck toward acromion)
│           │   └── left_acromion (67)
│           │       └── left_shoulder (5)
│           │           └── left_elbow (7)
│           │               ├── left_olecranon (63)
│           │               ├── left_cubital_fossa (65)
│           │               └── left_wrist (62)
│           │                   └── [left hand - 21 joints]
│           │
│           └── right_clavicle (5% from neck toward acromion)
│               └── right_acromion (68)
│                   └── right_shoulder (6)
│                       └── right_elbow (8)
│                           ├── right_olecranon (64)
│                           ├── right_cubital_fossa (66)
│                           └── right_wrist (41)
│                               └── [right hand - 21 joints]
│
├── left_hip (9)
│   └── left_knee (11)
│       └── left_ankle (13)
│           ├── left_heel (17)
│           ├── left_big_toe (15)
│           └── left_small_toe (16)
│
└── right_hip (10)
    └── right_knee (12)
        └── right_ankle (14)
            ├── right_heel (20)
            ├── right_big_toe (18)
            └── right_small_toe (19)
```

**Computed joints:**
- `root` = midpoint of left_hip (9) and right_hip (10)
- `spine1` = root + 1/3 * (neck - root)
- `spine2` = root + 2/3 * (neck - root)
- `head` = midpoint of left_ear (3) and right_ear (4)
- `left_clavicle` = neck + 0.05 * (left_acromion - neck)
- `right_clavicle` = neck + 0.05 * (right_acromion - neck)

## Rotation Computation

### Swing (pointing direction)
```python
bind_dir = normalize(bind_child_pos - bind_parent_pos)
detected_dir = normalize(detected_child_pos - detected_parent_pos)
swing = quaternion_from_two_vectors(bind_dir, detected_dir)
```

### Twist (rotation around bone axis)
Uses olecranon and cubital fossa for upper arm twist:
```python
elbow_orient = normalize(olecranon_pos - cubital_fossa_pos)
bind_orient = normalize(bind_olecranon - bind_cubital_fossa)

bone_dir = normalize(elbow_pos - shoulder_pos)
elbow_orient_proj = project_onto_plane(elbow_orient, bone_dir)
bind_orient_proj = project_onto_plane(bind_orient, bone_dir)

twist_angle = signed_angle(bind_orient_proj, elbow_orient_proj, bone_dir)
```

### Rotation Modes

| Mode | Flag | Description |
|------|------|-------------|
| Absolute | `--fbx_rotation_mode absolute` | World-space rotations from bone directions |
| Relative | `--fbx_rotation_mode relative` | First frame is bind pose, subsequent frames relative |
| Bind | `--fbx_rotation_mode bind --fbx_bind_pose <path>` | Relative to stored bind pose file |

## Animation Data

**Single image:** Skeleton at frame 0, no animation curves

**Image sequence:**
- Each image = one frame
- Keyframes on all joints per frame
- Configurable frame rate (default 24 fps)

## Mesh Handling

When `--export_fbx_mesh` is used:
- Include `pred_vertices` and `faces` as mesh object
- Parent to root with Armature modifier
- No skinning weights (mesh is pre-posed)
- For animation: mesh at first frame only (reference)

## CLI Flags

```
--export_fbx              Export FBX skeleton
--export_fbx_mesh         Include mesh in FBX export
--fbx_fps 24              Frame rate for animation (default: 24)
--fbx_rotation_mode       absolute | relative | bind (default: absolute)
--fbx_bind_pose <path>    Path to bind pose JSON (required if mode=bind)
--fbx_output <path>       Custom output path
```

## Error Handling

- **Missing keypoints:** Use parent position, log warning
- **No detections:** Skip image, continue; error if all empty
- **Multi-person:** Separate hierarchies (`skeleton_person0`, etc.), track by position proximity
- **Blender not found:** Clear error with install instructions

## Files

```
tools/
├── fbx_export.py           # Main export module
└── blender_fbx_export.py   # Blender script

data/bind_poses/
└── default_human.json      # Default T-pose bind pose
```
