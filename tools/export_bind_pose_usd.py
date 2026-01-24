"""Export SAM 3D Body bind pose as USD file for Maya import.

Usage:
    python tools/export_bind_pose_usd.py [--output path/to/output.usda]

Output:
    A USDA file with skeleton in T-pose matching data/bind_poses/default_human.json
"""

import argparse
from pathlib import Path


# Bind pose joint positions (from default_human.json)
# Y-up coordinate system, units in meters
BIND_POSE_JOINTS = {
    "root": [0.0, 1.0, 0.0],
    "spine1": [0.0, 1.17, 0.0],
    "spine2": [0.0, 1.33, 0.0],
    "neck": [0.0, 1.5, 0.0],
    "head": [0.0, 1.65, 0.0],
    "nose": [0.0, 1.7, 0.05],
    "left_eye": [-0.03, 1.72, 0.06],
    "right_eye": [0.03, 1.72, 0.06],
    "left_ear": [-0.08, 1.65, 0.0],
    "right_ear": [0.08, 1.65, 0.0],
    "left_clavicle": [-0.01, 1.5, 0.0],
    "left_shoulder": [-0.20, 1.5, 0.0],
    "left_elbow": [-0.45, 1.5, 0.0],
    "left_wrist": [-0.70, 1.5, 0.0],
    "left_thumb_third": [-0.72, 1.48, 0.02],
    "left_thumb_second": [-0.74, 1.46, 0.03],
    "left_thumb_first": [-0.76, 1.44, 0.04],
    "left_thumb_tip": [-0.78, 1.42, 0.05],
    "left_index_third": [-0.75, 1.5, 0.01],
    "left_index_second": [-0.78, 1.5, 0.01],
    "left_index_first": [-0.81, 1.5, 0.01],
    "left_index_tip": [-0.84, 1.5, 0.01],
    "left_middle_third": [-0.75, 1.5, 0.0],
    "left_middle_second": [-0.79, 1.5, 0.0],
    "left_middle_first": [-0.83, 1.5, 0.0],
    "left_middle_tip": [-0.87, 1.5, 0.0],
    "left_ring_third": [-0.75, 1.5, -0.01],
    "left_ring_second": [-0.78, 1.5, -0.01],
    "left_ring_first": [-0.81, 1.5, -0.01],
    "left_ring_tip": [-0.84, 1.5, -0.01],
    "left_pinky_third": [-0.74, 1.5, -0.02],
    "left_pinky_second": [-0.76, 1.5, -0.02],
    "left_pinky_first": [-0.78, 1.5, -0.02],
    "left_pinky_tip": [-0.80, 1.5, -0.02],
    "right_clavicle": [0.01, 1.5, 0.0],
    "right_shoulder": [0.20, 1.5, 0.0],
    "right_elbow": [0.45, 1.5, 0.0],
    "right_wrist": [0.70, 1.5, 0.0],
    "right_thumb_third": [0.72, 1.48, 0.02],
    "right_thumb_second": [0.74, 1.46, 0.03],
    "right_thumb_first": [0.76, 1.44, 0.04],
    "right_thumb_tip": [0.78, 1.42, 0.05],
    "right_index_third": [0.75, 1.5, 0.01],
    "right_index_second": [0.78, 1.5, 0.01],
    "right_index_first": [0.81, 1.5, 0.01],
    "right_index_tip": [0.84, 1.5, 0.01],
    "right_middle_third": [0.75, 1.5, 0.0],
    "right_middle_second": [0.79, 1.5, 0.0],
    "right_middle_first": [0.83, 1.5, 0.0],
    "right_middle_tip": [0.87, 1.5, 0.0],
    "right_ring_third": [0.75, 1.5, -0.01],
    "right_ring_second": [0.78, 1.5, -0.01],
    "right_ring_first": [0.81, 1.5, -0.01],
    "right_ring_tip": [0.84, 1.5, -0.01],
    "right_pinky_third": [0.74, 1.5, -0.02],
    "right_pinky_second": [0.76, 1.5, -0.02],
    "right_pinky_first": [0.78, 1.5, -0.02],
    "right_pinky_tip": [0.80, 1.5, -0.02],
    "left_hip": [-0.1, 1.0, 0.0],
    "left_knee": [-0.1, 0.55, 0.0],
    "left_ankle": [-0.1, 0.08, 0.0],
    "left_heel": [-0.1, 0.0, -0.04],
    "left_big_toe": [-0.08, 0.0, 0.1],
    "left_small_toe": [-0.12, 0.0, 0.08],
    "right_hip": [0.1, 1.0, 0.0],
    "right_knee": [0.1, 0.55, 0.0],
    "right_ankle": [0.1, 0.08, 0.0],
    "right_heel": [0.1, 0.0, -0.04],
    "right_big_toe": [0.08, 0.0, 0.1],
    "right_small_toe": [0.12, 0.0, 0.08],
}

# MHR70 names -> Mannequin names (Maya-safe, no special chars)
MHR70_TO_MANNEQUIN = {
    "root": "pelvis",
    "spine1": "spine_1",
    "spine2": "spine_2",
    "neck": "neck",
    "head": "head",
    "nose": "nose",
    "left_eye": "left_eye",
    "right_eye": "right_eye",
    "left_ear": "left_ear",
    "right_ear": "right_ear",
    "left_clavicle": "left_clavicle",
    "left_shoulder": "left_shoulder",
    "left_elbow": "left_elbow",
    "left_wrist": "left_wrist",
    "left_thumb_third": "left_thumb_1",
    "left_thumb_second": "left_thumb_2",
    "left_thumb_first": "left_thumb_3",
    "left_thumb_tip": "left_thumb_tip",
    "left_index_third": "left_index_finger_1",
    "left_index_second": "left_index_finger_2",
    "left_index_first": "left_index_finger_3",
    "left_index_tip": "left_index_finger_tip",
    "left_middle_third": "left_middle_finger_1",
    "left_middle_second": "left_middle_finger_2",
    "left_middle_first": "left_middle_finger_3",
    "left_middle_tip": "left_middle_finger_tip",
    "left_ring_third": "left_ring_finger_1",
    "left_ring_second": "left_ring_finger_2",
    "left_ring_first": "left_ring_finger_3",
    "left_ring_tip": "left_ring_finger_tip",
    "left_pinky_third": "left_pinky_finger_1",
    "left_pinky_second": "left_pinky_finger_2",
    "left_pinky_first": "left_pinky_finger_3",
    "left_pinky_tip": "left_pinky_finger_tip",
    "right_clavicle": "right_clavicle",
    "right_shoulder": "right_shoulder",
    "right_elbow": "right_elbow",
    "right_wrist": "right_wrist",
    "right_thumb_third": "right_thumb_1",
    "right_thumb_second": "right_thumb_2",
    "right_thumb_first": "right_thumb_3",
    "right_thumb_tip": "right_thumb_tip",
    "right_index_third": "right_index_finger_1",
    "right_index_second": "right_index_finger_2",
    "right_index_first": "right_index_finger_3",
    "right_index_tip": "right_index_finger_tip",
    "right_middle_third": "right_middle_finger_1",
    "right_middle_second": "right_middle_finger_2",
    "right_middle_first": "right_middle_finger_3",
    "right_middle_tip": "right_middle_finger_tip",
    "right_ring_third": "right_ring_finger_1",
    "right_ring_second": "right_ring_finger_2",
    "right_ring_first": "right_ring_finger_3",
    "right_ring_tip": "right_ring_finger_tip",
    "right_pinky_third": "right_pinky_finger_1",
    "right_pinky_second": "right_pinky_finger_2",
    "right_pinky_first": "right_pinky_finger_3",
    "right_pinky_tip": "right_pinky_finger_tip",
    "left_hip": "left_hip",
    "left_knee": "left_knee",
    "left_ankle": "left_ankle",
    "left_heel": "left_heel",
    "left_big_toe": "left_big_toe",
    "left_small_toe": "left_small_toe",
    "right_hip": "right_hip",
    "right_knee": "right_knee",
    "right_ankle": "right_ankle",
    "right_heel": "right_heel",
    "right_big_toe": "right_big_toe",
    "right_small_toe": "right_small_toe",
}

# Skeleton hierarchy (child -> parent) using MHR70 names
BONE_HIERARCHY = {
    "root": None,
    "spine1": "root",
    "spine2": "spine1",
    "neck": "spine2",
    "head": "neck",
    "nose": "head",
    "left_eye": "head",
    "right_eye": "head",
    "left_ear": "head",
    "right_ear": "head",
    "left_clavicle": "spine2",
    "left_shoulder": "left_clavicle",
    "left_elbow": "left_shoulder",
    "left_wrist": "left_elbow",
    "left_thumb_third": "left_wrist",
    "left_thumb_second": "left_thumb_third",
    "left_thumb_first": "left_thumb_second",
    "left_thumb_tip": "left_thumb_first",
    "left_index_third": "left_wrist",
    "left_index_second": "left_index_third",
    "left_index_first": "left_index_second",
    "left_index_tip": "left_index_first",
    "left_middle_third": "left_wrist",
    "left_middle_second": "left_middle_third",
    "left_middle_first": "left_middle_second",
    "left_middle_tip": "left_middle_first",
    "left_ring_third": "left_wrist",
    "left_ring_second": "left_ring_third",
    "left_ring_first": "left_ring_second",
    "left_ring_tip": "left_ring_first",
    "left_pinky_third": "left_wrist",
    "left_pinky_second": "left_pinky_third",
    "left_pinky_first": "left_pinky_second",
    "left_pinky_tip": "left_pinky_first",
    "right_clavicle": "spine2",
    "right_shoulder": "right_clavicle",
    "right_elbow": "right_shoulder",
    "right_wrist": "right_elbow",
    "right_thumb_third": "right_wrist",
    "right_thumb_second": "right_thumb_third",
    "right_thumb_first": "right_thumb_second",
    "right_thumb_tip": "right_thumb_first",
    "right_index_third": "right_wrist",
    "right_index_second": "right_index_third",
    "right_index_first": "right_index_second",
    "right_index_tip": "right_index_first",
    "right_middle_third": "right_wrist",
    "right_middle_second": "right_middle_third",
    "right_middle_first": "right_middle_second",
    "right_middle_tip": "right_middle_first",
    "right_ring_third": "right_wrist",
    "right_ring_second": "right_ring_third",
    "right_ring_first": "right_ring_second",
    "right_ring_tip": "right_ring_first",
    "right_pinky_third": "right_wrist",
    "right_pinky_second": "right_pinky_third",
    "right_pinky_first": "right_pinky_second",
    "right_pinky_tip": "right_pinky_first",
    "left_hip": "root",
    "left_knee": "left_hip",
    "left_ankle": "left_knee",
    "left_heel": "left_ankle",
    "left_big_toe": "left_ankle",
    "left_small_toe": "left_ankle",
    "right_hip": "root",
    "right_knee": "right_hip",
    "right_ankle": "right_knee",
    "right_heel": "right_ankle",
    "right_big_toe": "right_ankle",
    "right_small_toe": "right_ankle",
}


def build_joint_paths():
    """Build full USD paths for each joint."""
    paths = {}

    def get_path(mhr_name):
        if mhr_name in paths:
            return paths[mhr_name]

        mannequin_name = MHR70_TO_MANNEQUIN.get(mhr_name, mhr_name)
        parent = BONE_HIERARCHY.get(mhr_name)

        if parent is None:
            path = f"/SAM3D_Skeleton/{mannequin_name}"
        else:
            parent_path = get_path(parent)
            path = f"{parent_path}/{mannequin_name}"

        paths[mhr_name] = path
        return path

    for joint in BONE_HIERARCHY:
        get_path(joint)

    return paths


def export_usda(output_path, use_mannequin_names=True):
    """Export bind pose skeleton as USDA (ASCII USD) file."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build ordered joint list (parents before children)
    ordered_joints = []
    remaining = set(BONE_HIERARCHY.keys())
    while remaining:
        for joint in list(remaining):
            parent = BONE_HIERARCHY[joint]
            if parent is None or parent not in remaining:
                ordered_joints.append(joint)
                remaining.remove(joint)

    # Build joint paths
    joint_paths = build_joint_paths()

    # Generate USDA content
    lines = [
        '#usda 1.0',
        '(',
        '    defaultPrim = "SAM3D_Skeleton"',
        '    metersPerUnit = 1',
        '    upAxis = "Y"',
        ')',
        '',
        'def Xform "SAM3D_Skeleton" (',
        '    kind = "component"',
        ')',
        '{',
    ]

    # Track which joints we've written (for proper nesting)
    written = set()

    def write_joint(mhr_name, indent_level):
        """Write a joint and its children."""
        mannequin_name = MHR70_TO_MANNEQUIN.get(mhr_name, mhr_name) if use_mannequin_names else mhr_name
        world_pos = BIND_POSE_JOINTS[mhr_name]
        parent = BONE_HIERARCHY.get(mhr_name)

        # Compute local translation
        if parent and parent in BIND_POSE_JOINTS:
            parent_pos = BIND_POSE_JOINTS[parent]
            local_pos = [world_pos[i] - parent_pos[i] for i in range(3)]
        else:
            local_pos = world_pos

        indent = "    " * indent_level

        # Scale to centimeters for Maya (common convention)
        local_pos_cm = [p * 100 for p in local_pos]

        lines.append(f'{indent}def Xform "{mannequin_name}"')
        lines.append(f'{indent}{{')
        lines.append(f'{indent}    double3 xformOp:translate = ({local_pos_cm[0]}, {local_pos_cm[1]}, {local_pos_cm[2]})')
        lines.append(f'{indent}    uniform token[] xformOpOrder = ["xformOp:translate"]')

        # Find and write children
        children = [j for j, p in BONE_HIERARCHY.items() if p == mhr_name]
        for child in children:
            if child in BIND_POSE_JOINTS:
                lines.append('')
                write_joint(child, indent_level + 1)

        lines.append(f'{indent}}}')

    # Write from root
    write_joint("root", 1)

    lines.append('}')
    lines.append('')

    # Write file
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Exported USD to: {output_path}")
    print(f"  Joints: {len(ordered_joints)}")
    print(f"  Units: centimeters (Maya default)")

    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Export SAM 3D Body bind pose as USD"
    )
    parser.add_argument(
        "--output", "-o",
        default="./output/sam3d_bind_pose.usda",
        help="Output USD path (default: ./output/sam3d_bind_pose.usda)"
    )
    parser.add_argument(
        "--mhr70-names",
        action="store_true",
        help="Use MHR70 joint names instead of mannequin names"
    )
    args = parser.parse_args()

    export_usda(args.output, use_mannequin_names=not args.mhr70_names)


if __name__ == "__main__":
    main()
