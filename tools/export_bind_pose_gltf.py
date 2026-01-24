"""Export SAM 3D Body bind pose as GLTF file for import into Maya/Blender/etc.

Usage:
    python tools/export_bind_pose_gltf.py [--output path/to/output.gltf]

Output:
    A GLTF file with skeleton in T-pose matching data/bind_poses/default_human.json
"""

import argparse
import json
import struct
import base64
from pathlib import Path
import numpy as np


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

# MHR70 names -> Mannequin names
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


def compute_inverse_bind_matrix(world_position):
    """Compute 4x4 inverse bind matrix for a joint at world_position."""
    # The inverse bind matrix transforms from world space to bone space
    # For a joint with only translation (no rotation in bind pose),
    # it's just the inverse translation
    mat = np.eye(4, dtype=np.float32)
    mat[0, 3] = -world_position[0]
    mat[1, 3] = -world_position[1]
    mat[2, 3] = -world_position[2]
    return mat


def build_gltf(use_mannequin_names=True):
    """Build GLTF structure with skeleton."""

    # Order joints for processing (parents before children)
    ordered_joints = []
    remaining = set(BONE_HIERARCHY.keys())

    while remaining:
        for joint in list(remaining):
            parent = BONE_HIERARCHY[joint]
            if parent is None or parent not in remaining:
                ordered_joints.append(joint)
                remaining.remove(joint)

    # Build node index mapping
    joint_to_node_idx = {joint: i for i, joint in enumerate(ordered_joints)}

    # Build nodes array
    nodes = []
    for mhr_name in ordered_joints:
        world_pos = np.array(BIND_POSE_JOINTS[mhr_name])
        parent_mhr = BONE_HIERARCHY[mhr_name]

        # Compute local translation (offset from parent)
        if parent_mhr is not None:
            parent_pos = np.array(BIND_POSE_JOINTS[parent_mhr])
            local_translation = world_pos - parent_pos
        else:
            local_translation = world_pos

        # Get display name
        if use_mannequin_names:
            display_name = MHR70_TO_MANNEQUIN.get(mhr_name, mhr_name)
        else:
            display_name = mhr_name

        node = {
            "name": display_name,
            "translation": [float(local_translation[0]),
                           float(local_translation[1]),
                           float(local_translation[2])],
        }

        # Add children
        children_indices = [
            joint_to_node_idx[child]
            for child, parent in BONE_HIERARCHY.items()
            if parent == mhr_name
        ]
        if children_indices:
            node["children"] = children_indices

        nodes.append(node)

    # Build inverse bind matrices
    inverse_bind_matrices = []
    for mhr_name in ordered_joints:
        world_pos = BIND_POSE_JOINTS[mhr_name]
        ibm = compute_inverse_bind_matrix(world_pos)
        inverse_bind_matrices.append(ibm)

    # Pack matrices into binary buffer
    ibm_data = b''
    for mat in inverse_bind_matrices:
        # GLTF uses column-major order
        for col in range(4):
            for row in range(4):
                ibm_data += struct.pack('<f', mat[row, col])

    # Encode as base64 for embedded buffer
    ibm_base64 = base64.b64encode(ibm_data).decode('ascii')

    # Find root node (the one with no parent)
    root_idx = joint_to_node_idx["root"]

    # Add a scene root that contains the skeleton
    scene_root_idx = len(nodes)
    nodes.append({
        "name": "SAM3D_BindPose_Root",
        "children": [root_idx],
    })

    # Build skin
    skin = {
        "name": "SAM3D_Skeleton",
        "joints": list(range(len(ordered_joints))),
        "skeleton": root_idx,
        "inverseBindMatrices": 0,  # accessor index
    }

    # Build GLTF structure
    gltf = {
        "asset": {
            "version": "2.0",
            "generator": "SAM 3D Body Bind Pose Exporter",
        },
        "scene": 0,
        "scenes": [
            {
                "name": "SAM3D_BindPose",
                "nodes": [scene_root_idx],
            }
        ],
        "nodes": nodes,
        "skins": [skin],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,  # FLOAT
                "count": len(ordered_joints),
                "type": "MAT4",
            }
        ],
        "bufferViews": [
            {
                "buffer": 0,
                "byteLength": len(ibm_data),
                "byteOffset": 0,
            }
        ],
        "buffers": [
            {
                "uri": f"data:application/octet-stream;base64,{ibm_base64}",
                "byteLength": len(ibm_data),
            }
        ],
    }

    return gltf, ordered_joints


def export_gltf(output_path, use_mannequin_names=True):
    """Export bind pose skeleton as GLTF file."""
    gltf, joints = build_gltf(use_mannequin_names)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(gltf, f, indent=2)

    print(f"Exported GLTF to: {output_path}")
    print(f"  Joints: {len(joints)}")
    print(f"  Names: {'Mannequin' if use_mannequin_names else 'MHR70'}")

    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Export SAM 3D Body bind pose as GLTF"
    )
    parser.add_argument(
        "--output", "-o",
        default="./output/sam3d_bind_pose.gltf",
        help="Output GLTF path (default: ./output/sam3d_bind_pose.gltf)"
    )
    parser.add_argument(
        "--mhr70-names",
        action="store_true",
        help="Use MHR70 joint names instead of mannequin names"
    )
    args = parser.parse_args()

    export_gltf(args.output, use_mannequin_names=not args.mhr70_names)


if __name__ == "__main__":
    main()
