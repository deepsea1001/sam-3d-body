"""Blender script to create an armature matching the SAM 3D Body bind pose.

Usage in Blender:
    1. Open Blender
    2. Go to Scripting workspace
    3. Open this file or paste contents
    4. Run script (Alt+P or Run Script button)

The script creates an armature named "SAM3D_BindPose" with bones positioned
to match data/bind_poses/default_human.json.
"""

import bpy
import json
from mathutils import Vector

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
    "left_acromion": [-0.18, 1.5, 0.0],
    "left_shoulder": [-0.20, 1.5, 0.0],
    "left_elbow": [-0.45, 1.5, 0.0],
    "left_olecranon": [-0.45, 1.48, -0.02],
    "left_cubital_fossa": [-0.45, 1.52, 0.02],
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
    "right_acromion": [0.18, 1.5, 0.0],
    "right_shoulder": [0.20, 1.5, 0.0],
    "right_elbow": [0.45, 1.5, 0.0],
    "right_olecranon": [0.45, 1.48, -0.02],
    "right_cubital_fossa": [0.45, 1.52, 0.02],
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

# Joint name mapping: MHR70 names -> Mannequin names (for display)
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

# Skeleton hierarchy (child -> parent)
# These are the joints that map to mannequin; skipping acromion, olecranon, cubital_fossa
BONE_HIERARCHY = {
    # Spine chain
    "root": None,
    "spine1": "root",
    "spine2": "spine1",
    "neck": "spine2",
    "head": "neck",

    # Face (endpoints, no children)
    "nose": "head",
    "left_eye": "head",
    "right_eye": "head",
    "left_ear": "head",
    "right_ear": "head",

    # Left arm
    "left_clavicle": "spine2",
    "left_shoulder": "left_clavicle",
    "left_elbow": "left_shoulder",
    "left_wrist": "left_elbow",

    # Left hand - thumb
    "left_thumb_third": "left_wrist",
    "left_thumb_second": "left_thumb_third",
    "left_thumb_first": "left_thumb_second",
    "left_thumb_tip": "left_thumb_first",

    # Left hand - index
    "left_index_third": "left_wrist",
    "left_index_second": "left_index_third",
    "left_index_first": "left_index_second",
    "left_index_tip": "left_index_first",

    # Left hand - middle
    "left_middle_third": "left_wrist",
    "left_middle_second": "left_middle_third",
    "left_middle_first": "left_middle_second",
    "left_middle_tip": "left_middle_first",

    # Left hand - ring
    "left_ring_third": "left_wrist",
    "left_ring_second": "left_ring_third",
    "left_ring_first": "left_ring_second",
    "left_ring_tip": "left_ring_first",

    # Left hand - pinky
    "left_pinky_third": "left_wrist",
    "left_pinky_second": "left_pinky_third",
    "left_pinky_first": "left_pinky_second",
    "left_pinky_tip": "left_pinky_first",

    # Right arm
    "right_clavicle": "spine2",
    "right_shoulder": "right_clavicle",
    "right_elbow": "right_shoulder",
    "right_wrist": "right_elbow",

    # Right hand - thumb
    "right_thumb_third": "right_wrist",
    "right_thumb_second": "right_thumb_third",
    "right_thumb_first": "right_thumb_second",
    "right_thumb_tip": "right_thumb_first",

    # Right hand - index
    "right_index_third": "right_wrist",
    "right_index_second": "right_index_third",
    "right_index_first": "right_index_second",
    "right_index_tip": "right_index_first",

    # Right hand - middle
    "right_middle_third": "right_wrist",
    "right_middle_second": "right_middle_third",
    "right_middle_first": "right_middle_second",
    "right_middle_tip": "right_middle_first",

    # Right hand - ring
    "right_ring_third": "right_wrist",
    "right_ring_second": "right_ring_third",
    "right_ring_first": "right_ring_second",
    "right_ring_tip": "right_ring_first",

    # Right hand - pinky
    "right_pinky_third": "right_wrist",
    "right_pinky_second": "right_pinky_third",
    "right_pinky_first": "right_pinky_second",
    "right_pinky_tip": "right_pinky_first",

    # Left leg
    "left_hip": "root",
    "left_knee": "left_hip",
    "left_ankle": "left_knee",
    "left_heel": "left_ankle",
    "left_big_toe": "left_ankle",
    "left_small_toe": "left_ankle",

    # Right leg
    "right_hip": "root",
    "right_knee": "right_hip",
    "right_ankle": "right_knee",
    "right_heel": "right_ankle",
    "right_big_toe": "right_ankle",
    "right_small_toe": "right_ankle",
}


def create_bind_pose_armature(use_mannequin_names=True, prefix=""):
    """Create an armature with the SAM 3D Body bind pose.

    Args:
        use_mannequin_names: If True, use mannequin joint names (pelvis, spine_1, etc.)
                            If False, use MHR70 names (root, spine1, etc.)
    """
    # Create armature
    armature = bpy.data.armatures.new("SAM3D_BindPose")
    armature_obj = bpy.data.objects.new("SAM3D_BindPose", armature)

    # Link to scene
    bpy.context.collection.objects.link(armature_obj)
    bpy.context.view_layer.objects.active = armature_obj

    # Enter edit mode
    bpy.ops.object.mode_set(mode='EDIT')

    # Create bones
    edit_bones = armature.edit_bones

    # First pass: create all bones at their positions
    for mhr_name, position in BIND_POSE_JOINTS.items():
        # Skip joints not in hierarchy (acromion, olecranon, cubital_fossa)
        if mhr_name not in BONE_HIERARCHY:
            continue

        # Get display name
        if use_mannequin_names and mhr_name in MHR70_TO_MANNEQUIN:
            bone_name = prefix + MHR70_TO_MANNEQUIN[mhr_name]
        else:
            bone_name = prefix + mhr_name

        bone = edit_bones.new(bone_name)
        # Blender uses Z-up, our data is Y-up, so swap Y and Z
        bone.head = Vector((position[0], -position[2], position[1]))
        # Temporary tail - will be set properly in second pass
        bone.tail = bone.head + Vector((0, 0, 0.05))

    # Second pass: set parent relationships and proper tail positions
    for mhr_name, parent_mhr_name in BONE_HIERARCHY.items():
        if mhr_name not in BIND_POSE_JOINTS:
            continue

        # Get bone names
        if use_mannequin_names and mhr_name in MHR70_TO_MANNEQUIN:
            bone_name = prefix + MHR70_TO_MANNEQUIN[mhr_name]
        else:
            bone_name = prefix + mhr_name

        bone = edit_bones.get(bone_name)
        if not bone:
            continue

        # Set parent
        if parent_mhr_name:
            if use_mannequin_names and parent_mhr_name in MHR70_TO_MANNEQUIN:
                parent_name = prefix + MHR70_TO_MANNEQUIN[parent_mhr_name]
            else:
                parent_name = prefix + parent_mhr_name
            parent_bone = edit_bones.get(parent_name)
            if parent_bone:
                bone.parent = parent_bone

        # Find children to set tail direction
        children = [
            c for c, p in BONE_HIERARCHY.items()
            if p == mhr_name and c in BIND_POSE_JOINTS
        ]

        if children:
            # Point tail toward first child
            child_name = children[0]
            child_pos = BIND_POSE_JOINTS[child_name]
            # Convert Y-up to Z-up
            bone.tail = Vector((child_pos[0], -child_pos[2], child_pos[1]))
        else:
            # Endpoint bone - extend in parent direction
            if bone.parent:
                direction = bone.head - bone.parent.head
                if direction.length > 0.001:
                    direction.normalize()
                    bone.tail = bone.head + direction * 0.03
                else:
                    bone.tail = bone.head + Vector((0, 0, 0.03))
            else:
                bone.tail = bone.head + Vector((0, 0, 0.05))

    # Exit edit mode
    bpy.ops.object.mode_set(mode='OBJECT')

    print(f"Created armature 'SAM3D_BindPose' with {len(edit_bones)} bones")
    return armature_obj


def export_fbx(armature_obj, output_path):
    """Export the armature as FBX."""
    # Select only the armature
    bpy.ops.object.select_all(action='DESELECT')
    armature_obj.select_set(True)
    bpy.context.view_layer.objects.active = armature_obj

    # Export FBX
    bpy.ops.export_scene.fbx(
        filepath=output_path,
        use_selection=True,
        object_types={'ARMATURE'},
        add_leaf_bones=False,
        bake_anim=False,
        axis_forward='-Z',
        axis_up='Y',
    )
    print(f"Exported FBX to: {output_path}")


# Run the script
if __name__ == "__main__":
    import sys

    # Set to False to use MHR70 internal names instead of mannequin names
    USE_MANNEQUIN_NAMES = True

    # Check for command line args (for headless mode)
    output_fbx = None
    prefix = ""
    for i, arg in enumerate(sys.argv):
        if arg == "--output" and i + 1 < len(sys.argv):
            output_fbx = sys.argv[i + 1]
        if arg == "--prefix" and i + 1 < len(sys.argv):
            prefix = sys.argv[i + 1]

    armature = create_bind_pose_armature(use_mannequin_names=USE_MANNEQUIN_NAMES, prefix=prefix)
    print("Done! Armature created.")
    print("\nBone names use:", "Mannequin names" if USE_MANNEQUIN_NAMES else "MHR70 names")
    if prefix:
        print(f"Prefix: '{prefix}'")

    if output_fbx:
        export_fbx(armature, output_fbx)
    else:
        print("\nTo export as FBX: File > Export > FBX, select only Armature")
        print("\nOr run headless:")
        print("  blender --background --python tools/blender_bind_pose_armature.py -- --output output/sam3d_bind_pose.fbx --prefix bind_")
