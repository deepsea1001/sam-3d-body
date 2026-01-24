"""Blender script for FBX export.

Run with:
    blender --background --python blender_fbx_export.py -- input.json output.fbx
"""

import bpy
import json
import sys
import mathutils
from mathutils import Vector, Quaternion, Matrix


def cv_to_blender(pos):
    """Convert from CV coordinates (Y-down) to Blender (Z-up).

    SAM 3D Body uses Y-down: head at Y=-1.6, feet at Y=0
    Blender uses Z-up: head at positive Z, feet at Z=0

    Transformation: X->X, Y->-Z, Z->Y
    """
    x, y, z = pos
    return (x, z, -y)


def clear_scene():
    """Remove all objects from scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()


def create_armature(name, hierarchy, first_frame_positions):
    """Create armature with bone hierarchy.

    Args:
        name: armature name
        hierarchy: list of (joint_name, parent_name, keypoint_idx) tuples
        first_frame_positions: dict of joint positions for rest pose

    Returns:
        armature object
    """
    # Create armature
    bpy.ops.object.armature_add(enter_editmode=True)
    armature = bpy.context.object
    armature.name = name
    armature.data.name = name + "_data"

    # Remove default bone
    bpy.ops.armature.select_all(action='SELECT')
    bpy.ops.armature.delete()

    # Create bones
    edit_bones = armature.data.edit_bones
    bone_map = {}

    for joint_name, parent_name, _ in hierarchy:
        if joint_name not in first_frame_positions:
            print(f"  Skipping {joint_name} - not in positions")
            continue

        pos = first_frame_positions[joint_name]
        # Transform from CV coordinates to Blender coordinates
        head = Vector(cv_to_blender(pos))
        print(f"  Bone {joint_name}: raw={[round(p,3) for p in pos]} -> blender={[round(p,3) for p in head]}")

        # Create bone at world position
        bone = edit_bones.new(joint_name)
        bone.head = head

        # Find children to determine tail direction
        children = [j for j, p, _ in hierarchy if p == joint_name and j in first_frame_positions]

        if children:
            # Point toward first child
            child_pos = first_frame_positions[children[0]]
            bone.tail = Vector(cv_to_blender(child_pos))
        else:
            # Leaf bone - small offset in direction from parent
            if parent_name and parent_name in first_frame_positions:
                parent_pos = Vector(cv_to_blender(first_frame_positions[parent_name]))
                direction = head - parent_pos
                if direction.length > 0.001:
                    direction.normalize()
                    bone.tail = head + direction * 0.02
                else:
                    bone.tail = head + Vector((0, 0, 0.02))
            else:
                bone.tail = head + Vector((0, 0, 0.02))

        # Ensure minimum bone length
        if (bone.tail - bone.head).length < 0.001:
            bone.tail = bone.head + Vector((0, 0, 0.02))

        bone_map[joint_name] = bone

    # Set parents - use_connect=False preserves world position
    for joint_name, parent_name, _ in hierarchy:
        if joint_name in bone_map and parent_name in bone_map:
            bone_map[joint_name].parent = bone_map[parent_name]
            bone_map[joint_name].use_connect = False  # Don't snap to parent tail

    # Exit edit mode
    bpy.ops.object.mode_set(mode='OBJECT')

    return armature


def set_keyframes(armature, people_data, fps, rotation_mode="absolute"):
    """Set keyframes for animation.

    For single frame without bind pose: rest/reference pose (no rotations).
    For single frame with bind pose: apply rotations and scales relative to bind.
    For multi-frame: apply rotations and scales for animation.

    Args:
        armature: armature object
        people_data: list of person data with frames
        fps: frame rate
        rotation_mode: "absolute", "relative", or "bind"
    """
    bpy.context.scene.render.fps = fps

    if not people_data:
        return

    person = people_data[0]
    frames = person.get("frames", [])

    if not frames:
        return

    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = len(frames) - 1

    # Check if we have rotations or scales to apply
    has_rotations = any(frame.get("rotations") for frame in frames)
    has_scales = any(frame.get("scales") for frame in frames)

    if len(frames) == 1 and not has_rotations and not has_scales:
        print("Single frame bind pose - no rotations or scales")
        return

    if has_rotations or has_scales:
        print(f"Applying transforms ({rotation_mode} mode, {len(frames)} frames)")
        print(f"  Rotations: {has_rotations}, Scales: {has_scales}")

        bpy.context.view_layer.objects.active = armature
        bpy.ops.object.mode_set(mode='POSE')

        for frame_data in frames:
            frame_idx = frame_data["frame"]
            rotations = frame_data.get("rotations", {})
            scales = frame_data.get("scales", {})

            bpy.context.scene.frame_set(frame_idx)

            for bone_name in set(rotations.keys()) | set(scales.keys()):
                if bone_name not in armature.pose.bones:
                    continue

                pose_bone = armature.pose.bones[bone_name]

                # Set rotation (quaternion: w, x, y, z)
                if bone_name in rotations:
                    rotation = rotations[bone_name]
                    if len(rotation) == 4:
                        quat = Quaternion((rotation[0], rotation[1], rotation[2], rotation[3]))
                        pose_bone.rotation_mode = 'QUATERNION'
                        pose_bone.rotation_quaternion = quat
                        pose_bone.keyframe_insert(data_path="rotation_quaternion", frame=frame_idx)

                # Set scale (uniform scale along bone axis for length matching)
                if bone_name in scales:
                    scale_factor = scales[bone_name]
                    # Scale along Y axis (bone direction in Blender)
                    # Keep X and Z at 1.0 to preserve bone thickness
                    pose_bone.scale = (1.0, scale_factor, 1.0)
                    pose_bone.keyframe_insert(data_path="scale", frame=frame_idx)

        bpy.ops.object.mode_set(mode='OBJECT')
    else:
        print(f"No rotations or scales to apply")


def create_mesh(name, vertices, faces, armature):
    """Create mesh object and parent to armature.

    Args:
        name: mesh name
        vertices: list of [x, y, z] positions
        faces: list of triangle indices
        armature: parent armature

    Returns:
        mesh object
    """
    if not vertices or not faces:
        return None

    # Create mesh
    mesh = bpy.data.meshes.new(name + "_mesh")
    obj = bpy.data.objects.new(name, mesh)

    # Link to scene
    bpy.context.collection.objects.link(obj)

    # Set mesh data - transform from CV to Blender coordinates
    verts = [Vector(cv_to_blender(v)) for v in vertices]

    # Debug: print mesh bounds
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    zs = [v[2] for v in verts]
    print(f"  Mesh bounds after transform: X=[{min(xs):.3f}, {max(xs):.3f}] Y=[{min(ys):.3f}, {max(ys):.3f}] Z=[{min(zs):.3f}, {max(zs):.3f}]")

    mesh.from_pydata(verts, [], faces)
    mesh.update()

    # Parent to armature with armature modifier
    obj.parent = armature
    modifier = obj.modifiers.new(name="Armature", type='ARMATURE')
    modifier.object = armature

    # Create simple material
    mat = bpy.data.materials.new(name + "_material")
    mat.diffuse_color = (0.7, 0.7, 0.7, 1.0)
    obj.data.materials.append(mat)

    return obj


def export_fbx(filepath):
    """Export scene to FBX."""
    bpy.ops.export_scene.fbx(
        filepath=filepath,
        use_selection=False,
        global_scale=1.0,
        apply_unit_scale=True,
        apply_scale_options='FBX_SCALE_NONE',
        use_space_transform=False,  # We already transformed to Z-up
        bake_space_transform=False,
        object_types={'ARMATURE', 'MESH'},
        use_mesh_modifiers=True,
        use_mesh_modifiers_render=True,
        mesh_smooth_type='OFF',
        use_subsurf=False,
        use_mesh_edges=False,
        use_tspace=False,
        use_triangles=False,
        use_custom_props=False,
        add_leaf_bones=False,
        primary_bone_axis='Y',
        secondary_bone_axis='X',
        use_armature_deform_only=True,
        armature_nodetype='NULL',
        bake_anim=True,
        bake_anim_use_all_bones=True,
        bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=False,
        bake_anim_force_startend_keying=True,
        bake_anim_step=1.0,
        bake_anim_simplify_factor=1.0,
        path_mode='AUTO',
        embed_textures=False,
        batch_mode='OFF',
        use_batch_own_dir=True,
        use_metadata=True,
        axis_forward='-Z',
        axis_up='Y',
    )


def main():
    # Parse command line arguments
    # Arguments after "--" are passed to the script
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    if len(argv) < 2:
        print("Usage: blender --background --python blender_fbx_export.py -- input.json output.fbx")
        sys.exit(1)

    input_json = argv[0]
    output_fbx = argv[1]

    print(f"Loading skeleton data from: {input_json}")
    print(f"Exporting FBX to: {output_fbx}")

    # Load data
    with open(input_json) as f:
        data = json.load(f)

    hierarchy = data.get("hierarchy", [])
    people = data.get("people", [])
    fps = data.get("fps", 24)
    mesh_data = data.get("mesh")
    rotation_mode = data.get("rotation_mode", "absolute")
    bind_pose_positions = data.get("bind_pose_positions")

    if not people:
        print("No skeleton data found")
        sys.exit(1)

    # Get positions for armature rest pose
    first_person = people[0]
    first_frame = first_person["frames"][0] if first_person.get("frames") else {}

    # For bind mode, use bind pose positions; otherwise use first frame
    if rotation_mode == "bind" and bind_pose_positions:
        print("Using bind pose for armature rest pose")
        first_positions = bind_pose_positions
    else:
        first_positions = first_frame.get("positions", {})

    if not first_positions:
        print("No position data found")
        sys.exit(1)

    # Clear scene
    clear_scene()

    # Create armature
    print("Creating armature...")
    armature = create_armature("skeleton", hierarchy, first_positions)

    # Set keyframes
    print("Setting keyframes...")
    rotation_mode = data.get("rotation_mode", "absolute")
    set_keyframes(armature, people, fps, rotation_mode)

    # Create mesh if provided
    if mesh_data:
        print("Creating mesh...")
        vertices = mesh_data.get("vertices", [])
        faces = mesh_data.get("faces", [])
        create_mesh("body", vertices, faces, armature)

    # Export FBX
    print("Exporting FBX...")
    export_fbx(output_fbx)

    print("Done!")


if __name__ == "__main__":
    main()
