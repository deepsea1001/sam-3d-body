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


def set_keyframes(armature, people_data, fps):
    """Set keyframes for animation.

    For single frame: this is a bind/reference pose - no rotations needed.
    The bone positions in edit mode define the pose.

    For multi-frame: rotations would be applied relative to bind pose.

    Args:
        armature: armature object
        people_data: list of person data with frames
        fps: frame rate
    """
    # Set frame rate
    bpy.context.scene.render.fps = fps

    if not people_data:
        return

    person = people_data[0]
    frames = person.get("frames", [])

    if not frames:
        return

    # Set frame range
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = len(frames) - 1

    # For single frame (bind pose), no animation needed
    # The rest pose defined by bone positions IS the pose
    if len(frames) == 1:
        print("Single frame - using as bind pose (no rotations)")
        return

    # Multi-frame animation would go here
    # TODO: implement proper rotation animation for sequences
    print(f"Multi-frame animation: {len(frames)} frames (not yet implemented)")
    bpy.ops.object.mode_set(mode='OBJECT')


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

    if not people:
        print("No skeleton data found")
        sys.exit(1)

    # Get first frame positions for rest pose
    first_person = people[0]
    first_frame = first_person["frames"][0] if first_person.get("frames") else {}
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
    set_keyframes(armature, people, fps)

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
