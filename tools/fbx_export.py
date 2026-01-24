"""FBX export for Maya rigging workflow.

Uses Blender as intermediary to generate FBX files from SAM 3D Body outputs.
"""

import json
import numpy as np
import subprocess
import tempfile
import shutil
import os
from pathlib import Path

# Joint hierarchy definition
# Format: (joint_name, parent_name, keypoint_index or None for computed)
JOINT_HIERARCHY = [
    # Spine
    ("root", None, None),  # computed: midpoint of hips
    ("spine1", "root", None),  # computed: 1/3 from root to neck
    ("spine2", "spine1", None),  # computed: 2/3 from root to neck
    ("neck", "spine2", 69),  # pit of neck

    # Head
    ("head", "neck", None),  # computed: midpoint of ears
    ("left_ear", "head", 3),
    ("right_ear", "head", 4),
    ("left_eye", "head", 1),
    ("right_eye", "head", 2),
    ("nose", "head", 0),

    # Left arm chain
    ("left_clavicle", "neck", None),  # computed: 5% from neck to acromion
    ("left_acromion", "left_clavicle", 67),
    ("left_shoulder", "left_acromion", 5),
    ("left_elbow", "left_shoulder", 7),
    ("left_olecranon", "left_elbow", 63),
    ("left_cubital_fossa", "left_elbow", 65),
    ("left_wrist", "left_elbow", 62),

    # Left hand
    ("left_thumb_third", "left_wrist", 45),
    ("left_thumb_second", "left_thumb_third", 44),
    ("left_thumb_first", "left_thumb_second", 43),
    ("left_thumb_tip", "left_thumb_first", 42),
    ("left_index_third", "left_wrist", 49),
    ("left_index_second", "left_index_third", 48),
    ("left_index_first", "left_index_second", 47),
    ("left_index_tip", "left_index_first", 46),
    ("left_middle_third", "left_wrist", 53),
    ("left_middle_second", "left_middle_third", 52),
    ("left_middle_first", "left_middle_second", 51),
    ("left_middle_tip", "left_middle_first", 50),
    ("left_ring_third", "left_wrist", 57),
    ("left_ring_second", "left_ring_third", 56),
    ("left_ring_first", "left_ring_second", 55),
    ("left_ring_tip", "left_ring_first", 54),
    ("left_pinky_third", "left_wrist", 61),
    ("left_pinky_second", "left_pinky_third", 60),
    ("left_pinky_first", "left_pinky_second", 59),
    ("left_pinky_tip", "left_pinky_first", 58),

    # Right arm chain
    ("right_clavicle", "neck", None),  # computed: 5% from neck to acromion
    ("right_acromion", "right_clavicle", 68),
    ("right_shoulder", "right_acromion", 6),
    ("right_elbow", "right_shoulder", 8),
    ("right_olecranon", "right_elbow", 64),
    ("right_cubital_fossa", "right_elbow", 66),
    ("right_wrist", "right_elbow", 41),

    # Right hand
    ("right_thumb_third", "right_wrist", 24),
    ("right_thumb_second", "right_thumb_third", 23),
    ("right_thumb_first", "right_thumb_second", 22),
    ("right_thumb_tip", "right_thumb_first", 21),
    ("right_index_third", "right_wrist", 28),
    ("right_index_second", "right_index_third", 27),
    ("right_index_first", "right_index_second", 26),
    ("right_index_tip", "right_index_first", 25),
    ("right_middle_third", "right_wrist", 32),
    ("right_middle_second", "right_middle_third", 31),
    ("right_middle_first", "right_middle_second", 30),
    ("right_middle_tip", "right_middle_first", 29),
    ("right_ring_third", "right_wrist", 36),
    ("right_ring_second", "right_ring_third", 35),
    ("right_ring_first", "right_ring_second", 34),
    ("right_ring_tip", "right_ring_first", 33),
    ("right_pinky_third", "right_wrist", 40),
    ("right_pinky_second", "right_pinky_third", 39),
    ("right_pinky_first", "right_pinky_second", 38),
    ("right_pinky_tip", "right_pinky_first", 37),

    # Left leg
    ("left_hip", "root", 9),
    ("left_knee", "left_hip", 11),
    ("left_ankle", "left_knee", 13),
    ("left_heel", "left_ankle", 17),
    ("left_big_toe", "left_ankle", 15),
    ("left_small_toe", "left_ankle", 16),

    # Right leg
    ("right_hip", "root", 10),
    ("right_knee", "right_hip", 12),
    ("right_ankle", "right_knee", 14),
    ("right_heel", "right_ankle", 20),
    ("right_big_toe", "right_ankle", 18),
    ("right_small_toe", "right_ankle", 19),
]


def compute_joint_positions(keypoints_3d):
    """Compute all joint positions including derived joints.

    Args:
        keypoints_3d: (70, 3) array of keypoint positions

    Returns:
        dict mapping joint name to (x, y, z) position
    """
    kps = np.array(keypoints_3d)
    positions = {}

    # First pass: direct keypoint mappings
    for joint_name, parent_name, kp_idx in JOINT_HIERARCHY:
        if kp_idx is not None:
            positions[joint_name] = kps[kp_idx].tolist()

    # Computed joints
    left_hip = kps[9]
    right_hip = kps[10]
    neck = kps[69]
    left_ear = kps[3]
    right_ear = kps[4]
    left_acromion = kps[67]
    right_acromion = kps[68]

    # Root = midpoint of hips
    root = (left_hip + right_hip) / 2
    positions["root"] = root.tolist()

    # Spine joints - interpolate between root and neck
    spine_vec = neck - root
    positions["spine1"] = (root + spine_vec / 3).tolist()
    positions["spine2"] = (root + spine_vec * 2 / 3).tolist()

    # Head = midpoint of ears
    positions["head"] = ((left_ear + right_ear) / 2).tolist()

    # Clavicles = 5% from neck toward acromion
    positions["left_clavicle"] = (neck + 0.05 * (left_acromion - neck)).tolist()
    positions["right_clavicle"] = (neck + 0.05 * (right_acromion - neck)).tolist()

    return positions


def compute_rotation_absolute(parent_pos, child_pos):
    """Compute rotation to point from parent to child (world space).

    Returns quaternion as [w, x, y, z].
    """
    direction = np.array(child_pos) - np.array(parent_pos)
    length = np.linalg.norm(direction)

    if length < 1e-6:
        return [1, 0, 0, 0]  # Identity quaternion

    direction = direction / length

    # Default bone direction is +Y in Blender
    up = np.array([0, 1, 0])

    # Compute rotation from up to direction
    dot = np.dot(up, direction)

    if dot > 0.9999:
        return [1, 0, 0, 0]
    elif dot < -0.9999:
        return [0, 1, 0, 0]  # 180 degree rotation around X

    axis = np.cross(up, direction)
    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(np.clip(dot, -1, 1))

    # Quaternion from axis-angle
    w = np.cos(angle / 2)
    xyz = axis * np.sin(angle / 2)

    return [float(w), float(xyz[0]), float(xyz[1]), float(xyz[2])]


def compute_twist_rotation(shoulder_pos, elbow_pos, olecranon_pos, cubital_fossa_pos,
                           bind_olecranon=None, bind_cubital_fossa=None):
    """Compute twist rotation for upper arm using elbow orientation.

    Returns twist angle in radians.
    """
    if bind_olecranon is None or bind_cubital_fossa is None:
        return 0.0

    # Bone direction
    bone_dir = np.array(elbow_pos) - np.array(shoulder_pos)
    bone_length = np.linalg.norm(bone_dir)
    if bone_length < 1e-6:
        return 0.0
    bone_dir = bone_dir / bone_length

    # Elbow orientation vector (front to back)
    elbow_orient = np.array(olecranon_pos) - np.array(cubital_fossa_pos)
    bind_orient = np.array(bind_olecranon) - np.array(bind_cubital_fossa)

    # Project onto plane perpendicular to bone
    def project_onto_plane(v, normal):
        return v - np.dot(v, normal) * normal

    elbow_proj = project_onto_plane(elbow_orient, bone_dir)
    bind_proj = project_onto_plane(bind_orient, bone_dir)

    len_elbow = np.linalg.norm(elbow_proj)
    len_bind = np.linalg.norm(bind_proj)

    if len_elbow < 1e-6 or len_bind < 1e-6:
        return 0.0

    elbow_proj = elbow_proj / len_elbow
    bind_proj = bind_proj / len_bind

    # Signed angle between them
    dot = np.clip(np.dot(bind_proj, elbow_proj), -1, 1)
    angle = np.arccos(dot)

    # Determine sign using cross product
    cross = np.cross(bind_proj, elbow_proj)
    if np.dot(cross, bone_dir) < 0:
        angle = -angle

    return float(angle)


def build_children_map():
    """Build a map of joint -> list of children from JOINT_HIERARCHY."""
    children = {}
    for joint, parent, _ in JOINT_HIERARCHY:
        if parent is not None:
            if parent not in children:
                children[parent] = []
            children[parent].append(joint)
    return children


def prepare_skeleton_data(outputs, rotation_mode="absolute", bind_pose=None, fps=24):
    """Prepare skeleton data for Blender export.

    Args:
        outputs: list of dicts from SAM 3D Body pipeline (one per frame)
                 or list of lists for multi-person
        rotation_mode: "absolute", "relative", or "bind"
        bind_pose: dict with joint positions for bind mode
        fps: frame rate for animation

    Returns:
        dict with skeleton data for Blender
    """
    # Handle single frame vs sequence
    if not isinstance(outputs, list):
        outputs = [outputs]

    # Check if first element is a list (multi-person per frame)
    if outputs and isinstance(outputs[0], list):
        frames = outputs
    else:
        # Single person or single frame - wrap in list
        frames = [outputs]

    all_people = []

    # Process each person across frames
    # For simplicity, assume consistent person count and order
    if not frames or not frames[0]:
        return {"people": [], "fps": fps, "hierarchy": JOINT_HIERARCHY}

    num_people = len(frames[0])

    for person_idx in range(num_people):
        person_frames = []

        for frame_idx, frame_outputs in enumerate(frames):
            if person_idx >= len(frame_outputs):
                continue

            person_output = frame_outputs[person_idx]
            keypoints_3d = person_output.get("pred_keypoints_3d")

            if keypoints_3d is None:
                continue

            if not isinstance(keypoints_3d, np.ndarray):
                keypoints_3d = np.array(keypoints_3d)

            # Compute joint positions
            positions = compute_joint_positions(keypoints_3d)

            # Compute rotations based on mode
            rotations = {}
            scales = {}  # bone scale factors (for retargeting)

            if rotation_mode == "absolute":
                # Compute world-space rotations from bone directions
                for joint_name, parent_name, _ in JOINT_HIERARCHY:
                    if parent_name is None:
                        rotations[joint_name] = [1, 0, 0, 0]
                    elif joint_name in positions and parent_name in positions:
                        rotations[joint_name] = compute_rotation_absolute(
                            positions[parent_name], positions[joint_name]
                        )

            elif rotation_mode == "bind" and bind_pose:
                # Compute LOCAL rotations and SCALES relative to bind pose
                # FK animation: rotation on joint controls bone FROM that joint TO its child
                # Scale compensates for bone length differences (retargeting)

                children_map = build_children_map()
                world_rotations = {}

                for joint_name, parent_name, _ in JOINT_HIERARCHY:
                    # Get first child to determine bone direction
                    children = children_map.get(joint_name, [])
                    first_child = children[0] if children else None

                    if first_child is None:
                        # Leaf joint - no bone to rotate/scale
                        rotations[joint_name] = [1, 0, 0, 0]
                        scales[joint_name] = 1.0
                        world_rotations[joint_name] = world_rotations.get(parent_name, [1, 0, 0, 0])
                        continue

                    if joint_name not in positions or first_child not in positions:
                        rotations[joint_name] = [1, 0, 0, 0]
                        scales[joint_name] = 1.0
                        world_rotations[joint_name] = world_rotations.get(parent_name, [1, 0, 0, 0])
                        continue

                    if joint_name not in bind_pose or first_child not in bind_pose:
                        rotations[joint_name] = [1, 0, 0, 0]
                        scales[joint_name] = 1.0
                        world_rotations[joint_name] = world_rotations.get(parent_name, [1, 0, 0, 0])
                        continue

                    # Compute bone vectors: joint -> first_child
                    bind_vec_cv = np.array(bind_pose[first_child]) - np.array(bind_pose[joint_name])
                    curr_vec_cv = np.array(positions[first_child]) - np.array(positions[joint_name])

                    # Bone lengths for scaling
                    bind_length = np.linalg.norm(bind_vec_cv)
                    curr_length = np.linalg.norm(curr_vec_cv)

                    # Scale factor: target_length / bind_length
                    if bind_length > 1e-6:
                        scales[joint_name] = float(curr_length / bind_length)
                    else:
                        scales[joint_name] = 1.0

                    # Transform directions to Blender space
                    bind_dir = cv_to_blender_vec(bind_vec_cv)
                    curr_dir = cv_to_blender_vec(curr_vec_cv)

                    # Get parent's accumulated world rotation
                    parent_world_rot = world_rotations.get(parent_name, [1, 0, 0, 0])

                    # Compute world-space rotation from bind_dir to curr_dir
                    world_rot = quaternion_from_two_vectors(bind_dir, curr_dir)

                    # Local rotation = parent^-1 * world_rot
                    parent_inv = quat_conjugate(parent_world_rot)
                    local_rot = quat_multiply(parent_inv, world_rot)

                    # INVERT the rotation - Blender may apply it in opposite sense
                    local_rot = quat_conjugate(local_rot)

                    rotations[joint_name] = local_rot

                    # This bone's world rotation for its children
                    # Since we inverted local_rot, world = parent * local^-1 won't give world_rot
                    # But for hierarchy, we still track the TARGET world orientation
                    world_rotations[joint_name] = world_rot

            elif rotation_mode == "relative" and frame_idx == 0:
                # First frame becomes bind pose for relative mode
                bind_pose = positions.copy()
                for joint_name, parent_name, _ in JOINT_HIERARCHY:
                    rotations[joint_name] = [1, 0, 0, 0]

            elif rotation_mode == "relative" and bind_pose:
                # Subsequent frames relative to first - use local rotations
                # FK animation: rotation on joint controls bone FROM that joint TO its child
                children_map = build_children_map()
                world_rotations = {}

                for joint_name, parent_name, _ in JOINT_HIERARCHY:
                    children = children_map.get(joint_name, [])
                    first_child = children[0] if children else None

                    if first_child is None:
                        rotations[joint_name] = [1, 0, 0, 0]
                        world_rotations[joint_name] = world_rotations.get(parent_name, [1, 0, 0, 0])
                        continue

                    if joint_name not in positions or first_child not in positions:
                        rotations[joint_name] = [1, 0, 0, 0]
                        world_rotations[joint_name] = world_rotations.get(parent_name, [1, 0, 0, 0])
                        continue

                    if joint_name not in bind_pose or first_child not in bind_pose:
                        rotations[joint_name] = [1, 0, 0, 0]
                        world_rotations[joint_name] = world_rotations.get(parent_name, [1, 0, 0, 0])
                        continue

                    bind_dir_cv = np.array(bind_pose[first_child]) - np.array(bind_pose[joint_name])
                    curr_dir_cv = np.array(positions[first_child]) - np.array(positions[joint_name])
                    bind_dir = cv_to_blender_vec(bind_dir_cv)
                    curr_dir = cv_to_blender_vec(curr_dir_cv)

                    parent_world_rot = world_rotations.get(parent_name, [1, 0, 0, 0])

                    # World-space rotation from bind to current
                    world_rot = quaternion_from_two_vectors(bind_dir, curr_dir)

                    # Local rotation = parent^-1 * world_rot
                    parent_inv = quat_conjugate(parent_world_rot)
                    local_rot = quat_multiply(parent_inv, world_rot)
                    rotations[joint_name] = local_rot
                    world_rotations[joint_name] = world_rot

            # Include scales if computed (bind mode retargeting)
            frame_data = {
                "frame": frame_idx,
                "positions": positions,
                "rotations": rotations
            }
            if scales:
                frame_data["scales"] = scales

            person_frames.append(frame_data)

        if person_frames:
            all_people.append({
                "person_idx": person_idx,
                "frames": person_frames
            })

    # For bind mode, include bind_pose positions for armature creation
    result = {
        "people": all_people,
        "fps": fps,
        "hierarchy": [(j, p, k) for j, p, k in JOINT_HIERARCHY],
        "rotation_mode": rotation_mode,
    }

    # If using bind mode, armature should be created from bind pose
    if rotation_mode == "bind" and bind_pose:
        result["bind_pose_positions"] = bind_pose

    return result


def cv_to_blender_vec(v):
    """Transform vector from CV coordinates (Y-down) to Blender (Z-up).

    Same transformation as positions: X->X, Y->-Z, Z->Y
    """
    v = np.array(v, dtype=float)
    return np.array([v[0], v[2], -v[1]])


def quaternion_from_two_vectors(v1, v2):
    """Compute quaternion that rotates v1 to v2.

    Returns [w, x, y, z].
    """
    v1 = np.array(v1, dtype=float)
    v2 = np.array(v2, dtype=float)

    len1 = np.linalg.norm(v1)
    len2 = np.linalg.norm(v2)

    if len1 < 1e-6 or len2 < 1e-6:
        return [1, 0, 0, 0]

    v1 = v1 / len1
    v2 = v2 / len2

    dot = np.dot(v1, v2)

    if dot > 0.9999:
        return [1, 0, 0, 0]
    elif dot < -0.9999:
        # 180 degree rotation - find perpendicular axis
        perp = np.array([1, 0, 0]) if abs(v1[0]) < 0.9 else np.array([0, 1, 0])
        axis = np.cross(v1, perp)
        axis = axis / np.linalg.norm(axis)
        return [0, float(axis[0]), float(axis[1]), float(axis[2])]

    axis = np.cross(v1, v2)

    # Quaternion: w = 1 + dot, xyz = cross
    # Then normalize
    w = 1 + dot
    quat = np.array([w, axis[0], axis[1], axis[2]])
    quat = quat / np.linalg.norm(quat)

    return quat.tolist()


def quat_multiply(q1, q2):
    """Multiply two quaternions q1 * q2.

    Both quaternions are [w, x, y, z] format.
    Returns [w, x, y, z].
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return [w, x, y, z]


def quat_conjugate(q):
    """Return conjugate of quaternion (inverse for unit quaternions)."""
    return [q[0], -q[1], -q[2], -q[3]]


def quat_rotate_vector(q, v):
    """Rotate vector v by quaternion q.

    Args:
        q: quaternion [w, x, y, z]
        v: vector [x, y, z]

    Returns:
        rotated vector [x, y, z]
    """
    # Convert vector to quaternion with w=0
    v_quat = [0, v[0], v[1], v[2]]

    # Rotate: q * v * q^-1
    q_conj = quat_conjugate(q)
    result = quat_multiply(quat_multiply(q, v_quat), q_conj)

    return np.array([result[1], result[2], result[3]])


def find_blender():
    """Find Blender executable."""
    # Check environment variable first
    blender_path = os.environ.get("BLENDER_PATH")
    if blender_path and os.path.exists(blender_path):
        return blender_path

    # Common locations
    common_paths = [
        "/Applications/Blender.app/Contents/MacOS/Blender",  # macOS
        "blender",  # Linux/Windows PATH
        "/usr/bin/blender",
        "/usr/local/bin/blender",
    ]

    for path in common_paths:
        if shutil.which(path):
            return path
        if os.path.exists(path):
            return path

    return None


def export_fbx(outputs, output_path, include_mesh=False, rotation_mode="absolute",
               bind_pose_path=None, fps=24, faces=None, verbose=True):
    """Export skeleton to FBX using Blender.

    Args:
        outputs: pipeline outputs (single frame or list of frames)
        output_path: path to save FBX file
        include_mesh: whether to include mesh geometry
        rotation_mode: "absolute", "relative", or "bind"
        bind_pose_path: path to bind pose JSON (required if mode="bind")
        fps: frame rate for animation
        faces: mesh faces (required if include_mesh=True)
        verbose: print progress messages

    Returns:
        Path to exported FBX file
    """
    # Find Blender
    blender = find_blender()
    if not blender:
        raise RuntimeError(
            "Blender not found. Install from blender.org or set BLENDER_PATH environment variable"
        )

    # Load bind pose if needed
    bind_pose = None
    if rotation_mode == "bind":
        if not bind_pose_path:
            raise ValueError("bind_pose_path required when rotation_mode='bind'")
        bind_pose = load_bind_pose(bind_pose_path)
        if verbose:
            print(f"Loaded bind pose with {len(bind_pose)} joints")

    # Prepare skeleton data
    skeleton_data = prepare_skeleton_data(outputs, rotation_mode, bind_pose, fps)

    # Add mesh data if requested
    if include_mesh and outputs:
        first_output = outputs[0] if isinstance(outputs[0], list) else outputs
        if isinstance(first_output, list):
            first_output = first_output[0]

        if "pred_vertices" in first_output:
            vertices = first_output["pred_vertices"]
            if not isinstance(vertices, list):
                vertices = vertices.tolist()
            skeleton_data["mesh"] = {
                "vertices": vertices,
                "faces": faces.tolist() if faces is not None else None
            }

    # Write to temp file (also save a debug copy)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(skeleton_data, f, indent=2)
        temp_json = f.name

    # Save debug copy next to output
    debug_json = str(output_path).replace('.fbx', '_debug.json')
    with open(debug_json, 'w') as f:
        json.dump(skeleton_data, f, indent=2)
    if verbose:
        print(f"Debug JSON saved to: {debug_json}")

    try:
        # Get path to Blender script
        script_path = Path(__file__).parent / "blender_fbx_export.py"

        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Run Blender
        cmd = [
            blender,
            "--background",
            "--python", str(script_path),
            "--",
            temp_json,
            str(output_path)
        ]

        if verbose:
            print(f"Running Blender: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Blender stderr: {result.stderr}")
            raise RuntimeError(f"Blender export failed: {result.stderr}")

        if verbose:
            print(f"Exported FBX to: {output_path}")

        return output_path

    finally:
        # Clean up temp file
        os.unlink(temp_json)


def load_bind_pose(path):
    """Load bind pose from JSON file.

    Supports two formats:
    1. Web viewer JSON with raw keypoints:
       {"people": [{"keypoints": [[x,y,z], ...]}]}
    2. Named joints format:
       {"joints": {"root": [x,y,z], ...}}
    """
    with open(path) as f:
        data = json.load(f)

    # Check if it's web viewer format (has "people" with "keypoints")
    if "people" in data and data["people"]:
        person = data["people"][0]
        if "keypoints" in person:
            # Convert raw keypoints to joint positions
            keypoints = np.array(person["keypoints"])
            return compute_joint_positions(keypoints)

    # Otherwise expect named joints format
    return data.get("joints", data)
