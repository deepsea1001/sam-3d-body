"""JSON export utilities for web-based skeleton visualization."""

import json
import numpy as np
from pathlib import Path

# MHR70 keypoint names (indices 0-69)
MHR70_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_big_toe", "left_small_toe", "left_heel",
    "right_big_toe", "right_small_toe", "right_heel",
    # Right hand (21-41)
    "right_thumb_tip", "right_thumb_first", "right_thumb_second", "right_thumb_third",
    "right_index_tip", "right_index_first", "right_index_second", "right_index_third",
    "right_middle_tip", "right_middle_first", "right_middle_second", "right_middle_third",
    "right_ring_tip", "right_ring_first", "right_ring_second", "right_ring_third",
    "right_pinky_tip", "right_pinky_first", "right_pinky_second", "right_pinky_third",
    "right_wrist",
    # Left hand (42-62)
    "left_thumb_tip", "left_thumb_first", "left_thumb_second", "left_thumb_third",
    "left_index_tip", "left_index_first", "left_index_second", "left_index_third",
    "left_middle_tip", "left_middle_first", "left_middle_second", "left_middle_third",
    "left_ring_tip", "left_ring_first", "left_ring_second", "left_ring_third",
    "left_pinky_tip", "left_pinky_first", "left_pinky_second", "left_pinky_third",
    "left_wrist",
    # Extra (63-69)
    "left_olecranon", "right_olecranon",
    "left_cubital_fossa", "right_cubital_fossa",
    "left_acromion", "right_acromion", "neck"
]

# Bone connections as (start_idx, end_idx) pairs with colors
# Colors: green=[0,255,0] for left, orange=[255,128,0] for right, blue=[51,153,255] for center
MHR70_BONES = [
    # Body
    (13, 11, [0, 255, 0]),       # left_ankle -> left_knee
    (11, 9, [0, 255, 0]),        # left_knee -> left_hip
    (14, 12, [255, 128, 0]),     # right_ankle -> right_knee
    (12, 10, [255, 128, 0]),     # right_knee -> right_hip
    (9, 10, [51, 153, 255]),     # left_hip -> right_hip
    (5, 9, [51, 153, 255]),      # left_shoulder -> left_hip
    (6, 10, [51, 153, 255]),     # right_shoulder -> right_hip
    (5, 6, [51, 153, 255]),      # left_shoulder -> right_shoulder
    (5, 7, [0, 255, 0]),         # left_shoulder -> left_elbow
    (6, 8, [255, 128, 0]),       # right_shoulder -> right_elbow
    (7, 62, [0, 255, 0]),        # left_elbow -> left_wrist
    (8, 41, [255, 128, 0]),      # right_elbow -> right_wrist
    # Face
    (1, 2, [51, 153, 255]),      # left_eye -> right_eye
    (0, 1, [51, 153, 255]),      # nose -> left_eye
    (0, 2, [51, 153, 255]),      # nose -> right_eye
    (1, 3, [51, 153, 255]),      # left_eye -> left_ear
    (2, 4, [51, 153, 255]),      # right_eye -> right_ear
    (3, 5, [51, 153, 255]),      # left_ear -> left_shoulder
    (4, 6, [51, 153, 255]),      # right_ear -> right_shoulder
    # Feet
    (13, 15, [0, 255, 0]),       # left_ankle -> left_big_toe
    (13, 16, [0, 255, 0]),       # left_ankle -> left_small_toe
    (13, 17, [0, 255, 0]),       # left_ankle -> left_heel
    (14, 18, [255, 128, 0]),     # right_ankle -> right_big_toe
    (14, 19, [255, 128, 0]),     # right_ankle -> right_small_toe
    (14, 20, [255, 128, 0]),     # right_ankle -> right_heel
    # Left hand
    (62, 45, [255, 128, 0]),     # left_wrist -> left_thumb_third
    (45, 44, [255, 128, 0]),     # left_thumb chain
    (44, 43, [255, 128, 0]),
    (43, 42, [255, 128, 0]),
    (62, 49, [255, 153, 255]),   # left_wrist -> left_index_third
    (49, 48, [255, 153, 255]),
    (48, 47, [255, 153, 255]),
    (47, 46, [255, 153, 255]),
    (62, 53, [102, 178, 255]),   # left_wrist -> left_middle_third
    (53, 52, [102, 178, 255]),
    (52, 51, [102, 178, 255]),
    (51, 50, [102, 178, 255]),
    (62, 57, [255, 51, 51]),     # left_wrist -> left_ring_third
    (57, 56, [255, 51, 51]),
    (56, 55, [255, 51, 51]),
    (55, 54, [255, 51, 51]),
    (62, 61, [0, 255, 0]),       # left_wrist -> left_pinky_third
    (61, 60, [0, 255, 0]),
    (60, 59, [0, 255, 0]),
    (59, 58, [0, 255, 0]),
    # Right hand
    (41, 24, [255, 128, 0]),     # right_wrist -> right_thumb_third
    (24, 23, [255, 128, 0]),
    (23, 22, [255, 128, 0]),
    (22, 21, [255, 128, 0]),
    (41, 28, [255, 153, 255]),   # right_wrist -> right_index_third
    (28, 27, [255, 153, 255]),
    (27, 26, [255, 153, 255]),
    (26, 25, [255, 153, 255]),
    (41, 32, [102, 178, 255]),   # right_wrist -> right_middle_third
    (32, 31, [102, 178, 255]),
    (31, 30, [102, 178, 255]),
    (30, 29, [102, 178, 255]),
    (41, 36, [255, 51, 51]),     # right_wrist -> right_ring_third
    (36, 35, [255, 51, 51]),
    (35, 34, [255, 51, 51]),
    (34, 33, [255, 51, 51]),
    (41, 40, [0, 255, 0]),       # right_wrist -> right_pinky_third
    (40, 39, [0, 255, 0]),
    (39, 38, [0, 255, 0]),
    (38, 37, [0, 255, 0]),
]


def get_bones_and_colors():
    """Return bone index pairs and colors as separate lists."""
    bones = [[b[0], b[1]] for b in MHR70_BONES]
    bone_colors = [b[2] for b in MHR70_BONES]
    return bones, bone_colors


def export_skeleton_json(keypoints_3d, output_path, source_image=None):
    """
    Export single person 3D keypoints to JSON format for web visualization.

    Args:
        keypoints_3d: numpy array of shape (70, 3) or (N, 3) where N >= 70
        output_path: path to save the JSON file
        source_image: optional source image filename for metadata

    Returns:
        Path to the saved JSON file
    """
    return export_multi_skeleton_json([keypoints_3d], output_path, source_image)


def export_multi_skeleton_json(keypoints_list, output_path, source_image=None):
    """
    Export multiple people's 3D keypoints to JSON format for web visualization.

    Args:
        keypoints_list: list of numpy arrays, each of shape (70, 3) or (N, 3)
        output_path: path to save the JSON file
        source_image: optional source image filename for metadata

    Returns:
        Path to the saved JSON file
    """
    people = []
    for keypoints_3d in keypoints_list:
        # Ensure we have numpy array
        if not isinstance(keypoints_3d, np.ndarray):
            keypoints_3d = np.array(keypoints_3d)

        # Take first 70 keypoints if more are provided
        if keypoints_3d.shape[0] > 70:
            keypoints_3d = keypoints_3d[:70]

        people.append(keypoints_3d.tolist())

    # Build bone index pairs and colors (same for all people)
    bones, bone_colors = get_bones_and_colors()

    # Build JSON structure
    data = {
        "version": "1.0",
        "source_image": source_image or "",
        "people": people,
        "keypoint_names": MHR70_NAMES,
        "bones": bones,
        "bone_colors": bone_colors
    }

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write JSON
    with open(output_path, 'w') as f:
        json.dump(data, f)

    return output_path


def export_from_pipeline_outputs(outputs, output_path, source_image=None):
    """
    Export skeletons from SAM 3D Body pipeline output list.

    Args:
        outputs: list of dicts, each containing 'pred_keypoints_3d' key
        output_path: path to save the JSON file
        source_image: optional source image filename

    Returns:
        Path to the saved JSON file, or None if no keypoints found
    """
    keypoints_list = []
    for person_output in outputs:
        if "pred_keypoints_3d" in person_output:
            keypoints_list.append(person_output["pred_keypoints_3d"])

    if not keypoints_list:
        return None

    return export_multi_skeleton_json(keypoints_list, output_path, source_image)
