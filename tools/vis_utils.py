# Copyright (c) Meta Platforms, Inc. and affiliates.
import numpy as np
import cv2
from sam_3d_body.visualization.renderer import Renderer

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)

# Bone connections matching web viewer (from json_export.py)
# Format: (start_idx, end_idx, color_rgb)
MHR70_BONES = [
    # Torso bars (horizontal)
    (9, 10, [51, 153, 255]),     # left_hip -> right_hip
    (5, 6, [51, 153, 255]),      # left_shoulder -> right_shoulder
    # Legs
    (13, 11, [0, 255, 0]),       # left_ankle -> left_knee
    (11, 9, [0, 255, 0]),        # left_knee -> left_hip
    (14, 12, [255, 128, 0]),     # right_ankle -> right_knee
    (12, 10, [255, 128, 0]),     # right_knee -> right_hip
    # Arms
    (5, 7, [0, 255, 0]),         # left_shoulder -> left_elbow
    (6, 8, [255, 128, 0]),       # right_shoulder -> right_elbow
    (7, 62, [0, 255, 0]),        # left_elbow -> left_wrist
    (8, 41, [255, 128, 0]),      # right_elbow -> right_wrist
    # Left foot triangle
    (17, 15, [0, 255, 0]),       # left_heel -> left_big_toe
    (15, 16, [0, 255, 0]),       # left_big_toe -> left_small_toe
    (16, 17, [0, 255, 0]),       # left_small_toe -> left_heel
    (13, 17, [0, 255, 0]),       # left_ankle -> left_heel
    (13, 15, [0, 255, 0]),       # left_ankle -> left_big_toe
    (13, 16, [0, 255, 0]),       # left_ankle -> left_small_toe
    # Right foot triangle
    (20, 18, [255, 128, 0]),     # right_heel -> right_big_toe
    (18, 19, [255, 128, 0]),     # right_big_toe -> right_small_toe
    (19, 20, [255, 128, 0]),     # right_small_toe -> right_heel
    (14, 20, [255, 128, 0]),     # right_ankle -> right_heel
    (14, 18, [255, 128, 0]),     # right_ankle -> right_big_toe
    (14, 19, [255, 128, 0]),     # right_ankle -> right_small_toe
    # Left hand - fingers
    (45, 44, [255, 128, 0]),     # left_thumb: third -> second
    (44, 43, [255, 128, 0]),     # second -> first
    (43, 42, [255, 128, 0]),     # first -> tip
    (49, 48, [255, 153, 255]),   # left_index chain
    (48, 47, [255, 153, 255]),
    (47, 46, [255, 153, 255]),
    (53, 52, [102, 178, 255]),   # left_middle chain
    (52, 51, [102, 178, 255]),
    (51, 50, [102, 178, 255]),
    (57, 56, [255, 51, 51]),     # left_ring chain
    (56, 55, [255, 51, 51]),
    (55, 54, [255, 51, 51]),
    (61, 60, [0, 255, 0]),       # left_pinky chain
    (60, 59, [0, 255, 0]),
    (59, 58, [0, 255, 0]),
    # Left hand palm
    (62, 45, [200, 200, 200]),   # left_wrist -> left_thumb_third
    (44, 49, [200, 200, 200]),   # left_thumb_second -> left_index_third
    (49, 53, [200, 200, 200]),   # left_index_third -> left_middle_third
    (53, 57, [200, 200, 200]),   # left_middle_third -> left_ring_third
    (57, 61, [200, 200, 200]),   # left_ring_third -> left_pinky_third
    (61, 62, [200, 200, 200]),   # left_pinky_third -> left_wrist
    # Right hand - fingers
    (24, 23, [255, 128, 0]),     # right_thumb: third -> second
    (23, 22, [255, 128, 0]),     # second -> first
    (22, 21, [255, 128, 0]),     # first -> tip
    (28, 27, [255, 153, 255]),   # right_index chain
    (27, 26, [255, 153, 255]),
    (26, 25, [255, 153, 255]),
    (32, 31, [102, 178, 255]),   # right_middle chain
    (31, 30, [102, 178, 255]),
    (30, 29, [102, 178, 255]),
    (36, 35, [255, 51, 51]),     # right_ring chain
    (35, 34, [255, 51, 51]),
    (34, 33, [255, 51, 51]),
    (40, 39, [0, 255, 0]),       # right_pinky chain
    (39, 38, [0, 255, 0]),
    (38, 37, [0, 255, 0]),
    # Right hand palm
    (41, 24, [200, 200, 200]),   # right_wrist -> right_thumb_third
    (23, 28, [200, 200, 200]),   # right_thumb_second -> right_index_third
    (28, 32, [200, 200, 200]),   # right_index_third -> right_middle_third
    (32, 36, [200, 200, 200]),   # right_middle_third -> right_ring_third
    (36, 40, [200, 200, 200]),   # right_ring_third -> right_pinky_third
    (40, 41, [200, 200, 200]),   # right_pinky_third -> right_wrist
]

# Virtual bone connections for spine/neck/head (use virtual keypoint indices 70, 71, 72)
VIRTUAL_BONES = [
    (70, 71, [51, 153, 255]),    # midHip -> midShoulder (spine)
    (71, 72, [51, 153, 255]),    # midShoulder -> head (neck)
    (72, 3, [51, 153, 255]),     # head -> left_ear
    (72, 4, [51, 153, 255]),     # head -> right_ear
]


def compute_virtual_keypoints(kpts):
    """Compute virtual midpoints for spine/neck visualization.

    Returns keypoints array extended with:
    - 70: midHip (midpoint of left_hip and right_hip)
    - 71: midShoulder (midpoint of left_shoulder and right_shoulder)
    - 72: head (midpoint of left_ear and right_ear)
    """
    kpts = kpts.copy()
    # Indices: 5=left_shoulder, 6=right_shoulder, 9=left_hip, 10=right_hip, 3=left_ear, 4=right_ear
    mid_hip = (kpts[9] + kpts[10]) / 2
    mid_shoulder = (kpts[5] + kpts[6]) / 2
    head = (kpts[3] + kpts[4]) / 2

    # Extend keypoints array
    extended = np.vstack([kpts, mid_hip[np.newaxis], mid_shoulder[np.newaxis], head[np.newaxis]])
    return extended


def draw_skeleton_viewer_style(image, keypoints_2d, kpt_thr=0.3, line_width=2, radius=5):
    """Draw skeleton matching the web viewer style.

    Args:
        image: BGR image to draw on
        keypoints_2d: (N, 3) array with [x, y, confidence] for each keypoint
        kpt_thr: minimum confidence threshold
        line_width: line thickness
        radius: joint circle radius

    Returns:
        Image with skeleton drawn
    """
    image = image.copy()
    img_h, img_w = image.shape[:2]

    kpts = keypoints_2d[:, :2]
    scores = keypoints_2d[:, 2]

    # Compute virtual keypoints (midHip, midShoulder, head)
    # For virtual points, use average confidence of source points
    mid_hip_score = (scores[9] + scores[10]) / 2
    mid_shoulder_score = (scores[5] + scores[6]) / 2
    head_score = (scores[3] + scores[4]) / 2

    extended_kpts = compute_virtual_keypoints(kpts)
    extended_scores = np.concatenate([scores, [mid_hip_score, mid_shoulder_score, head_score]])

    def is_valid(idx):
        if idx >= len(extended_scores):
            return False
        if extended_scores[idx] < kpt_thr:
            return False
        x, y = extended_kpts[idx][:2]
        if x <= 0 or x >= img_w or y <= 0 or y >= img_h:
            return False
        return True

    # Check if this is a foot bone (both endpoints in 13-20 range)
    def is_foot_bone(start_idx, end_idx):
        return (13 <= start_idx <= 20) and (13 <= end_idx <= 20)

    # Draw all bones (regular + virtual)
    all_bones = MHR70_BONES + VIRTUAL_BONES

    for start_idx, end_idx, color_rgb in all_bones:
        if not is_valid(start_idx) or not is_valid(end_idx):
            continue

        pos1 = tuple(map(int, extended_kpts[start_idx][:2]))
        pos2 = tuple(map(int, extended_kpts[end_idx][:2]))

        # Convert RGB to BGR for OpenCV
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])

        # Foot bones at 0.5 opacity (blend with image)
        if is_foot_bone(start_idx, end_idx):
            overlay = image.copy()
            cv2.line(overlay, pos1, pos2, color_bgr, line_width)
            cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
        else:
            cv2.line(image, pos1, pos2, color_bgr, line_width)

    # Draw foot polygons (transparent)
    # Colors in RGB, will convert to BGR
    foot_indices = [
        ([13, 15, 16, 17], [0, 255, 0]),      # left foot (green)
        ([14, 18, 19, 20], [255, 128, 0]),    # right foot (orange)
    ]

    for indices, color_rgb in foot_indices:
        ankle_idx, big_toe_idx, small_toe_idx, heel_idx = indices
        if all(is_valid(i) for i in indices):
            pts = np.array([
                extended_kpts[ankle_idx][:2],
                extended_kpts[big_toe_idx][:2],
                extended_kpts[small_toe_idx][:2],
                extended_kpts[heel_idx][:2],
            ], dtype=np.int32)

            color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
            overlay = image.copy()
            cv2.fillPoly(overlay, [pts], color_bgr)
            cv2.addWeighted(overlay, 0.1, image, 0.9, 0, image)

    # Draw hand palm polygons (transparent)
    # Colors in RGB, will convert to BGR
    palm_indices = [
        # Left hand: wrist(62), thumb_third(45), thumb_second(44), index_third(49), middle_third(53), ring_third(57), pinky_third(61)
        ([62, 45, 44, 49, 53, 57, 61], [0, 255, 0]),      # green
        # Right hand: wrist(41), thumb_third(24), thumb_second(23), index_third(28), middle_third(32), ring_third(36), pinky_third(40)
        ([41, 24, 23, 28, 32, 36, 40], [255, 128, 0]),    # orange
    ]

    for indices, color_rgb in palm_indices:
        wrist, thumb_third, thumb_second, index_third, middle_third, ring_third, pinky_third = indices
        if all(is_valid(i) for i in indices):
            # Palm polygon: wrist -> thumb_third -> thumb_second -> index_third -> middle_third -> ring_third -> pinky_third -> wrist
            pts = np.array([
                extended_kpts[wrist][:2],
                extended_kpts[thumb_third][:2],
                extended_kpts[thumb_second][:2],
                extended_kpts[index_third][:2],
                extended_kpts[middle_third][:2],
                extended_kpts[ring_third][:2],
                extended_kpts[pinky_third][:2],
            ], dtype=np.int32)

            color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
            overlay = image.copy()
            cv2.fillPoly(overlay, [pts], color_bgr)
            cv2.addWeighted(overlay, 0.1, image, 0.9, 0, image)

    # Draw head circle with fill
    left_eye_idx, head_idx = 1, 72
    if is_valid(left_eye_idx) and is_valid(head_idx):
        head_pos = tuple(map(int, extended_kpts[head_idx][:2]))
        eye_pos = extended_kpts[left_eye_idx][:2]
        head_radius = int(np.sqrt(np.sum((extended_kpts[head_idx][:2] - eye_pos) ** 2)))

        if head_radius > 0:
            # Draw filled circle (background color approximation at 0.5 opacity)
            overlay = image.copy()
            cv2.circle(overlay, head_pos, int(head_radius * 0.9), (26, 26, 26), -1)
            cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)

            # Draw circle outline
            cv2.circle(image, head_pos, head_radius, (255, 153, 51), line_width)

    # Draw joints
    # Skip extra keypoints (63-69) and virtual keypoints (70+)
    for i in range(min(63, len(kpts))):
        if not is_valid(i):
            continue

        pos = tuple(map(int, kpts[i][:2]))

        # Eye joints are 1.5x larger (indices 1, 2)
        if i in [1, 2]:
            joint_radius = int(radius * 1.5)
        # Hand joints are 0.5x (indices 21-62)
        elif 21 <= i <= 62:
            joint_radius = radius // 2
        else:
            joint_radius = radius

        cv2.circle(image, pos, joint_radius, (255, 255, 255), -1)

    return image


def visualize_sample(img_cv2, outputs, faces):
    img_keypoints = img_cv2.copy()
    img_mesh = img_cv2.copy()

    rend_img = []
    for pid, person_output in enumerate(outputs):
        keypoints_2d = person_output["pred_keypoints_2d"]
        keypoints_2d = np.concatenate(
            [keypoints_2d, np.ones((keypoints_2d.shape[0], 1))], axis=-1
        )
        img1 = draw_skeleton_viewer_style(img_keypoints.copy(), keypoints_2d)

        img1 = cv2.rectangle(
            img1,
            (int(person_output["bbox"][0]), int(person_output["bbox"][1])),
            (int(person_output["bbox"][2]), int(person_output["bbox"][3])),
            (0, 255, 0),
            2,
        )

        if "lhand_bbox" in person_output:
            img1 = cv2.rectangle(
                img1,
                (
                    int(person_output["lhand_bbox"][0]),
                    int(person_output["lhand_bbox"][1]),
                ),
                (
                    int(person_output["lhand_bbox"][2]),
                    int(person_output["lhand_bbox"][3]),
                ),
                (255, 0, 0),
                2,
            )

        if "rhand_bbox" in person_output:
            img1 = cv2.rectangle(
                img1,
                (
                    int(person_output["rhand_bbox"][0]),
                    int(person_output["rhand_bbox"][1]),
                ),
                (
                    int(person_output["rhand_bbox"][2]),
                    int(person_output["rhand_bbox"][3]),
                ),
                (0, 0, 255),
                2,
            )

        renderer = Renderer(focal_length=person_output["focal_length"], faces=faces)
        img2 = (
            renderer(
                person_output["pred_vertices"],
                person_output["pred_cam_t"],
                img_mesh.copy(),
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
            )
            * 255
        )

        white_img = np.ones_like(img_cv2) * 255
        img3 = (
            renderer(
                person_output["pred_vertices"],
                person_output["pred_cam_t"],
                white_img,
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                side_view=True,
            )
            * 255
        )

        cur_img = np.concatenate([img_cv2, img1, img2, img3], axis=1)
        rend_img.append(cur_img)

    return rend_img

def visualize_sample_together(img_cv2, outputs, faces):
    # Render everything together
    img_keypoints = img_cv2.copy()
    img_mesh = img_cv2.copy()

    if len(outputs) == 0:
        # Return dict with original image for all views
        return {
            "ref": img_cv2,
            "skeleton": img_cv2,
            "mesh": img_cv2,
        }

    # First, sort by depth, furthest to closest
    all_depths = np.stack([tmp['pred_cam_t'] for tmp in outputs], axis=0)[:, 2]
    outputs_sorted = [outputs[idx] for idx in np.argsort(-all_depths)]

    # Then, draw all keypoints using viewer-style rendering
    for pid, person_output in enumerate(outputs_sorted):
        keypoints_2d = person_output["pred_keypoints_2d"]
        keypoints_2d = np.concatenate(
            [keypoints_2d, np.ones((keypoints_2d.shape[0], 1))], axis=-1
        )
        img_keypoints = draw_skeleton_viewer_style(img_keypoints, keypoints_2d)

    # Then, put all meshes together as one super mesh
    all_pred_vertices = []
    all_faces = []
    for pid, person_output in enumerate(outputs_sorted):
        all_pred_vertices.append(person_output["pred_vertices"] + person_output["pred_cam_t"])
        all_faces.append(faces + len(person_output["pred_vertices"]) * pid)
    all_pred_vertices = np.concatenate(all_pred_vertices, axis=0)
    all_faces = np.concatenate(all_faces, axis=0)

    # Pull out a fake translation; take the closest two
    fake_pred_cam_t = (np.max(all_pred_vertices[-2*18439:], axis=0) + np.min(all_pred_vertices[-2*18439:], axis=0)) / 2
    all_pred_vertices = all_pred_vertices - fake_pred_cam_t
    
    # Render front view
    renderer = Renderer(focal_length=person_output["focal_length"], faces=all_faces)
    img_mesh = (
        renderer(
            all_pred_vertices,
            fake_pred_cam_t,
            img_mesh,
            mesh_base_color=LIGHT_BLUE,
            scene_bg_color=(1, 1, 1),
        )
        * 255
    )

    return {
        "ref": img_cv2,
        "skeleton": img_keypoints,
        "mesh": img_mesh,
    }


def visualize_debug_detections(img_cv2, outputs):
    """
    Visualize SAM3 detections (bboxes and masks) for debugging.
    """
    debug_img = img_cv2.copy()
    overlay = img_cv2.copy()

    for pid, person_output in enumerate(outputs):
        bbox = person_output["bbox"]
        score = person_output.get("det_score", 0.0)
        bbox_score = person_output.get("bbox_score", score)
        mask = person_output.get("mask")

        # Draw mask
        if mask is not None:
            # Mask is [H, W, 1] usually or [H, W]
            mask_bool = mask.squeeze() > 0
            color = np.random.randint(0, 255, (3,)).tolist()
            overlay[mask_bool] = color
            
        # Draw bbox
        cv2.rectangle(
            debug_img,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            (0, 255, 0),
            2,
        )
        
        # Draw score
        label = f"ID:{pid} Det:{bbox_score:.2f}"
        if "mask_score" in person_output and person_output["mask_score"] is not None:
             label += f" Mask:{person_output['mask_score']:.2f}"
        cv2.putText(
            debug_img,
            label,
            (int(bbox[0]), int(bbox[1]) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

    # Blend overlay for masks
    alpha = 0.4
    debug_img = cv2.addWeighted(overlay, alpha, debug_img, 1 - alpha, 0)
    
    return debug_img
