import numpy as np
import os

class SkeletonExporter:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.frames_data = [] # List of frame data, each frame is a list of people
        os.makedirs(output_dir, exist_ok=True)

    def add_frame(self, frame_outputs):
        """
        frame_outputs: List of dicts, one for each person detected in the frame.
                       Each dict contains "pred_keypoints_3d" (num_kps, 3)
        """
        people_data = []
        for person_out in frame_outputs:
            if "pred_keypoints_3d" in person_out:
                # Assuming pred_keypoints_3d is (N_kps, 3)
                # MHR70 usually implies the first 70 keypoints are the skeleton
                # We save the full set just in case, but Blender script will use first 70
                kps = person_out["pred_keypoints_3d"]
                if kps.shape[0] > 70:
                    kps = kps[:70]
                people_data.append(kps)
        
        self.frames_data.append(people_data)

    def save_motion(self, filename="motion_export.npz"):
        """
        Saves the collected motion data to a .npz file.
        Format:
           data: array of shape (num_frames, max_people, 70, 3)
           num_people_per_frame: array of shape (num_frames,)
        """
        num_frames = len(self.frames_data)
        if num_frames == 0:
            print("No frames to export.")
            return

        max_people = 0
        for frame in self.frames_data:
            max_people = max(max_people, len(frame))
        
        # Initialize with NaNs or zeros
        # (Frames, People, Joints, 3)
        motion_array = np.zeros((num_frames, max_people, 70, 3), dtype=np.float32)
        num_people_per_frame = np.zeros((num_frames,), dtype=np.int32)
        
        for i, frame in enumerate(self.frames_data):
            num_people_per_frame[i] = len(frame)
            for p_idx, person_kps in enumerate(frame):
                # person_kps should be (70, 3)
                rows = min(70, person_kps.shape[0])
                motion_array[i, p_idx, :rows, :] = person_kps[:rows]

        output_path = os.path.join(self.output_dir, filename)
        np.savez_compressed(output_path, motion=motion_array, counts=num_people_per_frame)
        print(f"Skeleton motion saved to {output_path}")
