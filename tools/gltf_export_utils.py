import json
import struct
import os
import numpy as np

# MHR70 Hierarchy (Node Index -> Parent Index defined below)
# We need a linear list of nodes.
# 0..69 correspond to MHR keypoint IDs.
# We will create nodes 0..69 in the GLTF corresponding to these IDs.

# Skeleton Tree Structure for meaningful hierarchy
# (Child -> Parent)
PARENT_MAP = {
    0: 69, # nose -> neck
    1: 0, 2: 0, # eyes -> nose
    3: 1, 4: 2, # ears -> eyes
    69: -1, # neck -> root
    
    5: 69, 6: 69, # shoulders -> neck
    7: 5, 8: 6, # elbows -> shoulders
    41: 8, 62: 7, # wrists -> elbows
    
    # Hands (Right Wrist: 41)
    111: 41, 112: 111, 113: 112, 110: 113,
    115: 41, 116: 115, 117: 116, 114: 117,
    119: 41, 120: 119, 121: 120, 118: 121,
    123: 41, 124: 123, 125: 124, 122: 125,
    127: 41, 128: 127, 129: 128, 126: 129,

    # Legs
    9: -1, 10: -1, # hips -> root
    11: 9, 12: 10,
    13: 11, 14: 12,
    17: 13, 20: 14,
    15: 13, 18: 14,
    16: 13, 19: 14,
}

# The MHR70 set is 0..69. The hand indices above (110+) are from full MHR.
# We only have 0..69 available in the 70-set. 
# We need to map the available 0..69 points to a valid hierarchy.
# The 70-set has Hand points? 
# Lets check `mhr70.py` again.
# 21-40: Right Hand points.
# 42-61: Left Hand points.
# So we need to update the map for 21-40 and 42-61.

# Corrected Parent Map for 0-69
PARENT_MAP_70 = {
    0: 69, 1: 0, 2: 0, 3: 1, 4: 2, 69: -1,
    5: 69, 6: 69, 7: 5, 8: 6, 62: 7, 41: 8,
    9: -1, 10: -1, 11: 9, 12: 10, 13: 11, 14: 12,
    15: 13, 16: 13, 17: 13, 18: 14, 19: 14, 20: 14,
    
    # Right Hand (Wrist 41)
    21: 41, 22: 41, 23: 22, 24: 23, # Thumb
    25: 41, 26: 25, 27: 26, 28: 27, # Index (Indices from mhr70.py lines 31-34 -> 31,32,33,34)
    # Wait, mhr70.py indices 21..40 correspond to Right hand.
    # 21: r-thumb-tip, 22: r-thumb-1, 23: r-thumb-2, 24: r-thumb-3
    # Actually hierarchy is usually wrist -> 1 -> 2 -> 3 -> tip.
    # mhr70.py: 
    # 27: tip, 28: 1st, 29: 2nd, 30: 3rd. (Indices in list)
    # Let's just parent all hand points to Wrist for simplicity/robustness if order is unsure,
    # or just export flat points.
    # Parent-Child hierarchy allows "Bones" to be drawn in viewers.
}
# Auto-parent remaining hand points to respective wrists
for i in range(21, 41): PARENT_MAP_70[i] = 41 # Right Hand -> Right Wrist
for i in range(42, 62): PARENT_MAP_70[i] = 62 # Left Hand -> Left Wrist
# Add extra points
for i in [63, 64, 65, 66, 67, 68]: PARENT_MAP_70[i] = 69 # To Spine/Neck default

class GLBExporter:
    def __init__(self):
        self.nodes = []
        self.frames = [] # List of frames (people_poses)
        
    def add_frame(self, frame_outputs):
        # We need to track people consistently.
        # For simplicity, we assume Person 0 is consistent.
        # Handling multiple people in GLTF requires multiple skins/roots.
        # We will export ALL detected people in the frame as a single "Pose" 
        # (if multiple people exist, we create nodes for Person_0, Person_1...)
        
        # We need to pre-determine the MAX people to build the node hierarchy structure upfront?
        # GLTF structure (nodes) is usually static.
        # If number of people changes, we might have unused nodes. 
        # We will build the hierarchy at `save` time based on max people found.
        
        frame_data = []
        for p in frame_outputs:
            if "pred_keypoints_3d" in p:
                kps = p["pred_keypoints_3d"]
                if kps.shape[0] > 70: kps = kps[:70]
                frame_data.append(kps)
        self.frames.append(frame_data)

    def save(self, filename):
        if not self.frames: return
        
        # Static Export Mode:
        # Instead of animation, we export each frame/person as a static hierarchy in the scene.
        # Structure:
        # Scene Root
        #   -> Frame_0
        #       -> Person_0
        #       -> Person_1
        #   -> Frame_1 ...
        
        nodes = []
        scenes = [{"nodes": [0]}] # Root scene node
        
        # Root Node (0)
        nodes.append({"name": "Scene_Root", "children": []})
        current_node_idx = 1
        
        # Flattened list of (FrameIdx, PersonIdx, JointIdx) -> GLTF Node Index
        node_map = {} 
        
        for f_idx, frame in enumerate(self.frames):
            # Create Frame Node (Optional, maybe just group per frame or flatten)
            # Let's create a Frame node to organize
            frame_root_idx = current_node_idx
            nodes[0]["children"].append(frame_root_idx)
            nodes.append({"name": f"Frame_{f_idx}", "children": [], "translation": [0,0,0]})
            current_node_idx += 1
            
            for p_idx, person_kps in enumerate(frame):
                # Person Root
                p_root_idx = current_node_idx
                nodes[frame_root_idx]["children"].append(p_root_idx)
                nodes.append({"name": f"Frame_{f_idx}_Person_{p_idx}_Root", "children": []}) # Root at 0,0,0
                current_node_idx += 1
                
                # Create 70 joints
                base_idx = current_node_idx
                
                for j in range(70):
                    # We utilize the global positions from the frame directly.
                    # Calculate LOCAL translation relative to Parent.
                    
                    global_pos = person_kps[j]
                    parent_j = PARENT_MAP_70.get(j, -1)
                    
                    if parent_j != -1:
                        parent_global = person_kps[parent_j]
                        local_pos = global_pos - parent_global
                    else:
                        local_pos = global_pos
                        
                    # GLTF Y-up flip (x, -y, -z)
                    t = [float(local_pos[0]), float(-local_pos[1]), float(-local_pos[2])]
                    
                    nodes.append({"name": f"P{p_idx}_J{j}", "translation": t})
                    node_map[(f_idx, p_idx, j)] = base_idx + j
                
                current_node_idx += 70
                
                # Link hierarchy
                for j in range(70):
                    gltf_node_idx = node_map[(f_idx, p_idx, j)]
                    parent_j = PARENT_MAP_70.get(j, -1)
                    
                    if parent_j == -1:
                        nodes[p_root_idx]["children"].append(gltf_node_idx)
                    else:
                        parent_gltf_idx = node_map[(f_idx, p_idx, parent_j)]
                        if "children" not in nodes[parent_gltf_idx]:
                            nodes[parent_gltf_idx]["children"] = []
                        nodes[parent_gltf_idx]["children"].append(gltf_node_idx)
                        
        # JSON Construction only (No binary buffer needed since no animation/mesh data yet)
        # But GLB requires BIN chunk usually.
        # We'll put a dummy empty bin chunk or use None.
        # GLB spec: must have JSON. BIN is optional but 'GLB' implies binary container.
        # We can just write 0-length BIN.
        
        gltf = {
            "asset": {"version": "2.0", "generator": "SAM3D_GLB_Exporter"},
            "scenes": scenes,
            "scene": 0,
            "nodes": nodes,
        }
        
        json_str = json.dumps(gltf).encode('utf-8')
        json_padding = (4 - (len(json_str) % 4)) % 4
        json_str += b' ' * json_padding
        
        total_length = 12 + (8 + len(json_str)) + 8 # +8 for empty bin chunk header
        
        with open(filename, 'wb') as f:
            # Header
            f.write(struct.pack('<I', 0x46546C67)) # 'glTF'
            f.write(struct.pack('<I', 2)) # Version 2
            f.write(struct.pack('<I', total_length))
            
            # JSON Chunk
            f.write(struct.pack('<I', len(json_str)))
            f.write(struct.pack('<I', 0x4E4F534A)) # 'JSON'
            f.write(json_str)
            
            # BIN Chunk (Empty)
            f.write(struct.pack('<I', 0))
            f.write(struct.pack('<I', 0x004E4942)) # 'BIN\0'
            
        print(f"Exported Static GLB to {filename}")
