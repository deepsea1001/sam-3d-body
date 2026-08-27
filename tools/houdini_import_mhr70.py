import hou
import numpy as np
import os

# MHR70 Skeleton Definition (same as Blender script)
# MHR70 IDs
ID_MAP = {
    'LeftUpLeg': 9, 'LeftLeg': 11, 'LeftFoot': 13, 'LeftToeBase': 15,
    'RightUpLeg': 10, 'RightLeg': 12, 'RightFoot': 14, 'RightToeBase': 18,
    'Spine': 69, 'Head': 0,
    'LeftShoulder': 5, 'LeftArm': 7, 'LeftForeArm': 62,
    'RightShoulder': 6, 'RightArm': 8, 'RightForeArm': 41,
}

# Simple Parent Hierarchy for visualization
HIERARCHY = {
    'Hips': None,
    'Spine': 'Hips', 'Head': 'Spine',
    'LeftUpLeg': 'Hips', 'LeftLeg': 'LeftUpLeg', 'LeftFoot': 'LeftLeg', 'LeftToeBase': 'LeftFoot',
    'RightUpLeg': 'Hips', 'RightLeg': 'RightUpLeg', 'RightFoot': 'RightLeg', 'RightToeBase': 'RightFoot',
    'LeftShoulder': 'Spine', 'LeftArm': 'LeftShoulder', 'LeftForeArm': 'LeftArm',
    'RightShoulder': 'Spine', 'RightArm': 'RightShoulder', 'RightForeArm': 'RightArm',
}

def create_skeleton_nodes(container, person_idx):
    nodes = {}
    
    # Create a Null for the Hips (Root)
    hip_node = container.createNode("null", f"Person_{person_idx}_Hips")
    hip_node.setParms({'geoscale': 0.1})
    nodes['Hips'] = hip_node
    
    # Create nodes for others
    for name, parent in HIERARCHY.items():
        if name == 'Hips': continue
        
        node = container.createNode("null", f"Person_{person_idx}_{name}")
        node.setParms({'geoscale': 0.05})
        
        # Color coding: Left (Blue), Right (Red), Center (Yellow)
        color = (1, 1, 0)
        if 'Left' in name: color = (0, 0, 1)
        elif 'Right' in name: color = (1, 0, 0)
        node.setColor(hou.Color(color))
        
        nodes[name] = node
        
    # Link hierarchy if desired, BUT...
    # If we link hierarchy, local transforms become relative.
    # Since we only have WORLD positions, it's easier to keep them unparented 
    # (or parented to a master Root) and set world positions.
    # OR: We parent them for structure, but utilize separate logic to set world pos?
    # Houdini 'keep position when parenting' works in UI, but scripting requires inverse transform.
    
    # Simple approach: Create a flat hierarchy of Nulls following the points.
    # Then create a "Bone" geometry network that simply draws lines between them.
    
    return nodes

def import_motion_houdini(filepath):
    print(f"Loading {filepath}...")
    try:
        data = np.load(filepath)
        motion = data['motion'] # (Frames, Ppl, 70, 3)
        counts = data['counts']
    except Exception as e:
        hou.ui.displayMessage(f"Failed to load .npz: {e}")
        return

    obj = hou.node("/obj")
    subnet = obj.createNode("subnet", "MHR70_Import")
    
    max_people = motion.shape[1]
    
    people_nodes = []
    
    # Setup Nodes
    for p_idx in range(max_people):
        nodes = create_skeleton_nodes(subnet, p_idx)
        people_nodes.append(nodes)
        
    # Set Keyframes
    num_frames = motion.shape[0]
    
    # Optimization: Turn off update
    with hou.undos.disabler():
        for f in range(num_frames):
            # Frame is 1-based in Houdini usually, or match file index
            hou_frame = f + 1
            num_ppl = counts[f]
            
            for p_idx in range(max_people):
                if p_idx >= num_ppl: continue
                
                kps = motion[f, p_idx]
                nodes = people_nodes[p_idx]
                
                # Update Hips logic
                l_hip = kps[ID_MAP['LeftUpLeg']]
                r_hip = kps[ID_MAP['RightUpLeg']]
                center = (l_hip + r_hip) / 2.0
                
                # Coordinate swizzle:
                # SAM3D (from export_utils): XYZ
                # Blender import swizzled to: X, Z, -Y
                # Houdini is usually Y-up, Right-handed.
                # Assuming input is Y-down (image space converted).
                # Let's check common MHR output.
                # Usually standard computer vision is Y-down.
                # To Houdini Y-up: [x, -y, z] or [x, -y, -z]?
                # Let's try [x, -y, z] (flip Y). Or [x, y, z] if exported world coords are already good.
                # Safe bet matching Blender script logic: x, z, -y  (mapping Y->Z, Z->-Y implies Y-up source?)
                # Wait, Blender is Z-up. Source was likely Y-down.
                # So if Source Y-down -> Blender Z-up (y maps to -z).
                # Houdini is Y-up. So Source Y-down -> Houdini (y maps to -y).
                # Let's assume [x, -y, z]
                
                def to_hou(vec):
                    # Trying X, -Y, Z based on typical CV->CG
                    return hou.Vector3(float(vec[0]), float(-vec[1]), float(vec[2]))
                
                # Set Hips
                h_node = nodes['Hips']
                h_pos = to_hou(center)
                
                # Set Keyframe
                hk = hou.Keyframe()
                hk.setFrame(hou_frame)
                
                for axes in range(3):
                    hk.setValue(h_pos[axes])
                    h_node.parmTuple("t")[axes].setKeyframe(hk)
                    
                # Set other nodes
                for name, node in nodes.items():
                    if name == 'Hips': continue
                    
                    kp_id = ID_MAP[name]
                    raw = kps[kp_id]
                    pos = to_hou(raw)
                    
                    k = hou.Keyframe()
                    k.setFrame(hou_frame)
                    for axes in range(3):
                        k.setValue(pos[axes])
                        node.parmTuple("t")[axes].setKeyframe(hk)
                        
    # Layout
    subnet.layoutChildren()
    print("Import Done.")

# To run in Houdini:
# 1. Update path below.
# 2. Copy paste into Python Source Editor or shelf tool.
import_motion_houdini("/Users/scotteaton/Dropbox/CODE/sam-3d-body/output/samples/skeleton_motion.npz")
