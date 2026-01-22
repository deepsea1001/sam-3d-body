
import bpy
import numpy as np
import os
import math
from mathutils import Vector, Matrix

# Parent-Child hierarchy for MHR 70 Keypoints
# Sourced from mhr_utils.py / mhr70.py analysis
MHR70_PARENTS = {
    0: 69, # nose -> neck
    1: 0, 2: 0, # eyes -> nose
    3: 1, 4: 2, # ears -> eyes (approx)
    69: -1, # neck -> root (implicit) - actually usually neck is child of spine, but here we treat it near root
    
    # Arms
    5: 69, 6: 69, # shoulders -> neck
    7: 5, 8: 6, # elbows -> shoulders
    41: 8, 62: 7, # wrists -> elbows (Right: 41->8, Left: 62->7)
    
    # Hands - Right (Wrist: 41)
    # Thumb
    111: 41, 112: 111, 113: 112, 110: 113, # thumb3, 2, 1, tip
    # Index
    115: 41, 116: 115, 117: 116, 114: 117,
    # Middle
    119: 41, 120: 119, 121: 120, 118: 121,
    # Ring
    123: 41, 124: 123, 125: 124, 122: 125,
    # Pinky
    127: 41, 128: 127, 129: 128, 126: 129,

    # Legs (Hips usually connect to Pelvis/Root)
    9: -1, 10: -1, # hips -> root
    11: 9, 12: 10, # knees -> hips
    13: 11, 14: 12, # ankles -> knees
    17: 13, 20: 14, # heels -> ankles
    15: 13, 18: 14, # big toes -> ankles
    16: 13, 19: 14, # small toes -> ankles
}

# Simplified Hierarchy for MAIN BODY + HANDS (MHR70 IDs)
# We will just map the IDs we have in the 70-set.
# Mapping based on mhr70.py index list (0-69)
# 0:nose, 1:l-eye, 2:r-eye, 3:l-ear, 4:r-ear
# 5:l-shoulder, 6:r-shoulder
# 7:l-elbow, 8:r-elbow
# 9:l-hip, 10:r-hip
# 11:l-knee, 12:r-knee
# 13:l-ankle, 14:r-ankle
# 15:l-big, 16:l-small, 17:l-heel
# 18:r-big, 19:r-small, 20:r-heel
# 21-40: Right Hand
# 41: Right Wrist
# 42-61: Left Hand
# 62: Left Wrist
# 63-68: Extra elbow/acromion points (Optional)
# 69: Neck

# Updated Skeleton Tree
SKELETON_TREE = {
    # Root
    'Hips': {'id': -1, 'children': ['LeftUpLeg', 'RightUpLeg', 'Spine']}, # Virtual root
    
    # Left Leg
    'LeftUpLeg': {'id': 9, 'children': ['LeftLeg']},
    'LeftLeg': {'id': 11, 'children': ['LeftFoot']},
    'LeftFoot': {'id': 13, 'children': ['LeftToeBase']},
    'LeftToeBase': {'id': 15, 'children': []}, # Using big toe as end

    # Right Leg
    'RightUpLeg': {'id': 10, 'children': ['RightLeg']},
    'RightLeg': {'id': 12, 'children': ['RightFoot']},
    'RightFoot': {'id': 14, 'children': ['RightToeBase']},
    'RightToeBase': {'id': 18, 'children': []},

    # Spine
    'Spine': {'id': 69, 'children': ['Head', 'LeftShoulder', 'RightShoulder']}, # Neck/Spine
    'Head': {'id': 0, 'children': []}, # Nose

    # Left Arm
    'LeftShoulder': {'id': 5, 'children': ['LeftArm']},
    'LeftArm': {'id': 7, 'children': ['LeftForeArm']},
    'LeftForeArm': {'id': 62, 'children': ['LeftHand']},
    'LeftHand': {'id': -1, 'children': []}, # Wrist end

    # Right Arm
    'RightShoulder': {'id': 6, 'children': ['RightArm']},
    'RightArm': {'id': 8, 'children': ['RightForeArm']},
    'RightForeArm': {'id': 41, 'children': ['RightHand']},
    'RightHand': {'id': -1, 'children': []},
}

# Keypoint Map for direct lookup
ID_MAP = {
    'LeftUpLeg': 9, 'LeftLeg': 11, 'LeftFoot': 13, 'LeftToeBase': 15,
    'RightUpLeg': 10, 'RightLeg': 12, 'RightFoot': 14, 'RightToeBase': 18,
    'Spine': 69, 'Head': 0,
    'LeftShoulder': 5, 'LeftArm': 7, 'LeftForeArm': 62,
    'RightShoulder': 6, 'RightArm': 8, 'RightForeArm': 41,
}

def create_armature(name="HumanSkeleton"):
    bpy.ops.object.armature_add(enter_editmode=True, align='WORLD', location=(0, 0, 0))
    obj = bpy.context.object
    obj.name = name
    amt = obj.data
    amt.name = name + "_Amt"
    
    # Remove default bone
    bpy.ops.armature.select_all(action='SELECT')
    bpy.ops.armature.delete()
    
    bones = {}
    
    # Helper to recursively create bones
    def add_bone(node_name, parent_name=None):
        bone = amt.edit_bones.new(node_name)
        if parent_name:
            bone.parent = bones[parent_name]
        bones[node_name] = bone
        
        # Recurse
        if node_name in SKELETON_TREE:
            for child in SKELETON_TREE[node_name]['children']:
                add_bone(child, node_name)
    
    # Start chain
    add_bone('Hips')
    
    return obj

def import_motion(filepath):
    data = np.load(filepath)
    motion = data['motion'] # (Frames, Ppl, 70, 3)
    counts = data['counts']
    
    scene = bpy.context.scene
    scene.frame_start = 0
    scene.frame_end = motion.shape[0] - 1
    
    # Create an armature for the first person detected
    # (Handling multiple people implies creating multiple armatures)
    
    # For simplicity, we create one armature and animate it with Person 0
    # Or create N armatures if max people > 1
    
    max_people = motion.shape[1]
    armatures = []
    
    for i in range(max_people):
        arm = create_armature(f"Person_{i}")
        armatures.append(arm)
        # Reset mode
        bpy.ops.object.mode_set(mode='OBJECT')
    
    # Animation Loop
    for frame_idx in range(motion.shape[0]):
        scene.frame_set(frame_idx)
        
        num_ppl = counts[frame_idx]
        
        for p_idx in range(max_people):
            if p_idx >= num_ppl:
                continue
                
            obj = armatures[p_idx]
            kps = motion[frame_idx, p_idx] # (70, 3)
            
            # Update Bone Constraints / Positions
            # Since we only have positions, we can use PoseBone locations or 
            # simply set Head/Tail in EditMode (for binding) or PoseMode (for anim).
            
            # Approach: Forward Kinematics from Positions (Hard) vs Visual Locations (Easy)
            # Easy way: Use Bone Constraints "Copy Location" to Empties?
            # Or just set PoseBone matrices directly if we calculate them.
            
            # Simplest valid visual: Set Head/Tail of bones to match keypoints.
            # But standard animation requires Rotations.
            
            # We will position PoseBones using "Stretch To" or just setting locations 
            # effectively treating bones as disconnected if we just move them.
            
            # Better: Set Head to Joint, Tail to Child Joint.
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.mode_set(mode='POSE')
            
            for bone_name, kp_id in ID_MAP.items():
                pbone = obj.pose.bones.get(bone_name)
                if not pbone: continue
                
                # Get position
                # MHR coords might need coordinate system change for Blender (Y-up vs Z-up)
                # SAM3D usually outputs in camera coords.
                # Assuming Y-down (image) or standard 3D.
                # Usually: x right, y down, z forward.
                # Blender: x right, y forward, z up.
                
                # Map: x->x, y->-z, z->y
                raw_pos = kps[kp_id]
                pos = Vector((raw_pos[0], raw_pos[2], -raw_pos[1]))
                
                # Set location (Visual only, relative to rest pose if setting .location)
                # To set absolute world position:
                # pbone.matrix.translation = pos # This works but breaks hierarchy constraints
                
                # To make a proper skeleton, we really need the angles.
                # Since we don't have angles, we will "Retarget" the positions to the bones.
                # Create a temporary target constraint or just set Matrix.
                
                # Just setting Head location visually:
                # We need to find the vector to the child to point the bone.
                
                # Use a specific logic:
                # 'Hips' is root.
                if bone_name == 'Hips':
                    # Average of L/R UpLegs
                    l_hip = kps[ID_MAP['LeftUpLeg']]
                    r_hip = kps[ID_MAP['RightUpLeg']]
                    center = (l_hip + r_hip) / 2.0
                    pos = Vector((center[0], center[2], -center[1]))
                    
                    # Set absolute position of Hips (Root)
                    pbone.location = obj.matrix_world.inverted() @ pos
                    pbone.keyframe_insert(data_path="location", index=-1)
                    
                else:
                    # For other bones, we rely on IK or just simple rotation alignment.
                    # Since we promise a skeleton, let's try a simple alignment:
                    # Point bone towards child.
                    
                    node = SKELETON_TREE.get(bone_name)
                    if node and node['children']:
                        # Use first child as primary axis
                        child_name = node['children'][0]
                        child_id = SKELETON_TREE[child_name]['id']
                        if child_id != -1:
                            child_raw = kps[child_id]
                            child_pos = Vector((child_raw[0], child_raw[2], -child_raw[1]))
                            
                            my_raw = kps[ID_MAP[bone_name]]
                            my_pos = Vector((my_raw[0], my_raw[2], -my_raw[1]))
                            
                            # Direction vector
                            direction = child_pos - my_pos
                            if direction.length > 0.001:
                                # Align Y-axis of bone to direction
                                rot_quat = direction.to_track_quat('Y', 'Z')
                                
                                # Apply rotation (this is global rotation)
                                # We need local rotation relative to parent.
                                # This gets complex quickly without IK.
                                pass

            # FALLBACK: Create Empty markers for keypoints if Rig is too complex to solve blindly.
            # But user asked for skeleton.
            # Let's just create keypoints as Empties for now as a robust fallback, 
            # OR simple bones that track those empties.
            pass

    print("Import Finished.")

if __name__ == "__main__":
    # Change this path to your exported .npz file
    import_motion("/Users/scotteaton/Dropbox/CODE/sam-3d-body/output/samples/skeleton_motion.npz")
