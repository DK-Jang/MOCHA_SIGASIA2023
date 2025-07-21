import scipy.ndimage as ndimage
import scipy.signal as signal
from pathlib import Path
from itertools import chain
import numpy as np
import struct
import shutil
import os
import sys
sys.path.append('../motion')
sys.path.append('../etc')
import quat
import bvh
from utils import get_config

def softmax(x, **kw):
    softness = kw.pop('softness', 1.0)
    maxi, mini = np.max(x, **kw), np.min(x, **kw)
    return maxi + np.log(softness + np.exp(mini - maxi))

def softmin(x, **kw):
    return -softmax(-x, **kw)

""" Basic function for mirroring animation data with this particular skeleton structure """
def animation_mirror(lrot, lpos, names, parents):

    joints_mirror = np.array([(
        names.index('Left'+n[5:]) if n.startswith('Right') else (
        names.index('Right'+n[4:]) if n.startswith('Left') else 
        names.index(n))) for n in names])

    mirror_pos = np.array([-1, 1, 1])
    mirror_rot = np.array([[-1, -1, 1], [1, 1, -1], [1, 1, -1]])

    grot, gpos = quat.fk(lrot, lpos, parents)

    gpos_mirror = mirror_pos * gpos[:,joints_mirror]
    grot_mirror = quat.from_xform(mirror_rot * quat.to_xform(grot[:,joints_mirror]))
    
    return quat.ik(grot_mirror, gpos_mirror, parents)

def get_files(directory):
    return [os.path.join(directory, f) for f in sorted(list(os.listdir(directory)))
            if os.path.isfile(os.path.join(directory, f))
            and f.endswith('.bvh') and f != 'rest.bvh']

def get_character_dir(directory):
    return [os.path.join(directory, f) for f in sorted(list(os.listdir(directory)))
            if os.path.isdir(os.path.join(directory, f))]

def get_bvh_files(dname):
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
                          for ext in ['bvh']]))
    return fnames


def main():
    dataset_dir = '../bvh'
    dataset_save_dir = '../datasets/mocha60'
    if not os.path.exists(dataset_save_dir):
        os.makedirs(dataset_save_dir)
    dataset_cfg = get_config('../configs/dataset.yaml')

    bvh_files = np.array(get_bvh_files(dataset_dir))
    style_names = dataset_cfg["mocha_style_names"]
    action_names = dataset_cfg["mocha_action_names"]

    # We will accumulate data in these lists
    bone_positions = []
    bone_velocities = []
    bone_rotations = []
    bone_angular_velocities = []
    bone_parents = []
    bone_names = []
        
    range_starts = []
    range_stops = []

    contact_states = []
    style_labels = []
    action_labels = []

    for i, item in enumerate(bvh_files):
        for value in style_names:
            if value in item.stem:
                style_name = value
                break
        style_label = style_names.index(style_name)

        for value in action_names:
            if value in item.stem:
                action_name = value
                break
        action_label = action_names.index(action_name)

        for mirror in [False, True]:
        # for mirror in [False]:
            print('Processing %i of %i (%s)%s' % (i+1, len(bvh_files), item, "_Mirrored" if mirror else ""))

            # load bvh files
            bvh_data = bvh.load(item._str)

            positions = bvh_data['positions']
            rotations = quat.unroll(quat.from_euler(np.radians(bvh_data['rotations']), order=bvh_data['order']))

            # Convert from cm to m
            positions *= 0.01

            if mirror:
                rotations, positions = animation_mirror(rotations, positions, bvh_data['names'], bvh_data['parents'])
                rotations = quat.unroll(rotations)

            """ Extract Root Bone """
            # First compute world space positions/rotations
            global_rotations, global_positions = quat.fk(rotations, positions, bvh_data['parents'])
                
            # Specify joints to use for simulation bone 
            root_position_joint = bvh_data['names'].index("Spine2")
            root_rotation_joint = bvh_data['names'].index("Hips")
                
            # Position comes from spine joint
            root_position = np.array([1.0, 0.0, 1.0]) * global_positions[:,root_position_joint:root_position_joint+1]
            root_position = signal.savgol_filter(root_position, 15, 3, axis=0, mode='interp')

            sdr_l, sdr_r, hip_l, hip_r = \
                bvh_data['names'].index("LeftShoulder"), bvh_data['names'].index("RightShoulder"), \
                bvh_data['names'].index("LeftUpLeg"), bvh_data['names'].index("RightUpLeg")
            across = (
                (global_positions[:, sdr_l:sdr_l+1] - global_positions[:, sdr_r:sdr_r+1]) +
                (global_positions[:, hip_l:hip_l+1] - global_positions[:, hip_r:hip_r+1])
                )
            root_direction = np.array([1.0, 0.0, 1.0]) * np.cross(across, np.array([0, 1, 0]))

            # We need to re-normalize the direction after both projection and smoothing
            root_direction = root_direction / np.sqrt(np.sum(np.square(root_direction), axis=-1))[...,np.newaxis]
            root_direction = signal.savgol_filter(root_direction, 31, 3, axis=0, mode='interp')
            root_direction = root_direction / np.sqrt(np.sum(np.square(root_direction), axis=-1)[...,np.newaxis])
            
            # Extract rotation from direction
            root_rotation = quat.normalize(quat.between(np.array([0, 0, 1]), root_direction))

            # Transform joints to be local to root and append root as root bone
            positions[:,0:1] = quat.mul_vec(quat.inv(root_rotation), positions[:,0:1] - root_position)
            rotations[:,0:1] = quat.mul(quat.inv(root_rotation), rotations[:,0:1])

            positions = np.concatenate([root_position, positions], axis=1)
            rotations = np.concatenate([root_rotation, rotations], axis=1)

            bone_parents = np.concatenate([[-1], bvh_data['parents'] + 1])
            bone_names = ['Root'] + bvh_data['names']

            """ Compute Velocities """
            # Compute velocities via central difference
            velocities = np.empty_like(positions)
            velocities[1:-1] = (
                0.5 * (positions[2:  ] - positions[1:-1]) * 60.0 +
                0.5 * (positions[1:-1] - positions[ :-2]) * 60.0)
            velocities[ 0] = velocities[ 1] - (velocities[ 3] - velocities[ 2])
            velocities[-1] = velocities[-2] + (velocities[-2] - velocities[-3])
            
            # Same for angular velocities
            angular_velocities = np.zeros_like(positions)
            angular_velocities[1:-1] = (
                0.5 * quat.to_scaled_angle_axis(quat.abs(quat.mul_inv(rotations[2:  ], rotations[1:-1]))) * 60.0 +
                0.5 * quat.to_scaled_angle_axis(quat.abs(quat.mul_inv(rotations[1:-1], rotations[ :-2]))) * 60.0)
            angular_velocities[ 0] = angular_velocities[ 1] - (angular_velocities[ 3] - angular_velocities[ 2])
            angular_velocities[-1] = angular_velocities[-2] + (angular_velocities[-2] - angular_velocities[-3])

            """ Compute Contact Data """
            global_rotations, global_positions, global_velocities, global_angular_velocities = \
                quat.fk_vel(rotations, 
                            positions, 
                            velocities,
                            angular_velocities,
                            bone_parents)

            """ Foot Contacts """
            contact_velocity_threshold = 0.2
                
            contact_velocity = np.sqrt(np.sum(global_velocities[:,np.array([
                    bone_names.index("LeftToeBase"), 
                    bone_names.index("RightToeBase")])]**2, axis=-1))
                
            # Contacts are given for when contact bones are below velocity threshold
            contacts = contact_velocity < contact_velocity_threshold
            
            # Median filter here acts as a kind of "majority vote", and removes
            # small regions where contact is either active or inactive
            for ci in range(contacts.shape[1]):
                contacts[:,ci] = ndimage.median_filter(
                    contacts[:,ci], 
                    size=6, 
                    mode='nearest')
            
            """ Append to Database """
            bone_positions.append(positions)
            bone_velocities.append(velocities)
            bone_rotations.append(rotations)
            bone_angular_velocities.append(angular_velocities)
            
            offset = 0 if len(range_starts) == 0 else range_stops[-1] 

            range_starts.append(offset)
            range_stops.append(offset + len(positions))
            
            contact_states.append(contacts)
            style_labels.append(style_label)
            action_labels.append(action_label)

    """ Concatenate Data """
    bone_positions = np.concatenate(bone_positions, axis=0).astype(np.float32)
    bone_velocities = np.concatenate(bone_velocities, axis=0).astype(np.float32)
    bone_rotations = np.concatenate(bone_rotations, axis=0).astype(np.float32)
    bone_angular_velocities = np.concatenate(bone_angular_velocities, axis=0).astype(np.float32)
    bone_parents = bone_parents.astype(np.int32)

    range_starts = np.array(range_starts).astype(np.int32)
    range_stops = np.array(range_stops).astype(np.int32)
    style_labels = np.array(style_labels).astype(np.int32)
    action_labels = np.array(action_labels).astype(np.int32)

    contact_states = np.concatenate(contact_states, axis=0).astype(np.uint8)

    """ Write Database """

    print("Writing Database...")

    with open(os.path.join(dataset_save_dir, 'database.bin'), 'wb') as f:
        
        nframes = bone_positions.shape[0]
        nbones = bone_positions.shape[1]
        nranges = range_starts.shape[0]
        ncontacts = contact_states.shape[1]
        
        f.write(struct.pack('II', nframes, nbones) + bone_positions.ravel().tobytes())
        f.write(struct.pack('II', nframes, nbones) + bone_velocities.ravel().tobytes())
        f.write(struct.pack('II', nframes, nbones) + bone_rotations.ravel().tobytes())
        f.write(struct.pack('II', nframes, nbones) + bone_angular_velocities.ravel().tobytes())
        f.write(struct.pack('I', nbones) + bone_parents.ravel().tobytes())
        
        f.write(struct.pack('I', nranges) + range_starts.ravel().tobytes())
        f.write(struct.pack('I', nranges) + range_stops.ravel().tobytes())
        f.write(struct.pack('I', nranges) + style_labels.ravel().tobytes())
        f.write(struct.pack('I', nranges) + action_labels.ravel().tobytes())
        
        f.write(struct.pack('II', nframes, ncontacts) + contact_states.ravel().tobytes())


if __name__ == '__main__':
    main()