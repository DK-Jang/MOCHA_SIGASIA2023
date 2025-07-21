import scipy.ndimage as ndimage
import scipy.signal as signal
import numpy as np
import sys
sys.path.append('../motion')
sys.path.append('../etc')
import quat


def softmax(x, **kw):
    softness = kw.pop('softness', 1.0)
    maxi, mini = np.max(x, **kw), np.min(x, **kw)
    return maxi + np.log(softness + np.exp(mini - maxi))

def softmin(x, **kw):
    return -softmax(-x, **kw)

def pad_to_window(slice, window):
    def get_reflection(src, tlen):
        x = src.copy()
        x = np.flip(x, axis=0)
        ret = x.copy()
        while len(ret) < tlen:
            x = np.flip(x, axis=0)
            ret = np.concatenate((ret, x), axis=0)
        ret = ret[:tlen]
        return ret

    if len(slice) >= window:
        return slice
    left_len = (window - len(slice)) // 2 + (window - len(slice)) % 2
    right_len = (window - len(slice)) // 2
    left = np.flip(get_reflection(np.flip(slice, axis=0), left_len), axis=0)
    right = get_reflection(slice, right_len)
    slice = np.concatenate([left, slice, right], axis=0)
    assert len(slice) == window
    return slice

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

def divide_clip(input, window, window_step, vel_ang=False, divide=True):
    if not divide:  # return the whole clip
        t = ((input.shape[0]) // 4) * 4 + 4
        t = max(t, 12)
        if len(input) < t:
            input = pad_to_window(input, t)
        return [input]

    """ Slide over windows """
    windows = []
    for j in range(0, len(input)-window//4, window_step):
        """ If slice too small pad out by repeating start and end poses """
        slice = input[j:j+window]
        if len(slice) < window:
            left = slice[:1].repeat(
                (window-len(slice))//2 + (window-len(slice)) % 2, axis=0)
            right = slice[-1:].repeat((window-len(slice))//2, axis=0)
            if vel_ang:
                left[..., :] = 0.0
                right[..., :] = 0.0
            slice = np.concatenate([left, slice, right], axis=0)

        if len(slice) != window:
            raise Exception()

        windows.append(slice)

    return windows

def process_data(bvh_data, window=60, window_step=30, divide=True, mirror=False):        
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
    contact_velocity_threshold = 0.5
        
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
    
    """ Slide over windows """
    pos_windows = divide_clip(positions, window, window_step, divide=divide)
    vel_windows = divide_clip(velocities, window, window_step, vel_ang=True, divide=divide)
    rot_windows = divide_clip(rotations, window, window_step, divide=divide)
    ang_windows = divide_clip(angular_velocities, window, window_step, vel_ang=True, divide=divide)

    contacts_windows = divide_clip(contacts, window, window_step, divide=divide)

    return [pos_windows, vel_windows, rot_windows, ang_windows, contacts_windows], \
           bone_parents, bone_names

