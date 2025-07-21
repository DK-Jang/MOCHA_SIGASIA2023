# import os
import sys
import numpy as np
sys.path.append('./motion')
import quat

def length(x):
    return np.sqrt(np.sum(x * x, axis=-1))

def fast_negexpf(x):
    return 1.0 / (1.0 + x + 0.48*x*x + 0.235*x*x*x)

def halflife_to_damping(halflife,  eps=1e-5):
    return (4.0 * np.log(2.0)) / (halflife + eps)

# -------------------------------------------------------------------------------------
    
def decay_spring_damper_exact_pos(x, v, halflife, dt):
    y = halflife_to_damping(halflife) / 2.0
    j1 = v + x*y
    eydt = fast_negexpf(y*dt)

    x = eydt*(x + j1*dt)
    v = eydt*(v - j1*y*dt)

    return x, v

def decay_spring_damper_exact_rot(x, v, halflife, dt):
    y = halflife_to_damping(halflife) / 2.0
    j0 = quat.to_scaled_angle_axis(x)
    j1 = v + j0*y
    eydt = fast_negexpf(y*dt)

    x = quat.from_scaled_angle_axis(eydt*(j0 + j1*dt))
    v = eydt*(v - j1*y*dt)

    return x, v

def decay_spring_damper_exact(x, v, halflife, dt):
    y = halflife_to_damping(halflife) / 2.0
    eydt = fast_negexpf(y*dt)

    # when x is a scalar (float)
    if isinstance(x, (float, np.float32, np.float64)):
        j1 = v + x*y
        x = eydt*(x + j1*dt)
        v = eydt*(v - j1*y*dt)

    # when x is a vector (numpy array with shape 3)
    elif isinstance(x, np.ndarray) and x.shape[-1] == 3:
        j1 = v + x*y
        x = eydt*(x + j1*dt)
        v = eydt*(v - j1*y*dt)

    # when x is a quaternion (numpy array with shape 4)
    elif isinstance(x, np.ndarray) and x.shape[-1] == 4:
        j0 = quat.to_scaled_angle_axis(x)
        j1 = v + j0*y

        x = quat.from_scaled_angle_axis(eydt*(j0 + j1*dt))
        v = eydt*(v - j1*y*dt)

    else:
        raise TypeError("Invalid input type for x. It should be a scalar (float), \
                a numpy array with shape (3,) for vector, or shape (4,) for quaternion.")

    return x, v

# -------------------------------------------------------------------------------------

def inertialize_transition_pos(off_x, off_v, src_x, src_v, dst_x, dst_v):
    off_x = (src_x + off_x) - dst_x;
    off_v = (src_v + off_v) - dst_v;
    return off_x, off_v

def inertialize_update_pos(off_x, off_v, in_x, in_v, halflife, dt):
    off_x, off_v = decay_spring_damper_exact_pos(off_x, off_v, halflife, dt)
    out_x = in_x + off_x
    out_v = in_v + off_v
    return out_x, out_v, off_x, off_v

def inertialize_transition_rot(off_x, off_v, src_x, src_v, dst_x, dst_v):
    off_x = quat.abs(quat.mul(quat.mul(off_x, src_x), quat.inv(dst_x)))
    off_v = (off_v + src_v) - dst_v;
    return off_x, off_v

def inertialize_update_rot(off_x, off_v, in_x, in_v, halflife, dt):
    off_x, off_v = decay_spring_damper_exact_rot(off_x, off_v, halflife, dt)
    out_x = quat.mul(off_x, in_x)
    out_v = off_v + in_v
    return out_x, out_v, off_x, off_v

def inertialize_transition(off_x, off_v, src_x, src_v, dst_x, dst_v):
    # when off_x is a vector (numpy array with shape 3)
    if isinstance(off_x, np.ndarray) and off_x.shape[-1] == 3:
        off_x = (src_x + off_x) - dst_x
        off_v = (src_v + off_v) - dst_v

    # when off_x is a quaternion (numpy array with shape 4)
    elif isinstance(off_x, np.ndarray) and off_x.shape[-1] == 4:
        off_x = quat.abs(quat.mul(quat.mul(off_x, src_x), quat.inv(dst_x)))
        off_v = (off_v + src_v) - dst_v

    else:
        raise TypeError("Invalid input type for off_x. It should be a numpy array \
                        with shape (3,) for vector or shape (4,) for quaternion.")
    
    return off_x, off_v

def inertialize_update(out_x, out_v, off_x, off_v, in_x, in_v, halflife, dt):
    off_x, off_v = decay_spring_damper_exact(off_x, off_v, halflife, dt)

    # when out_x is a vector (numpy array with shape 3)
    if isinstance(out_x, np.ndarray) and out_x.shape[-1] == 3:
        out_x = in_x + off_x
        out_v = in_v + off_v

    # when out_x is a quaternion (numpy array with shape 4)
    elif isinstance(out_x, np.ndarray) and out_x.shape[-1] == 4:
        out_x = quat.mul(off_x, in_x)
        out_v = off_v + in_v

    else:
        raise TypeError("Invalid input type for out_x. It should be a numpy array \
                        with shape (3,) for vector or shape (4,) for quaternion.")

    return out_x, out_v, off_x, off_v

# -------------------------------------------------------------------------------------

# This function transitions the inertializer for the full character. 
# It takes as input the current offsets, as well as the root transition locations,
# current root state, and the full pose information 
# for the pose being transitioned from (src) as well as the pose being transitioned 
# to (dst) in their own animation spaces.
def pose_transition(bone_offset_positions,
                   bone_offset_velocities,
                   bone_offset_rotations,
                   bone_offset_angular_velocities,
                   root_position,
                   root_velocity,
                   root_rotation,
                   root_angular_velocity,
                   bone_src_positions,
                   bone_src_velocities,
                   bone_src_rotations,
                   bone_src_angular_velocities,
                   bone_dst_positions,
                   bone_dst_velocities,
                   bone_dst_rotations,
                   bone_dst_angular_velocities):
    
    # First we record the root position and rotation
    # in the animation data for the source and destination animation
    transition_dst_position = root_position;
    transition_dst_rotation = root_rotation;
    transition_src_position = bone_dst_positions[0];
    transition_src_rotation = bone_dst_rotations[0];

    # We then find the velocities so we can transition the root inertiaizers
    world_space_dst_velocity = quat.mul_vec(transition_dst_rotation, 
        quat.mul_vec(transition_src_rotation, bone_dst_velocities[0]))
    
    world_space_dst_angular_velocity = quat.mul_vec(transition_dst_rotation,
        quat.mul_vec(transition_src_rotation, bone_dst_angular_velocities[0]))

    # Transition inertializers recording the offsets for the root joint
    bone_offset_positions[0], bone_offset_velocities[0] = \
    inertialize_transition_pos(
        bone_offset_positions[0],
        bone_offset_velocities[0],
        root_position,
        root_velocity,
        root_position,
        world_space_dst_velocity)
    
    bone_offset_rotations[0], bone_offset_angular_velocities[0] = \
    inertialize_transition_rot(
        bone_offset_rotations[0],
        bone_offset_angular_velocities[0],
        root_rotation,
        root_angular_velocity,
        root_rotation,
        world_space_dst_angular_velocity)
    
    # Transition all the inertializers for each other bone
    for i in range(1, len(bone_offset_positions)):
        bone_offset_positions[i], bone_offset_velocities[i] = \
        inertialize_transition_pos(
            bone_offset_positions[i],
            bone_offset_velocities[i],
            bone_src_positions[i],
            bone_src_velocities[i],
            bone_dst_positions[i],
            bone_dst_velocities[i])
        
        bone_offset_rotations[i], bone_offset_angular_velocities[i] = \
        inertialize_transition_rot(
            bone_offset_rotations[i],
            bone_offset_angular_velocities[i],
            bone_src_rotations[i],
            bone_src_angular_velocities[i],
            bone_dst_rotations[i],
            bone_dst_angular_velocities[i])

    return (bone_offset_positions, bone_offset_velocities,
            bone_offset_rotations, bone_offset_angular_velocities,
            transition_src_position, transition_src_rotation,
            transition_dst_position, transition_dst_rotation)

# This function updates the inertializer states. 
# Here it outputs the smoothed animation (input plus offset) 
# as well as updating the offsets themselves. 
# It takes as input the current playing animation as well as 
# the root transition locations, a halflife, and a dt

def pose_update(bone_positions,
               bone_velocities,
               bone_rotations,
               bone_angular_velocities,
               bone_offset_positions,
               bone_offset_velocities,
               bone_offset_rotations,
               bone_offset_angular_velocities,
               bone_input_positions,
               bone_input_velocities,
               bone_input_rotations,
               bone_input_angular_velocities,
               transition_src_position,
               transition_src_rotation,
               transition_dst_position,
               transition_dst_rotation,
               halflife,
               dt):

    # First we find the next root position, velocity, rotation
    # and rotational velocity in the world space by transforming 
    # the input animation from it's animation space into the 
    # space of the currently playing animation.
    # FIXME: dst_position, dst_rotation부분은 지금은 필요없긴한데, 나중에 controller사용할때는
    # 필요함
    world_space_position = quat.mul_vec(transition_dst_rotation, 
        quat.inv_mul_vec(transition_src_rotation, 
            bone_input_positions[0] - transition_src_position)) + transition_dst_position
    
    world_space_velocity = quat.mul_vec(transition_dst_rotation, 
        quat.inv_mul_vec(transition_src_rotation, bone_input_velocities[0]))
    
    # Normalize here because quat inv mul can sometimes produce 
    # unstable returns when the two rotations are very close.
    world_space_rotation = quat.normalize(quat.mul(transition_dst_rotation, 
        quat.inv_mul(transition_src_rotation, bone_input_rotations[0])))
    
    world_space_angular_velocity = quat.mul_vec(transition_dst_rotation, 
        quat.inv_mul_vec(transition_src_rotation, bone_input_angular_velocities[0]))
    
    # Then we update these two inertializers with these new world space inputs
    bone_positions[0], bone_velocities[0], bone_offset_positions[0], bone_offset_velocities[0] = \
    inertialize_update_pos(
        bone_offset_positions[0],
        bone_offset_velocities[0],
        world_space_position,
        world_space_velocity,
        halflife,
        dt)
    
    bone_rotations[0], bone_angular_velocities[0], bone_offset_rotations[0], bone_offset_angular_velocities[0] = \
    inertialize_update_rot(
        bone_offset_rotations[0],
        bone_offset_angular_velocities[0],
        world_space_rotation,
        world_space_angular_velocity,
        halflife,
        dt)
    
    # Then we update the inertializers for the rest of the bones
    for i in range(1, len(bone_positions)):
        bone_positions[i], bone_velocities[i], bone_offset_positions[i], bone_offset_velocities[i] = \
        inertialize_update_pos(
            bone_offset_positions[i],
            bone_offset_velocities[i],
            bone_input_positions[i],
            bone_input_velocities[i],
            halflife,
            dt)
        
        bone_rotations[i], bone_angular_velocities[i], bone_offset_rotations[i], bone_offset_angular_velocities[i] = \
        inertialize_update_rot(
            bone_offset_rotations[i],
            bone_offset_angular_velocities[i],
            bone_input_rotations[i],
            bone_input_angular_velocities[i],
            halflife,
            dt)

    return bone_positions, bone_velocities, bone_rotations, bone_angular_velocities, \
           bone_offset_positions, bone_offset_velocities, bone_offset_rotations, bone_offset_angular_velocities


def contact_update(
    contact_state,
    contact_lock,
    contact_position,
    contact_velocity,
    contact_point,
    contact_target,
    contact_offset_position,
    contact_offset_velocity,
    input_contact_position,
    input_contact_state,
    unlock_radius,
    foot_height,
    halflife,
    dt,
    eps=1e-8):
    
    # First compute the input contact position velocity via finite difference
    input_contact_velocity = (input_contact_position - contact_target) / (dt + eps)
    contact_target = input_contact_position
    
    # Update the inertializer to tick forward in time
    contact_position, contact_velocity, contact_offset_position, contact_offset_velocity = \
    inertialize_update(
        contact_position,
        contact_velocity,
        contact_offset_position,
        contact_offset_velocity,
        # If locked we feed the contact point and zero velocity,
        # otherwise we feed the input from the animation
        contact_point if contact_lock else input_contact_position,
        np.zeros(3) if contact_lock else input_contact_velocity,
        halflife,
        dt)
    
    # If the contact point is too far from the current input position
    # then we need to unlock the contact
    unlock_contact = contact_lock and (
        length(contact_point - input_contact_position) > unlock_radius)
    
    # If the contact was previously inactive but is now active we
    # need to transition to the locked contact state
    if not contact_state and input_contact_state:
        # Contact point is given by the current position of
        # the foot projected onto the ground plus foot height
        contact_lock = True
        contact_point = contact_position.copy()
        contact_point[1] = foot_height
        
        contact_offset_position, contact_offset_velocity = \
        inertialize_transition(
            contact_offset_position,
            contact_offset_velocity,
            input_contact_position,
            input_contact_velocity,
            contact_point,
            np.zeros(3))
    
    # Otherwise if we need to unlock or we were previously in
    # contact but are no longer we transition to just taking
    # the input position as-is
    elif (contact_lock and contact_state and not input_contact_state) or unlock_contact:
        contact_lock = False
        
        contact_offset_position, contact_offset_velocity = \
        inertialize_transition(
            contact_offset_position,
            contact_offset_velocity,
            contact_point,
            np.zeros(3),
            input_contact_position,
            input_contact_velocity)
    
    # Update contact state
    contact_state = input_contact_state

    return contact_state, contact_lock, contact_position, contact_velocity, \
           contact_point, contact_target, contact_offset_position, contact_offset_velocity
