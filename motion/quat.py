import numpy as np

def _fast_cross(a, b):
    return np.concatenate([
        a[...,1:2]*b[...,2:3] - a[...,2:3]*b[...,1:2],
        a[...,2:3]*b[...,0:1] - a[...,0:1]*b[...,2:3],
        a[...,0:1]*b[...,1:2] - a[...,1:2]*b[...,0:1]], axis=-1)

def eye(shape, dtype=np.float32):
    return np.ones(list(shape) + [4], dtype=dtype) * np.asarray([1, 0, 0, 0], dtype=dtype)

def length(x):
    return np.sqrt(np.sum(x * x, axis=-1))

def normalize(x, eps=1e-8):
    return x / (length(x)[...,np.newaxis] + eps)

def abs(x):
    return np.where(x[...,0:1] > 0.0, x, -x)

def from_angle_axis(angle, axis):
    c = np.cos(angle / 2.0)[..., np.newaxis]
    s = np.sin(angle / 2.0)[..., np.newaxis]
    q = np.concatenate([c, s * axis], axis=-1)
    return q

def to_xform(x):

    qw, qx, qy, qz = x[...,0:1], x[...,1:2], x[...,2:3], x[...,3:4]
    
    x2, y2, z2 = qx + qx, qy + qy, qz + qz
    xx, yy, wx = qx * x2, qy * y2, qw * x2
    xy, yz, wy = qx * y2, qy * z2, qw * y2
    xz, zz, wz = qx * z2, qz * z2, qw * z2
    
    return np.concatenate([
        np.concatenate([1.0 - (yy + zz), xy - wz, xz + wy], axis=-1)[...,np.newaxis,:],
        np.concatenate([xy + wz, 1.0 - (xx + zz), yz - wx], axis=-1)[...,np.newaxis,:],
        np.concatenate([xz - wy, yz + wx, 1.0 - (xx + yy)], axis=-1)[...,np.newaxis,:],
    ], axis=-2)
    
def to_xform_xy(x):

    qw, qx, qy, qz = x[...,0:1], x[...,1:2], x[...,2:3], x[...,3:4]
    
    x2, y2, z2 = qx + qx, qy + qy, qz + qz
    xx, yy, wx = qx * x2, qy * y2, qw * x2
    xy, yz, wy = qx * y2, qy * z2, qw * y2
    xz, zz, wz = qx * z2, qz * z2, qw * z2
    
    return np.concatenate([
        np.concatenate([1.0 - (yy + zz), xy - wz], axis=-1)[...,np.newaxis,:],
        np.concatenate([xy + wz, 1.0 - (xx + zz)], axis=-1)[...,np.newaxis,:],
        np.concatenate([xz - wy, yz + wx], axis=-1)[...,np.newaxis,:],
    ], axis=-2)

def from_euler(e, order='zyx'):
    axis = {
        'x': np.asarray([1, 0, 0], dtype=np.float32),
        'y': np.asarray([0, 1, 0], dtype=np.float32),
        'z': np.asarray([0, 0, 1], dtype=np.float32)}

    q0 = from_angle_axis(e[..., 0], axis[order[0]])
    q1 = from_angle_axis(e[..., 1], axis[order[1]])
    q2 = from_angle_axis(e[..., 2], axis[order[2]])

    return mul(q0, mul(q1, q2))

def from_xform(ts):
    
    return normalize(
        np.where((ts[...,2,2] < 0.0)[...,np.newaxis],
            np.where((ts[...,0,0] >  ts[...,1,1])[...,np.newaxis],
                np.concatenate([
                    (ts[...,2,1]-ts[...,1,2])[...,np.newaxis], 
                    (1.0 + ts[...,0,0] - ts[...,1,1] - ts[...,2,2])[...,np.newaxis], 
                    (ts[...,1,0]+ts[...,0,1])[...,np.newaxis], 
                    (ts[...,0,2]+ts[...,2,0])[...,np.newaxis]], axis=-1),
                np.concatenate([
                    (ts[...,0,2]-ts[...,2,0])[...,np.newaxis], 
                    (ts[...,1,0]+ts[...,0,1])[...,np.newaxis], 
                    (1.0 - ts[...,0,0] + ts[...,1,1] - ts[...,2,2])[...,np.newaxis], 
                    (ts[...,2,1]+ts[...,1,2])[...,np.newaxis]], axis=-1)),
            np.where((ts[...,0,0] < -ts[...,1,1])[...,np.newaxis],
                np.concatenate([
                    (ts[...,1,0]-ts[...,0,1])[...,np.newaxis], 
                    (ts[...,0,2]+ts[...,2,0])[...,np.newaxis], 
                    (ts[...,2,1]+ts[...,1,2])[...,np.newaxis], 
                    (1.0 - ts[...,0,0] - ts[...,1,1] + ts[...,2,2])[...,np.newaxis]], axis=-1),
                np.concatenate([
                    (1.0 + ts[...,0,0] + ts[...,1,1] + ts[...,2,2])[...,np.newaxis], 
                    (ts[...,2,1]-ts[...,1,2])[...,np.newaxis], 
                    (ts[...,0,2]-ts[...,2,0])[...,np.newaxis], 
                    (ts[...,1,0]-ts[...,0,1])[...,np.newaxis]], axis=-1))))

def from_xform_xy(x):

    c2 = _fast_cross(x[...,0], x[...,1])
    c2 = c2 / np.sqrt(np.sum(np.square(c2), axis=-1))[...,np.newaxis]
    c1 = _fast_cross(c2, x[...,0])
    c1 = c1 / np.sqrt(np.sum(np.square(c1), axis=-1))[...,np.newaxis]
    c0 = x[...,0]
    
    return from_xform(np.concatenate([
        c0[...,np.newaxis], 
        c1[...,np.newaxis], 
        c2[...,np.newaxis]], axis=-1))

def inv(q):
    return np.asarray([1, -1, -1, -1], dtype=np.float32) * q

def mul(x, y):
    x0, x1, x2, x3 = x[..., 0:1], x[..., 1:2], x[..., 2:3], x[..., 3:4]
    y0, y1, y2, y3 = y[..., 0:1], y[..., 1:2], y[..., 2:3], y[..., 3:4]

    return np.concatenate([
        y0 * x0 - y1 * x1 - y2 * x2 - y3 * x3,
        y0 * x1 + y1 * x0 - y2 * x3 + y3 * x2,
        y0 * x2 + y1 * x3 + y2 * x0 - y3 * x1,
        y0 * x3 - y1 * x2 + y2 * x1 + y3 * x0], axis=-1)

def inv_mul(x, y):
    return mul(inv(x), y)

def mul_inv(x, y):
    return mul(x, inv(y))

def mul_vec(q, x):
    t = 2.0 * _fast_cross(q[..., 1:], x)
    return x + q[..., 0][..., np.newaxis] * t + _fast_cross(q[..., 1:], t)

def inv_mul_vec(q, x):
    return mul_vec(inv(q), x)

def unroll(x):
    y = x.copy()
    for i in range(1, len(x)):
        d0 = np.sum( y[i] * y[i-1], axis=-1)
        d1 = np.sum(-y[i] * y[i-1], axis=-1)
        y[i][d0 < d1] = -y[i][d0 < d1]
    return y

def between(x, y):
    return np.concatenate([
        np.sqrt(np.sum(x*x, axis=-1) * np.sum(y*y, axis=-1))[...,np.newaxis] + 
        np.sum(x * y, axis=-1)[...,np.newaxis], 
        _fast_cross(x, y)], axis=-1)
        
def log(x, eps=1e-5):
    length = np.sqrt(np.sum(np.square(x[...,1:]), axis=-1))[...,np.newaxis]
    halfangle = np.where(length < eps, np.ones_like(length), np.arctan2(length, x[...,0:1]) / length)
    return halfangle * x[...,1:]
    
def exp(x, eps=1e-5):
    halfangle = np.sqrt(np.sum(np.square(x), axis=-1))[...,np.newaxis]
    c = np.where(halfangle < eps, np.ones_like(halfangle), np.cos(halfangle))
    s = np.where(halfangle < eps, np.ones_like(halfangle), np.sinc(halfangle / np.pi))
    return np.concatenate([c, s * x], axis=-1)
    
def to_scaled_angle_axis(x, eps=1e-5):
    return 2.0 * log(x, eps)
    
def from_scaled_angle_axis(x, eps=1e-5):
    return exp(x / 2.0, eps)

def fk(lrot, lpos, parents):
    
    gp, gr = [lpos[...,:1,:]], [lrot[...,:1,:]]
    for i in range(1, len(parents)):
        gp.append(mul_vec(gr[parents[i]], lpos[...,i:i+1,:]) + gp[parents[i]])
        gr.append(mul    (gr[parents[i]], lrot[...,i:i+1,:]))
        
    return np.concatenate(gr, axis=-2), np.concatenate(gp, axis=-2)
    
def ik(grot, gpos, parents):
    
    return (
        np.concatenate([
            grot[...,:1,:],
            mul(inv(grot[...,parents[1:],:]), grot[...,1:,:]),
        ], axis=-2),
        np.concatenate([
            gpos[...,:1,:],
            mul_vec(
                inv(grot[...,parents[1:],:]),
                gpos[...,1:,:] - gpos[...,parents[1:],:]),
        ], axis=-2))
    
def fk_vel(lrot, lpos, lvel, lang, parents):
    
    gp, gr, gv, ga = [lpos[...,:1,:]], [lrot[...,:1,:]], [lvel[...,:1,:]], [lang[...,:1,:]]
    for i in range(1, len(parents)):
        gp.append(mul_vec(gr[parents[i]], lpos[...,i:i+1,:]) + gp[parents[i]])
        gr.append(mul    (gr[parents[i]], lrot[...,i:i+1,:]))
        gv.append(mul_vec(gr[parents[i]], lvel[...,i:i+1,:]) + 
            _fast_cross(ga[parents[i]], mul_vec(gr[parents[i]], lpos[...,i:i+1,:])) +
            gv[parents[i]])
        ga.append(mul_vec(gr[parents[i]], lang[...,i:i+1,:]) + ga[parents[i]])
        
    return (
        np.concatenate(gr, axis=-2), 
        np.concatenate(gp, axis=-2),
        np.concatenate(gv, axis=-2),
        np.concatenate(ga, axis=-2))

# Forward kinematics but also compute the velocities
def fk_vel_bone(
    bone_positions,
    bone_velocities,
    bone_rotations,
    bone_angular_velocities,
    bone_parents,
    bone):
    if bone_parents[bone] != -1:
        parent_position, parent_velocity, parent_rotation, parent_angular_velocity = \
            fk_vel_bone(bone_positions,
                        bone_velocities,
                        bone_rotations,
                        bone_angular_velocities,
                        bone_parents,
                        bone_parents[bone])

        bone_position = mul_vec(parent_rotation, bone_positions[bone]) + parent_position
        bone_velocity = (
            parent_velocity +
            mul_vec(parent_rotation, bone_velocities[bone]) +
            _fast_cross(parent_angular_velocity, mul_vec(parent_rotation, bone_positions[bone]))
        )
        bone_rotation = mul(parent_rotation, bone_rotations[bone])
        bone_angular_velocity = mul_vec(parent_rotation, bone_angular_velocities[bone]) + parent_angular_velocity
    else:
        bone_position = bone_positions[bone]
        bone_velocity = bone_velocities[bone]
        bone_rotation = bone_rotations[bone]
        bone_angular_velocity = bone_angular_velocities[bone]

    return bone_position, bone_velocity, bone_rotation, bone_angular_velocity

# Compute forward kinematics of just some joints using a
# mask to indicate which joints are already computed
def fk_partial(
    global_bone_positions,      # call by reference
    global_bone_rotations,
    global_bone_computed,
    local_bone_positions,
    local_bone_rotations,
    bone_parents,
    bone):
    
    if bone_parents[bone] == -1:
        global_bone_positions[bone] = local_bone_positions[bone]
        global_bone_rotations[bone] = local_bone_rotations[bone]
        global_bone_computed[bone] = True
        return global_bone_positions, global_bone_rotations, global_bone_computed
    
    if not global_bone_computed[bone_parents[bone]]:
        fk_partial(
            global_bone_positions,
            global_bone_rotations,
            global_bone_computed,
            local_bone_positions,
            local_bone_rotations,
            bone_parents,
            bone_parents[bone])
    
    parent_position = global_bone_positions[bone_parents[bone]]
    parent_rotation = global_bone_rotations[bone_parents[bone]]
    global_bone_positions[bone] = mul_vec(parent_rotation, local_bone_positions[bone]) + parent_position
    global_bone_rotations[bone] = mul(parent_rotation, local_bone_rotations[bone])
    global_bone_computed[bone] = True

    return global_bone_positions, global_bone_rotations, global_bone_computed

# Rotate a joint to look toward some 
# given target position
def ik_look_at(bone_rotation, 
               global_parent_rotation, 
               global_rotation, 
               global_position, 
               child_position, 
               target_position, 
               eps=1e-5):
    curr_dir = normalize(child_position - global_position)
    targ_dir = normalize(target_position - global_position)

    if np.abs(1.0 - np.dot(curr_dir, targ_dir)) > eps:
        bone_rotation = inv_mul(global_parent_rotation, 
            mul(between(curr_dir, targ_dir), global_rotation))
    
    return bone_rotation

# Basic two-joint IK in the style of https://theorangeduck.com/page/simple-two-joint
# Here I add a basic "forward vector" which acts like a kind of pole-vetor
# to control the bending direction
def ik_two_bone(bone_root_lr, 
                bone_mid_lr, 
                bone_root, 
                bone_mid, 
                bone_end, 
                target, 
                fwd, 
                bone_root_gr, 
                bone_mid_gr, 
                bone_par_gr, 
                max_length_buffer):
    max_extension = length(bone_root - bone_mid) + length(bone_mid - bone_end) - max_length_buffer

    target_clamp = target
    if length(target - bone_root) > max_extension:
        target_clamp = bone_root + max_extension * normalize(target - bone_root)

    axis_dwn = normalize(bone_end - bone_root)
    axis_rot = normalize(np.cross(axis_dwn, fwd))

    a = bone_root
    b = bone_mid
    c = bone_end
    t = target_clamp

    lab = length(b - a)
    lcb = length(b - c)
    lat = length(t - a)

    ac_ab_0 = np.arccos(np.clip(np.dot(normalize(c - a), normalize(b - a)), -1.0, 1.0))
    ba_bc_0 = np.arccos(np.clip(np.dot(normalize(a - b), normalize(c - b)), -1.0, 1.0))

    ac_ab_1 = np.arccos(np.clip((lab * lab + lat * lat - lcb * lcb) / (2.0 * lab * lat), -1.0, 1.0))
    ba_bc_1 = np.arccos(np.clip((lab * lab + lcb * lcb - lat * lat) / (2.0 * lab * lcb), -1.0, 1.0))

    r0 = from_angle_axis(ac_ab_1 - ac_ab_0, axis_rot)
    r1 = from_angle_axis(ba_bc_1 - ba_bc_0, axis_rot)

    c_a = normalize(bone_end - bone_root)
    t_a = normalize(target_clamp - bone_root)

    r2 = from_angle_axis(
        np.arccos(np.clip(np.dot(c_a, t_a), -1.0, 1.0)), 
        normalize(np.cross(c_a, t_a)))

    bone_root_lr = inv_mul(bone_par_gr, mul(r2, mul(r0, bone_root_gr)))
    bone_mid_lr = inv_mul(bone_root_gr, mul(r1, bone_mid_gr))

    return bone_root_lr, bone_mid_lr


def to_euler(x, order='xyz'):
    
    q0 = x[...,0:1]
    q1 = x[...,1:2]
    q2 = x[...,2:3]
    q3 = x[...,3:4]
    
    if order == 'xyz':
    
        return np.concatenate([
            np.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2)),
            np.arcsin((2 * (q0 * q2 - q3 * q1)).clip(-1,1)),
            np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))], axis=-1)
            
    elif order == 'yzx':
    
        return np.concatenate([
            np.arctan2(2 * (q1 * q0 - q2 * q3), -q1 * q1 + q2 * q2 - q3 * q3 + q0 * q0),
            np.arctan2(2 * (q2 * q0 - q1 * q3),  q1 * q1 - q2 * q2 - q3 * q3 + q0 * q0),
            np.arcsin((2 * (q1 * q2 + q3 * q0)).clip(-1,1))], axis=-1)
            
    else:
        raise NotImplementedError('Cannot convert from ordering %s' % order)
        