import numpy as np

def _fast_cross(a, b):
    return np.concatenate([
        a[...,1:2]*b[...,2:3] - a[...,2:3]*b[...,1:2],
        a[...,2:3]*b[...,0:1] - a[...,0:1]*b[...,2:3],
        a[...,0:1]*b[...,1:2] - a[...,1:2]*b[...,0:1]], axis=-1)

def mul(x, y):
    return np.matmul(x, y)
    
def mul_vec(x, v):
    return np.matmul(x, v[...,np.newaxis])[...,0]

def inv_mul(x, y):
    return np.matmul(x.transpose(-1, -2), y)
    
def inv_mul_vec(x, v):
    return np.matmul(x.transpose(-1, -2), v[...,np.newaxis])[...,0]

def from_xy(x):

    c2 = _fast_cross(x[...,0], x[...,1])
    c2 = c2 / np.sqrt(np.sum(np.square(c2), axis=-1))[...,np.newaxis]
    c1 = _fast_cross(c2, x[...,0])
    c1 = c1 / np.sqrt(np.sum(np.square(c1), axis=-1))[...,np.newaxis]
    c0 = x[...,0]
    
    return np.concatenate([
        c0[...,np.newaxis], 
        c1[...,np.newaxis], 
        c2[...,np.newaxis]], axis=-1)

def fk_vel(lrot, lpos, lvel, lang, parents):
    
    gp, gr, gv, ga = [lpos[...,:1,:]], [lrot[...,:1,:,:]], [lvel[...,:1,:]], [lang[...,:1,:]]
    for i in range(1, len(parents)):
        gp.append(mul_vec(gr[parents[i]], lpos[...,i:i+1,:]) + gp[parents[i]])
        gr.append(mul    (gr[parents[i]], lrot[...,i:i+1,:,:]))
        gv.append(mul_vec(gr[parents[i]], lvel[...,i:i+1,:]) + 
            np.cross(ga[parents[i]], mul_vec(gr[parents[i]], lpos[...,i:i+1,:]), axis=-1) +
            gv[parents[i]])
        ga.append(mul_vec(gr[parents[i]], lang[...,i:i+1,:]) + ga[parents[i]])
        
    return (
        np.concatenate(gr, axis=-3), 
        np.concatenate(gp, axis=-2),
        np.concatenate(gv, axis=-2),
        np.concatenate(ga, axis=-2))
