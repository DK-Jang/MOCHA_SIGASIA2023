import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
import matplotlib.patheffects as pe
sys.path.append('../motion')
import quat

joint_foot_indicies = [5, 24]  # left toe, right toe, mocha

def animation_plot(animations, fps=60):
    root_directions = [None] * len(animations)
    for ai in range(len(animations)):
        anim = animations[ai].copy()
        pos, rot, parents = anim[0], anim[1], anim[2]
        Grot, Gpos = quat.fk(rot, pos, parents)

        root_dir = quat.mul_vec(Grot[:, 0:1], np.array([0.0, 0.0, 1.0]))
        root_dir =  Gpos[:, 0:1] + root_dir*0.5

        animations[ai] = Gpos * 0.1
        root_directions[ai] = root_dir * 0.1
        
    scale = 1.25*((len(animations))/2)
    
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, projection='3d')
    rscale = scale * 30
    ax.set_xlim3d(-rscale, rscale)
    ax.set_zlim3d(0, rscale*2)
    ax.set_ylim3d(-rscale, rscale)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(20, -60) # (-40, 60): up view

    # checkerboard pane
    facec = (254, 254, 254)
    linec = (240, 240, 240)
    facec = list(np.array(facec) / 256.0) + [1.0]
    linec = list(np.array(linec) / 256.0) + [1.0]

    ax.w_zaxis.set_pane_color(facec)
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    X = np.arange(-rscale, rscale, 10)
    Y = np.arange(-rscale, rscale, 10)
    xlen = len(X)
    ylen = len(Y)
    X, Y = np.meshgrid(X, Y)
    Z = np.zeros(X.shape) # place it at a lower surface

    colortuple = (facec, linec)
    colors_pane = np.zeros((Z.shape + (4, )))
    for y in range(ylen):
        for x in range(xlen):
            colors_pane[y, x] = colortuple[(x + y) % len(colortuple)]

    # Plot the surface with face colors_pane taken from the array we made.
    surf = ax.plot_surface(X, Y, Z, facecolors=colors_pane, linewidth=0., zorder=-1, shade=False)

    ax.w_zaxis.line.set_lw(0.)
    ax.w_yaxis.line.set_lw(0.)
    ax.w_yaxis.line.set_color(linec)
    ax.w_xaxis.line.set_lw(0.)
    ax.w_xaxis.line.set_color(linec)


    acolors = list(sorted(colors.cnames.keys()))[::-1]
    acolors.pop(3)
    lines = []
    root_lines = []
    for ai, anim in enumerate(animations):
        lines.append([plt.plot([0,0], [0,0], [0,0], color=acolors[ai], zorder=3,
            lw=2, path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()]
            )[0] for _ in range(anim.shape[1])])
        root_lines.append(plt.plot([0,0], [0,0], [0,0], color='red', zorder=3,
            lw=2, path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()])[0])
    
    def animate(i):
        changed = []
        for ai in range(len(animations)):
            offset = 30*(ai-((len(animations))/2))
            for j in range(len(parents)):
                if parents[j] != -1:
                    lines[ai][j].set_data(
                        [ animations[ai][i,j,0]+offset, animations[ai][i,parents[j],0]+offset],
                        [-animations[ai][i,j,2],       -animations[ai][i,parents[j],2]])
                    lines[ai][j].set_3d_properties(
                        [ animations[ai][i,j,1],        animations[ai][i,parents[j],1]])
            changed += lines

            # root 
            root_lines[ai].set_data(
                        [ root_directions[ai][i,0,0]+offset, animations[ai][i,0,0]+offset],
                        [-root_directions[ai][i,0,2],       -animations[ai][i,0,2]])
            root_lines[ai].set_3d_properties(
                        [ root_directions[ai][i,0,1],        animations[ai][i,0,1]])
            changed += root_lines
            
        return changed
        
    plt.tight_layout()
        
    ani = animation.FuncAnimation(fig, animate, np.arange(len(animations[0])), interval=1000/fps)
        
    plt.show()


if __name__ == '__main__':
    import sys
    sys.path.append('../preprocess')
    sys.path.append('../etc')
    sys.path.append('../motion')
    from generate_database import process_data
    from utils import get_config
    import quat
    import bvh

    cfg = get_config('../configs/dataset.yaml')

    bvh_file = '../bvh_all/Loco_Walk_side_Neutral_Zombie_003.bvh'
    bvh_data = bvh.load(bvh_file)
    features, parents, names = \
                    process_data(bvh_data, divide=False, mirror=False)
    pos, vel, rot, ang, contact =  \
        features[0], features[1], features[2], features[3], features[4]
    motion1 = [pos, rot, vel, ang, contact, parents]

    bvh_file = '../bvh_all/Loco_Walk_side_Neutral_Zombie_003.bvh'
    bvh_data = bvh.load(bvh_file)
    features, parents, names = \
                    process_data(bvh_data, divide=False, mirror=True)
    pos, vel, rot, ang, contact =  \
        features[0], features[1], features[2], features[3], features[4]
    motion2 = [pos, rot, vel, ang, contact, parents]

    animation_plot([motion1, motion2])
    