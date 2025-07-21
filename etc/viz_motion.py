import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
import matplotlib.patheffects as pe
sys.path.append('../motion')
import quat

foot_indicies = [5, 24]  # left foot toe, right foot toe
shoulder_indicies = [10, 17]  # left shoulder, right shoulder
upleg_indicies = [2, 21]  # left upleg, right upleg

def animation_plot(animations, fps=60):
    root_directions = [None] * len(animations)
    foot_contacts = [None] * len(animations)
    for ai in range(len(animations)):
        anim = animations[ai].copy()

        if len(anim) == 7:
            pos, rot, vel, ang, contact, foot_indicies, parents = \
                anim[0], anim[1], anim[2], anim[3], anim[4], anim[5], anim[6]
            Grot, Gpos, Gvel, Gang = quat.fk_vel(rot, pos, vel, ang, parents)
        else:
            pos, rot, contact, foot_indicies, parents = \
                anim[0], anim[1], anim[2], anim[3], anim[4]
            Grot, Gpos = quat.fk(rot, pos, parents)

        root_dir = quat.mul_vec(Grot[:, 0:1], np.array([0.0, 0.0, 1.0]))
        root_dir_pos =  Gpos[:, 0:1] + root_dir*0.5

        animations[ai] = Gpos * 30
        root_directions[ai] = root_dir_pos * 30
        foot_contacts[ai] = contact
        
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

    # white_color = (1.0, 1.0, 1.0, 0.0)
    # ax.xaxis.set_pane_color(white_color)
    # ax.yaxis.set_pane_color(white_color)
    # # ax.zaxis.set_pane_color(white_color)

    # ax.xaxis.line.set_color(white_color)
    # ax.yaxis.line.set_color(white_color)
    # ax.zaxis.line.set_color(white_color)
    # # plt.xlabel('X')
    # # plt.ylabel('Y')
    # # ax.axis('off')

    # checkerboard pane
    facec = (254, 254, 254)
    linec = (240, 240, 240)
    facec = list(np.array(facec) / 256.0) + [1.0]
    linec = list(np.array(linec) / 256.0) + [1.0]

    ax.zaxis.set_pane_color(facec)
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

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

    ax.zaxis.line.set_lw(0.)
    ax.yaxis.line.set_lw(0.)
    ax.yaxis.line.set_color(linec)
    ax.xaxis.line.set_lw(0.)
    ax.xaxis.line.set_color(linec)

    acolors = list(sorted(colors.cnames.keys()))[::-1]
    acolors.pop(3)
    lines = []
    root_lines = []
    contact_dots = []
    for ai, anim in enumerate(animations):
        lines.append([plt.plot([0,0], [0,0], [0,0], color=acolors[ai], zorder=3,
            lw=2, path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()]
            )[0] for _ in range(anim.shape[1])])
        root_lines.append(plt.plot([0,0], [0,0], [0,0], color='red', zorder=3,
            lw=2, path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()])[0])
        contact_dots.append([plt.plot([0,0], [0,0], [0,0], color='white', zorder=3,
                    linewidth=2, linestyle='',
                    marker="o", markersize=1.5 * scale,
                    path_effects=[pe.Stroke(linewidth=2, foreground='black'), pe.Normal()]
                    )[0] for _ in range(contact.shape[1])])
    
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

            # foot contact
            for j, f_idx in enumerate(foot_indicies):
                contact_dots[ai][j].set_data([animations[ai][i,f_idx,0]+offset], [-animations[ai][i,f_idx,2]])      # left toe
                contact_dots[ai][j].set_3d_properties([animations[ai][i,f_idx,1]])
                color = 'red' if foot_contacts[ai][i, j] > 0.5 else 'blue'
                contact_dots[ai][j].set_color(color)
            
        return changed
        
    plt.tight_layout()
        
    ani = animation.FuncAnimation(fig, animate, np.arange(len(animations[0])), interval=1000/fps)
        
    plt.show()


if __name__ == '__main__':
    import sys
    sys.path.append('../etc')
    sys.path.append('../motion')
    sys.path.append('../preprocess')
    from generate_database import process_data
    import quat
    import bvh

    bvh_file = '../bvh/Loco_Run_Neutral_AverageJoe_001.bvh'
    bvh_data = bvh.load(bvh_file)
    features, parents, names = \
                    process_data(bvh_data, divide=False, mirror=False)
    pos, vel, rot, ang, contact =  \
        features[0], features[1], features[2], features[3], features[4]
    motion1 = [pos[0], rot[0], vel[0], ang[0], contact[0], parents]

    bvh_file = '../bvh/Loco_Run_Neutral_AverageJoe_001.bvh'
    bvh_data = bvh.load(bvh_file)
    features, parents, names = \
                    process_data(bvh_data, divide=False, mirror=True)
    pos, vel, rot, ang, contact =  \
        features[0], features[1], features[2], features[3], features[4]
    motion2 = [pos[0], rot[0], vel[0], ang[0], contact[0], parents]

    animation_plot([motion1, motion2])
    