import os
import sys
import argparse
import torch
import torch.nn.functional as F
from sklearn.neighbors import BallTree
import numpy as np
sys.path.append('./etc')
sys.path.append('./motion')
sys.path.append('./preprocess')
sys.path.append('./net')
from utils import ensure_dirs, get_config
import quat
import bvh
import Inertialization as inert
from utils import set_seed
from viz_motion import animation_plot
from generate_database import process_data
from trainer import Trainer
from model_CVAE import CVAE
from transformer import mean_variance_norm

torch.set_grad_enabled(False);
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def initialize_path(config):
    config['main_dir'] = os.path.join('.', config['name'])
    config['model_dir'] = os.path.join(config['main_dir'], "pth")
    ensure_dirs([config['main_dir'], config['model_dir']])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', 
                        type=str, 
                        default='model_ours/info/config.yaml',
                        help='Path to the config file.')
    args = parser.parse_args()

    # initialize path
    cfg = get_config(args.config)
    initialize_path(cfg)

    set_seed(1777)

    # load model
    trainer = Trainer(cfg)
    epochs = trainer.load_checkpoint('model_ours/pth/gen_125.pt')
    model = trainer.gen_ema.eval()

    cvae_save_dir = 'Neutral_AverageJoe2Neutral_Princess'
    network_cvae = CVAE(output_seq=90,
                        latent_dim=256, depth=2, nheads=4,
                        feedforward_dim=512, dropout=0.1, 
                        activation=F.relu).to(device)
    network_cvae.load_state_dict(torch.load(
        os.path.join(cvae_save_dir, 'cvae_020000.pt'), map_location=device))
    network_cvae.eval()

    dataset_dir = cfg['data_dir']
    dataset_cfg = get_config('./configs/dataset.yaml')
    style_names = dataset_cfg["mocha_style_names"]

    # load generator norm
    ndata_npz_path = os.path.join(dataset_dir, 'norm.npz')
    norm = np.load(ndata_npz_path, allow_pickle=True)
    norm_np = {}
    for key, value in norm.items():
        norm_np[key] = value[np.newaxis, np.newaxis] 
    X_mean, X_std = norm_np['X_mean'], norm_np['X_std']
    Y_mean, Y_std = norm_np['Y_mean'], norm_np['Y_std']

    # import all dataset cnt norm
    cnt_norm_path = os.path.join(dataset_dir, 'cnt_norm.npz')
    cnt_norm = np.load(cnt_norm_path, allow_pickle=True)
    cnt_mean, cnt_std = cnt_norm['mean'], cnt_norm['std']

    # import norm dataset
    cvae_norm_path = os.path.join(cvae_save_dir, 'cvae_norm.npz')
    cvae_norm = np.load(cvae_norm_path, allow_pickle=True)
    temp_weight = cvae_norm['std_weight']
    src_cnt_mean = cvae_norm['src_cnt_mean']
    src_cnt_std = cvae_norm['src_cnt_std']
    cha_cnt_mean = cvae_norm['cha_cnt_mean']
    cha_cnt_std = cvae_norm['cha_cnt_std']
    cha_encoded_mean = cvae_norm['cha_encoded_mean']
    cha_encoded_std = cvae_norm['cha_encoded_std']

    cnt_std /= temp_weight
    src_cnt_std /= temp_weight
    cha_cnt_std /= temp_weight
    cha_encoded_std /= temp_weight

    src_cnt_mean = torch.from_numpy(src_cnt_mean).to(device)
    src_cnt_std = torch.from_numpy(src_cnt_std).to(device)
    cha_cnt_mean = torch.from_numpy(cha_cnt_mean).to(device)
    cha_cnt_std = torch.from_numpy(cha_cnt_std).to(device)
    cha_encoded_mean = torch.from_numpy(cha_encoded_mean).to(device)
    cha_encoded_std = torch.from_numpy(cha_encoded_std).to(device)

    parents_original = np.array(cfg['dataset']['mocha']['parents'])
    parents = np.concatenate([[-1], parents_original + 1])
    nbones = len(parents)
    contact_bones = np.array([5, 24])
    dt = 1.0 / 60.0
    window = 60

    # IK
    ik_enabled = True
    ik_max_length_buffer = 0.015
    ik_foot_height = 0.02
    ik_toe_length = 0.15
    ik_unlock_radius = 0.2
    ik_blending_halflife = 0.1

    src_bvh_file = 'bvh/Loco_Walk_Neutral_AverageJoe_001.bvh'
    cha_bvh_file = 'bvh/Loco_Walk_Neutral_Princess_002.bvh'
    # src_bvh_file = 'bvh/Jump_Neutral_AverageJoe_001.bvh'
    # cha_bvh_file = 'bvh/Jump_Neutral_Princess_001.bvh'
    # src_bvh_file = 'bvh/Stats_Sit_Neutral_AverageJoe_001.bvh'
    # cha_bvh_file = 'bvh/Stats_Sit_Neutral_Princess_002.bvh'
    
    # load src bvh
    src_bvh_data = bvh.load(src_bvh_file)

    pos, vel, rot, ang, contacts = [], [], [], [], []
    feature_clips, _, _ = \
        process_data(src_bvh_data, window=60, window_step=1, divide=True, mirror=False)
    pos += feature_clips[0]
    vel += feature_clips[1]
    rot += feature_clips[2]
    ang += feature_clips[3]
    contacts += feature_clips[4]

    Ypos = np.array(pos, dtype=np.float32)
    Yvel = np.array(vel, dtype=np.float32)
    Yrot = np.array(rot, dtype=np.float32)
    Yang = np.array(ang, dtype=np.float32)
    Yextra = np.array(contacts, dtype=np.uint8)

    # Compute local root velocity (body coordinate)
    Yrvel = quat.inv_mul_vec(Yrot[:,:,0], Yvel[:,:,0])
    Yrang = quat.inv_mul_vec(Yrot[:,:,0], Yang[:,:,0])

    # Compute world space
    Grot, Gpos, Gvel, Gang = quat.fk_vel(Yrot, Ypos, Yvel, Yang, parents)

    Gpos[:,:,0:1] = np.repeat(Gpos[:, -1:, 0:1], window, axis=1)
    Grot[:,:,0:1] = np.repeat(Grot[:, -1:, 0:1], window, axis=1)
    Gvel[:,:,0:1] = np.repeat(Gvel[:, -1:, 0:1], window, axis=1)
    Gang[:,:,0:1] = np.repeat(Gang[:, -1:, 0:1], window, axis=1)

    # Compute x local to character
    Xpos = quat.inv_mul_vec(Grot[:,:,0:1], Gpos - Gpos[:,:,0:1])
    Xrot = quat.inv_mul(Grot[:,:,0:1], Grot)
    Xtxy = quat.to_xform_xy(Xrot).astype(np.float32)
    Xvel = quat.inv_mul_vec(Grot[:,:,0:1], Gvel)
    Xang = quat.inv_mul_vec(Grot[:,:,0:1], Gang)

    Yrot, Ypos = quat.ik(Xrot, Xpos, parents)
    Ytxy = quat.to_xform_xy(Yrot).astype(np.float32)

    # Compute velocities via central difference
    Yvel = np.empty_like(Ypos)
    Yvel[:,1:-1] = (
        0.5 * (Ypos[:,2:  ] - Ypos[:,1:-1]) * 60.0 +
        0.5 * (Ypos[:,1:-1] - Ypos[:, :-2]) * 60.0)
    Yvel[:, 0] = Yvel[:, 1] - (Yvel[:, 3] - Yvel[:, 2])
    Yvel[:,-1] = Yvel[:,-2] + (Yvel[:,-2] - Yvel[:,-3])    
        
    # Same for angular velocities
    Yang = np.zeros_like(Ypos)
    Yang[:,1:-1] = (
        0.5 * quat.to_scaled_angle_axis(quat.abs(quat.mul_inv(Yrot[:,2:  ], Yrot[:,1:-1]))) * 60.0 +
        0.5 * quat.to_scaled_angle_axis(quat.abs(quat.mul_inv(Yrot[:,1:-1], Yrot[:, :-2]))) * 60.0)
    Yang[:, 0] = Yang[:, 1] - (Yang[:, 3] - Yang[:, 2])
    Yang[:,-1] = Yang[:,-2] + (Yang[:,-2] - Yang[:,-3])

    b, ns, nj, _, _ = Xtxy.shape
    X = np.concatenate([
                Xpos,
                Xtxy.reshape(b, ns, nj, -1),
                Xvel,
                Xang,
            ], axis=-1)
    X = (X[:,:,1:] - X_mean[:,:,1:]) / X_std[:,:,1:]

    with torch.no_grad():
        X = torch.from_numpy(X).to(device)
        tokens = model.mot_embedding(X)
        tokens = tokens + model.pos_emb[:, :tokens.shape[1]]
        encoded = model.encoder(tokens)
        cnt = mean_variance_norm(encoded.permute(0, 2, 1))
        cnt = cnt.permute(0, 2, 1).cpu().numpy()
    src_encoded = encoded.clone()
    src_cnt = cnt.copy()
    src_Ypos = Ypos.copy()
    src_Yrot = Yrot.copy()
    src_Yvel = Yvel.copy()
    src_Yang = Yang.copy()
    src_Yrvel = Yrvel.copy()
    src_Yrang = Yrang.copy()
    src_contact = Yextra.copy()

    # load cha bvh
    cha_bvh_data = bvh.load(cha_bvh_file)

    pos, vel, rot, ang, contacts = [], [], [], [], []
    feature_clips, _, _ = \
        process_data(cha_bvh_data, window=60, window_step=1, divide=True, mirror=False)
    pos += feature_clips[0]
    vel += feature_clips[1]
    rot += feature_clips[2]
    ang += feature_clips[3]
    contacts += feature_clips[4]

    Ypos = np.array(pos, dtype=np.float32)
    Yvel = np.array(vel, dtype=np.float32)
    Yrot = np.array(rot, dtype=np.float32)
    Yang = np.array(ang, dtype=np.float32)
    Yextra = np.array(contacts, dtype=np.uint8)

    # Compute local root velocity (body coordinate)
    Yrvel = quat.inv_mul_vec(Yrot[:,:,0], Yvel[:,:,0])
    Yrang = quat.inv_mul_vec(Yrot[:,:,0], Yang[:,:,0])

    # Compute world space
    Grot, Gpos, Gvel, Gang = quat.fk_vel(Yrot, Ypos, Yvel, Yang, parents)

    # Compute X local to current root
    Gpos[:,:,0:1] = np.repeat(Gpos[:, -1:, 0:1], window, axis=1)
    Grot[:,:,0:1] = np.repeat(Grot[:, -1:, 0:1], window, axis=1)
    Gvel[:,:,0:1] = np.repeat(Gvel[:, -1:, 0:1], window, axis=1)
    Gang[:,:,0:1] = np.repeat(Gang[:, -1:, 0:1], window, axis=1)

    # Compute x local to character
    Xpos = quat.inv_mul_vec(Grot[:,:,0:1], Gpos - Gpos[:,:,0:1])
    Xrot = quat.inv_mul(Grot[:,:,0:1], Grot)
    Xtxy = quat.to_xform_xy(Xrot).astype(np.float32)
    Xvel = quat.inv_mul_vec(Grot[:,:,0:1], Gvel)
    Xang = quat.inv_mul_vec(Grot[:,:,0:1], Gang)

    Yrot, Ypos = quat.ik(Xrot, Xpos, parents)
    Ytxy = quat.to_xform_xy(Yrot).astype(np.float32)

    # Compute velocities via central difference
    Yvel = np.empty_like(Ypos)
    Yvel[:,1:-1] = (
        0.5 * (Ypos[:,2:  ] - Ypos[:,1:-1]) * 60.0 +
        0.5 * (Ypos[:,1:-1] - Ypos[:, :-2]) * 60.0)
    Yvel[:, 0] = Yvel[:, 1] - (Yvel[:, 3] - Yvel[:, 2])
    Yvel[:,-1] = Yvel[:,-2] + (Yvel[:,-2] - Yvel[:,-3])    
        
    # Same for angular velocities
    Yang = np.zeros_like(Ypos)
    Yang[:,1:-1] = (
        0.5 * quat.to_scaled_angle_axis(quat.abs(quat.mul_inv(Yrot[:,2:  ], Yrot[:,1:-1]))) * 60.0 +
        0.5 * quat.to_scaled_angle_axis(quat.abs(quat.mul_inv(Yrot[:,1:-1], Yrot[:, :-2]))) * 60.0)
    Yang[:, 0] = Yang[:, 1] - (Yang[:, 3] - Yang[:, 2])
    Yang[:,-1] = Yang[:,-2] + (Yang[:,-2] - Yang[:,-3])

    b, ns, nj, _, _ = Xtxy.shape
    X = np.concatenate([
                Xpos,
                Xtxy.reshape(b, ns, nj, -1),
                Xvel,
                Xang,
            ], axis=-1)
    X = (X[:,:,1:] - X_mean[:,:,1:]) / X_std[:,:,1:]

    with torch.no_grad():
        X = torch.from_numpy(X).to(device)
        tokens = model.mot_embedding(X)
        tokens = tokens + model.pos_emb[:, :tokens.shape[1]]
        encoded = model.encoder(tokens)
        cnt = mean_variance_norm(encoded.permute(0, 2, 1))
        cnt = cnt.permute(0, 2, 1).cpu().numpy()
    cha_encoded = encoded.clone()
    cha_cnt = cnt.copy()
    cha_Ypos = Ypos.copy()
    cha_Yrot = Yrot.copy()
    cha_Yvel = Yvel.copy()
    cha_Yang = Yang.copy()
    cha_Yrvel = Yrvel.copy()
    cha_Yrang = Yrang.copy()
    cha_contact = Yextra.copy()

    with torch.no_grad():
        # initialize, first frame
        curr_src_encoded = src_encoded[0:1]
        curr_src_cnt = src_cnt[0:1]

        cha_cnt_nm = (cha_cnt - cnt_mean[np.newaxis]) / cnt_std[np.newaxis]
        tree_cnt = BallTree(cha_cnt_nm.reshape(cha_cnt_nm.shape[0], -1))
        curr_src_cnt_nm = (curr_src_cnt - cnt_mean[np.newaxis]) / cnt_std[np.newaxis]
        frame_index = tree_cnt.query(curr_src_cnt_nm.reshape(curr_src_cnt_nm.shape[0], -1), k=1, return_distance=False)[:,0][0]

        curr_cha_encoded = cha_encoded[frame_index:frame_index+1]

        # Decode
        trans_decoded_tokens = model.decoder(curr_src_encoded, curr_cha_encoded)
        trans_Ytil = model.to_mot(trans_decoded_tokens)
        trans_Ytil = trans_Ytil[0].cpu().numpy() * Y_std[0,:,1:]  + Y_mean[0,:,1:]
        trans_Ypos = trans_Ytil[-1, :, :3]
        trans_Ytxy = trans_Ytil[-1, :, 3:9].reshape(trans_Ypos.shape[0], 3, 2)
        trans_Yvel = trans_Ytil[..., 9:12]      # for control the velocity
        trans_Yang = trans_Ytil[-1, :, 12:15]     # for control the velocity
        trans_Yrot = quat.from_xform_xy(trans_Ytxy)

        # Decode
        cm_trans_decoded_tokens = model.decoder(curr_src_encoded, curr_cha_encoded)
        cm_trans_Ytil = model.to_mot(cm_trans_decoded_tokens)
        cm_trans_Ytil = cm_trans_Ytil[0].cpu().numpy() * Y_std[0,:,1:]  + Y_mean[0,:,1:]
        cm_trans_Ypos = cm_trans_Ytil[-1, :, :3]
        cm_trans_Ytxy = cm_trans_Ytil[-1, :, 3:9].reshape(cm_trans_Ypos.shape[0], 3, 2)
        cm_trans_Yvel = cm_trans_Ytil[..., 9:12]      # for control the velocity
        cm_trans_Yang = cm_trans_Ytil[-1, :, 12:15]     # for control the velocity
        cm_trans_Yrot = quat.from_xform_xy(cm_trans_Ytxy)
        
        # initialize the list
        src_rootvel = quat.mul_vec(np.array([1, 0, 0, 0]), src_Yrvel[0,-1])
        src_rootang = quat.mul_vec(np.array([1, 0, 0, 0]), src_Yrang[0,-1])
        src_rootpos = np.array([0, 0, 0]) + src_rootvel * dt
        src_rootrot = quat.mul(np.array([1, 0, 0, 0]), quat.from_scaled_angle_axis(src_rootang * dt))
        src_Ypos[0,-1,0] = src_rootpos
        src_Yvel[0,-1,0] = src_rootvel
        src_Yrot[0,-1,0] = src_rootrot
        src_Yang[0,-1,0] = src_rootang

        src_Ypos_list = [src_Ypos[0,-1]]
        src_Yvel_list = [src_Yvel[0,-1]]
        src_Yrot_list = [src_Yrot[0,-1]]
        src_Yang_list = [src_Yang[0,-1]]
        src_contact_list = [src_contact[0,-1]]
    
        # Integrate root displacement
        trans_Yrvel_ratio = (np.linalg.norm(trans_Yvel[:,0], axis=1).mean() \
                           / np.linalg.norm(src_Yvel[0,:,1], axis=1).mean())
        if trans_Yrvel_ratio > 3.0 or trans_Yrvel_ratio < 0.33:
            trans_Yrvel_ratio = 1.0
        trans_Yrvel = src_Yrvel[0,-1] * trans_Yrvel_ratio
        # trans_Yrvel = src_Yrvel[0,-1]
        trans_Yrang = src_Yrang[0,-1]

        trans_rootvel = quat.mul_vec(np.array([1, 0, 0, 0]), trans_Yrvel)
        trans_rootang = quat.mul_vec(np.array([1, 0, 0, 0]), trans_Yrang)
        trans_rootpos = np.array([0, 0, 0]) + trans_rootvel * dt
        trans_rootrot = quat.mul(np.array([1, 0, 0, 0]), quat.from_scaled_angle_axis(trans_rootang * dt))
        
        trans_Ypos = np.concatenate([trans_rootpos[np.newaxis], trans_Ypos], axis=0)
        trans_Yvel = np.concatenate([trans_rootvel[np.newaxis], trans_Yvel[-1]], axis=0)
        trans_Yrot = np.concatenate([trans_rootrot[np.newaxis], trans_Yrot], axis=0)
        trans_Yang = np.concatenate([trans_rootang[np.newaxis], trans_Yang], axis=0)
        trans_contact = src_contact[0,-1]

        # Integrate root displacement
        cm_trans_Yrvel_ratio = (np.linalg.norm(cm_trans_Yvel[:,0], axis=1).mean() \
                           / np.linalg.norm(src_Yvel[0,:,1], axis=1).mean())
        if cm_trans_Yrvel_ratio > 3.0 or cm_trans_Yrvel_ratio < 0.33:
            cm_trans_Yrvel_ratio = 1.0
        cm_trans_Yrvel = src_Yrvel[0,-1] * cm_trans_Yrvel_ratio
        cm_trans_Yrang = src_Yrang[0,-1]

        cm_trans_rootvel = quat.mul_vec(np.array([1, 0, 0, 0]), cm_trans_Yrvel)
        cm_trans_rootang = quat.mul_vec(np.array([1, 0, 0, 0]), cm_trans_Yrang)
        cm_trans_rootpos = np.array([0, 0, 0]) + cm_trans_rootvel * dt
        cm_trans_rootrot = quat.mul(np.array([1, 0, 0, 0]), quat.from_scaled_angle_axis(cm_trans_rootang * dt))
        
        cm_trans_Ypos = np.concatenate([cm_trans_rootpos[np.newaxis], cm_trans_Ypos], axis=0)
        cm_trans_Yvel = np.concatenate([cm_trans_rootvel[np.newaxis], cm_trans_Yvel[-1]], axis=0)
        cm_trans_Yrot = np.concatenate([cm_trans_rootrot[np.newaxis], cm_trans_Yrot], axis=0)
        cm_trans_Yang = np.concatenate([cm_trans_rootang[np.newaxis], cm_trans_Yang], axis=0)
        cm_trans_contact = src_contact[0,-1]

        trans_Ypos_list = [trans_Ypos]
        trans_Yvel_list = [trans_Yvel]
        trans_Yrot_list = [trans_Yrot]
        trans_Yang_list = [trans_Yang]
        trans_contact_list = [trans_contact]

        ik_trans_Ypos_list = [trans_Ypos]
        ik_trans_Yrot_list = [trans_Yrot]
        ik_trans_contact_list = [trans_contact]

        cm_trans_Ypos_list = [cm_trans_Ypos]
        cm_trans_Yvel_list = [cm_trans_Yvel]
        cm_trans_Yrot_list = [cm_trans_Yrot]
        cm_trans_Yang_list = [cm_trans_Yang]
        cm_trans_contact_list = [cm_trans_contact]

        global_bone_positions = np.zeros((len(parents), 3))
        global_bone_velocities = np.zeros((len(parents), 3))
        global_bone_rotations = np.zeros((len(parents), 4))
        global_bone_angular_velocities = np.zeros((len(parents), 3))
        global_bone_computed = np.zeros(len(parents), dtype=bool)

        bone_positions = trans_Ypos.copy()
        bone_velocities = trans_Yvel.copy()
        bone_rotations = trans_Yrot.copy()
        bone_angular_velocities = trans_Yang.copy()

        # initialize contact states
        contact_states = np.zeros(contact_bones.size, dtype=bool)
        contact_locks = np.zeros(contact_bones.size, dtype=bool)
        contact_positions = np.zeros((contact_bones.size, 3))
        contact_velocities = np.zeros((contact_bones.size, 3))
        contact_points = np.zeros((contact_bones.size, 3))
        contact_targets = np.zeros((contact_bones.size, 3))
        contact_offset_positions = np.zeros((contact_bones.size, 3))
        contact_offset_velocities = np.zeros((contact_bones.size, 3))

        for bs in range(contact_bones.size):
            bone_position, bone_velocity, bone_rotation, bone_angular_velocity = \
                quat.fk_vel_bone(
                    bone_positions,
                    bone_velocities,
                    bone_rotations,
                    bone_angular_velocities,
                    parents,
                    contact_bones[bs]
                )
            
            # contact reset
            contact_states[bs] = False
            contact_locks[bs] = False
            contact_positions[bs] = bone_position
            contact_velocities[bs] = bone_velocity
            contact_points[bs] = bone_position
            contact_targets[bs] = bone_position
            contact_offset_positions[bs] = np.zeros(3)
            contact_offset_velocities[bs] = np.zeros(3)

        adjusted_bone_positions = trans_Ypos.copy()
        adjusted_bone_rotations = trans_Yrot.copy()
        
        # Go
        prev_cha_encoded = curr_cha_encoded.clone()
        for i in range(1, len(src_encoded)):
            # use nearest neighbor to find the corresponding frame
            curr_src_encoded = src_encoded[i:i+1]
            curr_src_cnt = src_cnt[i:i+1]
            curr_src_cnt_nm = (curr_src_cnt - cnt_mean[np.newaxis]) / cnt_std[np.newaxis]
            frame_index = tree_cnt.query(curr_src_cnt_nm.reshape(curr_src_cnt_nm.shape[0], -1), k=1, return_distance=False)[:,0][0]

            # use cvae
            condition = torch.cat([(torch.as_tensor(curr_src_cnt).to(device) - src_cnt_mean.unsqueeze(0)) / src_cnt_std.unsqueeze(0), 
                        (prev_cha_encoded - cha_encoded_mean.unsqueeze(0)) / cha_encoded_std.unsqueeze(0)], dim=1)
            vae_output = network_cvae.sample(condition, deterministic=False)
            curr_cha_encoded = vae_output *  cha_encoded_std.unsqueeze(0) + cha_encoded_mean.unsqueeze(0)
            
            # update prev_cha_encoded for next condition
            prev_cha_encoded = curr_cha_encoded.clone()

            # compute current trans pose using CVAE
            trans_decoded_tokens = model.decoder(curr_src_encoded, curr_cha_encoded)
            trans_Ytil = model.to_mot(trans_decoded_tokens)
            trans_Ytil = trans_Ytil[0].cpu().numpy() * Y_std[0,:,1:]  + Y_mean[0,:,1:]
            trans_Ypos = trans_Ytil[-1, :, :3]
            trans_Ytxy = trans_Ytil[-1, :, 3:9].reshape(trans_Ypos.shape[0], 3, 2)
            trans_Yvel = trans_Ytil[..., 9:12]      # for control the velocity
            trans_Yang = trans_Ytil[-1, :, 12:15]     # for control the velocity
            trans_Yrot = quat.from_xform_xy(trans_Ytxy)

            # compute current trans pose using NN
            cm_trans_decoded_tokens = model.decoder(curr_src_encoded, cha_encoded[frame_index:frame_index+1])
            cm_trans_Ytil = model.to_mot(cm_trans_decoded_tokens)
            cm_trans_Ytil = cm_trans_Ytil[0].cpu().numpy() * Y_std[0,:,1:]  + Y_mean[0,:,1:]
            cm_trans_Ypos = cm_trans_Ytil[-1, :, :3]
            cm_trans_Ytxy = cm_trans_Ytil[-1, :, 3:9].reshape(cm_trans_Ypos.shape[0], 3, 2)
            cm_trans_Yvel = cm_trans_Ytil[..., 9:12]      # for control the velocity
            cm_trans_Yang = cm_trans_Ytil[-1, :, 12:15]     # for control the velocity
            cm_trans_Yrot = quat.from_xform_xy(cm_trans_Ytxy)

            # src pose
            # Extract root velocities and put in world space
            src_rootvel = quat.mul_vec(src_Yrot_list[-1][0], src_Yrvel[i,-1])
            src_rootang = quat.mul_vec(src_Yrot_list[-1][0], src_Yrang[i,-1])
            src_rootpos = src_Ypos_list[-1][0] + src_rootvel * dt
            src_rootrot = quat.mul(src_Yrot_list[-1][0], quat.from_scaled_angle_axis(src_rootang * dt))
            src_Ypos[i,-1,0] = src_rootpos
            src_Yvel[i,-1,0] = src_rootvel
            src_Yrot[i,-1,0] = src_rootrot
            src_Yang[i,-1,0] = src_rootang

            src_Ypos_list += [src_Ypos[i,-1]]
            src_Yvel_list += [src_Yvel[i,-1]]
            src_Yrot_list += [src_Yrot[i,-1]]
            src_Yang_list += [src_Yang[i,-1]]
            src_contact_list += [src_contact[i,-1]]

            # tans pose
            trans_Yrvel_ratio = (np.linalg.norm(trans_Yvel[:,0], axis=1).mean() \
                               / np.linalg.norm(src_Yvel[i,:,1], axis=1).mean())
            if trans_Yrvel_ratio > 3.0 or trans_Yrvel_ratio < 0.33:
                trans_Yrvel_ratio = 1.0
            trans_Yrvel = src_Yrvel[i,-1] * trans_Yrvel_ratio
            # trans_Yrvel = src_Yrvel[i,-1]
            trans_Yrang = src_Yrang[i,-1]

            trans_rootvel = quat.mul_vec(trans_Yrot_list[-1][0], trans_Yrvel)
            trans_rootang = quat.mul_vec(trans_Yrot_list[-1][0], trans_Yrang)
            trans_rootpos = trans_Ypos_list[-1][0] + trans_rootvel * dt
            trans_rootrot = quat.mul(trans_Yrot_list[-1][0], quat.from_scaled_angle_axis(trans_rootang * dt))
            
            trans_Ypos = np.concatenate([trans_rootpos[np.newaxis], trans_Ypos], axis=0)
            trans_Yvel = np.concatenate([trans_rootvel[np.newaxis], trans_Yvel[-1]], axis=0)
            trans_Yrot = np.concatenate([trans_rootrot[np.newaxis], trans_Yrot], axis=0)
            trans_Yang = np.concatenate([trans_rootang[np.newaxis], trans_Yang], axis=0)
            trans_contact = src_contact[i,-1]

            # context matching pose
            cm_trans_Yrvel_ratio = (np.linalg.norm(cm_trans_Yvel[:,0], axis=1).mean() \
                                  / np.linalg.norm(src_Yvel[i,:,1], axis=1).mean())
            if cm_trans_Yrvel_ratio > 3.0 or cm_trans_Yrvel_ratio < 0.33:
                cm_trans_Yrvel_ratio = 1.0
            cm_trans_Yrvel = src_Yrvel[i,-1] * cm_trans_Yrvel_ratio
            cm_trans_Yrang = src_Yrang[i,-1]

            cm_trans_rootvel = quat.mul_vec(cm_trans_Yrot_list[-1][0], cm_trans_Yrvel)
            cm_trans_rootang = quat.mul_vec(cm_trans_Yrot_list[-1][0], cm_trans_Yrang)
            cm_trans_rootpos = cm_trans_Ypos_list[-1][0] + cm_trans_rootvel * dt
            cm_trans_rootrot = quat.mul(cm_trans_Yrot_list[-1][0], quat.from_scaled_angle_axis(cm_trans_rootang * dt))
            
            cm_trans_Ypos = np.concatenate([cm_trans_rootpos[np.newaxis], cm_trans_Ypos], axis=0)
            cm_trans_Yvel = np.concatenate([cm_trans_rootvel[np.newaxis], cm_trans_Yvel[-1]], axis=0)
            cm_trans_Yrot = np.concatenate([cm_trans_rootrot[np.newaxis], cm_trans_Yrot], axis=0)
            cm_trans_Yang = np.concatenate([cm_trans_rootang[np.newaxis], cm_trans_Yang], axis=0)
            cm_trans_contact = src_contact[i,-1]

            # Contact fixup with foot locking and IK
            # adjusted_bone_positions = trans_Ypos.copy()
            adjusted_bone_positions = ((ik_trans_Ypos_list[-1]+trans_Yvel*dt)*0.5 + trans_Ypos*0.5).copy()
            adjusted_bone_rotations = trans_Yrot.copy()
            # bone_positions = trans_Ypos.copy()
            bone_positions = ((ik_trans_Ypos_list[-1]+trans_Yvel*dt)*0.5 + trans_Ypos*0.5).copy()
            bone_rotations = trans_Yrot.copy()
            curr_bone_contacts = trans_contact.copy().astype(bool)
            if ik_enabled:
                for bs in range(contact_bones.size):
                    toe_bone = contact_bones[bs]
                    heel_bone = parents[toe_bone]
                    knee_bone = parents[heel_bone]
                    hip_bone = parents[knee_bone]
                    root_bone = parents[hip_bone]

                    # Compute the world space position for the toe
                    global_bone_computed = np.zeros(global_bone_computed.shape, dtype=bool)

                    global_bone_positions, global_bone_rotations, global_bone_computed = \
                    quat.fk_partial(
                        global_bone_positions,
                        global_bone_rotations,
                        global_bone_computed,
                        bone_positions,
                        bone_rotations,
                        parents,
                        toe_bone)

                    # Update the contact state
                    contact_states[bs], contact_locks[bs], \
                    contact_positions[bs], contact_velocities[bs], \
                    contact_points[bs], contact_targets[bs], \
                    contact_offset_positions[bs], contact_offset_velocities[bs] = \
                    inert.contact_update(
                        contact_states[bs],
                        contact_locks[bs],
                        contact_positions[bs],  
                        contact_velocities[bs],
                        contact_points[bs],
                        contact_targets[bs],
                        contact_offset_positions[bs],
                        contact_offset_velocities[bs],
                        global_bone_positions[toe_bone],
                        curr_bone_contacts[bs],
                        ik_unlock_radius,
                        ik_foot_height,
                        ik_blending_halflife,
                        dt)
            
                    # Ensure contact position never goes through floor
                    contact_position_clamp = contact_positions[bs]
                    contact_position_clamp[1] = np.max([contact_position_clamp[1], ik_foot_height])

                    # Re-compute toe, heel, knee, hip, and root bone positions
                    for bone in [heel_bone, knee_bone, hip_bone, root_bone]:
                        global_bone_positions, global_bone_rotations, global_bone_computed = \
                            quat.fk_partial(
                                global_bone_positions,
                                global_bone_rotations,
                                global_bone_computed,
                                bone_positions,
                                bone_rotations,
                                parents,
                                bone)
                    
                    # Perform simple two-joint IK to place heel
                    adjusted_bone_rotations[hip_bone], adjusted_bone_rotations[knee_bone] = \
                    quat.ik_two_bone(
                        adjusted_bone_rotations[hip_bone],
                        adjusted_bone_rotations[knee_bone],
                        global_bone_positions[hip_bone],
                        global_bone_positions[knee_bone],
                        global_bone_positions[heel_bone],
                        contact_position_clamp + (global_bone_positions[heel_bone] - global_bone_positions[toe_bone]),
                        quat.mul_vec(global_bone_rotations[knee_bone], np.array([0.0, 1.0, 0.0], dtype=np.float32)),
                        global_bone_rotations[hip_bone],
                        global_bone_rotations[knee_bone],
                        global_bone_rotations[root_bone],
                        ik_max_length_buffer)

                    # Re-compute toe, heel, and knee positions 
                    global_bone_computed = np.zeros(global_bone_computed.shape, dtype=bool)

                    for bone in [toe_bone, heel_bone, knee_bone]:
                        global_bone_positions, global_bone_rotations, global_bone_computed = \
                            quat.fk_partial(
                                global_bone_positions,
                                global_bone_rotations,
                                global_bone_computed,
                                adjusted_bone_positions,
                                adjusted_bone_rotations,
                                parents,
                                bone)
            
            # assign to list
            trans_Ypos_list += [(trans_Ypos_list[-1]+trans_Yvel*dt)*0.5 + trans_Ypos*0.5]
            # trans_Ypos_list += [trans_Ypos]
            trans_Yvel_list += [trans_Yvel]
            trans_Yrot_list += [trans_Yrot]
            trans_Yang_list += [trans_Yang]
            trans_contact_list += [trans_contact]

            ik_trans_Ypos_list += [adjusted_bone_positions]
            ik_trans_Yrot_list += [adjusted_bone_rotations]
            ik_trans_contact_list += [trans_contact] 

            cm_trans_Ypos_list += [cm_trans_Ypos]
            cm_trans_Yvel_list += [cm_trans_Yvel]
            cm_trans_Yrot_list += [cm_trans_Yrot]
            cm_trans_Yang_list += [cm_trans_Yang]
            cm_trans_contact_list += [cm_trans_contact]
            
    src_Ypos = np.stack(src_Ypos_list, axis=0)
    src_Yvel = np.stack(src_Yvel_list, axis=0)
    src_Yrot = np.stack(src_Yrot_list, axis=0)
    src_Yang = np.stack(src_Yang_list, axis=0)
    src_contact = np.stack(src_contact_list, axis=0)

    trans_Ypos = np.stack(trans_Ypos_list, axis=0)
    trans_Yvel = np.stack(trans_Yvel_list, axis=0)
    trans_Yrot = np.stack(trans_Yrot_list, axis=0)
    trans_Yang = np.stack(trans_Yang_list, axis=0)
    trans_contact = np.stack(trans_contact_list, axis=0)

    ik_trans_Ypos = np.stack(ik_trans_Ypos_list, axis=0)
    ik_trans_Yrot = np.stack(ik_trans_Yrot_list, axis=0)
    ik_trans_contact = np.stack(ik_trans_contact_list, axis=0)

    cm_trans_Ypos = np.stack(cm_trans_Ypos_list, axis=0)
    cm_trans_Yvel = np.stack(cm_trans_Yvel_list, axis=0)
    cm_trans_Yrot = np.stack(cm_trans_Yrot_list, axis=0)
    cm_trans_Yang = np.stack(cm_trans_Yang_list, axis=0)
    cm_trans_contact = np.stack(cm_trans_contact_list, axis=0)

    src = [src_Ypos, src_Yrot, src_contact, contact_bones, parents]
    trans = [trans_Ypos, trans_Yrot, src_contact, contact_bones, parents]
    ik_trans = [ik_trans_Ypos, ik_trans_Yrot, src_contact, contact_bones, parents]
    cm_trans = [cm_trans_Ypos, cm_trans_Yrot, src_contact, contact_bones, parents]

    animation_plot([src, cm_trans, trans, ik_trans])

    src_glb_rot, src_glb_pos = quat.fk(src_Yrot, src_Ypos, parents)
    src_Ypos = src_Ypos[:, 1:]
    src_Ypos[:, 0] = src_glb_pos[:, 1]
    src_Yrot = src_Yrot[:, 1:]
    src_Yrot[:, 0] = src_glb_rot[:, 1]

    trans_glb_rot, trans_glb_pos = quat.fk(trans_Yrot, trans_Ypos, parents)
    trans_Ypos = trans_Ypos[:, 1:]
    trans_Ypos[:, 0] = trans_glb_pos[:, 1]
    trans_Yrot = trans_Yrot[:, 1:]
    trans_Yrot[:, 0] = trans_glb_rot[:, 1]

    ik_trans_glb_rot, ik_trans_glb_pos = quat.fk(ik_trans_Yrot, ik_trans_Ypos, parents)
    ik_trans_Ypos = ik_trans_Ypos[:, 1:]
    ik_trans_Ypos[:, 0] = ik_trans_glb_pos[:, 1]
    ik_trans_Yrot = ik_trans_Yrot[:, 1:]
    ik_trans_Yrot[:, 0] = ik_trans_glb_rot[:, 1]

    cm_trans_glb_rot, cm_trans_glb_pos = quat.fk(cm_trans_Yrot, cm_trans_Ypos, parents)
    cm_trans_Ypos = cm_trans_Ypos[:, 1:]
    cm_trans_Ypos[:, 0] = cm_trans_glb_pos[:, 1]
    cm_trans_Yrot = cm_trans_Yrot[:, 1:]
    cm_trans_Yrot[:, 0] = cm_trans_glb_rot[:, 1]

    # # Write BVH 
    bvh_data = bvh.load('bvh/Loco_Walk_Neutral_AverageJoe_001.bvh')    # for bvh name
    if not os.path.exists('./results/'):
        os.makedirs('./results/')

    try:
        bvh.save(os.path.join('./results/', 'Src_' + src_bvh_file.split('/')[-1]), {
            'rotations': np.degrees(quat.to_euler(src_Yrot)),
            'positions': src_Ypos,
            'offsets': src_Ypos[0],
            'parents': parents_original,
            'names': bvh_data['names'],
            'order': 'zyx'
        })

        bvh.save(os.path.join('./results/', 'Ours_' + src_bvh_file.split('/')[-1][:-4] + '_To_' + cha_bvh_file.split('/')[-1]), {
            'rotations': np.degrees(quat.to_euler(ik_trans_Yrot)),
            'positions': ik_trans_Ypos,
            'offsets': ik_trans_Ypos[0],
            'parents': parents_original,
            'names': bvh_data['names'],
            'order': 'zyx'
        })

    except IOError as e:
        print(e)


if __name__ == '__main__':
    main()