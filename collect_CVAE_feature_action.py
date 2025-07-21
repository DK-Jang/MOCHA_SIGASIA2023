import os
import sys
import argparse
from pathlib import Path
from itertools import chain
import torch
import numpy as np
sys.path.append('./etc')
sys.path.append('./motion')
sys.path.append('./preprocess')
sys.path.append('./net')
from utils import ensure_dirs, get_config, load_database
import quat
from trainer import Trainer
from net.transformer import mean_variance_norm

torch.set_grad_enabled(False);
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_bvh_files(dname):
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
                          for ext in ['bvh']]))
    return fnames

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

    # Trainer
    trainer = Trainer(cfg)
    epochs = trainer.load_checkpoint('model_ours/pth/gen_125.pt')
    model = trainer.gen_ema.eval()

    target_character = 'Neutral_Princess'
    target_action_list = ['Jump', 'Crawling', 'Run', 'Walk', 'Sit']
    
    dataset_save_dir = './CVAE_transformer'
    if not os.path.exists(dataset_save_dir):
        os.makedirs(dataset_save_dir)

    # ----------------------------------------------------------------------------------------
    dataset_dir = cfg['data_dir']
    dataset_cfg = get_config('./configs/dataset.yaml')
    style_names = dataset_cfg['mocha_style_names']
    action_names = dataset_cfg['mocha_action_names']
    target_style_label = [i for i, sm in enumerate(style_names) if target_character in sm]
    # target_style_label = [style_names.index(target_style) for target_style in target_character]
    target_action_label = [action_names.index(target_action) for target_action in target_action_list]

    parents_no_root = np.array(cfg['dataset']['mocha']['parents'])
    parents = np.concatenate([[-1], parents_no_root + 1])
    
    # import norm dataset
    ndata_npz_path = os.path.join(dataset_dir, 'norm.npz')
    norm = np.load(ndata_npz_path, allow_pickle=True)
    norm_np = {}
    for key, value in norm.items():
        norm_np[key] = value[np.newaxis] 
    X_mean, X_std = norm_np['X_mean'][np.newaxis], norm_np['X_std'][np.newaxis]

    """ dataset """
    database = load_database(os.path.join(dataset_dir, 'database.bin'))
        
    parents = database['bone_parents']
    contacts = database['contact_states']
    range_starts = database['range_starts']
    range_stops = database['range_stops']
    style_labels = database['style_labels']
    action_labels = database['action_labels']

    Ypos = database['bone_positions'].astype(np.float32)
    Yrot = database['bone_rotations'].astype(np.float32)
    Yvel = database['bone_velocities'].astype(np.float32)
    Yang = database['bone_angular_velocities'].astype(np.float32)

    del database

    nframes = Ypos.shape[0]
    njoints = Ypos.shape[1]
    nextra = contacts.shape[1]
    total_len = len(range_starts)

    window = 60

    Ypos_slice, Yvel_slice, Yrot_slice, Yang_slice = [], [], [], []
    contact_slice = []
    action_label_slice = []
    cha_range_starts, cha_range_stops = [], []
    for i in range(total_len):
        if not style_labels[i] in target_style_label:
            continue
        if not action_labels[i] in target_action_label:
            continue
        start = range_starts[i]
        stop = range_stops[i]
        total_frames = stop - start
        print('Character processing clip %d/%d' % (i+1, total_len))
        print('  start: %d' % start)
        print('  stop: %d' % stop)
        print('  length: %d' % total_frames)
        print('  style: %s' % style_names[style_labels[i]])
        print('  action: %s' % action_names[action_labels[i]])

        for j in range(window, total_frames):
            Ypos_slice.append(Ypos[start:stop][j-window:j])
            Yrot_slice.append(Yrot[start:stop][j-window:j])
            Yvel_slice.append(Yvel[start:stop][j-window:j])
            Yang_slice.append(Yang[start:stop][j-window:j])
            contact_slice.append(contacts[start:stop][j-window:j])
            action_label_slice.append(action_labels[i])

        offset = 0 if len(cha_range_starts) == 0 else cha_range_stops[-1] 
        cha_range_starts.append(offset)
        cha_range_stops.append(offset + (total_frames-window))
    print('Total number of motion clips: %d' % len(Ypos_slice))
    
    # collect train dataset
    Ypos = np.array(Ypos_slice, dtype=np.float32)
    Yvel = np.array(Yvel_slice, dtype=np.float32)
    Yrot = np.array(Yrot_slice, dtype=np.float32)
    Yang = np.array(Yang_slice, dtype=np.float32)
    contacts = np.array(contact_slice, dtype=np.float32)
    action_label = np.array(action_label_slice, dtype=np.int32)

    cha_range_starts = np.array(cha_range_starts).astype(np.int32)
    cha_range_stops = np.array(cha_range_stops).astype(np.int32)

    # Compute world space
    Grot, Gpos, Gvel, Gang = quat.fk_vel(Yrot, Ypos, Yvel, Yang, parents)

    # Compute X local to (current - N/2) root
    Gpos[:,:,0:1] = np.repeat(Gpos[:, -1:, 0:1], window, axis=1)
    Grot[:,:,0:1] = np.repeat(Grot[:, -1:, 0:1], window, axis=1)
    Gvel[:,:,0:1] = np.repeat(Gvel[:, -1:, 0:1], window, axis=1)
    Gang[:,:,0:1] = np.repeat(Gang[:, -1:, 0:1], window, axis=1)

    Xpos = quat.inv_mul_vec(Grot[:,:,0:1], Gpos - Gpos[:,:,0:1])
    Xrot = quat.inv_mul(Grot[:,:,0:1], Grot)
    Xtxy = quat.to_xform_xy(Xrot).astype(np.float32)
    Xvel = quat.inv_mul_vec(Grot[:,:,0:1], Gvel)
    Xang = quat.inv_mul_vec(Grot[:,:,0:1], Gang)

    b, ns, nj, _, _ = Xtxy.shape
    X = np.concatenate([
                Xpos,
                Xtxy.reshape(b, ns, nj, -1),
                Xvel,
                Xang,
            ], axis=-1)
    X = (X[:,:,1:] - X_mean[:,:,1:]) / X_std[:,:,1:]

    encoded_list, cnt_list = [], []
    for i in range(0, len(X), 32):
        X_batch = X[i:i+32]
        X_batch = torch.from_numpy(X_batch).to(device)
        with torch.no_grad():
            tokens = model.mot_embedding(X_batch)
            tokens = tokens + model.pos_emb[:, :tokens.shape[1]]
            encoded = model.encoder(tokens)
            cnt = mean_variance_norm(encoded.permute(0, 2, 1))
            cnt = cnt.permute(0, 2, 1)
        encoded_list.append(encoded.cpu().numpy())
        cnt_list.append(cnt.cpu().numpy())
    encoded = np.concatenate(encoded_list, axis=0)
    cnt = np.concatenate(cnt_list, axis=0)

    file_name = target_character + '_5action_feature.npz'
    # file_name =  'Neutral_Princess_5action_feature.npz'
    save_path = os.path.join(dataset_save_dir, file_name)
    np.savez_compressed(save_path, encoded=encoded,
                                   cnt=cnt,
                                   range_starts=cha_range_starts,
                                   range_stops=cha_range_stops,
                                   action_label=action_label)
    print('Save ' + save_path)


if __name__ == '__main__':
    main()