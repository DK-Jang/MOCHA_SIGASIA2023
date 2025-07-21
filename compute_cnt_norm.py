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

def divide_clip(input, window, window_step):
    """ Slide over windows """
    windows = []
    for j in range(0, len(input)-window//4, window_step):
        """ If slice too small pad out by repeating start and end poses """
        slice = input[j:j+window]
        if len(slice) < window:
            break
        if len(slice) != window:
            raise Exception()
        windows.append(slice)
    return windows


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

    # ----------------------------------------------------------------------------------------
    dataset_dir = cfg['data_dir']
    dataset_cfg = get_config('./configs/dataset.yaml')

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
    window_step = 20

    Ypos_ws, Yvel_ws, Yrot_ws, Yang_ws = [], [], [], []
    contacts_ws, style_label_ws, action_label_ws = [], [], []
    for i in range(total_len):
        start = range_starts[i]
        stop = range_stops[i]
        print('Processing clip %d/%d' % (i+1, total_len))
        print('  start: %d' % start)
        print('  stop: %d' % stop)
        print('  length: %d' % (stop-start))

        # Divide clip into windows
        n_ws = (stop - start - window) // window_step + 1
        Ypos_ws += divide_clip(Ypos[start:stop], window, window_step)
        Yvel_ws += divide_clip(Yvel[start:stop], window, window_step)
        Yrot_ws += divide_clip(Yrot[start:stop], window, window_step)
        Yang_ws += divide_clip(Yang[start:stop], window, window_step)
        contacts_ws += divide_clip(contacts[start:stop], window, window_step)
        style_label_ws += [style_labels[i]] * n_ws
        action_label_ws += [action_labels[i]] * n_ws

    # collect train dataset
    Ypos = np.array(Ypos_ws, dtype=np.float32)
    Yvel = np.array(Yvel_ws, dtype=np.float32)
    Yrot = np.array(Yrot_ws, dtype=np.float32)
    Yang = np.array(Yang_ws, dtype=np.float32)
    contacts = np.array(contacts_ws, dtype=np.float32)
    style_labels = np.array(style_label_ws, dtype=np.int32)
    action_labels = np.array(action_label_ws, dtype=np.int32)

    # Compute world space
    Grot, Gpos, Gvel, Gang = quat.fk_vel(Yrot, Ypos, Yvel, Yang, parents)

    # Compute X local to root at current frame
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

    cnt_list = []
    encoded_list = []
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

    del cnt_list
    del X, Xpos, Xrot, Xtxy, Xvel, Xang

    encoded_mean, encoded_std = np.mean(encoded, axis=0), np.std(encoded, axis=0)
    cnt_mean, cnt_std = np.mean(cnt, axis=0), np.std(cnt, axis=0)

    # Save data
    dataset_path = os.path.join('./datasets/mocha60', 'cnt_norm.npz')
    np.savez_compressed(dataset_path, mean=cnt_mean, std=cnt_std)


if __name__ == '__main__':
    main()

