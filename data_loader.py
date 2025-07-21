import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from etc.utils import load_database
import motion.quat as quat


class MotionDataset(Dataset):
    def __init__(self, data_dir, phase='train'):
        super(MotionDataset, self).__init__()
        if phase == 'test':
            database_bin_path = os.path.join(data_dir, 'database_test.bin')
        else:
            database_bin_path = os.path.join(data_dir, 'database.bin')
        norm_npz_path = os.path.join(data_dir, 'norm.npz')

        database = load_database(database_bin_path)
        parents = database['bone_parents']
        contacts = database['contact_states']
        range_starts = database['range_starts']
        range_stops = database['range_stops']
        style_labels = database['style_labels']

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
        contacts_ws, labels_ws = [], []
        for i in range(total_len):
            start = range_starts[i]
            stop = range_stops[i]

            # Divide clip into windows
            n_ws = (stop - start - window) // window_step + 1
            Ypos_ws += divide_clip(Ypos[start:stop], window, window_step)
            Yvel_ws += divide_clip(Yvel[start:stop], window, window_step)
            Yrot_ws += divide_clip(Yrot[start:stop], window, window_step)
            Yang_ws += divide_clip(Yang[start:stop], window, window_step)
            contacts_ws += divide_clip(
                contacts[start:stop], window, window_step)
            labels_ws += [style_labels[i]] * n_ws

        # collect train dataset
        Ypos = np.array(Ypos_ws, dtype=np.float32)
        Yvel = np.array(Yvel_ws, dtype=np.float32)
        Yrot = np.array(Yrot_ws, dtype=np.float32)
        Yang = np.array(Yang_ws, dtype=np.float32)
        contacts = np.array(contacts_ws, dtype=np.float32)
        labels = np.array(labels_ws, dtype=np.int32)

        # Compute two-column transformation matrix
        Ytxy = quat.to_xform_xy(Yrot).astype(np.float32)

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

        Xpos = quat.inv_mul_vec(Grot[:,:,0:1], Gpos - Gpos[:,:,0:1])
        Xrot = quat.inv_mul(Grot[:,:,0:1], Grot)
        Xtxy = quat.to_xform_xy(Xrot).astype(np.float32)
        Xvel = quat.inv_mul_vec(Grot[:,:,0:1], Gvel)
        Xang = quat.inv_mul_vec(Grot[:,:,0:1], Gang)

        # 이렇게 하면 root joint value들 다 0됨
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

        # Compute means/stds
        if not os.path.exists(norm_npz_path):
            # generator mean and std in
            X_mean = np.concatenate([
                Xpos.mean(axis=(0, 1)),
                Xtxy.mean(axis=(0, 1)).reshape((njoints, -1)),
                Xvel.mean(axis=(0, 1)),
                Xang.mean(axis=(0, 1)),
            ], axis=-1).astype(np.float32)

            X_std = np.concatenate([
                Xpos.std(axis=(0, 1)),
                Xtxy.std(axis=(0, 1)).reshape((njoints, -1)),
                Xvel.std(axis=(0, 1)),
                Xang.std(axis=(0, 1)),
            ], axis=-1).astype(np.float32) + 1e-6

            Y_mean = np.concatenate([
                Ypos.mean(axis=(0, 1)),
                Ytxy.mean(axis=(0, 1)).reshape((njoints, -1)),
                Yvel.mean(axis=(0, 1)),
                Yang.mean(axis=(0, 1)),
            ], axis=-1).astype(np.float32)

            Y_std = np.concatenate([
                Ypos.std(axis=(0, 1)),
                Ytxy.std(axis=(0, 1)).reshape((njoints, -1)),
                Yvel.std(axis=(0, 1)),
                Yang.std(axis=(0, 1)),
            ], axis=-1).astype(np.float32) + 1e-6

            # root mean and std
            root_mean = np.concatenate([
                Yrvel.mean(axis=(0, 1)),
                Yrang.mean(axis=(0, 1)),
            ], axis=-1).astype(np.float32)

            root_std = np.concatenate([
                Yrvel.std(axis=(0, 1)),
                Yrang.std(axis=(0, 1)),
            ], axis=-1).astype(np.float32)

            np.savez_compressed(norm_npz_path,
                                X_mean=X_mean, X_std=X_std,
                                Y_mean=Y_mean, Y_std=Y_std,
                                root_mean=root_mean, root_std=root_std)
            print('Save norm dataset')

        b, ns, nj, _, _ = Xtxy.shape
        self.X = np.concatenate([
            Xpos,
            Xtxy.reshape(b, ns, nj, -1),
            Xvel,
            Xang,
        ], axis=-1)

        self.Y = np.concatenate([
            Ypos,
            Ytxy.reshape(b, ns, nj, -1),
            Yvel,
            Yang,
        ], axis=-1)

        self.root = np.concatenate([
            Yrvel,
            Yrang,
        ], axis=-1)

        self.contact = contacts
        self.label = labels

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        data = {
            'X': self.X[index],
            'Y': self.Y[index],
            'root': self.root[index],
            'contact': self.contact[index],
            'label': self.label[index],
        }

        return data


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


def get_dataloader(phase, config, seed=None, shuffle=None):
    dataset = MotionDataset(config['data_dir'], phase)
    batch_size = config['batch_size'] if phase == 'train' else 1
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=(phase == 'train') if shuffle is None else shuffle,
                      num_workers=config['num_workers'] if phase == 'train' else 0,
                      worker_init_fn=np.random.seed(seed) if seed else None,
                      pin_memory=True,
                      drop_last=True)


if __name__ == '__main__':
    import sys
    from etc.utils import print_composite
    sys.path.append('./motion')
    sys.path.append('./etc')
    from utils import get_config
    from viz_motion import animation_plot    # for checking dataloader
    # from viz_motion_X import animation_plot    # for checking dataloader
    data_dir = './datasets/mocha60/'
    batch_size = 4
    dataset = MotionDataset(data_dir, 'test')
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    ndata_npz_path = os.path.join(data_dir, 'norm.npz')
    norm = np.load(ndata_npz_path, allow_pickle=True)
    norm_tensor = {}
    for key, value in norm.items():
        norm_tensor[key] = torch.as_tensor(value).unsqueeze(0).unsqueeze(0)

    cfg = get_config('./configs/dataset.yaml')
    parents = np.array(cfg['mocha_parents'])
    parents = np.concatenate([[-1], parents + 1])
    Xparents = np.array([-1] + [0]*24)
    foot_indicies = [5, 24]

    for batch in data_loader:
        print_composite(batch)
        print(batch['label'])

        Y = batch['Y']
        Ypos = Y[..., :3].cpu().numpy()
        Ytxy = Y[..., 3:9].reshape(
            Y.shape[0], Y.shape[1], Y.shape[2], 3, 2).cpu().numpy()
        Yrot = quat.from_xform_xy(Ytxy)
        Yvel = Y[..., 9:12].cpu().numpy()
        Yang = Y[..., 12:15].cpu().numpy()

        X = batch['X']
        Xpos = X[..., :3].cpu().numpy()
        Xtxy = X[..., 3:9].reshape(
            X.shape[0], X.shape[1], X.shape[2], 3, 2).cpu().numpy()
        Xrot = quat.from_xform_xy(Xtxy)
        Xvel = X[..., 9:12].cpu().numpy()
        Xang = X[..., 12:15].cpu().numpy()

        root = batch['root']
        Yrvel = root[..., :3].cpu().numpy()
        Yrang = root[..., 3:6].cpu().numpy()

        contact = batch['contact'].cpu().numpy()
        dt = 1.0 / 60.0

        motions = []
        for i in range(len(Ypos)):
            motions.append([Ypos[i], Yrot[i], Yvel[i], Yang[i], contact[i], foot_indicies, parents])
            # motions.append([Xpos[i], Xrot[i], Xvel[i], Xang[i], contact[i], Xparents])

        animation_plot(motions)
