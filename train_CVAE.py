import os
import sys
import numpy as np
from einops import rearrange
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from etc.utils import get_config, set_seed
from sklearn.neighbors import BallTree
from model_CVAE import CVAE
from net.transformer import mean_variance_norm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def kl_normal(mu_po, logvar_po, mu_pr, logvar_pr):
    """
    Computes the elem-wise KL divergence between two normal distributions KL(q || p) and
    sum over the last dimension
    """
    element_wise = 0.5 * (logvar_pr - logvar_po + logvar_po.exp() / logvar_pr.exp() \
        + (mu_po - mu_pr).pow(2) / logvar_pr.exp() - 1)
    kl = element_wise.sum(-1).clamp(min=0)
    return kl

def main():
    dataset_cfg = get_config('./configs/dataset.yaml')
    action_names = dataset_cfg['mocha_action_names']

    source_name = 'Neutral_5action'
    character_name = 'Neutral_Princess_5action'
    target_action_list = ['Jump', 'Crawling', 'Run', 'Walk', 'Sit']
    target_action_label = [action_names.index(target_action) for target_action in target_action_list]
    data_path = './CVAE_transformer'

    model_save_dir = os.path.join(data_path, source_name+'2'+character_name+'_z_token_temp_weight_1to3_KL_1e-2_autodrop0.8_noise')
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    # import all dataset cnt norm
    cnt_norm_path = os.path.join(dataset_cfg['data_dir'], 'cnt_norm.npz')
    cnt_norm = np.load(cnt_norm_path, allow_pickle=True)
    cnt_mean, cnt_std = cnt_norm['mean'], cnt_norm['std']

    src_feature_path = os.path.join(data_path, source_name + '_feature.npz')
    src_feature = np.load(src_feature_path, allow_pickle=True)
    src_cnt = src_feature['cnt']
    # src_encoded = src_feature['encoded']
    src_action_label = src_feature['action_label']
    src_range_starts = src_feature['range_starts']
    src_range_stops = src_feature['range_stops']
    del src_feature

    cha_feature_path = os.path.join(data_path, character_name + '_feature.npz')
    cha_feature = np.load(cha_feature_path, allow_pickle=True)
    cha_cnt = cha_feature['cnt']
    cha_encoded = cha_feature['encoded']
    cha_action_label = cha_feature['action_label']
    cha_range_starts = cha_feature['range_starts']
    cha_range_stops = cha_feature['range_stops']
    del cha_feature

    # for temp-weighted normalization for context feature
    temp_weight = np.linspace(1.0, 3.0, num=15)   # set weight 4 times lager for current frame
    temp_weight = np.repeat(temp_weight[:, np.newaxis], 6*256, axis=1)
    temp_weight = rearrange(temp_weight, 't (v c) -> (t v) c', v=6)

    # compute mean and std
    src_cnt_mean = src_cnt.mean(axis=0).astype(np.float32)
    src_cnt_std = src_cnt.std(axis=0).astype(np.float32)
    cha_cnt_mean = cha_cnt.mean(axis=0).astype(np.float32)
    cha_cnt_std = cha_cnt.std(axis=0).astype(np.float32)
    cha_encoded_mean = cha_encoded.mean(axis=0).astype(np.float32)
    cha_encoded_std = cha_encoded.std(axis=0).astype(np.float32)

    # Save norm data
    norm_path = os.path.join(model_save_dir, 'cvae_norm.npz')
    if not os.path.exists(norm_path):
        np.savez_compressed(norm_path,
                            std_weight=temp_weight,
                            src_cnt_mean=src_cnt_mean,
                            src_cnt_std=src_cnt_std,
                            cha_cnt_mean=cha_cnt_mean,
                            cha_cnt_std=cha_cnt_std,
                            cha_encoded_mean=cha_encoded_mean,
                            cha_encoded_std=cha_encoded_std)
    
    # scale std by temp_weight
    cnt_std /= temp_weight
    src_cnt_std /= temp_weight
    cha_cnt_std /= temp_weight
    cha_encoded_std /= temp_weight
    
    # add noise to src_cnt_std
    src_cnt_noise_std = cnt_std + 1.0
    
    src_cnt_mean = torch.from_numpy(src_cnt_mean).to(device)
    src_cnt_std = torch.from_numpy(src_cnt_std).to(device)
    cha_cnt_mean = torch.from_numpy(cha_cnt_mean).to(device)
    cha_cnt_std = torch.from_numpy(cha_cnt_std).to(device)
    cha_encoded_mean = torch.from_numpy(cha_encoded_mean).to(device)
    cha_encoded_std = torch.from_numpy(cha_encoded_std).to(device)
    
    # learning parameters
    seed = 1777
    teacher_iters = 10000
    ramping_iters = 10000
    student_iters = 20000
    num_iters = teacher_iters + ramping_iters + student_iters
    batch_size = 32
    initial_lr = 1e-4
    weight_decay = 1e-4
    use_kl_anneal = True
    kl_loss_anneal_start = 0
    kl_loss_anneal_end = 5000
    kl_w = 1e-2

    # model parameters
    nseq = 90
    latent_dim = 256
    feedforward_dim = 512
    num_steps_per_rollout = 10

    # construct dataset
    src_indices = []
    src_window_step = 5
    for i in range(len(src_range_starts)):
        total_frames = src_range_stops[i] - src_range_starts[i]
        for j in range(0, total_frames-num_steps_per_rollout, src_window_step):
            src_indices.append(np.arange(
                src_range_starts[i]+j, src_range_starts[i]+j+num_steps_per_rollout))
    src_indices = np.array(src_indices)
    src_action_indices = src_action_label[src_indices[:, 0]]

    cha_indices = []
    cha_window_step = 5
    for i in range(len(cha_range_starts)):
        total_frames = cha_range_stops[i] - cha_range_starts[i]
        for j in range(0, total_frames-num_steps_per_rollout, cha_window_step):
            cha_indices.append(np.arange(
                cha_range_starts[i]+j, cha_range_starts[i]+j+num_steps_per_rollout))
    cha_indices = np.array(cha_indices)
    cha_action_indices = cha_action_label[cha_indices[:, 0]]

    # model
    cvae_network = CVAE(output_seq=nseq,
                        latent_dim=latent_dim, depth=2, nheads=4,
                        feedforward_dim=feedforward_dim, dropout=0.1, 
                        activation=F.relu).to(device)
    cvae_network.train()

    # Train
    writer = SummaryWriter(os.path.join(model_save_dir, 'log'))

    optimizer = torch.optim.AdamW(
        cvae_network.parameters(),
        lr=initial_lr,
        weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    sample_schedule = torch.cat(
        (
            # First part is pure teacher forcing
            torch.zeros(teacher_iters),
            # Second part with schedule sampling
            torch.linspace(0.0, 1.0, ramping_iters),
            # last part is pure student
            torch.ones(student_iters),
        )
    )

    sys.stdout.write('\n')
    set_seed(seed)
    for iter in range(num_iters):
        iter_encoded_loss = 0
        iter_kl_loss = 0
        iter_cnt_loss = 0
        iter_dist_loss = 0
        
        picked_action_label =  np.random.choice(target_action_label, 1)[0]

        src_picked_action_indices = np.where(
            src_action_indices == picked_action_label)[0]
        if len(src_picked_action_indices) < batch_size:
            continue
        samples = np.random.choice(src_picked_action_indices, batch_size)
        batch_indices = src_indices[samples]
        src_cnt_batch = src_cnt[batch_indices]

        # add noise
        nsigma = np.random.uniform(size=[batch_size, 1, 1, 1]).astype(np.float32)
        noise = np.random.normal(size=[batch_size, num_steps_per_rollout, nseq, latent_dim]).astype(np.float32)
        src_cnt_hat_batch = src_cnt_batch + \
            src_cnt_noise_std[np.newaxis, np.newaxis] * nsigma * noise

        # Find nearest
        cha_picked_action_indices = np.where(
            cha_action_indices == picked_action_label)[0]
        if len(cha_picked_action_indices) < 1:      # check if there is no data
            continue
        cha_picked_indices = cha_indices[cha_picked_action_indices]
        cha_cnt_picked = cha_cnt[cha_picked_indices]
        cha_encoded_picked = cha_encoded[cha_picked_indices]

        # first index of cnt
        cha_cnt_picked_nm = (cha_cnt_picked[:,0] - cnt_mean[np.newaxis]) / cnt_std[np.newaxis]
        tree = BallTree(cha_cnt_picked_nm.reshape(len(cha_picked_indices), -1))
        src_cnt_hat_batch_nm = (src_cnt_hat_batch[:,0] - cnt_mean[np.newaxis]) / cnt_std[np.newaxis]
        nearest = tree.query(
            src_cnt_hat_batch_nm.reshape(batch_size,-1), k=1, return_distance=False)[:,0]

        cha_encoded_gnd = (torch.as_tensor(cha_encoded_picked[nearest]).to(device) 
            - cha_encoded_mean.unsqueeze(0).unsqueeze(0)) / cha_encoded_std.unsqueeze(0).unsqueeze(0)
        src_cnt_hat_batch = (torch.as_tensor(src_cnt_hat_batch).to(device) 
            - src_cnt_mean.unsqueeze(0).unsqueeze(0)) / src_cnt_std.unsqueeze(0).unsqueeze(0)
        cha_cnt_gnd = (torch.as_tensor(cha_cnt_picked[nearest]).to(device) 
            - cha_cnt_mean.unsqueeze(0).unsqueeze(0)) / cha_cnt_std.unsqueeze(0).unsqueeze(0)
        Dgnd = torch.sqrt(torch.sum(torch.square(
            src_cnt_hat_batch - cha_cnt_gnd), dim=-1)).to(device)

        offset = 0
        condition = torch.cat([src_cnt_hat_batch[:,offset+1], 
                               F.dropout(cha_encoded_gnd[:,offset], p=0.8)], dim=1)
        for offset in range(1, num_steps_per_rollout):
            use_student = torch.rand(1) < sample_schedule[iter]
      
            vae_output, po_dist, pr_dist = cvae_network(cha_encoded_gnd[:,offset], condition)
            (mu_po, logvar_po), (mu_pr, logvar_pr) = po_dist, pr_dist
            cha_encoded_til = vae_output
            cha_encoded_til_un = cha_encoded_til * cha_encoded_std.unsqueeze(0) + cha_encoded_mean.unsqueeze(0)
            cha_cnt_til = ((mean_variance_norm(cha_encoded_til_un.permute(0,2,1))).permute(0,2,1)
                - cha_cnt_mean.unsqueeze(0)) / cha_cnt_std.unsqueeze(0)
            Dtil = torch.sqrt(torch.sum(torch.square(
                src_cnt_hat_batch[:,offset] - cha_cnt_til), dim=-1))

            # loss
            kl_loss = kl_normal(mu_po, logvar_po, mu_pr, logvar_pr)     # divergence between predicted posterior and prior
            kl_loss = kl_loss.mean()
            encoded_loss = torch.mean(torch.abs(cha_encoded_til - cha_encoded_gnd[:,offset]))
            cnt_loss = torch.mean(torch.abs(cha_cnt_til - cha_cnt_gnd[:,offset]))
            dist_loss = torch.mean(torch.abs(Dtil - Dgnd[:,offset]))
            
            if offset < num_steps_per_rollout - 1:
                next_frame = vae_output if use_student else cha_encoded_gnd[:,offset]
                condition = torch.cat([src_cnt_hat_batch[:,offset+1], 
                                       F.dropout(next_frame.clone().detach(), p=0.8)], dim=1)
            
            if use_kl_anneal:
                if iter >= kl_loss_anneal_start:
                    anneal_weight = (iter - kl_loss_anneal_start) / \
                        (kl_loss_anneal_end - kl_loss_anneal_start)
                else:
                    anneal_weight = 0.0
                anneal_weight = 1.0 if anneal_weight > 1.0 else anneal_weight
                
            optimizer.zero_grad()
            (encoded_loss + anneal_weight*kl_w*kl_loss + 0.1*dist_loss).backward()
            optimizer.step()

            iter_encoded_loss += encoded_loss.item()
            iter_kl_loss += kl_loss.item()
            iter_cnt_loss += cnt_loss.item()
            iter_dist_loss += dist_loss.item()
        
        iter_encoded_loss = iter_encoded_loss / num_steps_per_rollout
        iter_kl_loss = iter_kl_loss / num_steps_per_rollout
        iter_cnt_loss = iter_cnt_loss / num_steps_per_rollout
        iter_dist_loss = iter_dist_loss / num_steps_per_rollout

        # Logging
        writer.add_scalar('cvae/encoded_loss', iter_encoded_loss, iter)
        writer.add_scalar('cvae/kl_loss', iter_kl_loss, iter)
        writer.add_scalar('cvae/kl_weight', anneal_weight*kl_w, iter)
        writer.add_scalar('cvae/cnt_loss', iter_cnt_loss, iter)
        writer.add_scalar('cvae/dist_loss', iter_dist_loss, iter)
        
        if (iter+1) % 5 == 0:
            sys.stdout.write(
                '\rIter: %7i cnt_loss: %5.3f encoded_loss: %5.3f kl_loss: %5.3f dist_loss: %5.3f' 
                % ((iter+1), iter_cnt_loss, iter_encoded_loss, iter_kl_loss, iter_dist_loss))
        
        if (iter+1) % 5000 == 0:
            cvae_path = os.path.join(model_save_dir, 'cvae_%06i.pt' % (iter+1))
            torch.save(cvae_network.state_dict(), cvae_path)
            
        if iter % 200 == 0:
            scheduler.step()


if __name__ == '__main__':
    main()