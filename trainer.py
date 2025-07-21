import os
import sys
import copy
import logging
from tqdm import tqdm
from etc.utils import get_model_list
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append('./motion')
import txform
import tquat
logger = logging.getLogger(__name__)

from model import (Generator, Projector)


class Trainer(nn.Module):
    def __init__(self, config):
        super(Trainer, self).__init__()
        self.gen = Generator(config['model'])
        self.gen_ema = copy.deepcopy(self.gen)
        self.prj_cnt = Projector(config['model'], mode='all')
        
        self.model_dir = config['model_dir']
        self.config = config
        parents = np.array(config['dataset']['mocha']['parents'])
        self.parents = np.concatenate([[-1], parents + 1])

        lr_gen = config['lr_gen']
        weight_decay_gen = config['weight_decay_gen']
        lr_drop = config['lr_drop']
        
        gen_params = list(self.gen.parameters()) \
                   + list(self.prj_cnt.parameters())
        self.gen_opt = torch.optim.AdamW([p for p in gen_params if p.requires_grad], 
                                          lr=lr_gen,
                                          weight_decay=weight_decay_gen)
        self.gen_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.gen_opt, lr_drop)

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.gen = nn.DataParallel(self.gen).to(self.device)
            self.gen_ema = nn.DataParallel(self.gen_ema).to(self.device)
            self.prj_cnt = nn.DataParallel(self.prj_cnt).to(self.device)

    def train(self, loader, writer):
        config = self.config
        norm  = loader['norm']
        for key, value in norm.items():
            norm[key] = value.to(self.device)

        def run_epoch(epoch):
            self.gen.train()
            self.prj_cnt.train()
            
            pbar = tqdm(enumerate(zip(loader['train_src'], loader['train_cha'])), 
                        total=len(loader['train_src']))
            
            for it, (src_data, cha_data) in pbar:
                # to cuda tensor
                for key, value in src_data.items():
                    if key == 'bvh_name':
                        break
                    src_data[key] = value.to(self.device)
                for key, value in cha_data.items():
                    if key == 'bvh_name':
                        break
                    cha_data[key] = value.to(self.device)
                      
                # train the generator
                gen_loss_total, gen_loss_dict = \
                    self.compute_gen_loss(src_data, cha_data, norm)
                self.gen_opt.zero_grad()
                gen_loss_total.backward()
                torch.nn.utils.clip_grad_norm_(self.gen.parameters(), 1.0)
                self.gen_opt.step()
                update_average(self.gen_ema, self.gen)

                # report progress
                log = "Epoch [%i/%i], " % (epoch+1, config['max_epochs'])
                all_losses = dict()
                for loss in [gen_loss_dict]:
                    for key, value in loss.items():
                        if key.find('total') > -1:
                            all_losses[key] = value
                log += ' '.join(['%s: [%.2f]' % (key, value) for key, value in all_losses.items()])
                pbar.set_description(log)

                if (it+1) % config['log_every'] == 0:
                    for k, v in gen_loss_dict.items():
                        writer.add_scalar(k, v, epoch*len(loader['train_src'])+it)
                        
        for epoch in range(config['max_epochs']):
            run_epoch(epoch)
            self.gen_lr_scheduler.step()

            if (epoch+1) % config['save_every'] == 0:
                self.save_checkpoint(epoch+1)

    def compute_gen_loss(self, src_data, cha_data, norm):
        config = self.config
        X_mean, X_std = norm['X_mean'], norm['X_std']
        Y_mean, Y_std = norm['Y_mean'], norm['Y_std']

        # src_style_label, cha_style_label = src_data['style_label'], cha_data['style_label']
        # src_action_label, cha_action_label = src_data['action_label'], cha_data['action_label']

        src_X, cha_X = src_data['X'], cha_data['X']
        src_Y, cha_Y = src_data['Y'], cha_data['Y']
        # src_fc, cha_fc = src_data['contact'], cha_data['contact']
        
        # input
        src_X_in = (src_X[:,:,1:] - X_mean[:,:,1:]) / X_std[:,:,1:]
        cha_X_in = (cha_X[:,:,1:] - X_mean[:,:,1:]) / X_std[:,:,1:]

        # generate
        # trans_Ytil, trans_fc_logit = self.gen(src_X_in, cha_X_in, foot_contact=True)
        trans_Ytil = self.gen(src_X_in, cha_X_in)
        recon_src_Ytil = self.gen(src_X_in, src_X_in)
        recon_cha_Ytil = self.gen(cha_X_in, cha_X_in)

        # convert Ytil to X_in
        trans_Ytil = trans_Ytil * Y_std[:,:,1:] + Y_mean[:,:,1:]
        trans_X = convert_YtilToX(trans_Ytil, src_Y[:,:,0:1], self.parents)
        trans_X_in = (trans_X[:,:,1:] - X_mean[:,:,1:]) / X_std[:,:,1:]
        recon_src_Ytil = recon_src_Ytil * Y_std[:,:,1:] + Y_mean[:,:,1:]
        recon_cha_Ytil = recon_cha_Ytil * Y_std[:,:,1:] + Y_mean[:,:,1:]
        
        # recon loss
        loss_recon_cha = recon_criterion(recon_cha_Ytil, cha_Y, self.parents)
        loss_recon_src = recon_criterion(recon_src_Ytil, src_Y, self.parents)
        loss_recon = 0.5 * (loss_recon_src + loss_recon_cha)

        # contrastive loss for context preservation
        _, _, src_cnt, trans_cnt = self.gen(src_X_in, trans_X_in, extract_feature=True)
        feat_k_cnt, sample_id = self.prj_cnt(trans_cnt, None)
        feat_q_cnt, _ = self.prj_cnt(src_cnt, sample_id)
        loss_nce_cnt, logits_nce_cnt, labels_nce_cnt = self.patch_nce_loss(feat_q_cnt, feat_k_cnt)
        top1, top5 = contrastive_acc(logits_nce_cnt, labels_nce_cnt, topk=(1, 5))

        # cyc
        cyc_src_Ytil = self.gen(trans_X_in, src_X_in)
        cyc_cha_Ytil = self.gen(cha_X_in, trans_X_in)
        cyc_src_Ytil = cyc_src_Ytil * Y_std[:,:,1:] + Y_mean[:,:,1:]
        cyc_cha_Ytil = cyc_cha_Ytil * Y_std[:,:,1:] + Y_mean[:,:,1:]
        loss_cyc_cha = recon_criterion(cyc_cha_Ytil, cha_Y, self.parents)
        loss_cyc_src = recon_criterion(cyc_src_Ytil, src_Y, self.parents)
        loss_cyc = 0.5 * (loss_cyc_src + loss_cyc_cha)

        # summary
        l_total = (config['rec_w'] * loss_recon
                 + config['nce_w'] * loss_nce_cnt
                 + config['cyc_w'] * loss_cyc
                )
            
        l_dict = {'gen/loss_total': l_total,
                  'gen/loss_recon': loss_recon,
                  'gen/loss_nce_cnt': loss_nce_cnt,
                  'gen/cnt_acc_top1': top1[0],
                  'gen/cnt_acc_top5': top5[0],
                  'gen/loss_cyc': loss_cyc,
                  }

        return l_total, l_dict

    def patch_nce_loss(self, feat_q, feat_k, temp=0.07, nce_includes_all_negatives_from_minibatch=True):
        config = self.config
        nce_includes_all_negatives_from_minibatch = config['nce_includes_all_negatives_from_minibatch']
        num_patches = feat_q.shape[0]
        dim = feat_q.shape[1]

        feat_q = F.normalize(feat_q, dim=1)
        feat_k = F.normalize(feat_k, dim=1)

        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(
            feat_q.view(num_patches, 1, -1), feat_k.view(num_patches, -1, 1))
        l_pos = l_pos.view(num_patches, 1)

        # neg logit
        if nce_includes_all_negatives_from_minibatch:
            # reshape features as if they are all negatives of minibatch of size 1.
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = config['batch_size']

        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=torch.bool)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        logits = torch.cat((l_pos, l_neg), dim=1) / temp
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=feat_q.device)
        loss = F.cross_entropy(logits, labels, reduction='none')

        return loss.mean(), logits, labels
    
    def save_checkpoint(self, epoch):
        gen_path = os.path.join(self.model_dir, 'gen_%03d.pt' % epoch)

        # DataParallel wrappers keep raw model object in .module attribute
        raw_gen = self.gen.module if hasattr(self.gen, "module") else self.gen
        raw_gen_ema = self.gen_ema.module if hasattr(self.gen_ema, "module") else self.gen_ema

        logger.info("saving %s", gen_path)
        torch.save({'gen': raw_gen.state_dict(), 
                    'gen_ema': raw_gen_ema.state_dict(),
                    'gen_opt': self.gen_opt.state_dict()}, gen_path)
        
        print('Saved model at epoch %d' % epoch)
    
    def load_checkpoint(self, model_path=None, resume=False):
        if not model_path:
            model_dir = self.model_dir
            model_path = get_model_list(model_dir, "gen")   # last model

        map_location = lambda storage, loc: storage
        if torch.cuda.is_available():
            map_location = None
        state_dict = torch.load(model_path, map_location=map_location)

        # if self.device == 'cpu':
        #     state_dict = torch.load(model_path, map_location=self.device)
        # else: 
        #     state_dict = torch.load(model_path)

        self.gen.load_state_dict(state_dict['gen'])
        self.gen_ema.load_state_dict(state_dict['gen_ema'])
        if resume:
            self.gen_opt.load_state_dict(state_dict['gen_opt'])

        epochs = int(model_path[-6:-3])
        print('Load from epoch %d' % epochs)

        return epochs

def recon_criterion(Ytil, Ygt, parents):
    dt = 1.0 / 60.0

    # ground truth
    Ygt_pos = Ygt[..., :3]
    Ygt_txy = Ygt[..., 3:9].reshape(Ygt.shape[0], Ygt.shape[1],
                                        Ygt.shape[2], 3, 2)
    Ygt_xfm = txform.from_xy(Ygt_txy)
    Ygt_vel = Ygt[..., 9:12]
    Ygt_ang = Ygt[..., 12:15]

    # Ytil
    Ytil_pos = Ytil[..., :3]
    Ytil_txy = Ytil[..., 3:9].reshape(Ytil.shape[0], 
                                      Ytil.shape[1], 
                                      Ytil.shape[2], 3, 2)
    Ytil_vel = Ytil[..., 9:12]
    Ytil_ang = Ytil[..., 12:15]

    # Add root bone from ground truth
    Ytil_pos = torch.cat([Ygt_pos[:,:,0:1], Ytil_pos], dim=2)
    Ytil_txy = torch.cat([Ygt_txy[:,:,0:1], Ytil_txy], dim=2)
    Ytil_xfm = txform.from_xy(Ytil_txy)
    Ytil_vel = torch.cat([Ygt_vel[:,:,0:1], Ytil_vel], dim=2)
    Ytil_ang = torch.cat([Ygt_ang[:,:,0:1], Ytil_ang], dim=2)

    # Do FK
    Ggt_xfm, Ggt_pos, Ggt_vel, Ggt_ang = txform.fk_vel(
        Ygt_xfm, Ygt_pos, Ygt_vel, Ygt_ang, parents)
    Gtil_xfm, Gtil_pos, Gtil_vel, Gtil_ang = txform.fk_vel(
        Ytil_xfm, Ytil_pos, Ytil_vel, Ytil_ang, parents)
    
    # Compute Character Space, local to current frame
    Qgt_xfm = txform.inv_mul(Ggt_xfm[:,:,0:1], Ggt_xfm)
    Qgt_pos = txform.inv_mul_vec(Ggt_xfm[:,:,0:1], Ggt_pos - Ggt_pos[:,:,0:1])
    Qgt_vel = txform.inv_mul_vec(Ggt_xfm[:,:,0:1], Ggt_vel)
    Qgt_ang = txform.inv_mul_vec(Ggt_xfm[:,:,0:1], Ggt_ang)
    Qtil_xfm = txform.inv_mul(Gtil_xfm[:,:,0:1], Gtil_xfm)
    Qtil_pos = txform.inv_mul_vec(Gtil_xfm[:,:,0:1], Gtil_pos - Gtil_pos[:,:,0:1])
    Qtil_vel = txform.inv_mul_vec(Gtil_xfm[:,:,0:1], Gtil_vel)
    Qtil_ang = txform.inv_mul_vec(Gtil_xfm[:,:,0:1], Gtil_ang)

    # Compute deltas
    Ygt_dpos = (Ygt_pos[:,1:] - Ygt_pos[:,:-1]) / dt
    Ygt_drot = (Ygt_txy[:,1:] - Ygt_txy[:,:-1]) / dt
    Qgt_dpos = (Qgt_pos[:,1:] - Qgt_pos[:,:-1]) / dt
    Qgt_drot = (Qgt_xfm[:,1:] - Qgt_xfm[:,:-1]) / dt
    
    Ytil_dpos = (Ytil_pos[:,1:] - Ytil_pos[:,:-1]) / dt
    Ytil_drot = (Ytil_txy[:,1:] - Ytil_txy[:,:-1]) / dt
    Qtil_dpos = (Qtil_pos[:,1:] - Qtil_pos[:,:-1]) / dt
    Qtil_drot = (Qtil_xfm[:,1:] - Qtil_xfm[:,:-1]) / dt

    # losses
    loss_loc_pos = torch.mean(75.0 * torch.abs(Ygt_pos - Ytil_pos))
    loss_loc_txy = torch.mean(10.0 * torch.abs(Ygt_txy - Ytil_txy))
    loss_loc_vel = torch.mean(10.0 * torch.abs(Ygt_vel - Ytil_vel))
    loss_loc_ang = torch.mean(1.25 * torch.abs(Ygt_ang - Ytil_ang))
    # loss_loc_rvel = torch.mean(2.0 * torch.abs(Ygt_rvel - Ytil_rvel))
    # loss_loc_rang = torch.mean(2.0 * torch.abs(Ygt_rang - Ytil_rang))
    # loss_loc_extra = torch.mean(2.0 * torch.abs(Ygt_extra - Ytil_extra))
    
    loss_chr_pos = torch.mean(15.0 * torch.abs(Qgt_pos - Qtil_pos))
    loss_chr_xfm = torch.mean( 5.0 * torch.abs(Qgt_xfm - Qtil_xfm))
    loss_chr_vel = torch.mean( 2.0 * torch.abs(Qgt_vel - Qtil_vel))
    loss_chr_ang = torch.mean(0.75 * torch.abs(Qgt_ang - Qtil_ang))
    
    loss_lvel_pos = torch.mean(10.0 * torch.abs(Ygt_dpos - Ytil_dpos))
    loss_lvel_rot = torch.mean(1.75 * torch.abs(Ygt_drot - Ytil_drot))
    loss_cvel_pos = torch.mean(2.0  * torch.abs(Qgt_dpos - Qtil_dpos))
    loss_cvel_rot = torch.mean(0.75 * torch.abs(Qgt_drot - Qtil_drot))

    loss_recon = (loss_loc_pos + 
                  loss_loc_txy + 
                  loss_loc_vel + 
                  loss_loc_ang + 
              #   loss_loc_rvel + 
              #   loss_loc_rang + 
              #   loss_loc_extra + 
                  loss_chr_pos + 
                  loss_chr_xfm + 
                  loss_chr_vel + 
                  loss_chr_ang + 
                  loss_lvel_pos + 
                  loss_lvel_rot + 
                  loss_cvel_pos + 
                  loss_cvel_rot)

    return loss_recon

def convert_YtilToX(Ytil, Ygrd, parents):
    Ygnd_pos = Ygrd[..., :3]
    Ygnd_txy = Ygrd[..., 3:9].reshape(Ygrd.shape[0], Ygrd.shape[1], Ygrd.shape[2], 3, 2)
    Ygnd_vel = Ygrd[..., 9:12]
    Ygnd_ang = Ygrd[..., 12:15]

    Ytil_pos = Ytil[..., :3]
    Ytil_txy = Ytil[..., 3:9].reshape(Ytil.shape[0], Ytil.shape[1], Ytil.shape[2], 3, 2)
    Ytil_vel = Ytil[..., 9:12]
    Ytil_ang = Ytil[..., 12:15]

    # Add root bone from ground
    Ytil_pos = torch.cat([Ygnd_pos, Ytil_pos], dim=2)
    Ytil_txy = torch.cat([Ygnd_txy, Ytil_txy], dim=2)
    Ytil_rot = tquat.from_xform_xy(Ytil_txy)
    Ytil_vel = torch.cat([Ygnd_vel, Ytil_vel], dim=2)
    Ytil_ang = torch.cat([Ygnd_ang, Ytil_ang], dim=2)

    # Do FK and compute world space
    Gtil_rot, Gtil_pos, Gtil_vel, Gtil_ang = tquat.fk_vel(
        Ytil_rot, Ytil_pos, Ytil_vel, Ytil_ang, parents)

    Xpos = tquat.inv_mul_vec(Gtil_rot[:,:,0:1], Gtil_pos - Gtil_pos[:,:,0:1])
    Xrot = tquat.inv_mul(Gtil_rot[:,:,0:1], Gtil_rot)
    Xtxy = tquat.to_xform_xy(Xrot)
    Xvel = tquat.inv_mul_vec(Gtil_rot[:,:,0:1], Gtil_vel)
    Xang = tquat.inv_mul_vec(Gtil_rot[:,:,0:1], Gtil_ang)

    X = torch.cat([
                Xpos,
                Xtxy.flatten(-2),
                Xvel,
                Xang,
                ], dim=-1)

    return X

def update_average(model_tgt, model_src, beta=0.999):
    with torch.no_grad():
        param_dict_src = dict(model_src.named_parameters())
        for p_name, p_tgt in model_tgt.named_parameters():
            p_src = param_dict_src[p_name]
            assert(p_src is not p_tgt)
            p_tgt.copy_(beta*p_tgt + (1. - beta)*p_src)

def contrastive_acc(output, target, topk=(1,)):
    """Computes the contrastive_acc over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    import argparse
    from etc.utils import get_config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to the config file.')
    args = parser.parse_args()
    config = get_config(args.config)
    config['main_dir'] = os.path.join('.', config['name'])
    config['model_dir'] = os.path.join(config['main_dir'], "pth")

    trainer = Trainer(config)

    src_motion = torch.randn(8, 12, 23, 120)
    cha_motion = torch.randn(8, 12, 23, 120)

    # print(in_xb1)

    