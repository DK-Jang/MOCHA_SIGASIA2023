import sys
import numpy as np
import math
import torch
from torch import nn
from einops.layers.torch import Rearrange
sys.path.append('./net')
sys.path.append('./motion')
from graph import (Graph_Joint, Graph_Bodypart, 
                   PoolJointToBodypart, UnpoolBodypartToJoint)
from transformer import Transformer, mean_variance_norm
from blocks import (STGCN_Block)


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        mot_in_dim = config['mot_in_dim']
        nframes = config['nframes']
        njoints = config['njoints']
        temporal_patch_size = config['temporal_patch_size']

        encoder_dim = config['encoder_dim']
        encoder_depth = config['encoder_depth']
        encoder_heads = config['encoder_heads']
        encoder_dim_head = config['encoder_dim_head']
        encoder_mlp_dim = config['encoder_mlp_dim']
        decoder_dim = config['decoder_dim']
        decoder_depth = config['decoder_depth']
        decoder_heads = config['decoder_heads']
        decoder_dim_head = config['decoder_dim_head']
        decoder_mlp_dim = config['decoder_mlp_dim']
        graph_cfg = config['graph']

        num_temp = (nframes // temporal_patch_size)     # temporal_patch_size: 3
        nbody = 6
        num_tokens = nbody * num_temp

        # positional encoding for encdoer
        self.pos_emb = nn.Parameter(torch.randn(1, num_tokens, encoder_dim))

        self.mot_embedding = nn.Sequential(
            Rearrange('b t v c -> b c t v'),
            nn.Conv2d(mot_in_dim, encoder_dim//temporal_patch_size, (1, 1)),
            JointBlock(encoder_dim//temporal_patch_size, encoder_dim, graph_cfg),
            PoolJointToBodypart('mocha'),
            nn.AvgPool2d(kernel_size=(temporal_patch_size, 1)),     # kernel size=stride size
            BodyBlock(encoder_dim, encoder_dim, graph_cfg),
            Rearrange('b c t v -> b (t v) c')
        )

        # encoder
        self.encoder = Transformer(dim = encoder_dim,
                                   depth = encoder_depth,
                                   heads = encoder_heads, 
                                   dim_head = encoder_dim_head, 
                                   mlp_dim = encoder_mlp_dim,
                                   dropout = 0.1,
                                   adain = False)
        
        # decoder
        self.decoder = Transformer(dim = decoder_dim,
                                   depth = decoder_depth, 
                                   heads = decoder_heads, 
                                   dim_head = decoder_dim_head, 
                                   mlp_dim = decoder_mlp_dim,
                                   dropout = 0.1,
                                   adain = True)
                                #    adain = False)   # for ablation study

        self.to_mot = nn.Sequential(
            Rearrange('b (t v) c -> b c t v', t=num_temp),
            BodyBlock(decoder_dim, decoder_dim, graph_cfg),
            Interpolate(scale_factor=(temporal_patch_size, 1), mode='nearest'),
            UnpoolBodypartToJoint('mocha'),
            JointBlock(decoder_dim, decoder_dim//temporal_patch_size, graph_cfg),
            nn.LeakyReLU(0.2),
            nn.Conv2d(decoder_dim//temporal_patch_size, mot_in_dim, (1, 1)),
            Rearrange('b c t v -> b t v c')
        )
    
    def forward(self, src_X, cha_X, extract_feature=False):
        # from mot
        src_tokens = self.mot_embedding(src_X)
        cha_tokens = self.mot_embedding(cha_X)

        # positional encoding
        src_tokens = src_tokens + self.pos_emb[:, :src_tokens.shape[1]]
        cha_tokens = cha_tokens + self.pos_emb[:, :cha_tokens.shape[1]]

        # propagate through the encoder transformer
        src_encoded = self.encoder(src_tokens)
        cha_encoded = self.encoder(cha_tokens)

        if extract_feature:
            src_cnt = mean_variance_norm(src_encoded.permute(0, 2, 1))
            cha_cnt = mean_variance_norm(cha_encoded.permute(0, 2, 1))
            return src_encoded, cha_encoded, src_cnt.permute(0, 2, 1), cha_cnt.permute(0, 2, 1)

        # propagate through the decoder transformer
        trans_decoded = self.decoder(src_encoded, cha_encoded)

        # to motion
        trans_Ytil = self.to_mot(trans_decoded)

        return trans_Ytil


class JointBlock(nn.Module):
    def __init__(self, in_channels, 
                       out_channels,
                       graph_cfg):
        super().__init__()

        self.graph_j = Graph_Joint(**graph_cfg['joint'])
        A_j = torch.tensor(self.graph_j.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A_j', A_j)

        # build networks
        spatial_kernel_size_j = self.A_j.size(0)     # ex) subset K= 0, 1, 2 -> 3
        ks_joint = (5, spatial_kernel_size_j)
        
        # joint level
        self.blk = STGCN_Block(in_channels,
                                 out_channels, 
                                 kernel_size=ks_joint, 
                                 t_stride=1,
                                 t_padding=True,
                                 norm='none',
                                 activation='lrelu')

    def forward(self, x):       # (N, C, T, V)
        x = self.blk(x, self.A_j)
        return x


class BodyBlock(nn.Module):
    def __init__(self, in_channels, 
                       out_channels,
                       graph_cfg):
        super().__init__()

        self.graph_b = Graph_Bodypart(**graph_cfg['bodypart'])
        A_b = torch.tensor(self.graph_b.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A_b', A_b)

        # build networks
        spatial_kernel_size_b = self.A_b.size(0)    # 2
        ks_bodypart = (3, spatial_kernel_size_b)
        
        # body level
        self.blk = STGCN_Block(in_channels,
                                 out_channels, 
                                 kernel_size=ks_bodypart, 
                                 t_stride=1,
                                 t_padding=True,
                                 norm='none',
                                 activation='lrelu')

    def forward(self, x):       # (N, C, T, V)
        x = self.blk(x, self.A_b)
        return x


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode='nearest'):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class Projector(nn.Module):
    def __init__(self, config, mode):
        super(Projector, self).__init__()
        self.num_patches = config['num_patches']
        nframes = config['nframes']
        temporal_patch_size = config['temporal_patch_size']
        encoder_dim = config['encoder_dim']
        prj_dim = config['prj_dim']
        nbody = 6
        num_temp = (nframes // temporal_patch_size)     # temporal_patch_size: 3

        self.mode = mode
        if mode == "spatial":
            self.m_dim = num_temp
        elif mode == "temp":
            self.m_dim = nbody
        elif mode == "all":
            self.m_dim = 1
        elif mode == 'style':
            self.m_dim = 2
        elif mode == "no_patches":
            self.m_dim = num_temp * nbody
        else:
            raise NotImplementedError('mode is wrong')

        # projection head
        self.mlp = nn.Sequential(nn.Linear(self.m_dim*encoder_dim, 1024),
                                 nn.ReLU(),
                                 nn.Linear(1024, prj_dim))
        # self.mlp = nn.Sequential(nn.Linear(self.m_dim*encoder_dim, prj_dim),
        #                          nn.ReLU(),
        #                          nn.Linear(prj_dim, prj_dim))

    def forward(self, feat, patch_id=None):
        b, s, c = feat.shape
        if (self.mode == "spatial" or self.mode == "temp" or self.mode == "all"):
            feat = feat.reshape(b, -1, self.m_dim * c)
            if patch_id is None:
                patch_id = np.random.permutation(feat.shape[1])
                patch_id = patch_id[:int(min(self.num_patches, patch_id.shape[0]))] if self.num_patches != -1 \
                        else patch_id[:patch_id.shape[0]]
            patch_id = torch.tensor(patch_id, dtype=torch.long, device=feat.device)
            feat_sample = feat[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])
        elif self.mode == 'style':
            std, mean = torch.std_mean(feat, dim=1)
            feat_sample = torch.cat([std, mean], dim=1)
        else:
            feat_sample = feat.reshape(b, self.m_dim * c)
        
        feat_sample = self.mlp(feat_sample)

        return feat_sample, patch_id


class Projector_noMLP(nn.Module):
    def __init__(self, config, mode):
        super(Projector_noMLP, self).__init__()
        self.num_patches = config['num_patches']
        nframes = config['nframes']
        temporal_patch_size = config['temporal_patch_size']
        encoder_dim = config['encoder_dim']
        prj_dim = config['prj_dim']
        nbody = 6
        num_temp = (nframes // temporal_patch_size)     # temporal_patch_size: 3

        self.mode = mode
        if mode == "spatial":
            self.m_dim = num_temp
        elif mode == "temp":
            self.m_dim = nbody
        elif mode == "all":
            self.m_dim = 1
        elif mode == 'style':
            self.m_dim = 2
        elif mode == "no_patches":
            self.m_dim = num_temp * nbody
        else:
            raise NotImplementedError('mode is wrong')

    def forward(self, feat, patch_id=None):
        b, s, c = feat.shape
        if (self.mode == "spatial" or self.mode == "temp" or self.mode == "all"):
            feat = feat.reshape(b, -1, self.m_dim * c)
            if patch_id is None:
                patch_id = np.random.permutation(feat.shape[1])
                patch_id = patch_id[:int(min(self.num_patches, patch_id.shape[0]))] if self.num_patches != -1 \
                        else patch_id[:patch_id.shape[0]]
            patch_id = torch.tensor(patch_id, dtype=torch.long, device=feat.device)
            feat_sample = feat[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])
        elif self.mode == 'style':
            std, mean = torch.std_mean(feat, dim=1)
            feat_sample = torch.cat([std, mean], dim=1)
        else:
            feat_sample = feat.reshape(b, self.m_dim * c)
        
        return feat_sample, patch_id
        

if __name__ == '__main__':
    import time
    import argparse
    sys.path.append('./etc')
    from utils import get_config

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/config.yaml',
                        help='Path to the config file.')
    args = parser.parse_args()
    config = get_config(args.config)

    G = Generator(config['model']).to(device)
    src_X = torch.randn(1, 60, 24, 15).to(device)
    cha_X = torch.randn(1, 60, 24, 15).to(device)

    start = time.time()
    x_out = G(src_X, cha_X)
    end = time.time()

    print(f"Runtime of the program is {end - start}")