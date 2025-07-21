import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def mean_variance_norm(input, eps=1e-5):
    size = input.size()
    input = input.view(size[0], size[1], -1)
    mean = input.mean(-1, keepdim=True)
    std = input.std(-1, keepdim=True)
    input = (input - mean) / (std + eps)
    input = input.view(size)
    return input


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.1, adain=False):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.mapping_function = nn.Sequential(Rearrange('b s c -> b c s'),
                                            #   nn.InstanceNorm1d(dim),
                                              InstanceNorm1d(),
                                              Rearrange('b c s -> b s c')) if adain else nn.Identity()

        self.to_q = nn.Sequential(self.mapping_function, nn.Linear(dim, inner_dim, bias=False))
        self.to_k = nn.Sequential(self.mapping_function, nn.Linear(dim, inner_dim, bias=False))
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
    
    def forward(self, src, tar=None):
        tar = default(tar, src)
        qkv = (self.to_q(src), self.to_k(tar), self.to_v(tar))

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.1, adain=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.adain = adain
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                AdaIN(dim, dim) if adain else nn.Identity(),
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, adain=adain),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x, sty=None):
        for nm, attn, ff in self.layers:
            x = nm(x, sty) if (exists(sty) and self.adain) else nm(x)
            x = attn(x, sty) + x
            x = ff(x) + x
        return x


class AdaIN(nn.Module):
    def __init__(self, fin, style_dim=512):
        super().__init__()
        # self.norm = nn.InstanceNorm1d(fin, affine=False)
        self.norm = InstanceNorm1d()
        self.style = nn.Sequential(nn.AdaptiveAvgPool1d(1),
                                   Rearrange('b c 1 -> b c'),
                                   nn.Linear(style_dim, style_dim*2),
                                   nn.LeakyReLU(0.2),
                                   nn.Linear(style_dim*2, fin*2))
    def forward(self, input, style):
        style = self.style(style.permute(0,2,1)).unsqueeze(2)
        gamma, beta = style.chunk(2, 1)
        out = self.norm(input.permute(0, 2, 1))
        out = (1 + gamma) * out + beta
        return out.permute(0, 2, 1)


class InstanceNorm1d(nn.Module):
    def __init__(self, eps=1e-5):
        super(InstanceNorm1d, self).__init__()
        self.eps = eps
    def forward(self, input):
        return mean_variance_norm(input, self.eps)


if __name__ == '__main__':
    import time

    dim = 256
    dim_head = 64
    heads = 4

    trans = Transformer(dim = dim,
                        depth=2,
                        heads=heads, 
                        dim_head=dim_head, 
                        mlp_dim=dim*2,
                        dropout=0.1,
                        adain=True)

    src = torch.randn(1, 120, 256)
    tar = torch.randn(1, 120, 256)

    start = time.time()
    out = trans(src, tar)
    end = time.time()

    print(f"Runtime of the program is {end - start}")
