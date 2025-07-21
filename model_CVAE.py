import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat


class CVAE(nn.Module):
    def __init__(
        self,
        output_seq,
        latent_dim=256, depth=2, nheads=4,      # transformer
        feedforward_dim=512, dropout=0.1, activation=F.relu,
        ):
        super().__init__()

        args = (
            latent_dim,
            depth,
            nheads,
            feedforward_dim,
            dropout,
            activation
        )

        self.prior_net = PriorNet(*args)
        self.encoder = Encoder(*args)
        self.decoder = Decoder(output_seq, *args)
    
    def prior(self, c):
        _, mu, logvar = self.prior_net(c)
        return mu, logvar
    
    def encode(self, x, c):     # posterior
        _, mu, logvar = self.encoder(x, c)
        return mu, logvar

    def forward(self, x, c):
        z_po, mu_po, logvar_po = self.encoder(x, c)
        z_pr, mu_pr, logvar_pr = self.prior_net(c)
        out = self.decoder(z_po, c)
        return out, (mu_po, logvar_po), (mu_pr, logvar_pr)   # out, (posterior), (prior)

    def sample(self, c, deterministic=False):
        z_pr, _, _ = self.prior_net(c, deterministic)
        return self.decoder(z_pr, c)
    

class PriorNet(nn.Module):
    def __init__(
        self,
        latent_dim=256, depth=2, nheads=4,
        feedforward_dim=512, dropout=0.1, activation=F.relu,
        ):
        super().__init__()

        self.pos_encoder = PositionalEncoding(latent_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim,
                                                   nhead=nheads,
                                                   dim_feedforward=feedforward_dim,
                                                   dropout=dropout,
                                                   activation=activation,
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer,
                                             num_layers=depth)

        self.mu_token = nn.Parameter(torch.randn(1, 1, latent_dim))
        self.logvar_token = nn.Parameter(torch.randn(1, 1, latent_dim))

    def encode(self, c):
        b = c.shape[0]
        mu_token = repeat(self.mu_token, '1 1 d -> b 1 d', b = b)
        logvar_token = repeat(self.logvar_token, '1 1 d -> b 1 d', b = b)

        tokens = torch.cat((mu_token, logvar_token, c), dim=1)
        tokens = self.pos_encoder(tokens)
        out = self.encoder(tokens)
        mu, logvar = out[:, 0], out[:, 1]
        return mu, logvar

    def reparameterize(self, mu, logvar, deterministic=False):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        if deterministic:
            return mu
        else:
            return mu + eps * std

    def forward(self, c, deterministic=False):
        mu, logvar = self.encode(c)
        z = self.reparameterize(mu, logvar, deterministic)
        return z, mu, logvar


class Encoder(nn.Module):
    def __init__(
        self,
        latent_dim=256, depth=2, nheads=4,      # transformer
        feedforward_dim=512, dropout=0.1, activation=F.relu,
        ):
        super().__init__()

        self.pos_encoder = PositionalEncoding(latent_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim,
                                                   nhead=nheads,
                                                   dim_feedforward=feedforward_dim,
                                                   dropout=dropout,
                                                   activation=activation,
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer,
                                             num_layers=depth)

        self.mu_token = nn.Parameter(torch.randn(1, 1, latent_dim))
        self.logvar_token = nn.Parameter(torch.randn(1, 1, latent_dim))

    def encode(self, x, c):
        b = x.shape[0]
        mu_token = repeat(self.mu_token, '1 1 d -> b 1 d', b = b)
        logvar_token = repeat(self.logvar_token, '1 1 d -> b 1 d', b = b)

        tokens = torch.cat((mu_token, logvar_token, c, x), dim=1)
        tokens = self.pos_encoder(tokens)
        out = self.encoder(tokens)
        mu, logvar = out[:, 0], out[:, 1]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class Decoder(nn.Module):
    def __init__(
        self,
        output_seq,
        latent_dim=256, depth=2, nheads=4,
        feedforward_dim=512, dropout=0.1, activation=F.relu,
        ):
        super().__init__()

        self.pos_encoder = PositionalEncoding(latent_dim, dropout)
        decoder_layer = nn.TransformerDecoderLayer(d_model=latent_dim,
                                                   nhead=nheads,
                                                   dim_feedforward=feedforward_dim,
                                                   dropout=dropout,
                                                   activation=activation,
                                                   batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer,
                                             num_layers=depth)

        self.output_seq = output_seq

    def forward(self, z, c):
        batch, n_cseq, latent_dim = c.shape
        z_c = torch.cat((z.unsqueeze(1), c), dim=1)
        query = torch.zeros(batch, self.output_seq, latent_dim, device=z_c.device)
        query = self.pos_encoder(query)
        output = self.decoder(tgt=query, memory=z_c)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
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


if __name__ == '__main__':
    import time
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cvae = CVAE(output_seq=180,
                latent_dim=256, depth=2, nheads=4,      # transformer
                feedforward_dim=512, dropout=0.1, activation=F.relu).to(device)

    x = torch.randn(1, 90, 256).to(device)
    c = torch.randn(1, 180, 256).to(device)

    start = time.time()
    out, _, _ = cvae(x, c)
    # out = cvae.sample(c)
    end = time.time()

    print(f"Runtime of the program is {end - start}")
