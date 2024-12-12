import math
import torch
import numpy as np
import torch.nn as nn

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    assert embed_dim % 2 == 0, "Embedding dimension must be even."

    # Generate 2D grid
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  
    grid = np.stack(grid, axis=0).reshape(2, -1) 

    # Compute sine-cosine embeddings for each axis
    def _1d_sincos_pos_embed(dim, pos):
        omega = 1.0 / (10000 ** (np.arange(dim // 2, dtype=np.float32) / (dim // 2)))
        pos = pos.reshape(-1) 
        out = np.einsum('m,d->md', pos, omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    emb_h = _1d_sincos_pos_embed(embed_dim // 2, grid[0]) 
    emb_w = _1d_sincos_pos_embed(embed_dim // 2, grid[1])  

    pos_embed = np.concatenate([emb_h, emb_w], axis=1)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]),
                                    pos_embed],
                                    axis=0)
    return pos_embed

def unnormalize(tensor, device='cpu'):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    mean = torch.tensor(mean).view(-1, 1, 1).to(device)
    std = torch.tensor(std).view(-1, 1, 1).to(device)
    return tensor * std + mean

# Sinusoidal positional embeds
# detail could refer to:
# https://arxiv.org/abs/1706.03762 and https://arxiv.org/abs/2009.09761

@staticmethod
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    
class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class LabelEmbedder(nn.Module):
    def __init__(self, n_class, hidden_size, dropout_prob=0.1):
        super().__init__()
        # adding one extra class to the embedding layer.
        # this extra class represents the unconditioned generation ε_θ(x_t, ∅).
        
        self.n_class = n_class
        self.dropout_prob = dropout_prob
        self.embedding = nn.Embedding(n_class + 1, hidden_size) 

    def forward(self, labels, train=True):
        if train and self.dropout_prob > 0:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            labels = torch.where(drop_ids, self.n_class, labels)
        emb = self.embedding(labels)
        return emb
    
class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)
