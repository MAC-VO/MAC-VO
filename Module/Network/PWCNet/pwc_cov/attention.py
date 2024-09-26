import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from torch import einsum


class PositionalEncoding2D(nn.Module):

    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 4) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None)

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = self._get_emb(sin_inp_x).unsqueeze(1)
        emb_y = self._get_emb(sin_inp_y)
        emb = torch.zeros((x, y, self.channels * 2), device=tensor.device).type(
            tensor.type()
        )
        emb[:, :, : self.channels] = emb_x
        emb[:, :, self.channels : 2 * self.channels] = emb_y

        self.cached_penc = emb[None, :, :, :orig_ch].repeat(tensor.shape[0], 1, 1, 1)
        return self.cached_penc

    def _get_emb(self, sin_inp):
        """
        Gets a base embedding for one dimension with sin and cos intertwined
        """
        emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
        return torch.flatten(emb, -2, -1)


class MultiHeadAttention(nn.Module):

    def __init__(self, dim, heads):
        super(MultiHeadAttention, self).__init__()
        self.dim = dim
        self.heads = heads
        self.scale = (dim / heads) ** -0.5
        self.attend = nn.Softmax(dim=-1)

    def attend_with_rpe(self, Q, K):
        Q = rearrange(Q, "b i (heads d) -> b heads i d", heads=self.heads)
        K = rearrange(K, "b j (heads d) -> b heads j d", heads=self.heads)

        dots = (
            einsum("bhid, bhjd -> bhij", Q, K) * self.scale
        )  # (b hw) heads 1 pointnum

        return self.attend(dots)

    def forward(self, Q, K, V):
        attn = self.attend_with_rpe(Q, K)
        B, HW, _ = Q.shape

        V = rearrange(V, "b j (heads d) -> b heads j d", heads=self.heads)

        out = einsum("bhij, bhjd -> bhid", attn, V)
        out = rearrange(out, "b heads hw d -> b hw (heads d)", b=B, hw=HW)

        return out


class AttentionLayer(nn.Module):

    def __init__(self, cfg):
        super(AttentionLayer, self).__init__()
        self.cfg = cfg
        self.dim = cfg.dim
        self.norm1 = nn.LayerNorm(cfg.dim)
        self.norm2 = nn.LayerNorm(cfg.dim)
        self.proj = nn.Linear(cfg.dim, cfg.dim)
        self.pos = PositionalEncoding2D(cfg.dim)
        self.q, self.k, self.v = (
            nn.Linear(cfg.dim, cfg.dim),
            nn.Linear(cfg.dim, cfg.dim),
            nn.Linear(cfg.dim, cfg.dim),
        )
        self.att = MultiHeadAttention(cfg.dim, cfg.num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.dim, cfg.dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.dim, cfg.dim),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, query, key, value, memory, cov):
        if key is None and value is None:
            key, value = self.k(memory), self.v(memory)
            B, C, H, W = key.shape
            key = key.reshape(B, H * W * C // self.dim, self.dim)
            value = value.reshape(B, H * W * C // self.dim, self.dim)
        B, C, H, W = query.shape

        query = query.reshape(B, H * W * C // self.dim, self.dim)

        shortcut = query
        query = self.norm1(query)

        cov_pos_embed = self.pos(cov.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        cov = torch.cat([cov, cov_pos_embed], dim=1)
        B, C, H, W = cov.shape
        cov = cov.reshape(B, H * W * C // self.dim, self.dim)
        q = self.q(torch.cat([query, cov], dim=1))

        k, v = key, value

        x = self.att(q, k, v)

        x = self.proj(torch.cat([x, shortcut], dim=1))
        x = x + self.ffn(self.norm2(x))
        C = 4 * self.cfg.mixtures + 4
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        return x, k, v
