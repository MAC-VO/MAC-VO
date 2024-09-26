import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import AttentionLayer


def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def initialize_flow(img):
    """Flow is represented as difference between two means flow = mean1 - mean0"""
    N, C, H, W = img.shape
    mean = coords_grid(N, H, W).to(img.device)
    mean_init = coords_grid(N, H, W).to(img.device)

    # optical flow computed as difference: flow = mean1 - mean0
    return mean, mean_init


class GaussianGRU(nn.Module):

    def __init__(self, cfg):
        super(GaussianGRU, self).__init__()
        self.cfg = cfg
        self.iters = cfg.gru_iters
        self.kernel_size = cfg.kernel_size
        # downsample x2
        self.proj = nn.Sequential(
            nn.Conv2d(128, cfg.dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.mem_proj = nn.Conv2d(597, 64, 1, padding=0)
        self.att = AttentionLayer(cfg)
        self.gaussian = GaussianUpdateBlock(cfg, hidden_dim=cfg.dim)

    def upsample_flow(self, flow, mask):
        N, C, H, W = flow.shape
        mask = mask.view(N, 1, self.kernel_size * self.kernel_size, 8, 8, H, W)
        mask = F.softmax(mask, dim=2)
        hr_key = F.unfold(
            self.kernel_size * flow,
            (self.kernel_size, self.kernel_size),
            padding=self.kernel_size // 2,
        )
        hr_key = hr_key.view(N, C, self.kernel_size * self.kernel_size, 1, 1, H, W)
        up_key = torch.sum(mask * hr_key, dim=2)
        up_key = up_key.permute(0, 1, 4, 2, 5, 3)
        return up_key.reshape(N, C, 8 * H, 8 * W)

    def forward(self, context, memory, cost_map):
        cov_preds = []
        memory = self.mem_proj(memory)
        memory = memory.permute(0, 2, 3, 1)
        covs0, covs1 = initialize_flow(context)
        covs0 = covs0.repeat(1, self.cfg.mixtures, 1, 1)
        covs1 = covs1.repeat(1, self.cfg.mixtures, 1, 1)
        context = self.proj(context)
        net, inp = torch.split(context, [self.cfg.dim, self.cfg.dim], dim=1)
        net = torch.tanh(net)
        inp = torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)(inp)

        key, value = None, None

        for _ in range(self.iters):
            cost, key, value = self.att(cost_map, key, value, memory, covs1)
            corr = torch.cat([cost, cost_map], dim=1)  # C:4*mixtures+6
            cov = covs1 - covs0
            net, delta_covs, up_mask = self.gaussian(net, inp, corr, cov)
            covs1 = covs0 + delta_covs
            cov_up = self.upsample_flow(covs1 - covs0, up_mask)
            cov_preds.append(cov_up)
        return cov_preds


class GaussianHead(nn.Module):

    def __init__(self, input_dim=128, hidden_dim=256, mixtures=9):
        super(GaussianHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2 * mixtures, 3, padding=1)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class SepConvGRU(nn.Module):

    def __init__(self, hidden_dim=128, input_dim=192 + 128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convr1 = nn.Conv2d(input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(input_dim, hidden_dim, (1, 5), padding=(0, 2))

        self.convz2 = nn.Conv2d(input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convr2 = nn.Conv2d(input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(input_dim, hidden_dim, (5, 1), padding=(2, 0))

    def forward(self, h, x):
        # horizontal

        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        return h


class GaussianEncoder(nn.Module):

    def __init__(self, cfg):
        super(GaussianEncoder, self).__init__()

        self.convc1 = nn.Conv2d(4 * cfg.mixtures + 6, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2 * cfg.mixtures, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64 + 192, 128 - 2, 3, padding=1)

    def forward(self, cov, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(cov))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = self.conv(cor_flo)
        return torch.cat([out, cov], dim=1)  # 126+2*mixtures


class GaussianUpdateBlock(nn.Module):

    def __init__(self, cfg, hidden_dim=128):
        super().__init__()
        self.cfg = cfg
        self.encoder = GaussianEncoder(cfg)
        self.gaussian = SepConvGRU(
            hidden_dim=hidden_dim, input_dim=126 + 2 * cfg.dim + 2 * cfg.mixtures
        )
        self.gaussian_head = GaussianHead(
            hidden_dim, hidden_dim=hidden_dim, mixtures=cfg.mixtures
        )

        self.mask = nn.Sequential(
            nn.Conv2d(cfg.dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0),
        )

    def forward(self, net, inp, corr, cov):
        features = self.encoder(cov, corr)
        inp = torch.cat([inp, features], dim=1)  # C:126+2*mixtures+dim

        net = self.gaussian(net, inp)  # (B,128,H,W)
        delta_covs = self.gaussian_head(net)
        up_mask = 0.25 * self.mask(net)
        return net, delta_covs, up_mask
