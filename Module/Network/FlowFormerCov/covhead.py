import torch
import torch.nn as nn

from ..FlowFormer.core.FlowFormer.LatentCostFormer.decoder import (
    MemoryDecoder,
    initialize_flow,
)
from ..FlowFormer.core.FlowFormer.LatentCostFormer.gru import SepConvGRU


class CovHead(nn.Module):

    def __init__(self, input_dim=128, hidden_dim=256):
        super(CovHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim // 2, hidden_dim // 4, 3, padding=1)
        self.conv4 = nn.Conv2d(hidden_dim // 4, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv2(self.relu(self.conv1(x)))
        x = self.conv4(self.relu(self.conv3(x)))
        return x


class CovUpdateBlock(nn.Module):

    def __init__(self, args, hidden_dim=128):
        super().__init__()
        self.args = args
        self.gru = SepConvGRU(
            hidden_dim=hidden_dim, input_dim=128 + hidden_dim + hidden_dim
        )
        self.cov_head = CovHead(hidden_dim, hidden_dim=256)
        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0),
        )

    def forward(self, covs_net, inp_cat):
        covs_net = self.gru(covs_net, inp_cat)
        delta_covs = self.cov_head(covs_net)
        mask = 0.25 * self.mask(covs_net)
        return covs_net, delta_covs, mask


class MemoryCovDecoder(MemoryDecoder):
    def __init__(self, cfg):
        super(MemoryCovDecoder, self).__init__(cfg)
        self.cov_update = CovUpdateBlock(self.cfg, hidden_dim=128)

    def forward(self, cost_memory, context, data, flow_init=None, mode="cov"):  # pyright: ignore[reportIncompatibleMethodOverride]
        """
        memory: [B*H1*W1, H2'*W2', C]
        context: [B, D, H1, W1]
        """
        cost_maps = data["cost_maps"]
        coords0, coords1 = initialize_flow(context)
        cov0, cov1       = initialize_flow(context)
        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        cov_predictions = []

        context = self.proj(context)
        net, inp = torch.split(context, [128, 128], dim=1)
        net = torch.tanh(net)
        cov_net = net.clone()
        inp = torch.relu(inp)
        attention = None
        if self.cfg.gma:
            attention = self.att(inp)

        size = net.shape
        key, value = None, None

        for _ in range(self.depth):
            coords1 = coords1.detach()
            cost_forward = self.encode_flow_token(cost_maps, coords1)

            query = self.flow_token_encoder(cost_forward)
            query = (
                query.permute(0, 2, 3, 1)
                .contiguous()
                .view(size[0] * size[2] * size[3], 1, self.dim)
            )
            cost_global, key, value = self.decoder_layer(
                query, key, value, cost_memory, coords1, size, self.cfg.query_latent_dim
            )
            if self.cfg.only_global:
                corr = cost_global
            else:
                corr = torch.cat([cost_global, cost_forward], dim=1)

            flow = coords1 - coords0

            if self.cfg.gma:
                motion_feat = self.update_block.encoder(flow, corr)
                motion_feat_global = self.update_block.aggregator(
                    attention, motion_feat
                )
                inp_cat = torch.cat([inp, motion_feat, motion_feat_global], dim=1)
                net = self.update_block.gru(net, inp_cat)
                delta_flow = self.update_block.flow_head(net)
                up_mask = 0.25 * self.update_block.mask(net)

            else:
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)
                inp_cat = None

            cov_net, delta_cov, mask = self.cov_update(cov_net, inp_cat)
            cov1 = cov1 + delta_cov
            coords1 = coords1 + delta_flow
            flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            flow_predictions.append(flow_up)
            cov_up = self.upsample_flow(cov1 - cov0, mask)
            cov_predictions.append(cov_up)
        if self.training:
            return flow_predictions, cov_predictions
        else:
            return (flow_predictions[-1], coords1 - coords0), (cov_predictions[-1], cov1 - cov0)
