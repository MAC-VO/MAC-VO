import math
import torch
import torch.nn as nn
from collections import OrderedDict

from .pwc_cov.gru import GaussianGRU
from .pwc.correlation import FunctionCorrelation
from .pwc.pwc_model import PWCDCNet


class PWCFeature(PWCDCNet):
    def __init__(self, md=4, flow_norm=20.0):
        super(PWCFeature, self).__init__(md=md, flow_norm=flow_norm)
        self.memory = None
        self.context = None
        self.costMap = None
        self.feature = None
        self.flow_norm = flow_norm

    def encoder(self, im1, im2):
        c11 = self.conv1b(self.conv1aa(self.conv1a(im1)))
        c21 = self.conv1b(self.conv1aa(self.conv1a(im2)))
        c12 = self.conv2b(self.conv2aa(self.conv2a(c11)))
        c22 = self.conv2b(self.conv2aa(self.conv2a(c21)))
        c13 = self.conv3b(self.conv3aa(self.conv3a(c12)))
        c23 = self.conv3b(self.conv3aa(self.conv3a(c22)))
        c14 = self.conv4b(self.conv4aa(self.conv4a(c13)))
        c24 = self.conv4b(self.conv4aa(self.conv4a(c23)))
        c15 = self.conv5b(self.conv5aa(self.conv5a(c14)))
        c25 = self.conv5b(self.conv5aa(self.conv5a(c24)))
        c16 = self.conv6b(self.conv6a(self.conv6aa(c15)))
        c26 = self.conv6b(self.conv6a(self.conv6aa(c25)))

        return c12, c22, c13, c23, c14, c24, c15, c25, c26, c16

    def forward(self, x, feature=None):
        im1 = x[0]
        im2 = x[1]
        if feature is None:
            c12, c22, c13, c23, c14, c24, c15, c25, c26, c16 = self.encoder(im1, im2)
            self.feature = [c12, c22, c13, c23, c14, c24, c15, c25, c26, c16]
        else:
            c12, c22, c13, c23, c14, c24, c15, c25, c26, c16 = feature
        # corr6 = self.corr(c16, c26)
        corr6 = FunctionCorrelation(tenFirst=c16, tenSecond=c26)

        corr6 = self.leakyRELU(corr6)

        x = torch.cat((self.conv6_0(corr6), corr6), 1)
        x = torch.cat((self.conv6_1(x), x), 1)
        x = torch.cat((self.conv6_2(x), x), 1)
        x = torch.cat((self.conv6_3(x), x), 1)
        x = torch.cat((self.conv6_4(x), x), 1)
        flow6 = self.predict_flow6(x)
        up_flow6 = self.deconv6(flow6)
        up_feat6 = self.upfeat6(x)

        warp5 = self.warp(c25, up_flow6 * 0.625)
        # corr5 = self.corr(c15, warp5)
        corr5 = FunctionCorrelation(tenFirst=c15, tenSecond=warp5)
        corr5 = self.leakyRELU(corr5)
        x = torch.cat((corr5, c15, up_flow6, up_feat6), 1)
        x = torch.cat((self.conv5_0(x), x), 1)
        x = torch.cat((self.conv5_1(x), x), 1)
        x = torch.cat((self.conv5_2(x), x), 1)
        x = torch.cat((self.conv5_3(x), x), 1)
        x = torch.cat((self.conv5_4(x), x), 1)
        flow5 = self.predict_flow5(x)
        up_flow5 = self.deconv5(flow5)
        up_feat5 = self.upfeat5(x)

        warp4 = self.warp(c24, up_flow5 * 1.25)  # 1.25
        # corr4 = self.corr(c14, warp4)
        corr4 = FunctionCorrelation(tenFirst=c14, tenSecond=warp4)
        corr4 = self.leakyRELU(corr4)
        x = torch.cat((corr4, c14, up_flow5, up_feat5), 1)
        x = torch.cat((self.conv4_0(x), x), 1)
        x = torch.cat((self.conv4_1(x), x), 1)
        x = torch.cat((self.conv4_2(x), x), 1)
        x = torch.cat((self.conv4_3(x), x), 1)
        x = torch.cat((self.conv4_4(x), x), 1)
        flow4 = self.predict_flow4(x)
        up_flow4 = self.deconv4(flow4)
        up_feat4 = self.upfeat4(x)

        warp3 = self.warp(c23, up_flow4 * 2.5)  # 2.5
        # corr3 = self.corr(c13, warp3)
        corr3 = FunctionCorrelation(tenFirst=c13, tenSecond=warp3)
        corr3 = self.leakyRELU(corr3)

        x = torch.cat((corr3, c13, up_flow4, up_feat4), 1)
        x = torch.cat((self.conv3_0(x), x), 1)
        x = torch.cat((self.conv3_1(x), x), 1)
        x = torch.cat((self.conv3_2(x), x), 1)
        x = torch.cat((self.conv3_3(x), x), 1)
        x = torch.cat((self.conv3_4(x), x), 1)
        flow3 = self.predict_flow3(x)
        up_flow3 = self.deconv3(flow3)
        up_feat3 = self.upfeat3(x)

        self.costMap = flow3
        self.memory = x
        self.context = torch.cat([c13, c23], dim=1)

        warp2 = self.warp(c22, up_flow3 * 5.0)
        # corr2 = self.corr(c12, warp2)
        corr2 = FunctionCorrelation(tenFirst=c12, tenSecond=warp2)
        corr2 = self.leakyRELU(corr2)
        x = torch.cat((corr2, c12, up_flow3, up_feat3), 1)
        x = torch.cat((self.conv2_0(x), x), 1)
        x = torch.cat((self.conv2_1(x), x), 1)
        x = torch.cat((self.conv2_2(x), x), 1)
        x = torch.cat((self.conv2_3(x), x), 1)
        x = torch.cat((self.conv2_4(x), x), 1)
        flow2 = self.predict_flow2(x)

        x = self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x))))
        flow2 = flow2 + self.dc_conv7(self.dc_conv6(self.dc_conv5(x)))

        return flow2 * self.flow_norm, self.context, self.memory, self.costMap


class RAFTFlowCovNet(nn.Module):
    def __init__(self, args, device: str):
        super(RAFTFlowCovNet, self).__init__()
        self.args = args
        self.device = device
        self.feature = PWCFeature()
        self.netGaussian = GaussianGRU(args)

    def forward(self, first: torch.Tensor, second: torch.Tensor):
        assert first.is_contiguous() and second.is_contiguous()
        assert (first.size(2) % 64 == 0) and (first.size(3) % 64 == 0)
        assert (second.size(2) % 64 == 0) and (second.size(3) % 64 == 0)

        flow, context, memory, costmap = self.feature((first, second))
        cov = self.netGaussian(context, memory, costmap)

        return flow, cov

    def load_ddp_state_dict(self, ckpt: OrderedDict) -> None:
        new_ckpt = OrderedDict()
        for key in ckpt:
            if key.startswith("module."):
                new_ckpt[key[7:]] = ckpt[key]
            else:
                new_ckpt[key] = ckpt[key]
        self.load_state_dict(new_ckpt)

    @torch.no_grad()
    @torch.inference_mode()
    def inference(self, first: torch.Tensor, second: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pre_im0, shape0 = self.preprocess(first)
        pre_im1, _ = self.preprocess(second)
        
        flow, flow_covs = self(pre_im0.to(self.device), pre_im1.to(self.device))
        
        flow_covs = [
            cov.view(cov.shape[0], 2, cov.shape[1] // 2, cov.shape[2], cov.shape[3])
            for cov in flow_covs
        ]
        cov_preds = [cov.mean(dim=2) for cov in flow_covs]
        flow_cov = cov_preds[-1]
        
        post_flow = self.postprocess(flow, shape0, isflow=True)
        post_flow_cov = (self.postprocess(flow_cov, shape0, isflow=False) * 2).exp()
        
        return post_flow, post_flow_cov

    @staticmethod
    def preprocess(img):
        _, _, H, W = img.shape
        W_ = int(math.floor(math.ceil(W / 64.0) * 64.0))
        H_ = int(math.floor(math.ceil(H / 64.0) * 64.0))
        img = torch.nn.functional.interpolate(input=img,
                            size=(H_, W_),
                            mode='bilinear',
                            align_corners=False)

        return img, [W, H, W_, H_]

    @staticmethod
    def postprocess(img, shape, isflow=False):
        W, H, W_, H_ = shape
        img = torch.nn.functional.interpolate(img, size=(H, W), mode='bilinear', align_corners=False)
        if isflow:
            img[:, 0, :, :] *= float(W) / float(W_)
            img[:, 1, :, :] *= float(H) / float(H_)
        return img
