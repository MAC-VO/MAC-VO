import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as v2
from collections import OrderedDict

from .decoder import HourglassDecoder
from .StereoNet import StereoNet7

from DataLoader import StereoData

torchvision.disable_beta_transforms_warning()


class StereoFeature(nn.Module):

    def __init__(self, normalize_factor=0.02):
        super(StereoFeature, self).__init__()
        self.stereo = StereoNet7()
        self.normalize_factor = normalize_factor

    @torch.no_grad()
    @staticmethod
    def getCropMargin(shape: torch.Size):
        h, w = shape[-2], shape[-1]
        h64, w64 = (h // 64) * 64, (w // 64) * 64

        # Well, I assumed that the image's h/w must be even number, if not
        # the margin here will have off-by-one error.
        h_margin, w_margin = (h - h64) // 2, (w - w64) // 2
        return (h_margin, w_margin), (h64, w64)

    def forward(self, tenOne, tenTwo):
        _, cropsize = StereoFeature.getCropMargin(tenOne.shape)
        transform = v2.Compose(
            [
                v2.CenterCrop(cropsize),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )
        tenOne, tenTwo = transform(tenOne).cuda(), transform(tenTwo).cuda()
        stereo, hourglass_context, cats = self.stereo(
            torch.cat((tenOne, tenTwo), dim=1)
        )
        return (stereo / self.normalize_factor,hourglass_context,cats)


class StereoCovNet(nn.Module):

    def __init__(self, cfg):
        super(StereoCovNet, self).__init__()
        self.cfg = cfg
        self.feature = StereoFeature()
        self.decoder = HourglassDecoder(exp=cfg.exp)

    def forward(self, tenOne, tenTwo):
        stereo, hourglass_context, cats = self.feature(tenOne, tenTwo)
        cov_preds = self.decoder(hourglass_context, cats)
        return stereo, cov_preds

    @torch.no_grad()
    @torch.inference_mode()
    def inference(self, stereo: StereoData):
        disparity, disparity_cov = self(stereo.imageL, stereo.imageR)
        depth = ((stereo.frame_baseline * stereo.fx) / disparity)

        disparity_2 = disparity.square()
        error_rate_2 = disparity_cov / disparity_2
        depth_var = ((stereo.frame_baseline * stereo.fx) ** 2) * (error_rate_2 / disparity_2)

        return depth, depth_var

    def load_ddp_state_dict(self, ckpt):
        cvt_ckpt = OrderedDict()
        for k in ckpt:
            if k.startswith("module."):
                cvt_ckpt[k[7:]] = ckpt[k]
            else:
                cvt_ckpt[k] = ckpt[k]
        self.load_state_dict(cvt_ckpt)
