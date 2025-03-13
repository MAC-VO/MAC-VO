from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from .Utility import (
    Compose, DownscaleFlow, Normalize, ToTensor, CropCenter,
    make_device_intrinsic_layer, make_intrinsics_layer
)
from DataLoader import StereoData, StereoFrame
from Utility.Utils import centerCropTo


class TartanStereoVONetInterface:
    """
    Description
    ---
    An interface class used to build connection between MAC-VO and TartanVO codebase.
    """

    def __init__(self, weight: Path | str, eval_mode: bool, device: str = "cuda"):
        assert eval_mode is not None

        from .StereoVO import StereoVONet
        self.device = device
        self.model = StereoVONet(
            flowNormFactor=1.0, stereoNormFactor=0.02, poseDepthNormFactor=0.25
        )
        self.transform = Compose(
            [
                # Utility.CropCenter((448, 640), fix_ratio=False, scale_w=1.0, scale_disp=False),
                DownscaleFlow(scale=4),
                Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    rgbbgr=False,
                    keep_old=True,
                ),
                ToTensor(),
            ]
        )
        self.loadWeight(weight)
        self.model = self.model.to(device)
        if eval_mode:
            self.model = self.model.eval()
        elif (not eval_mode) and (eval_mode is not None):
            self.model = self.model.train()
        else:
            raise ValueError(f"Receive {eval_mode}, expected to be [True | False]")
        self.pose_norm = torch.tensor(
            [0.13, 0.13, 0.13, 0.013, 0.013, 0.013], dtype=torch.float, device=device
        )
        self.flow_norm = 0.05

    def loadWeight(self, weight: Path | str):
        state_dict = torch.load(weight, map_location="cpu", weights_only=True)
        converted_dict = OrderedDict()
        for key in state_dict:
            if key.startswith("module."):
                converted_dict[key[7:]] = state_dict[key]
        self.model.load_state_dict(converted_dict)

    @staticmethod
    def frame2Sample(
        meta: StereoData, imageL: torch.Tensor, imageR: torch.Tensor | None
    ):
        """
        Adapt the SourceDataFrame used in this project into the "sample" format used in TartanVO
        """
        assert (
            imageL.size(0) == 1
        ), "The interface will not handle batch dimension correctly."

        if imageR is not None:
            return {
                "img0": [imageL.squeeze().permute(1, 2, 0).cpu().numpy()],
                "img1": [imageR.squeeze().permute(1, 2, 0).cpu().numpy()],
                "intrinsic": [
                    make_intrinsics_layer(
                        int(meta.cx * 2),
                        int(meta.cy * 2),
                        meta.fx,
                        meta.fy,
                        meta.cx,
                        meta.cy,
                    )
                ],
                "blxfx": np.array([meta.fx * meta.frame_baseline], dtype=np.float32),
            }
        else:
            return {
                "img0": [imageL.squeeze().permute(1, 2, 0).numpy()],
                "intrinsic": [
                    make_intrinsics_layer(
                        int(meta.cx * 2),
                        int(meta.cy * 2),
                        meta.fx,
                        meta.fy,
                        meta.cx,
                        meta.cy,
                    )
                ],
                "blxfx": np.array([meta.fx * meta.frame_baseline], dtype=np.float32),
            }

    @staticmethod
    def getCropMargin(shape: torch.Size):
        h, w = shape[-2], shape[-1]
        h64, w64 = (h // 64) * 64, (w // 64) * 64

        # Well, I assumed that the image's h/w must be even number, if not
        # the margin here will have off-by-one error.
        h_margin, w_margin = (h - h64) // 2, (w - w64) // 2
        return (h_margin, w_margin), (h64, w64)

    @staticmethod
    def getCropMarginTo(shape: torch.Size, target_shape: tuple[int, int]):
        h, w = shape[-2], shape[-1]
        h_margin, w_margin = (h - target_shape[0]) // 2, (w - target_shape[1]) // 2
        return (h_margin, w_margin), target_shape


class TartanStereoVOMatch(TartanStereoVONetInterface):
    @torch.no_grad()
    @torch.inference_mode()
    def inference(self, meta: StereoData, img0: torch.Tensor, img1: torch.Tensor) -> torch.Tensor:
        margin, crop_size = self.getCropMargin(img0.shape)

        # Preprocess
        cropcenter_transform = CropCenter(
            crop_size, fix_ratio=False, scale_w=1.0, scale_disp=False
        )

        sample0 = self.frame2Sample(meta, img0, None)
        sample1 = self.frame2Sample(meta, img1, None)

        sample0 = self.transform(cropcenter_transform(sample0))
        sample1 = self.transform(cropcenter_transform(sample1))
        img0_flow, img1_flow = sample0["img0"].cuda(), sample1["img0"].cuda()

        #
        flow_output = self.model.forward_flow(img0_flow, img1_flow)
        flow_output = flow_output / self.flow_norm
        
        # Postprocess
        flow_output_resized = torch.nn.functional.interpolate(
            flow_output, scale_factor=4.0, mode="nearest"
        )

        return flow_output_resized[0]


class TartanStereoVOMotion(TartanStereoVONetInterface):
    def __init__(self, weight: Path, eval_mode: bool, device: str = "cuda"):
        super().__init__(weight, eval_mode, device)
    
    @staticmethod
    def cropAndResize(x: torch.Tensor, target_shape: tuple[int, int]) -> torch.Tensor:
        orig_height, orig_width = x.shape[-2], x.shape[-1]
        targ_height, targ_width = target_shape
        
        # Calculate scaing factor
        scale_factor = min(int(orig_height / targ_height), int(orig_width / targ_width))
        
        crop_targ_height, crop_targ_width = targ_height * scale_factor, targ_width * scale_factor
        x = centerCropTo(x, [crop_targ_height, crop_targ_width], [-2, -1])
        
        # Scale down
        x = torch.nn.functional.interpolate(x, size=(targ_height, targ_width), mode='bilinear', align_corners=True)
        return x
    
    @torch.inference_mode()
    def inference(self, frame0: StereoFrame, flow: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:        
        meta = frame0.stereo
        tensor_intrinsic = make_device_intrinsic_layer(
            meta.height, meta.width, meta.fx, meta.fy, meta.cx, meta.cy, torch.device(self.device)
        ).unsqueeze(0).permute(0, 3, 1, 2)
        
        # (112, 160) is the size of feature map received by the PoseNet in original TartanVO paper.
        # See paper at https://arxiv.org/abs/2011.00359
        tensor_intrinsic_resize = self.cropAndResize(tensor_intrinsic, (112, 160))
        depth_resize = self.cropAndResize(depth, (112, 160))
        flow_resize  = self.cropAndResize(flow, (112, 160)) * self.flow_norm
        
        stereo = (meta.frame_baseline * meta.fx) / depth_resize
        stereo = torch.nan_to_num(stereo * self.model.stereoNormFactor, nan=0.0).clamp(min=0.0)

        depth_resize = stereo / (meta.frame_baseline * meta.fx) / float(self.model.stereoNormFactor * self.model.poseDepthNormFactor)

        inputTensor = torch.cat((flow_resize, depth_resize, tensor_intrinsic_resize), dim=1).to(self.device)
        pose = self.model.flowPoseNet(inputTensor, scale_disp=1.0)
        
        return pose.squeeze() * self.pose_norm
