import torch
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Literal, get_args
from types import SimpleNamespace

from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize, center_crop

from .Interface import StereoFrame, StereoData, T_Data
from Utility.Extensions import ConfigTestableSubclass
from Utility.Config     import build_dynamic_config, DynamicConfigSpec


T_from = TypeVar("T_from")
T_to   = TypeVar("T_to")


class IDataTransform(Generic[T_from, T_to], ABC, ConfigTestableSubclass, torch.nn.Module):
    def __init__(self, config: SimpleNamespace | None | DynamicConfigSpec) -> None:
        super().__init__()
        if config is None:
            self.config = SimpleNamespace()
        elif isinstance(config, SimpleNamespace):
            self.config = config
        else: 
            self.config, _ = build_dynamic_config(config)
    
    @abstractmethod
    def forward(self, frame: T_from) -> T_to: ...


class NoTransform(IDataTransform[T_Data, T_Data]):
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        return

    def forward(self, frame: T_Data) -> T_Data:
        return frame


class ScaleFrame(IDataTransform[StereoFrame, StereoFrame]):
    """
    Scale the image & ground truths on u and v direction and modify the camera intrinsic accordingly.
    """
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "scale_u": lambda v: isinstance(v, (float, int)) and v > 0,
            "scale_v": lambda v: isinstance(v, (float, int)) and v > 0,
            "interp" : lambda v: v in {"nearest", "bilinear"}
        })
    
    @staticmethod
    def scale_stereo(data: StereoData, scale_u: float, scale_v: float, interpolate: Literal["nearest", "bilinear"]) -> StereoData:
        match interpolate:
            case "bilinear": interp = InterpolationMode.BILINEAR
            case "nearest" : interp = InterpolationMode.NEAREST_EXACT
        
        raw_height = data.height
        raw_width  = data.width
        
        target_h   = int(raw_height / scale_v)
        target_w   = int(raw_width  / scale_u)
        
        round_scale_v = raw_height / target_h
        round_scale_u = raw_width  / target_w
        
        data.K = data.K.clone()
        data.height = target_h
        data.width  = target_w
        data.K[:, 0] /= round_scale_u
        data.K[:, 1] /= round_scale_v
        
        data.imageL = resize(data.imageL, [target_h, target_w], interpolation=interp)
        data.imageR = resize(data.imageR, [target_h, target_w], interpolation=interp)
        
        if data.gt_flow is not None:
            data.gt_flow = resize(data.gt_flow, [target_h, target_w], interpolation=interp)
            data.gt_flow[:, 0] /= round_scale_u
            data.gt_flow[:, 1] /= round_scale_v
        
        if data.flow_mask is not None:
            data.flow_mask = resize(data.flow_mask, [target_h, target_w], interpolation=interp)
        
        if data.gt_depth is not None:
            data.gt_depth = resize(data.gt_depth, [target_h, target_w], interpolation=interp)
        
        return data

    def forward(self, frame: StereoFrame) -> StereoFrame:
        frame.stereo = self.scale_stereo(
            frame.stereo, scale_u=self.config.scale_u, scale_v=self.config.scale_v, interpolate=self.config.interp
        )
        return frame


class CenterCropFrame(IDataTransform[StereoFrame, StereoFrame]):
    """
    Center crop the image and modify ground truth & camera intrinsic accordingly.
    """
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "height": lambda v: isinstance(v, int) and v > 0,
            "width": lambda v: isinstance(v, int) and v > 0
        })
    
    @staticmethod
    def crop_stereo(data: StereoData, target_h: int, target_w: int) -> StereoData:
        orig_h, orig_w = data.height, data.width
        data.imageL = center_crop(data.imageL, [target_h, target_w])
        data.imageR = center_crop(data.imageR, [target_h, target_w])
        
        if data.gt_flow is not None:
            data.gt_flow   = center_crop(data.gt_flow, [target_h, target_w])
        if data.flow_mask is not None:
            data.flow_mask = center_crop(data.flow_mask, [target_h, target_w])
        if data.gt_depth is not None:
            data.gt_depth  = center_crop(data.gt_depth, [target_h, target_w])
        
        data.K = data.K.clone()
        data.height = target_h
        data.width  = target_w
        data.K[:, 0, 2] -= (orig_w - target_w) / 2.
        data.K[:, 1, 2] -= (orig_h - target_h) / 2.
        
        return data

    def forward(self, frame: StereoFrame) -> StereoFrame:
        frame.stereo = self.crop_stereo(
            frame.stereo, target_h=self.config.height, target_w=self.config.width
        )
        return frame


class AddImageNoise(IDataTransform[StereoFrame, StereoFrame]):
    """
    Add noise to image color. Note that the `stdv` is on scale of [0-255] image instead of
    on the scale of [0-1]. (That is, we will divide stdv by 255 when applying noise on image)
    """
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "stdv": lambda v: isinstance(v, (int, float)) and v > 0
        })
    
    def forward(self, frame: StereoFrame) -> StereoFrame:
        frame.stereo.imageL = (frame.stereo.imageL + (self.config.stdv / 255) * torch.randn_like(frame.stereo.imageL)).clamp(0.0, 1.0)
        frame.stereo.imageR = (frame.stereo.imageR + (self.config.stdv / 255) * torch.randn_like(frame.stereo.imageR)).clamp(0.0, 1.0)
        return frame


class CastDataType(IDataTransform[StereoFrame, StereoFrame]):
    T_SUPPORT = Literal["fp16", "fp32", "bf16"]
    def __init__(self, config) -> None:
        super().__init__(config)
        self.dtype: torch.dtype = self.cast_dtype(self.config.dtype)
    
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "dtype": lambda v: v in get_args(CastDataType.T_SUPPORT)
        })
    
    @staticmethod
    def cast_dtype(dtype: T_SUPPORT) -> torch.dtype:
        match dtype:
            case "bf16": return torch.bfloat16
            case "fp16": return torch.float16
            case "fp32": return torch.float32
    
    def forward(self, frame: StereoFrame) -> StereoFrame:
        frame.stereo.imageL = frame.stereo.imageL.to(dtype=self.dtype)
        frame.stereo.imageR = frame.stereo.imageR.to(dtype=self.dtype)
        if frame.stereo.gt_flow is not None  : frame.stereo.gt_flow   = frame.stereo.gt_flow.to(dtype=self.dtype)
        if frame.stereo.gt_depth is not None : frame.stereo.gt_depth  = frame.stereo.gt_depth.to(dtype=self.dtype)
        if frame.stereo.flow_mask is not None: frame.stereo.flow_mask = frame.stereo.flow_mask.to(dtype=self.dtype)
        return frame


class SmartResizeFrame(IDataTransform[StereoFrame, StereoFrame]):
    """
    Automatically resize and crop the frame to target height and width to
    maximize the fov of resulted frame while achieving target shape.
    
    This process will maintein the aspect ratio of the image (i.e. the image 
    will not be stretched)
    """
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "height": lambda v: isinstance(v, int) and v > 0,
            "width": lambda v: isinstance(v, int) and v > 0,
            "interp" : lambda v: v in {"nearest", "bilinear"},
        })
    
    def forward(self, frame: StereoFrame) -> StereoFrame:
        orig_height, orig_width = frame.stereo.height, frame.stereo.width
        targ_height, targ_width = self.config.height, self.config.width
        
        scale_factor = min(orig_height / targ_height, orig_width / targ_width)
        frame.stereo = ScaleFrame.scale_stereo(
            frame.stereo, scale_u=scale_factor, scale_v=scale_factor, interpolate=self.config.interp
        )
        frame.stereo = CenterCropFrame.crop_stereo(
            frame.stereo, target_h=targ_height, target_w=targ_width
        )
        
        return frame
