from __future__ import annotations

import torch
from types import SimpleNamespace
from typing import TypeVar, Generic, TypedDict, overload
from abc import ABC, abstractmethod

from DataLoader import SourceDataFrame
from Utility.Utils import padTo
from Utility.Extensions import ConfigTestableSubclass

# Stereo Depth interface ###
# T_Context = The internal state of stereo depth estimator
T_Context = TypeVar("T_Context")


class IStereoDepth(ABC, Generic[T_Context], ConfigTestableSubclass):
    """
    Estimate dense depth map of current stereo image.
    
    `IStereoDepth(frame: SourceDataFrame) -> depth, depth_cov`

    Given a frame with imageL, imageR being Bx3xHxW, return `output` where    

    * depth         - Bx1xHxW shaped torch.Tensor, estimated depth map
                    maybe padded with `nan` if model can't output prediction with same shape as input image.
    * depth_cov     - Bx1xHxW shaped torch.Tensor or None, estimated covariance of depth map (if provided)
                    maybe padded with `nan` if model can't output prediction with same shape as input image.
    """
    def __init__(self, config: SimpleNamespace):
        self.config : SimpleNamespace = config
        self.context: T_Context       = self.init_context()
    
    @property
    @abstractmethod
    def provide_cov(self) -> bool: ...
    
    @abstractmethod
    def init_context(self) -> T_Context: ...
    
    @abstractmethod
    def estimate(self, frame: SourceDataFrame) -> tuple[torch.Tensor, torch.Tensor | None]: ...
    
    def __call__(self, frame: SourceDataFrame) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.estimate(frame)

    @overload
    @staticmethod
    def retrieve_pixels(pixel_uv: torch.Tensor, scalar_map: torch.Tensor, interpolate: bool=False) -> torch.Tensor: ...
    @overload
    @staticmethod
    def retrieve_pixels(pixel_uv: torch.Tensor, scalar_map: None, interpolate: bool=False) -> None: ...

    @staticmethod
    def retrieve_pixels(pixel_uv: torch.Tensor, scalar_map: torch.Tensor | None, interpolate: bool=False) -> torch.Tensor | None:
        """
        Given a pixel_uv (Nx2) tensor, retrieve the pixel values (1, N) from scalar_map (Bx1xHxW).
        
        #### Note that the pixel_uv is in (x, y) format, not (row, col) format.
        
        #### Note that only first sample of scalar_map is used. (Batch idx=0)
        """
        if scalar_map is None: return None
        
        if interpolate:
            raise NotImplementedError("Not implemented yet")
        else:
            values = scalar_map[0, ..., pixel_uv[..., 1].long(), pixel_uv[..., 0].long()]
            return values

# End #######################


# Stereo Depth Implementation ###
# Contexts

class ModelContext(TypedDict):
    model: torch.nn.Module

# Implementations

class GTDepth(IStereoDepth[None]):
    """
    Always returns the ground truth depth. input frame must have `gtDepth` attribute non-empty.
    """
    @property
    def provide_cov(self) -> bool: return False
    
    def init_context(self) -> None: return None
    
    def estimate(self, frame: SourceDataFrame) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert frame.gtDepth is not None
        gt_depthmap = padTo(frame.gtDepth, (frame.meta.height, frame.meta.width), dim=(-2, -1), value=float('nan'))
        
        return gt_depthmap, None

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None: return


class FlowFormerDepth(IStereoDepth[ModelContext]):
    """
    Use FlowFormer to estimate disparity between rectified stereo image. Does not generate depth_cov.
    
    See FlowFormerCovDepth for jointly estimating depth and depth_cov
    """
    @property
    def provide_cov(self) -> bool: return False
    
    def init_context(self) -> ModelContext:
        from ..Network.FlowFormer.configs.submission import get_cfg
        from ..Network.FlowFormer.core.FlowFormer import build_flowformer
        model = build_flowformer(get_cfg(), self.config.device)
        ckpt  = torch.load(self.config.weight, weights_only=True)
        model.load_ddp_state_dict(ckpt)
        model.to(self.config.device)
        
        model.eval()
        return ModelContext(model=model)
        
    @torch.inference_mode()
    def estimate(self, frame: SourceDataFrame) -> tuple[torch.Tensor, torch.Tensor | None]:
        est_flow, _ = self.context["model"].inference(frame.imageL, frame.imageR)
        disparity = est_flow[:1].abs().unsqueeze(0)
        depth_map = ((frame.meta.baseline * frame.meta.fx) / disparity)
        
        return depth_map, None
    
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
                "weight"    : lambda s: isinstance(s, str),
                "device"    : lambda s: isinstance(s, str) and (("cuda" in s) or (s == "cpu")),
            })


class FlowFormerCovDepth(IStereoDepth[ModelContext]):
    """
    Use modified FlowFormer to estimate diparity between rectified stereo image and uncertainty of disparity.
    """
    @property
    def provide_cov(self) -> bool: return True
    
    def init_context(self) -> ModelContext:
        from ..Network.FlowFormer.configs.submission import get_cfg
        from ..Network.FlowFormerCov import build_flowformer
        model = build_flowformer(get_cfg(), self.config.device)
        ckpt  = torch.load(self.config.weight)
        model.load_ddp_state_dict(ckpt)
        model.to(self.config.device)
        
        model.eval()
        return ModelContext(model=model)
        
    @torch.inference_mode()
    def estimate(self, frame: SourceDataFrame) -> tuple[torch.Tensor, torch.Tensor | None]:        
        est_flow, est_cov = self.context["model"].inference(frame.imageL, frame.imageR)
        disparity, disparity_cov = est_flow[:, :1].abs(), est_cov[:, :1]
        
        # Propagate disparity covariance to depth covariance
        # See Appendix A.1 of paper
        disparity_2 = disparity.square()
        error_rate_2 = disparity_cov / disparity_2
        depth_map = ((frame.meta.baseline * frame.meta.fx) / disparity)
        depth_cov = (((frame.meta.baseline * frame.meta.fx) ** 2) * (error_rate_2 / disparity_2))
        
        return depth_map, depth_cov

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
                "weight"    : lambda s: isinstance(s, str),
                "device"    : lambda s: isinstance(s, str) and (("cuda" in s) or (s == "cpu")),
            })


class TartanVODepth(IStereoDepth[ModelContext]):
    """
    Use the StereoNet from TartanVO to estimate diparity between stereo image. 
    
    Does not estimate depth_cov if config.cov_mode set to 'None'.
    where config.cov_mode = {'None', 'Est'}
    """
    @property
    def provide_cov(self) -> bool: return self.config.cov_mode == "Est"
    
    def init_context(self) -> ModelContext:
        from Utility.Config import build_dynamic_config
        from Module.Network.StereoCov import StereoCovNet
        
        cfg, _ = build_dynamic_config({"exp": False, "decoder": "hourglass"})
        model = StereoCovNet(cfg)
        ckpt = torch.load(self.config.weight, weights_only=True)
        model.load_ddp_state_dict(ckpt)
        model.to(self.config.device)
        
        model.eval()
        return ModelContext(model=model)
        
    @torch.inference_mode()
    def estimate(self, frame: SourceDataFrame) -> tuple[torch.Tensor, torch.Tensor | None]:        
        imgL_cu, imgR_cu = frame.imageL.to(self.config.device), frame.imageR.to(self.config.device)
        depth, depth_cov = self.context["model"].inference(frame.meta, imgL_cu, imgR_cu)
        
        depth_map = padTo(depth, (frame.meta.height, frame.meta.width), dim=(-2, -1), value=float('nan'))
        
        if self.config.cov_mode == "Est":
            depth_cov = padTo(depth_cov, (frame.meta.height, frame.meta.width), dim=(-2, -1), value=float('nan'))
            return depth_map, depth_cov
        else:
            return depth_map, None
        
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
                "weight"    : lambda s: isinstance(s, str),
                "device"    : lambda s: isinstance(s, str) and (("cuda" in s) or (s == "cpu")),
                "cov_mode"  : lambda s: s in {"Est", "None"}
            })


class ApplyGTCov(IStereoDepth[IStereoDepth]):
    """
    A higher-order-module that encapsulates a IStereoDepth module. 
    
    Always compare the estimated output of encapsulated IStereoDepth with ground truth depth and convert
    error in estimation to 'estimated' covariance.
    
    Will raise AssertionError if frame does not have gtDepth.
    """
    @property
    def provide_cov(self) -> bool: return True
    
    def init_context(self) -> IStereoDepth:
        internal_module = IStereoDepth.instantiate(self.config.module.name, self.config.module.args)
        return internal_module
    
    @torch.inference_mode()
    def estimate(self, frame: SourceDataFrame) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert frame.gtDepth is not None
        
        est_depth, _ = self.context(frame)
        error = (frame.gtDepth - est_depth).abs()
        gt_cov = error.square()
        
        return est_depth, gt_cov

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        assert config is not None
        IStereoDepth.is_valid_config(config.module)
