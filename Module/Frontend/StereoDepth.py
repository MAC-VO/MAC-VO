from __future__ import annotations

import torch
import jaxtyping as Jt
from typeguard import typechecked
from types import SimpleNamespace
from typing import overload
from abc import ABC, abstractmethod
from dataclasses import dataclass

from DataLoader import StereoData
from Utility.Utils import padTo, reflect_torch_dtype
from Utility.Extensions import ConfigTestableSubclass, OnCallCompiler

# Stereo Depth interface ###


class IStereoDepth(ABC, ConfigTestableSubclass):
    """
    Estimate dense depth map of current stereo image.
    
    `IStereoDepth.estimate(frame: StereoData) -> IStereoDepth.Output`

    Given a frame with imageL, imageR being Bx3xHxW, return `output` where    

    * depth         - Bx1xHxW shaped torch.Tensor, estimated depth map
                    maybe padded with `nan` if model can't output prediction with same shape as input image.
    * cov           - Bx1xHxW shaped torch.Tensor or None, estimated covariance of depth map (if provided)
                    maybe padded with `nan` if model can't output prediction with same shape as input image.
    * mask          - Bx1xHxW shaped torch.Tensor or None, the position with `True` value means valid (not occluded) 
                    prediction regions.
    """
    @Jt.jaxtyped(typechecker=typechecked)
    @dataclass
    class Output:
        depth    : Jt.Float32[torch.Tensor, "B 1 H W"]                
        disparity: Jt.Float32[torch.Tensor, "B 1 H W"] | None = None  # None if not applicable  
        cov      : Jt.Float32[torch.Tensor, "B 1 H W"] | None = None  # None if not applicable
        mask     : Jt.Bool   [torch.Tensor, "B 1 H W"] | None = None  # None if not applicable
        disparity_uncertainty: Jt.Float32[torch.Tensor, "B 1 H W"] | None = None
    
    def __init__(self, config: SimpleNamespace):
        self.config : SimpleNamespace = config
    
    @property
    @abstractmethod
    def provide_cov(self) -> bool: ...
    
    @abstractmethod
    def estimate(self, frame: StereoData) -> Output: ...

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


class GTDepth(IStereoDepth):
    """
    Always returns the ground truth depth. input frame must have `gtDepth` attribute non-empty.
    """
    @property
    def provide_cov(self) -> bool: return False
    
    def estimate(self, frame: StereoData) -> IStereoDepth.Output:
        assert frame.gt_depth is not None
        gt_depthmap = padTo(frame.gt_depth, (frame.height, frame.width), dim=(-2, -1), value=float('nan'))
        
        return IStereoDepth.Output(depth=gt_depthmap)

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None: return


class FlowFormerDepth(IStereoDepth):
    """
    Use FlowFormer to estimate disparity between rectified stereo image. Does not generate depth_cov.
    
    See FlowFormerCovDepth for jointly estimating depth and depth_cov
    """
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        from ..Network.FlowFormer.configs.submission import get_cfg
        from ..Network.FlowFormer.core import build_flowformer
        model = build_flowformer(get_cfg())
        ckpt  = torch.load(self.config.weight, weights_only=True)
        model.load_ddp_state_dict(ckpt)
        model.to(self.config.device)
        
        model.eval()
        self.model = model 
    
    @property
    def provide_cov(self) -> bool: return False
        
    @torch.inference_mode()
    def estimate(self, frame: StereoData) -> IStereoDepth.Output:
        est_flow, _ = self.model.inference(
            frame.imageL.to(self.config.device),
            frame.imageR.to(self.config.device),
        )
        disparity = est_flow[:1].abs().unsqueeze(0)
        depth_map = disparity_to_depth(disparity, frame.frame_baseline, frame.fx)
        return IStereoDepth.Output(depth=depth_map, disparity=disparity)
    
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
                "weight"    : lambda s: isinstance(s, str),
                "device"    : lambda s: isinstance(s, str) and (("cuda" in s) or (s == "cpu")),
            })


class FlowFormerCovDepth(IStereoDepth):
    """
    Use modified FlowFormer to estimate diparity between rectified stereo image and uncertainty of disparity.
    """
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        
        from ..Network.FlowFormer.configs.submission import get_cfg
        from ..Network.FlowFormerCov import build_flowformer
        model = build_flowformer(
            get_cfg(),
            reflect_torch_dtype(config.enc_dtype),
            reflect_torch_dtype(config.dec_dtype)
        )
        ckpt  = torch.load(self.config.weight, weights_only=True)
        model.load_ddp_state_dict(ckpt)
        model.to(self.config.device)
        model.eval()
        
        self.model = model

    @property
    def provide_cov(self) -> bool: return True
        
    @torch.inference_mode()
    def estimate(self, frame: StereoData) -> IStereoDepth.Output:
        est_flow, est_cov = self.model.inference(
            frame.imageL.to(self.config.device),
            frame.imageR.to(self.config.device),
        )
        disparity, disparity_cov = est_flow[:, :1].abs(), est_cov[:, :1]
        
        depth_map = disparity_to_depth(disparity, frame.frame_baseline, frame.fx)
        depth_cov = disparity_to_depth_cov(disparity, disparity_cov, frame.frame_baseline, frame.fx)
        
        return IStereoDepth.Output(depth=depth_map, cov=depth_cov, 
                                   disparity=disparity, disparity_uncertainty=disparity_cov)

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
                "weight"    : lambda s: isinstance(s, str),
                "device"    : lambda s: isinstance(s, str) and (("cuda" in s) or (s == "cpu")),
                "enc_dtype" : lambda s: s in {"fp16", "bf16", "fp32"},  # Precision casting for encoder, the network's input and output will still be in fp32.
                "dec_dtype" : lambda s: s in {"fp16", "bf16", "fp32"},  # Precision casting for decoder, the network's input and output will still be in fp32.
            })


class TartanVODepth(IStereoDepth):
    """
    Use the StereoNet from TartanVO to estimate diparity between stereo image. 
    
    Does not estimate depth_cov if config.cov_mode set to 'None'.
    where config.cov_mode = {'None', 'Est'}
    """
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        
        from Utility.Config import build_dynamic_config
        from Module.Network.StereoCov import StereoCovNet
        
        cfg, _ = build_dynamic_config({"exp": False, "decoder": "hourglass"})
        model = StereoCovNet(cfg)
        ckpt = torch.load(self.config.weight, weights_only=True)
        model.load_ddp_state_dict(ckpt)
        model.to(self.config.device)
        model.eval()
        
        self.model = model
    
    @property
    def provide_cov(self) -> bool: return self.config.cov_mode == "Est"
        
    @torch.inference_mode()
    def estimate(self, frame: StereoData) -> IStereoDepth.Output:
        depth, depth_cov = self.model.inference(frame)
        
        depth_map = padTo(depth, (frame.height, frame.width), dim=(-2, -1), value=float('nan'))
        
        if self.config.cov_mode == "Est":
            depth_cov = padTo(depth_cov, (frame.height, frame.width), dim=(-2, -1), value=float('nan'))
            return IStereoDepth.Output(depth=depth_map, cov=depth_cov)
        else:
            return IStereoDepth.Output(depth=depth_map)
        
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
                "weight"    : lambda s: isinstance(s, str),
                "device"    : lambda s: isinstance(s, str) and (("cuda" in s) or (s == "cpu")),
                "cov_mode"  : lambda s: s in {"Est", "None"}
            })


# Modifier - modifies the input/output of another estimator
# Modifier(IStereoDepth) -> IStereoDepth'


class ApplyGTDepthCov(IStereoDepth):
    """
    A higher-order-module that encapsulates a IStereoDepth module. 
    
    Always compare the estimated output of encapsulated IStereoDepth with ground truth depth and convert
    error in estimation to 'estimated' covariance.
    
    Will raise AssertionError if frame does not have gtDepth.
    """
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.internal_module = IStereoDepth.instantiate(self.config.module.type, self.config.module.args)

    @property
    def provide_cov(self) -> bool: return True
    
    @torch.inference_mode()
    def estimate(self, frame: StereoData) -> IStereoDepth.Output:
        assert frame.gt_depth is not None
        
        output = self.internal_module.estimate(frame)
        error = frame.gt_depth.to(output.depth.device) - output.depth
        gt_cov = error.square()
        
        output.cov = gt_cov
        return output

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        assert config is not None
        IStereoDepth.is_valid_config(config.module)

# Common Routeins

@OnCallCompiler()
def disparity_to_depth(disp: torch.Tensor, bl: float, fx: float) -> torch.Tensor:
    return (bl * fx) * disp.reciprocal()


@OnCallCompiler()
def disparity_to_depth_cov(disp: torch.Tensor, disp_cov: torch.Tensor, bl: float, fx: float) -> torch.Tensor:
    # Propagate disparity covariance to depth covariance
    # See Appendix A.1 of the MAC-VO paper
    disparity_2 = disp.square()
    error_rate_2 = disp_cov * disparity_2.reciprocal()
    depth_cov  = (((bl * fx) ** 2) * (error_rate_2 / disparity_2))
    return depth_cov
