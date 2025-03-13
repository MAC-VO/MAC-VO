from __future__ import annotations

import torch
from types import SimpleNamespace
from typing import TypeVar, Generic, TypedDict, overload, Any
from abc import ABC, abstractmethod

from dataclasses import dataclass

from DataLoader import StereoData
from Utility.Utils import padTo
from Utility.Extensions import ConfigTestableSubclass
from Utility.Config import build_dynamic_config

# Matching interface ###
T_Context = TypeVar("T_Context")


class IMatcher(ABC, Generic[T_Context], ConfigTestableSubclass):
    @dataclass
    class Output:
        flow: torch.Tensor                 # B x 2 x H x W, float32
        cov : torch.Tensor | None = None   # B x 3 x H x W, float32 OR None if not applicable
        mask: torch.Tensor | None = None   # B x 1 x H x W, bool    OR None if not applicable
    
        @property
        def as_full_cov(self) -> "IMatcher.Output":
            if self.cov is None or self.cov.size(1) == 3: return self
            B, C, H, W = self.cov.shape
            assert C == 2, f"number of channel for cov must be either 2 or 3, get {C=}"
            return IMatcher.Output(
                flow=self.flow,
                cov =torch.cat([self.cov, torch.zeros((B, 1, H, W), device=self.cov.device, dtype=self.cov.dtype)], dim=1),
                mask=self.mask
            )
        
    """
    Estimate the optical flow map between two frames. (Use left-frame of stereo pair)
    
    `IMatcher.estimate(frame_t1: StereoData, frame_t2: StereoData) -> IMatcher.Output`

    Given a frame with imageL, imageR being Bx3xHxW, return `output` where    

    * flow      - Bx2xHxW shaped torch.Tensor, estimated optical flow map
                maybe padded with `nan` if model can't output prediction with same shape as input image.
    * cov       - Bx3xHxW shaped torch.Tensor or None, estimated covariance of optical flow map map (if provided)
                maybe padded with `nan` if model can't output prediction with same shape as input image.
                The three channels are uu, vv, and uv respectively, such that the 2x2 covariance matrix will be:
                    Sigma = [[uu, uv], [uv, vv]]
    * mask      - Bx1xHxW shaped torch.Tensor or None, the position with `True` value means valid (not occluded) 
                prediction regions.
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
    def forward(self, frame_t1: StereoData, frame_t2: StereoData) -> IMatcher.Output: ...

    def estimate(self, frame_t1: StereoData, frame_t2: StereoData) -> IMatcher.Output:
        with torch.no_grad(), torch.inference_mode():
            return self.forward(frame_t1, frame_t2)

    @overload
    @staticmethod
    def retrieve_pixels(pixel_uv: torch.Tensor, scalar_map: torch.Tensor, interpolate: bool=False) -> torch.Tensor: ...
    @overload
    @staticmethod
    def retrieve_pixels(pixel_uv: torch.Tensor, scalar_map: None, interpolate: bool=False) -> None: ...

    @staticmethod
    def retrieve_pixels(pixel_uv: torch.Tensor, scalar_map: torch.Tensor | None, interpolate: bool=False) -> torch.Tensor | None:
        """
        Given a pixel_uv (Nx2) tensor, retrieve the pixel values (CxN) from scalar_map (BxCxHxW).
        
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

# Dense Matching Implementation ###
# Contexts

class ModelContext(TypedDict):
    model: torch.nn.Module

# Implementations

class GTMatcher(IMatcher[None]):
    """
    A matcher that returns ground truth optical flow.
    
    Will raise AssertionError if ground truth optical flow is not provided.
    """
    @property
    def provide_cov(self) -> bool: return False
    
    def init_context(self) -> None: return None
    
    def forward(self, frame_t1: StereoData, frame_t2: StereoData) -> IMatcher.Output:
        assert frame_t1.gt_flow is not None
        
        gt_flow = padTo(frame_t1.gt_flow, (frame_t1.height, frame_t1.width), (-2, -1), float('nan'))
        return IMatcher.Output(flow=gt_flow)
    
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None: return


class FlowFormerMatcher(IMatcher[ModelContext]):
    """
    Use FlowFormer to estimate optical flow betweeen two frames. Does not generate match_cov.
    
    See FlowFormerCovMatcher for jointly estimating depth and match_cov
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
    
    def forward(self, frame_t1: StereoData, frame_t2: StereoData) -> IMatcher.Output:
        flow, _ = self.context["model"].inference(frame_t1.imageL, frame_t2.imageL)
        return IMatcher.Output(flow=flow.unsqueeze(0))
    
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
                "weight"    : lambda s: isinstance(s, str),
                "device"    : lambda s: isinstance(s, str) and (("cuda" in s) or (s == "cpu")),
            })


class FlowFormerCovMatcher(IMatcher[ModelContext]):
    """
    Use the modified FlowFormer proposed by us to jointly estimate optical flow betweeen two frames.
    """
    @property
    def provide_cov(self) -> bool: return True
    
    def init_context(self) -> ModelContext:
        from ..Network.FlowFormer.configs.submission import get_cfg
        from ..Network.FlowFormerCov import build_flowformer
        
        model = build_flowformer(get_cfg(), self.config.device)
        ckpt  = torch.load(self.config.weight, weights_only=True)
        model.load_ddp_state_dict(ckpt)
        model.to(self.config.device)
        
        model.eval()
        return ModelContext(model=model)

    def forward(self, frame_t1: StereoData, frame_t2: StereoData) -> IMatcher.Output:
        flow, flow_cov = self.context["model"].inference(frame_t1.imageL, frame_t2.imageL)
        return IMatcher.Output(flow=flow, cov=flow_cov).as_full_cov
    
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
                "weight"    : lambda s: isinstance(s, str),
                "device"    : lambda s: isinstance(s, str) and (("cuda" in s) or (s == "cpu")),
            })


class TartanVOMatcher(IMatcher[ModelContext]):
    """
    Use TartanVO to estimate optical flow between two frames.
    """
    @property
    def provide_cov(self) -> bool: return False
    
    def init_context(self) -> ModelContext:
        from ..Network.TartanVOStereo.StereoVO_Interface import TartanStereoVOMatch
        model = TartanStereoVOMatch(self.config.weight, True, self.config.device)
        return ModelContext(model=model)    #type: ignore

    def forward(self, frame_t1: StereoData, frame_t2: StereoData) -> IMatcher.Output:
        flow = self.context["model"].inference(frame_t1, frame_t1.imageL, frame_t2.imageL).unsqueeze(0)
        
        mask = torch.zeros_like(flow[:, :1], dtype=torch.bool)
        pad_height = (frame_t1.height - flow.size(-2)) // 2
        pad_width  = (frame_t1.width  - flow.size(-1)) // 2
        mask[..., pad_height:-pad_height, pad_width:-pad_width] = True
        
        flow = padTo(flow, (frame_t1.height, frame_t1.width), dim=(-2, -1), value=float('nan'))
        
        return IMatcher.Output(flow=flow, mask=mask)
    
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:    
        cls._enforce_config_spec(config, {
                "weight"    : lambda s: isinstance(s, str),
                "device"    : lambda s: isinstance(s, str) and (("cuda" in s) or (s == "cpu")),
            })


class TartanVOCovMatcher(IMatcher[ModelContext]):
    """
    Use a modified version of TartanVO frontend network to jointly estimate optical flow
    and its covariance between two frames.
    """
    @property
    def provide_cov(self) -> bool: return True
    
    def init_context(self) -> ModelContext:
        from Module.Network.PWCNet import RAFTFlowCovNet
        cfg, _ = build_dynamic_config({
            "decoder": "raft", "dim": 64, "dropout": 0.1,
            "num_heads": 4, "mixtures": 4, "gru_iters": 12, "kernel_size": 3,
        })
        ckpt = torch.load(self.config.weight, map_location="cpu", weights_only=True)
        model = RAFTFlowCovNet(cfg, self.config.device)
        model.load_ddp_state_dict(ckpt)

        model.eval()
        model = model.to(self.config.device)
        return ModelContext(model=model)
    
    def forward(self, frame_t1: StereoData, frame_t2: StereoData) -> IMatcher.Output:
        flow, flow_cov = self.context["model"].inference(frame_t1.imageL, frame_t2.imageL)
        
        mask = torch.zeros_like(flow[:, :1], dtype=torch.bool)
        pad_height = (frame_t1.height - flow.size(-2)) // 2
        pad_width  = (frame_t1.width  - flow.size(-1)) // 2
        mask[..., pad_height:-pad_height, pad_width:-pad_width] = True
        
        flow     = padTo(flow    , (frame_t1.height, frame_t1.width), dim=(-2, -1), value=float('nan'))
        flow_cov = padTo(flow_cov, (frame_t1.height, frame_t1.width), dim=(-2, -1), value=float('nan'))
        return IMatcher.Output(flow=flow, cov=flow_cov, mask=mask).as_full_cov

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:    
        cls._enforce_config_spec(config, {
                "weight"    : lambda s: isinstance(s, str),
                "device"    : lambda s: isinstance(s, str) and (("cuda" in s) or (s == "cpu")),
            })


# Modifier
# Modifier(IMatcher) -> IMatcher'


class ApplyGTMatchCov(IMatcher[IMatcher]):
    """
    A higher-order-module that encapsulates a IMatcher module. 
    
    Always compare the estimated output of encapsulated IMatcher with ground truth matching and convert
    error in estimation to 'estimated' covariance.
    
    Will raise AssertionError if frame does not have gtFlow.
    
    NOTE: This modifier only creates estimation to Sigma matrix as a diagonal form, since the optimum 
    covariance matrix (that maximize log-likelihood of ground truth flow) is degenerated for a full
    2x2 matrix setup.
    """
    @property
    def provide_cov(self) -> bool: return True
    
    def init_context(self) -> IMatcher:
        internal_module = IMatcher.instantiate(self.config.module.type, self.config.module.args)
        return internal_module
    
    def forward(self, frame_t1: StereoData, frame_t2: StereoData) -> IMatcher.Output:
        assert frame_t1.gt_flow is not None
        out = self.context.estimate(frame_t1, frame_t2)
        
        flow_error = out.flow - frame_t1.gt_flow.to(out.flow.device)
        flow_cov   = flow_error.square()
        out.cov = flow_cov
        return out.as_full_cov
    
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        assert config is not None
        IMatcher.is_valid_config(config.module)
