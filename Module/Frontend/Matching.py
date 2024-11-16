from __future__ import annotations

import torch
from types import SimpleNamespace
from typing import TypeVar, Generic, TypedDict, overload
from abc import ABC, abstractmethod

from DataLoader import SourceDataFrame
from Utility.Utils import padTo
from Utility.Extensions import ConfigTestableSubclass
from Utility.Config import build_dynamic_config

from ..Network.TartanVOStereo.StereoVO_Interface import TartanStereoVOMatch

# Matching interface ###
# T_Context = The internal state of matcher
T_Context = TypeVar("T_Context")


class IMatcher(ABC, Generic[T_Context], ConfigTestableSubclass):
    """
    Estimate the optical flow map between two frames. (Use left-frame of stereo pair)
    
    `IMatcher(frame_t1: SourceDataFrame, frame_t2: SourceDataFrame) -> flow, flow_cov`

    Given a frame with imageL, imageR being Bx3xHxW, return `output` where    

    * flow      - Bx2xHxW shaped torch.Tensor, estimated optical flow map
                maybe padded with `nan` if model can't output prediction with same shape as input image.
    * flow_cov  - Bx2xHxW shaped torch.Tensor or None, estimated covariance of optical flow map map (if provided)
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
    def estimate(self, frame_t1: SourceDataFrame, frame_t2: SourceDataFrame) -> tuple[torch.Tensor, torch.Tensor | None]: ...
    
    def __call__(self, frame_t1: SourceDataFrame, frame_t2: SourceDataFrame) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.estimate(frame_t1, frame_t2)

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

class TartanVOContext(TypedDict):
    model: TartanStereoVOMatch

# Implementations

class GTMatcher(IMatcher[None]):
    """
    A matcher that returns ground truth optical flow.
    
    Will raise AssertionError if ground truth optical flow is not provided.
    """
    @property
    def provide_cov(self) -> bool: return False
    
    def init_context(self) -> None: return None
    
    def estimate(self, frame_t1: SourceDataFrame, frame_t2: SourceDataFrame) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert frame_t1.gtFlow is not None
        
        gt_flow = padTo(frame_t1.gtFlow, (frame_t1.meta.height, frame_t1.meta.width), (-2, -1), float('nan'))
        return gt_flow, None
    
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
    
    @torch.inference_mode()
    def estimate(self, frame_t1: SourceDataFrame, frame_t2: SourceDataFrame) -> tuple[torch.Tensor, torch.Tensor | None]:
        flow, _ = self.context["model"].inference(frame_t1.imageL, frame_t2.imageL)
        return flow.unsqueeze(0), None
    
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
        ckpt  = torch.load(self.config.weight)
        model.load_ddp_state_dict(ckpt)
        model.to(self.config.device)
        
        model.eval()
        return ModelContext(model=model)

    @torch.inference_mode()
    def estimate(self, frame_t1: SourceDataFrame, frame_t2: SourceDataFrame) -> tuple[torch.Tensor, torch.Tensor | None]:
        flow, flow_cov = self.context["model"].inference(frame_t1.imageL, frame_t2.imageL)
        return flow, flow_cov
    
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
                "weight"    : lambda s: isinstance(s, str),
                "device"    : lambda s: isinstance(s, str) and (("cuda" in s) or (s == "cpu")),
            })


class TartanVOMatcher(IMatcher[TartanVOContext]):
    """
    Use TartanVO to estimate optical flow between two frames.
    """
    @property
    def provide_cov(self) -> bool: return False
    
    def init_context(self) -> TartanVOContext:
        model = TartanStereoVOMatch(self.config.weight, True, self.config.device)
        return TartanVOContext(model=model)

    @torch.inference_mode()
    def estimate(self, frame_t1: SourceDataFrame, frame_t2: SourceDataFrame) -> tuple[torch.Tensor, torch.Tensor | None]:
        meta = frame_t1.meta
        flow = self.context["model"].inference(meta, frame_t1.imageL, frame_t2.imageL).unsqueeze(0)
        flow = padTo(flow, (meta.height, meta.width), dim=(-2, -1), value=float('nan'))
        return flow, None
    
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
        ckpt = torch.load(self.config.weight, map_location="cpu")
        model = RAFTFlowCovNet(cfg, self.config.device)
        model.load_ddp_state_dict(ckpt)

        model.eval()
        model = model.to(self.config.device)
        return ModelContext(model=model)
    
    @torch.inference_mode()
    def estimate(self, frame_t1: SourceDataFrame, frame_t2: SourceDataFrame) -> tuple[torch.Tensor, torch.Tensor | None]:
        meta = frame_t1.meta
        flow_map, flow_cov_map = self.context["model"].inference(frame_t1.imageL, frame_t2.imageL)
        flow_map     = padTo(flow_map[0]    , (meta.height, meta.width), dim=(-2, -1), value=float('nan'))
        flow_cov_map = padTo(flow_cov_map[0], (meta.height, meta.width), dim=(-2, -1), value=float('nan'))
        return flow_map, flow_cov_map

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:    
        cls._enforce_config_spec(config, {
                "weight"    : lambda s: isinstance(s, str),
                "device"    : lambda s: isinstance(s, str) and (("cuda" in s) or (s == "cpu")),
            })


class ApplyGTMatchCov(IMatcher[IMatcher]):
    """
    A higher-order-module that encapsulates a IMatcher module. 
    
    Always compare the estimated output of encapsulated IMatcher with ground truth matching and convert
    error in estimation to 'estimated' covariance.
    
    Will raise AssertionError if frame does not have gtFlow.
    """
    @property
    def provide_cov(self) -> bool: return True
    
    def init_context(self) -> IMatcher:
        internal_module = IMatcher.instantiate(self.config.module.name, self.config.module.args)
        return internal_module
    
    def estimate(self, frame_t1: SourceDataFrame, frame_t2: SourceDataFrame) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert frame_t1.gtFlow is not None
        flow, _ = self.context.estimate(frame_t1, frame_t2)
        
        flow_error = flow - frame_t1.gtFlow
        flow_cov   = flow_error.square()
        return flow, flow_cov
    
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        IMatcher.is_valid_config(config)
