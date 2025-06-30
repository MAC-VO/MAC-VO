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
from Utility.Extensions import ConfigTestableSubclass
from Utility.Config import build_dynamic_config

# Matching interface ###


class IMatcher(ABC, ConfigTestableSubclass):
    @Jt.jaxtyped(typechecker=typechecked)
    @dataclass
    class Output:
        flow: Jt.Float32[torch.Tensor, "B 2 H W"]                 # B x 2 x H x W, float32
        cov : Jt.Float32[torch.Tensor, "B 3 H W"] | None = None   # B x 3 x H x W, float32 OR None if not applicable
        mask: Jt.Bool   [torch.Tensor, "B 1 H W"] | None = None   # B x 1 x H x W, bool    OR None if not applicable
        
        @classmethod
        def from_partial_cov(cls,
            flow: Jt.Float32[torch.Tensor, "B 2 H W"],
            cov : Jt.Float32[torch.Tensor, "B 2 H W"],
            mask: Jt.Bool   [torch.Tensor, "B 1 H W"] | None = None 
        ) -> "IMatcher.Output":
            B, C, H, W = cov.shape
            assert C == 2, "Partial cov is the matcher output where only \\sigma_uu, \\sigma_vv are available."
            return cls(
                flow=flow,
                cov =torch.cat([cov, torch.zeros((B, 1, H, W)).to(cov)], dim=1),
                mask=mask
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
    
    @property
    @abstractmethod
    def provide_cov(self) -> bool: ...
    
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
# Implementations

class GTMatcher(IMatcher):
    """
    A matcher that returns ground truth optical flow.
    
    Will raise AssertionError if ground truth optical flow is not provided.
    """
    @property
    def provide_cov(self) -> bool: return False
    
    def forward(self, frame_t1: StereoData, frame_t2: StereoData) -> IMatcher.Output:
        assert frame_t1.gt_flow is not None
        
        gt_flow = padTo(frame_t1.gt_flow, (frame_t1.height, frame_t1.width), (-2, -1), float('nan'))
        return IMatcher.Output(flow=gt_flow)
    
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None: return


class FlowFormerMatcher(IMatcher):
    """
    Use FlowFormer to estimate optical flow betweeen two frames. Does not generate match_cov.
    
    See FlowFormerCovMatcher for jointly estimating depth and match_cov
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
    
    def forward(self, frame_t1: StereoData, frame_t2: StereoData) -> IMatcher.Output:
        flow, _ = self.model.inference(
            frame_t1.imageL.to(self.config.device),
            frame_t2.imageL.to(self.config.device),
        )
        return IMatcher.Output(flow=flow.unsqueeze(0))
    
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "weight"    : lambda s: isinstance(s, str),
            "device"    : lambda s: isinstance(s, str) and (("cuda" in s) or (s == "cpu")),
        })


class FlowFormerCovMatcher(IMatcher):
    """
    Use the modified FlowFormer proposed by us to jointly estimate optical flow betweeen two frames.
    """
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        
        from ..Network.FlowFormer.configs.submission import get_cfg
        from ..Network.FlowFormerCov import build_flowformer
        
        model = build_flowformer(
            get_cfg(),
            encoder_dtype=reflect_torch_dtype(self.config.enc_dtype),
            decoder_dtype=reflect_torch_dtype(self.config.dec_dtype)
        )
        ckpt  = torch.load(self.config.weight, weights_only=True)
        model.load_ddp_state_dict(ckpt)
        model.to(self.config.device)
        model.eval()
        
        self.model = model
    
    @property
    def provide_cov(self) -> bool: return True

    def forward(self, frame_t1: StereoData, frame_t2: StereoData) -> IMatcher.Output:
        flow, flow_cov = self.model.inference(
            frame_t1.imageL.to(self.config.device),
            frame_t2.imageL.to(self.config.device),
        )
        return IMatcher.Output.from_partial_cov(flow=flow, cov=flow_cov)
    
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
                "weight"    : lambda s: isinstance(s, str),
                "device"    : lambda s: isinstance(s, str) and (("cuda" in s) or (s == "cpu")),
                "enc_dtype" : lambda s: s in {"fp16", "bf16", "fp32"},  # Precision casting for encoder, the network's input and output will still be in fp32.
                "dec_dtype" : lambda s: s in {"fp16", "bf16", "fp32"},  # Precision casting for decoder, the network's input and output will still be in fp32.
            })


class TartanVOMatcher(IMatcher):
    """
    Use TartanVO to estimate optical flow between two frames.
    """
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        
        from ..Network.TartanVOStereo.StereoVO_Interface import TartanStereoVOMatch
        model = TartanStereoVOMatch(self.config.weight, True, self.config.device)
        self.model = model
    
    @property
    def provide_cov(self) -> bool: return False        

    def forward(self, frame_t1: StereoData, frame_t2: StereoData) -> IMatcher.Output:
        flow = self.model.inference(frame_t1, frame_t1.imageL, frame_t2.imageL).unsqueeze(0)
        
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


class TartanVOCovMatcher(IMatcher):
    """
    Use a modified version of TartanVO frontend network to jointly estimate optical flow
    and its covariance between two frames.
    """
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        
        from Module.Network.PWCNet import RAFTFlowCovNet
        cfg, _ = build_dynamic_config({
            "decoder": "raft", "dim": 64, "dropout": 0.1,
            "num_heads": 4, "mixtures": 4, "gru_iters": 12, "kernel_size": 3,
        })
        ckpt = torch.load(self.config.weight, map_location="cpu", weights_only=True)
        model = RAFTFlowCovNet(cfg, self.config.device)
        model.load_ddp_state_dict(ckpt)
        model = model.to(self.config.device)
        model.eval()

        self.model = model
    
    @property
    def provide_cov(self) -> bool: return True
    
    def forward(self, frame_t1: StereoData, frame_t2: StereoData) -> IMatcher.Output:
        flow, flow_cov = self.model.inference(frame_t1.imageL, frame_t2.imageL)
        
        mask = torch.zeros_like(flow[:, :1], dtype=torch.bool)
        pad_height = (frame_t1.height - flow.size(-2)) // 2
        pad_width  = (frame_t1.width  - flow.size(-1)) // 2
        mask[..., pad_height:-pad_height, pad_width:-pad_width] = True
        
        flow     = padTo(flow    , (frame_t1.height, frame_t1.width), dim=(-2, -1), value=float('nan'))
        flow_cov = padTo(flow_cov, (frame_t1.height, frame_t1.width), dim=(-2, -1), value=float('nan'))
        return IMatcher.Output.from_partial_cov(flow=flow, cov=flow_cov, mask=mask)

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:    
        cls._enforce_config_spec(config, {
                "weight"    : lambda s: isinstance(s, str),
                "device"    : lambda s: isinstance(s, str) and (("cuda" in s) or (s == "cpu")),
            })


# Modifier
# Modifier(IMatcher) -> IMatcher'


class ApplyGTMatchCov(IMatcher):
    """
    A higher-order-module that encapsulates a IMatcher module. 
    
    Always compare the estimated output of encapsulated IMatcher with ground truth matching and convert
    error in estimation to 'estimated' covariance.
    
    Will raise AssertionError if frame does not have gtFlow.
    
    NOTE: This modifier only creates estimation to Sigma matrix as a diagonal form, since the optimum 
    covariance matrix (that maximize log-likelihood of ground truth flow) is degenerated for a full
    2x2 matrix setup.
    """
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        
        self.internal_module = IMatcher.instantiate(self.config.module.type, self.config.module.args)
    
    @property
    def provide_cov(self) -> bool: return True
    
    def forward(self, frame_t1: StereoData, frame_t2: StereoData) -> IMatcher.Output:
        assert frame_t1.gt_flow is not None
        out = self.internal_module.estimate(frame_t1, frame_t2)
        
        flow_error = out.flow - frame_t1.gt_flow.to(out.flow.device)
        flow_cov   = flow_error.square()
        return IMatcher.Output.from_partial_cov(flow=out.flow, cov=flow_cov, mask=out.mask)
    
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        assert config is not None
        IMatcher.is_valid_config(config.module)


class ApplyGTMatchMask(IMatcher):
    """
    A higher-order-module that encapsulates a IMatcher module. 
    
    Always compare the estimated output of encapsulated IMatcher with ground truth matching and convert
    error in estimation to 'estimated' covariance.
    
    Will raise AssertionError if frame does not have gtFlow.
    
    NOTE: This modifier only creates estimation to Sigma matrix as a diagonal form, since the optimum 
    covariance matrix (that maximize log-likelihood of ground truth flow) is degenerated for a full
    2x2 matrix setup.
    """
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.internal_module = IMatcher.instantiate(self.config.module.type, self.config.module.args)
    
    @property
    def provide_cov(self) -> bool: return self.internal_module.provide_cov
    
    def forward(self, frame_t1: StereoData, frame_t2: StereoData) -> IMatcher.Output:
        assert frame_t1.flow_mask is not None
        out = self.internal_module.estimate(frame_t1, frame_t2)
        out.mask = frame_t1.flow_mask
        return out
    
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        assert config is not None
        IMatcher.is_valid_config(config.module)
