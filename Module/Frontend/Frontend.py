"""
What is Frontend?
    - up to now (2024/06) it's just a combination of StereoDepth and Matcher

Why we need Frontend?
    - Sometime the depth estimation and matching are tightly coupled, so we need a way to combine them.
    
      For instance, if depth (using disparity) and matching uses the same network with same weight, instead of
      inference twice in sequential mannor, we can compose a batch with size of 2 and inference once.
    
How to use this?
    - If there's no specific need (e.g. for performance improvement mentioned above), just use the `FrontendCompose`
      to combine an IStereoDepth and an IMatcher. This should work just fine.
    
    - Otherwise implement a new IFrontend and plug it in the pipeline.
"""

from __future__ import annotations

import time
import torch
from types import SimpleNamespace
from typing import TypeVar, Generic, TypedDict, overload, Any
from abc import ABC, abstractmethod

from DataLoader import SourceDataFrame
from Utility.PrettyPrint import Logger
from Utility.Extensions import ConfigTestableSubclass
from Utility.Utils import reflect_torch_dtype

from .StereoDepth import IStereoDepth
from .Matching    import IMatcher

# Frontend interface ###
# T_Context = The internal state of frontend (DepthEstimator + MatchEstimator)
T_Context = TypeVar("T_Context")


class IFrontend(ABC, Generic[T_Context], ConfigTestableSubclass):
    """
    Jointly estimate dense depth map, dense matching and potentially their covariances given two pairs of stereo images.
    
    `IFrontend(frame_t1: SourceDataFrame, frame_t2: SourceDataFrame) -> (depth, depth_cov, match, match_cov)`

    Given two frames with imageL, imageR with shape of Bx3xHxW, return `output` where

    * `depth    `   - Bx1xHxW shaped torch.Tensor, estimated depth map for **frame_t2**
    * `depth_cov`   - Bx1xHxW shaped torch.Tensor or None, estimated covariance of depth map (if provided) for **frame_t2**
    * `match    `   - Bx2xHxW shaped torch.Tensor, estimated optical flow map from **frame_t1** to **frame_t2**
    * `match_cov`   - Bx2xHxW shaped torch.Tensor or None, estimated covariance of optical flow map (if provided) from **frame_t1** to **frame_t2**
    
    If frame_t1 is None, return only `depth` and `depth_cov` and leave `match` and `match_cov` as None.

    #### All outputs maybe padded with `nan` if model can't output prediction with same shape as input image.
    """
    
    def __init__(self, config: SimpleNamespace):
        self.config : SimpleNamespace = config
        self.context: T_Context       = self.init_context()
        
    @overload
    def __call__(self, frame_t1: None, frame_t2: SourceDataFrame) -> tuple[torch.Tensor, torch.Tensor | None, None, None]: ...
    @overload
    def __call__(self, frame_t1: SourceDataFrame, frame_t2: SourceDataFrame) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None]: ...
    
    def __call__(self, frame_t1: SourceDataFrame | None, frame_t2: SourceDataFrame) -> \
        tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
            return self.estimate(frame_t1, frame_t2)
    
    @property
    @abstractmethod
    def provide_cov(self) -> tuple[bool, bool]: ...
    
    @abstractmethod
    def init_context(self) -> T_Context: ...
    
    @overload
    @abstractmethod
    def estimate(self, frame_t1: None, frame_t2: SourceDataFrame) -> \
        tuple[torch.Tensor, torch.Tensor | None, None, None]: ...
    
    @overload
    @abstractmethod
    def estimate(self, frame_t1: SourceDataFrame, frame_t2: SourceDataFrame) -> \
        tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None]: ...
    
    @abstractmethod
    def estimate(self, frame_t1: SourceDataFrame | None, frame_t2: SourceDataFrame) -> \
        tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
            """
            Given two frames with imageL, imageR with shape of Bx3xHxW, return `output` where

            * `depth    `   - Bx1xHxW shaped torch.Tensor, estimated depth map for **frame_t2**
            * `depth_cov`   - Bx1xHxW shaped torch.Tensor or None, estimated covariance of depth map (if provided) for **frame_t2**
            * `match    `   - Bx2xHxW shaped torch.Tensor, estimated optical flow map from **frame_t1** to **frame_t2**
            * `match_cov`   - Bx2xHxW shaped torch.Tensor or None, estimated covariance of optical flow map (if provided) from **frame_t1** to **frame_t2**
            
            If frame_t1 is None, return only `depth` and `depth_cov` and leave `match` and `match_cov` as None.

            #### All outputs maybe padded with `nan` if model can't output prediction with same shape as input image.
            """
            ...
    
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

class ComposeContext(TypedDict):
    depth: IStereoDepth
    match: IMatcher

class ModelContext(TypedDict):
    model: torch.nn.Module
    cuda_graph: tuple[torch.Size, torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None
    compiled_model: Any | None

# Implementations

class FrontendCompose(IFrontend[ComposeContext]):
    def init_context(self) -> ComposeContext:
        depth_estimator = IStereoDepth.instantiate(self.config.depth.type, self.config.depth.args)
        match_estimator = IMatcher.instantiate(self.config.match.type, self.config.match.args)
        return ComposeContext(depth=depth_estimator, match=match_estimator)

    @property
    def provide_cov(self) -> tuple[bool, bool]:
        return self.context["depth"].provide_cov, self.context["match"].provide_cov
    
    @overload
    def estimate(self, frame_t1: None, frame_t2: SourceDataFrame) -> \
        tuple[torch.Tensor, torch.Tensor | None, None, None]: ...
    
    @overload
    def estimate(self, frame_t1: SourceDataFrame, frame_t2: SourceDataFrame) -> \
        tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None]: ...
    
    def estimate(self, frame_t1: SourceDataFrame | None, frame_t2: SourceDataFrame) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        depth, depth_cov = self.context["depth"](frame_t2)
        if frame_t1 is None:
            return depth, depth_cov, None, None
        else:
            match, match_cov = self.context["match"](frame_t1, frame_t2)
            return depth, depth_cov, match, match_cov

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        assert config is not None
        IMatcher.is_valid_config(config.match)
        IStereoDepth.is_valid_config(config.depth)


class FlowFormerCovFrontend(IFrontend[ModelContext]):    
    def init_context(self) -> ModelContext:
        from ..Network.FlowFormer.configs.submission import get_cfg
        from ..Network.FlowFormerCov import build_flowformer
        
        if self.config.use_jit:
            Logger.write("info", "Using JIT for Frontend inference, meanwhile,\n"
                         "\tSet float32 matmul precision to medium\n"
                         "\tEnable Tensor Cores for matmul and linalg ops\n"
                         "for maximum speedup.")
            torch.backends.cuda.matmul.allow_tf32 = True    # Allow tensor cores
            torch.backends.cudnn.allow_tf32 = True          # Allow tensor cores
            torch.set_float32_matmul_precision("medium")    # Reduced precision for higher throughput
            torch.backends.cuda.preferred_linalg_library = "cusolver"   # For faster linalg ops
        
        model = build_flowformer(get_cfg(), self.config.device, use_inference_jit=self.config.use_jit)
        ckpt  = torch.load(self.config.weight, map_location=self.config.device, weights_only=True)
        
        model.eval()
        model.to(self.config.device)
        model.to(reflect_torch_dtype(self.config.dtype))
        model.load_ddp_state_dict(ckpt)
        
        return ModelContext(model=model, cuda_graph=None, compiled_model=None)
    
    @property
    def provide_cov(self) -> tuple[bool, bool]:
        return True, True
    
    @overload
    def estimate(self, frame_t1: None, frame_t2: SourceDataFrame) -> \
        tuple[torch.Tensor, torch.Tensor | None, None, None]: ...
    
    @overload
    def estimate(self, frame_t1: SourceDataFrame, frame_t2: SourceDataFrame) -> \
        tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None]: ...
    
    @torch.inference_mode()
    def estimate(self, frame_t1: SourceDataFrame | None, frame_t2: SourceDataFrame) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        # Joint inference
        depth_pair = (frame_t2.imageL, frame_t2.imageR)
        
        if frame_t1 is not None:
            match_pair = (frame_t1.imageL, frame_t2.imageL)
            input_A = torch.cat([depth_pair[0], match_pair[0]], dim=0)
            input_B = torch.cat([depth_pair[1], match_pair[1]], dim=0)
        else:
            input_A, input_B = depth_pair
        
        input_A = input_A.to(reflect_torch_dtype(self.config.dtype))
        input_B = input_B.to(reflect_torch_dtype(self.config.dtype))
        
        if frame_t1 is not None and self.config.use_jit and ('cuda' in self.config.device.lower()):
            est_flow, est_cov = self.cuda_graph_estimate(input_A, input_B)
        else:
            est_flow, est_cov = self.context["model"].inference(input_A, input_B)
        
        est_flow, est_cov = est_flow.float(), est_cov.float()
        
        # Depth estimation
        disparity, disparity_cov = est_flow[0:1, :1].abs(), est_cov[0:1, :1]
        
        # Propagate disparity covariance to depth covariance
        # See Appendix A.1 of paper
        disparity_2 = disparity.square()
        error_rate_2 = disparity_cov / disparity_2
        depth_map = ((frame_t2.meta.baseline * frame_t2.meta.fx) / disparity)
        depth_cov = (((frame_t2.meta.baseline * frame_t2.meta.fx) ** 2) * (error_rate_2 / disparity_2))
        
        if self.config.enforce_positive_disparity:
            bad_mask = est_flow[0:1, :1] > 0
            depth_map[bad_mask] = -1
        else:
            bad_mask = None
        
        # Matching estimation
        if frame_t1 is not None:
            match_map = est_flow[1:2]
            match_cov = est_cov[1:2]
            if bad_mask is not None:
                match_cov[bad_mask.expand(-1, 2, -1, -1)] = 1e5
            return depth_map, depth_cov, match_map, match_cov
        else:
            return depth_map, depth_cov, None, None

    def cuda_graph_estimate(self, inp_A: torch.Tensor, inp_B: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        If does not exist a cuda graph
            build one and run inference through it. 
            Store the resulted graph in frontend for future use.
        If does exist a cuda graph
            Launch graph with new input.
        """
        if self.context["cuda_graph"] is None:
            Logger.write("info", "Building CUDAGraph for FlowFormerCovFrontend")
            static_input_A, static_input_B   = torch.empty_like(inp_A, device='cuda'), torch.empty_like(inp_A, device='cuda')
            static_output, static_output_cov = torch.empty_like(inp_A, device='cuda'), torch.empty_like(inp_A, device='cuda')
            
            static_input_A.copy_(inp_A)
            static_input_B.copy_(inp_B)
            
            output_val: None | torch.Tensor = None
            output_cov: None | torch.Tensor = None
            
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())  #type: ignore
            with torch.cuda.stream(s):                  #type: ignore
                for _ in range(3):
                    output_val, output_cov = self.context["model"].inference(static_input_A, static_input_B)
            torch.cuda.current_stream().wait_stream(s)
            assert output_val is not None and output_cov is not None
            
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_output, static_output_cov = self.context["model"].inference(static_input_A, static_input_B)
            
            self.context["cuda_graph"] = (inp_A.shape, graph, static_input_A, static_input_B, static_output, static_output_cov)
            Logger.write("info", "CUDAGraph Built. Will use CUDAGraph for accelerated inference.")
            
            return output_val, output_cov
        else:
            input_shape, cuda_graph, static_input_A, static_input_B, static_output, static_output_cov = self.context["cuda_graph"]
            result_val = torch.empty_like(static_output)
            result_cov = torch.empty_like(static_output_cov)
            
            assert inp_A.shape == input_shape, f"Input shape mismatch for CUDAGraph replay: {inp_A.shape} != {input_shape}"
            
            static_input_A.copy_(inp_A)
            static_input_B.copy_(inp_B)
            
            cuda_graph.replay()
            time.sleep(0.0) # Hint OS scheduler for context switch
            
            result_val.copy_(static_output)
            result_cov.copy_(static_output_cov)
            
        return result_val, result_cov

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "use_jit"   : lambda b: isinstance(b, bool),
            "weight"    : lambda s: isinstance(s, str),
            "device"    : lambda s: isinstance(s, str) and (("cuda" in s) or (s == "cpu")),
        })
