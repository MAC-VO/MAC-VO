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

import torch
import time
from pathlib import Path
from types import SimpleNamespace
from typing import TypeVar, Generic, TypedDict, overload, Literal, get_args, Mapping, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass

from DataLoader import StereoData
from Utility.PrettyPrint import Logger
from Utility.Timer import Timer
from Utility.Extensions import ConfigTestableSubclass
from Utility.Utils import reflect_torch_dtype

from .StereoDepth import IStereoDepth, disparity_to_depth, disparity_to_depth_cov
from .Matching    import IMatcher

# Frontend interface ###
# T_Context = The internal state of frontend (DepthEstimator + MatchEstimator)
T_Context = TypeVar("T_Context")


class IFrontend(ABC, Generic[T_Context], ConfigTestableSubclass):
    """
    Jointly estimate dense depth map, dense matching and potentially their covariances given two pairs of stereo images.
    
    `IFrontend(frame_t1: StereoData, frame_t2: StereoData) -> IStereoDepth.Output, IMatcher.Output`

    Given two frames with imageL, imageR with shape of Bx3xHxW, return `output` where

    * [0] - IStereoDepth.Output, the predicted depth (and potentially depth covariance & validity mask)
    * [1] - IMatcher.Output or None, the predicted flow (potentially flow covariance & mask)
    
    If frame_t1 is None, return only `IStereoDepth.Output` and leave [1] as None.

    #### All outputs maybe padded with `nan` if model can't output prediction with same shape as input image.
    """
    
    def __init__(self, config: SimpleNamespace):
        self.config : SimpleNamespace = config
        self.context: T_Context       = self.init_context()
    
    @property
    @abstractmethod
    def provide_cov(self) -> tuple[bool, bool]: ...
    
    @abstractmethod
    def init_context(self) -> T_Context: ...
    
    @overload
    @abstractmethod
    def estimate(self, frame_t1: None, frame_t2: StereoData) -> tuple[IStereoDepth.Output, None]: ...
    
    @overload
    @abstractmethod
    def estimate(self, frame_t1: StereoData, frame_t2: StereoData) -> tuple[IStereoDepth.Output, IMatcher.Output]: ...
    
    @abstractmethod
    def estimate(self, frame_t1: StereoData | None, frame_t2: StereoData) -> tuple[IStereoDepth.Output, IMatcher.Output | None]:
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

@dataclass
class CUDAGraphHandler:
    graph: torch.cuda.CUDAGraph
    shape: torch.Size
    static_input: dict[str, torch.Tensor]
    static_ouput: dict[str, torch.Tensor]

class ComposeContext(TypedDict):
    depth: IStereoDepth
    match: IMatcher

class ModelContext(TypedDict):
    model: torch.nn.Module

# Implementations

class FrontendCompose(IFrontend[ComposeContext]):
    def init_context(self) -> ComposeContext:
        depth_estimator = IStereoDepth.instantiate(self.config.depth.type, self.config.depth.args)
        match_estimator = IMatcher.instantiate(self.config.match.type, self.config.match.args)
        return ComposeContext(depth=depth_estimator, match=match_estimator)

    @property
    def provide_cov(self) -> tuple[bool, bool]:
        return self.context["depth"].provide_cov, self.context["match"].provide_cov
    
    # To make static type checker happy and better describe the behavior of API
    @overload
    def estimate(self, frame_t1: None, frame_t2: StereoData) -> tuple[IStereoDepth.Output, None]: ...
    @overload
    def estimate(self, frame_t1: StereoData, frame_t2: StereoData) -> tuple[IStereoDepth.Output, IMatcher.Output]: ...
    
    @Timer.cpu_timeit("Frontend.estimate")
    @Timer.gpu_timeit("Frontend.estimate")
    def estimate(self, frame_t1: StereoData | None, frame_t2: StereoData) -> tuple[IStereoDepth.Output, IMatcher.Output | None]:
        depth_output = self.context["depth"].estimate(frame_t2)
        if frame_t1 is None:
            return depth_output, None
        else:
            match_output = self.context["match"].estimate(frame_t1, frame_t2)
            return depth_output, match_output

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        assert config is not None
        IMatcher.is_valid_config(config.match)
        IStereoDepth.is_valid_config(config.depth)


class FlowFormerCovFrontend(IFrontend[ModelContext]):
    TENSOR_RT_AOT_RESULT_PATH = Path("./cache/FlowFormerCov_TRTCache")
    T_SUPPORT_DTYPE = Literal["fp32", "bf16", "fp16"]
    
    def init_context(self) -> ModelContext:
        from ..Network.FlowFormer.configs.submission import get_cfg
        from ..Network.FlowFormerCov import build_flowformer
        
        model = build_flowformer(get_cfg(), self.config.device, use_inference_jit=False)
        ckpt  = torch.load(self.config.weight, map_location=self.config.device, weights_only=True)
        
        model.eval()
        model.to(self.config.device)
        model.to(reflect_torch_dtype(self.config.dtype))
        model.load_ddp_state_dict(ckpt)
        model = model.to(self._get_dtype(self.config.dtype))
        
        return ModelContext(model=model)
    
    @property
    def provide_cov(self) -> tuple[bool, bool]:
        return True, True
    
    @staticmethod
    def _get_dtype(t: T_SUPPORT_DTYPE) -> torch.dtype:
        match t:
            case "bf16": return torch.bfloat16
            case "fp32": return torch.float32
            case "fp16": return torch.float16
            case _: raise ValueError(f"dtype can only be {get_args(FlowFormerCovFrontend.T_SUPPORT_DTYPE)}, but received {t}")

    @overload
    def estimate(self, frame_t1: None, frame_t2: StereoData) -> tuple[IStereoDepth.Output, None]: ...
    @overload
    def estimate(self, frame_t1: StereoData, frame_t2: StereoData) -> tuple[IStereoDepth.Output, IMatcher.Output]: ...
    
    @Timer.cpu_timeit("Frontend.estimate")
    @Timer.gpu_timeit("Frontend.estimate")
    @torch.inference_mode()
    def estimate(self, frame_t1: StereoData | None, frame_t2: StereoData) -> tuple[IStereoDepth.Output, IMatcher.Output | None]:
        # Joint inference
        depth_pair = (frame_t2.imageL, frame_t2.imageR)
        
        if frame_t1 is not None:
            match_pair = (frame_t1.imageL, frame_t2.imageL)
            input_A = torch.cat([depth_pair[0], match_pair[0]], dim=0)
            input_B = torch.cat([depth_pair[1], match_pair[1]], dim=0)
        else:
            input_A, input_B = depth_pair
        
        input_A = input_A.to(self._get_dtype(self.config.dtype))
        input_B = input_B.to(self._get_dtype(self.config.dtype))
        est_flow, est_cov = self.context["model"].inference(input_A, input_B)
        
        est_flow = est_flow.float()
        est_cov  = est_cov.float()
        
        # Depth estimation
        disparity, disparity_cov = est_flow[0:1, :1].abs(), est_cov[0:1, :1]

        depth_map = disparity_to_depth(disparity, frame_t2.frame_baseline, frame_t2.fx)
        depth_cov = disparity_to_depth_cov(disparity, disparity_cov, frame_t2.frame_baseline, frame_t2.fx)
        
        if self.config.enforce_positive_disparity:
            bad_mask = est_flow[0:1, :1] <= 0
        else:
            bad_mask = None
        
        # Matching estimation
        if frame_t1 is not None:
            match_map = est_flow[1:2]
            match_cov = est_cov[1:2]
            if self.config.max_flow != -1:
                match_mask        = est_flow < self.config.max_flow
                match_mask        = torch.logical_and(match_mask[1:2, :1], match_mask[1:2, 1:])
            else:
                match_mask = None
            
            return IStereoDepth.Output(depth=depth_map, cov=depth_cov, disparity=disparity, disparity_uncertainty=disparity_cov,mask=bad_mask), \
                    IMatcher.Output(flow=match_map, cov=match_cov, mask=match_mask).as_full_cov
        else:
            return IStereoDepth.Output(depth=depth_map, cov=depth_cov, disparity=disparity, disparity_uncertainty=disparity_cov,mask=bad_mask), None

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "weight"    : lambda s: isinstance(s, str),
            "device"    : lambda s: isinstance(s, str) and (("cuda" in s) or (s == "cpu")),
            "dtype"     : lambda b: isinstance(b, str) and b in ("fp32", "fp16", "bf16"),
            "enforce_positive_disparity": lambda b: isinstance(b, bool),
            "max_flow"  : lambda v: isinstance(v, (int, float)) and (v > 0) or (v == -1)
        })


class CUDAGraph_FlowFormerCovFrontend(FlowFormerCovFrontend):
    """
    FlowformerCov Frontend, but using CUDAGraph acceleration to improve inference speed.
    """
    
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.cuda_graph: CUDAGraphHandler | None = None
        assert "cuda" in self.config.device.lower(), "CUDAGraph_FlowFormerCovFrontend can only run on CUDA device."
        
        torch.backends.cuda.matmul.allow_tf32 = True    # Allow tensor cores
        torch.backends.cudnn.allow_tf32 = True          # Allow tensor cores
        torch.set_float32_matmul_precision("medium")    # Reduced precision for higher throughput
        torch.backends.cuda.preferred_linalg_library = "cusolver"   # For faster linalg ops
    
    def init_context(self) -> ModelContext:
        from ..Network.FlowFormer.configs.submission import get_cfg
        from ..Network.FlowFormerCov import build_flowformer
        
        model = build_flowformer(get_cfg(), self.config.device, use_inference_jit=True)
        ckpt  = torch.load(self.config.weight, map_location=self.config.device, weights_only=True)
        
        model.eval()
        model.to(self.config.device)
        model.to(reflect_torch_dtype(self.config.dtype))
        model.load_ddp_state_dict(ckpt)
        model = model.to(self._get_dtype(self.config.dtype))
        
        return ModelContext(model=model)
    
    @overload
    def estimate(self, frame_t1: None, frame_t2: StereoData) -> tuple[IStereoDepth.Output, None]: ...
    @overload
    def estimate(self, frame_t1: StereoData, frame_t2: StereoData) -> tuple[IStereoDepth.Output, IMatcher.Output]: ...
    
    @Timer.cpu_timeit("Frontend.estimate")
    @Timer.gpu_timeit("Frontend.estimate")
    def estimate(self, frame_t1: StereoData | None, frame_t2: StereoData) -> tuple[IStereoDepth.Output, IMatcher.Output | None]:
        if frame_t1 is None:
            # Will not trigger optimization on the first frame (since tensor shapes are different from the next N frames)
            return super().estimate(frame_t1, frame_t2)

        # Joint inference
        depth_pair = (frame_t2.imageL, frame_t2.imageR)
        match_pair = (frame_t1.imageL, frame_t2.imageL)
        input_A = torch.cat([depth_pair[0], match_pair[0]], dim=0)
        input_B = torch.cat([depth_pair[1], match_pair[1]], dim=0)
        
        input_A = input_A.to(self._get_dtype(self.config.dtype))
        input_B = input_B.to(self._get_dtype(self.config.dtype))

        est_flow, est_cov = self.cuda_graph_estimate(input_A, input_B)
        time.sleep(0.0) # Hint OS scheduler for context switch
        
        est_flow = est_flow.float()
        est_cov  = est_cov.float()
        
        # Depth estimation
        disparity, disparity_cov = est_flow[0:1, :1].abs(), est_cov[0:1, :1]
        depth_map = disparity_to_depth(disparity, frame_t2.frame_baseline, frame_t2.fx)
        depth_cov = disparity_to_depth_cov(disparity, disparity_cov, frame_t2.frame_baseline, frame_t2.fx)
        
        if self.config.enforce_positive_disparity:
            bad_mask = est_flow[0:1, :1] <= 0
        else:
            bad_mask = None
        
        # Matching estimation
        match_map = est_flow[1:2]
        match_cov = est_cov[1:2]
        if self.config.max_flow != -1:
            match_mask        = est_flow < self.config.max_flow
            match_mask        = torch.logical_and(match_mask[1:2, :1], match_mask[1:2, 1:])
        else:
            match_mask = None
            
        return IStereoDepth.Output(depth=depth_map, cov=depth_cov, disparity=disparity, disparity_uncertainty=disparity_cov,mask=bad_mask), \
                IMatcher.Output(flow=match_map, cov=match_cov, mask=match_mask).as_full_cov

    def cuda_graph_estimate(self, inp_A: torch.Tensor, inp_B: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        If does not exist a cuda graph
            build one and run inference through it. 
            Store the resulted graph in frontend for future use.
        If does exist a cuda graph
            Launch graph with new input.
        """
        if self.cuda_graph is None:
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
            
            self.cuda_graph = CUDAGraphHandler(
                graph, inp_A.shape,
                static_input={"input_A": static_input_A, "input_B": static_input_B},
                static_ouput={"flow": static_output, "flow_cov": static_output_cov}
            )
            Logger.write("info", "CUDAGraph Built. Will use CUDAGraph for accelerated inference.")
            
            return output_val, output_cov
        else:
            g_context = self.cuda_graph
            
            assert inp_A.shape == g_context.shape, f"Input shape mismatch for CUDAGraph replay: {inp_A.shape} != {g_context.shape}"
            
            g_context.static_input["input_A"].copy_(inp_A)
            g_context.static_input["input_B"].copy_(inp_B)
            
            g_context.graph.replay()
            time.sleep(0.0) # Hint OS scheduler for context switch
            
            result_val = g_context.static_ouput["flow"].clone()
            result_cov = g_context.static_ouput["flow_cov"].clone()
            
        return result_val, result_cov
