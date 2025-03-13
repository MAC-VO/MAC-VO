import torch
import pypose as pp

from dataclasses import dataclass
from Module.Map import VisualMap, FrameNode, MatchObs, PointNode
from .Interface import IOptimizer

from typing import Any, cast

from pypose.optim import LM
from pypose.optim.corrector import FastTriggs
from pypose.optim.kernel import Huber
from pypose.optim.scheduler import StopOnPlateau
from pypose.optim.solver import PINV
from pypose.optim.strategy import TrustRegion

from Utility.Timer import Timer
from Utility.Point import pixel2point_NED

# Define arguments

@dataclass
class GraphInput:
    frame2opt: FrameNode
    observations: MatchObs
    points: PointNode
    images_intrinsic: torch.Tensor
    edges_index: torch.Tensor
    device: str


@dataclass
class GraphOutput:
    frame2opt: FrameNode


class TwoFramePoseOptimizer(IOptimizer[GraphInput, dict, GraphOutput]):
    """
    Implements a simple two-frame pose graph optimization with ICP-like residual.
    """
    def _get_graph_data(self, global_map: VisualMap, frame_idx: torch.Tensor) -> GraphInput:
        frame2opt = global_map.frames[frame_idx]

        obs = global_map.get_frame2match(frame2opt)
        pts = global_map.get_match2point(obs)
        im_intrinsics = frame2opt.data["K"]
        
        lengths = global_map.frame2match.ranges[frame2opt.index, :, 1].flatten()
        lengths = lengths[lengths >= 0]
        edges_idx = torch.repeat_interleave(torch.arange(lengths.size(0)), lengths.long())        

        return GraphInput(frame2opt, obs, pts, im_intrinsics, edges_idx, "cpu")
    
    @staticmethod
    def init_context(config: Any) -> dict:
        return {
            "kernel": Huber(delta=0.1),
            "solver": PINV(),
            "strategy": TrustRegion(radius=1e3),
            "corrector": FastTriggs(Huber(delta=0.1)),
            "vectorize": config.vectorize,
            "device": config.device
        }

    @staticmethod
    def _optimize(context: dict, graph_data: GraphInput) -> tuple[dict, GraphOutput]:
        with Timer.CPUTimingContext("TwoframePGO"), Timer.GPUTimingContext("TwoframePGO", torch.cuda.current_stream()):
            graph = PoseGraph(graph_data).to(context["device"])
            
            optimizer = LM(graph.double(), solver=context["solver"],
                            strategy=context["strategy"], kernel=context["kernel"],
                            corrector=context["corrector"],
                            min=1e-6, vectorize=context["vectorize"])
            scheduler = StopOnPlateau(optimizer, steps=10, patience=2, decreasing=1e-5, verbose=False)
            
            while scheduler.continual():
                weight = torch.block_diag(*(
                    torch.pinverse(graph.covariance_array().to(context["device"]).double())
                ))
                loss = optimizer.step(input=(), weight=weight)
                scheduler.step(loss)
        return context, graph.write_back()

    @staticmethod
    def _write_map(result: GraphOutput | None, global_map: VisualMap) -> None:
        if result is None: return
        global_map.frames[result.frame2opt.index] = result.frame2opt


class PoseGraph(torch.nn.Module):
    def __init__(self, graph_data: GraphInput) -> None:
        super().__init__()
        self.device                = graph_data.device
        self.frame2opt: FrameNode = graph_data.frame2opt
        
        self.pose2opt       = pp.Parameter(pp.SE3(self.frame2opt.data["pose"]))
        self.edges_index    = graph_data.edges_index
        
        # ICP-based residual
        self.pts = graph_data.points
        self.obs = graph_data.observations
        
        self.register_buffer("K", graph_data.images_intrinsic)
        self.register_buffer("points_Tc",
            pixel2point_NED(self.obs.data["pixel2_uv"], self.obs.data["pixel2_d"].squeeze(-1), self.K)
        )
        self.register_buffer("points_Tw", self.pts.data["pos_Tw"])
        self.register_buffer("obs_covTc", self.obs.data["obs2_covTc"])
        self.register_buffer("pts_covTw", self.pts.data["cov_Tw"])
        

    def forward(self) -> torch.Tensor:
        frame_pose = cast(pp.LieTensor, self.pose2opt[self.edges_index])
        return frame_pose.Act(self.points_Tc) - self.points_Tw
    
    @torch.no_grad()
    @torch.inference_mode()
    def covariance_array(self) -> torch.Tensor:
        frame_pose = cast(pp.LieTensor, self.pose2opt[self.edges_index])
        R  = frame_pose.rotation().matrix()
        RT = R.transpose(-2, -1)
        return (R @ self.obs_covTc @ RT) + self.pts_covTw

    @torch.no_grad()
    @torch.inference_mode()
    def write_back(self) -> GraphOutput:
        self.frame2opt.data["pose"].copy_(self.pose2opt.data)
        return GraphOutput(frame2opt=self.frame2opt)
