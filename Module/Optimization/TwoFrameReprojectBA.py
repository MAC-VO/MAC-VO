import torch
import pypose as pp
from dataclasses import dataclass

from Module.Map import VisualMap, FrameNode, MatchObs, PointNode
from .Interface import IOptimizer

from typing import Any

from pypose.optim import LM
from pypose.optim.corrector import FastTriggs
from pypose.optim.kernel import Huber
from pypose.optim.scheduler import StopOnPlateau
from pypose.optim.solver import PINV
from pypose.optim.strategy import TrustRegion

from Utility.Timer import Timer
from Utility.Point import point2pixel_NED

# Define arguments

@dataclass
class GraphInput:
    frame2fix: FrameNode    # In BA setup, we also need the fixed previous frame as 'anchor'.
    frame2opt: FrameNode
    observations: MatchObs
    points: PointNode
    images_intrinsic: torch.Tensor
    edges_index: torch.Tensor


@dataclass
class GraphOutput:
    frame2opt: FrameNode
    points   : PointNode


class ReprojectBAOptimizer(IOptimizer[GraphInput, dict, GraphOutput]):
    """
    Implements a simple two-frame pose graph optimization with Reprojection Error formulation.
    """
    def _get_graph_data(self, global_map: VisualMap, frame_idx: torch.Tensor) -> GraphInput:
        frame2opt = global_map.frames[frame_idx]

        obs = global_map.get_frame2match(frame2opt)
        pts = global_map.get_match2point(obs)
        im_intrinsics = frame2opt.data["K"][0]
        
        lengths = global_map.frame2match.ranges[frame2opt.index, :, 1].flatten()
        lengths = lengths[lengths >= 0]
        edges_idx = torch.repeat_interleave(torch.arange(lengths.size(0)), lengths.long())        

        return GraphInput(
            global_map.frames[frame_idx - 1],
            global_map.frames[frame_idx],   
            obs, pts, im_intrinsics, edges_idx
        )
    
    @staticmethod
    def init_context(config: Any) -> dict:
        return {
            "kernel": Huber(delta=0.1),
            "solver": PINV(),
            "strategy": TrustRegion(radius=1e3),
            "corrector": FastTriggs(Huber(delta=0.1)),
            "vectorize": config.vectorize,
            "device": config.device,
            "constraint": config.constraint,
        }

    @staticmethod
    def _optimize(context: dict, graph_data: GraphInput) -> tuple[dict, GraphOutput]:
        with Timer.CPUTimingContext("TwoframeBA"), Timer.GPUTimingContext("TwoframeBA", torch.cuda.current_stream()):
            match context["constraint"]:
                case "none":
                    graph = PoseGraph(graph_data).to(context["device"])
                case "network":
                    graph = PoseGraph_withDepthConstraint(graph_data, use_network_dcov=True).to(context["device"])
                case "match_aware":
                    graph = PoseGraph_withDepthConstraint(graph_data, use_network_dcov=False).to(context["device"])
                case "disparity":
                    graph = PoseGraph_withDisparityConstraint(graph_data).to(context["device"])
                case _: raise Exception('Expect to be one of "none", "network", "match_aware"')
            
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
        global_map.points[result.points.index]    = result.points


class PoseGraph(torch.nn.Module):
    def __init__(self, graph_data: GraphInput) -> None:
        super().__init__()
        self.frame2opt: FrameNode = graph_data.frame2opt
        self.point2opt: PointNode = graph_data.points
        
        self.pose2opt       = pp.Parameter(pp.SE3(self.frame2opt.data["pose"]))
        self.pos_Tw         = torch.nn.Parameter(graph_data.points.data["pos_Tw"])
        self.edges_index    = graph_data.edges_index
        
        # Reprojection-based Residual
        self.register_buffer("pose2fix", pp.SE3(graph_data.frame2fix.data["pose"]))
        self.register_buffer("K"       , graph_data.images_intrinsic)
        self.register_buffer("cov_Tw"  , graph_data.points.data["cov_Tw"])
        self.register_buffer("kp1"     , graph_data.observations.data["pixel1_uv"])
        self.register_buffer("kp2"     , graph_data.observations.data["pixel2_uv"])
        
        N = graph_data.observations.data["pixel2_uv_cov"].size(0)
        self.register_buffer("cov_kp2" , torch.empty((N, 2, 2)))
        self.cov_kp2[:, 0, 0] = graph_data.observations.data["pixel2_uv_cov"][:, 0]
        self.cov_kp2[:, 1, 1] = graph_data.observations.data["pixel2_uv_cov"][:, 1]
        self.cov_kp2[:, 0, 1] = graph_data.observations.data["pixel2_uv_cov"][:, 2]
        self.cov_kp2[:, 1, 0] = graph_data.observations.data["pixel2_uv_cov"][:, 2]
        
        self.register_buffer("cov_kp1", torch.empty((N, 2, 2)))
        self.cov_kp1[:, 0, 0] = graph_data.observations.data["pixel1_uv_cov"][:, 0]
        self.cov_kp1[:, 1, 1] = graph_data.observations.data["pixel1_uv_cov"][:, 1]
        self.cov_kp1[:, 0, 1] = graph_data.observations.data["pixel1_uv_cov"][:, 2]
        self.cov_kp1[:, 1, 0] = graph_data.observations.data["pixel1_uv_cov"][:, 2]
        
        self.register_buffer("cov", torch.cat((self.cov_kp1, self.cov_kp2), dim=0))

    def forward(self) -> torch.Tensor:
        frame1_reprojerr = point2pixel_NED(self.pose2fix.Inv().Act(self.pos_Tw), self.K) - self.kp1
        frame2_reprojerr = point2pixel_NED(self.pose2opt.Inv().Act(self.pos_Tw), self.K) - self.kp2
        return torch.cat((frame1_reprojerr, frame2_reprojerr), dim=0)
    
    @torch.no_grad()
    @torch.inference_mode()
    def covariance_array(self) -> torch.Tensor:
        return self.cov

    @torch.no_grad()
    @torch.inference_mode()
    def write_back(self) -> GraphOutput:
        self.frame2opt.data["pose"].copy_(self.pose2opt.data)
        self.point2opt.data["pos_Tw"].copy_(self.pos_Tw.data)
        return GraphOutput(frame2opt=self.frame2opt, points=self.point2opt)


class PoseGraph_withDepthConstraint(PoseGraph):
    def __init__(self, graph_data: GraphInput, use_network_dcov: bool) -> None:
        super().__init__(graph_data)
        self.register_buffer("kp1_depth", graph_data.observations.data["pixel1_d"].squeeze(-1))
        self.register_buffer("kp2_depth", graph_data.observations.data["pixel2_d"].squeeze(-1))
        
        self.register_buffer("kp1_d_cov", graph_data.observations.data["pixel1_d_cov"].squeeze(-1))
        if use_network_dcov:
            self.register_buffer("kp2_d_cov", graph_data.observations.data["pixel2_d_cov"].squeeze(-1))
        else:
            self.register_buffer("kp2_d_cov", graph_data.observations.data["obs2_covTc"][:, 0, 0])
        
        new_cov = torch.zeros((self.cov.size(0), 3, 3))
        new_cov[:, :2, :2] = self.cov
        new_cov[:, 2, 2]   = torch.cat((self.kp1_d_cov, self.kp2_d_cov), dim=0)
        
        self.cov = new_cov
    
    def forward(self) -> torch.Tensor:
        frame1_deptherr  = self.pose2fix.Inv().Act(self.pos_Tw)[:, 0] - self.kp1_depth
        frame2_deptherr  = self.pose2opt.Inv().Act(self.pos_Tw)[:, 0] - self.kp2_depth
        frame1_reprojerr = point2pixel_NED(self.pose2fix.Inv().Act(self.pos_Tw), self.K) - self.kp1
        frame2_reprojerr = point2pixel_NED(self.pose2opt.Inv().Act(self.pos_Tw), self.K) - self.kp2
        
        result = torch.cat([
            torch.cat((frame1_reprojerr, frame2_reprojerr), dim=0),
            torch.cat((frame1_deptherr, frame2_deptherr), dim=0).unsqueeze(-1),
        ], dim=1)
        return result


class PoseGraph_withDisparityConstraint(PoseGraph):
    def __init__(self, graph_data: GraphInput) -> None:
        super().__init__(graph_data)
        self.register_buffer("blfx_1", graph_data.frame2fix.data["baseline"] * graph_data.frame2fix.data["K"][0, 0, 0])
        self.register_buffer("blfx_2", graph_data.frame2opt.data["baseline"] * graph_data.frame2opt.data["K"][0, 0, 0])
        self.register_buffer("kp1_disparity", graph_data.observations.data["pixel1_disp"].squeeze(-1))
        self.register_buffer("kp2_disparity", graph_data.observations.data["pixel2_disp"].squeeze(-1))
        
        self.register_buffer("kp1_d_cov", graph_data.observations.data["pixel1_disp_cov"].squeeze(-1))
        self.register_buffer("kp2_d_cov", graph_data.observations.data["pixel2_disp_cov"].squeeze(-1))
        
        new_cov = torch.zeros((self.cov.size(0), 3, 3))
        new_cov[:, :2, :2] = self.cov
        new_cov[:, 2, 2]   = torch.cat((self.kp1_d_cov, self.kp2_d_cov), dim=0)
        
        self.cov = new_cov
    
    def forward(self) -> torch.Tensor:
        frame1_disperr  = (self.pose2fix.Inv().Act(self.pos_Tw)[:, 0].reciprocal() * self.blfx_1) - self.kp1_disparity
        frame2_disperr  = (self.pose2opt.Inv().Act(self.pos_Tw)[:, 0].reciprocal() * self.blfx_2) - self.kp2_disparity
        frame1_reprojerr = point2pixel_NED(self.pose2fix.Inv().Act(self.pos_Tw), self.K) - self.kp1
        frame2_reprojerr = point2pixel_NED(self.pose2opt.Inv().Act(self.pos_Tw), self.K) - self.kp2
        
        result = torch.cat([
            torch.cat((frame1_reprojerr, frame2_reprojerr), dim=0),
            torch.cat((frame1_disperr  , frame2_disperr  ), dim=0).unsqueeze(-1),
        ], dim=1)
        return result
