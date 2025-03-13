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
    frame2opt: FrameNode
    observations: MatchObs
    points: PointNode
    images_intrinsic: torch.Tensor
    baseline: torch.Tensor
    edges_index: torch.Tensor


@dataclass
class GraphOutput:
    frame2opt: FrameNode


class ReprojectOptimizer(IOptimizer[GraphInput, dict, GraphOutput]):
    """
    Implements a simple two-frame pose graph optimization with Reprojection Error formulation.
    """
    def _get_graph_data(self, global_map: VisualMap, frame_idx: torch.Tensor) -> GraphInput:
        frame2opt = global_map.frames[frame_idx]

        obs = global_map.get_frame2match(frame2opt)
        pts = global_map.get_match2point(obs)
        im_intrinsics = frame2opt.data["K"][0]
        baseline = frame2opt.data["baseline"]
        
        lengths = global_map.frame2match.ranges[frame2opt.index, :, 1].flatten()
        lengths = lengths[lengths >= 0]
        edges_idx = torch.repeat_interleave(torch.arange(lengths.size(0)), lengths.long())        
        
        return GraphInput(frame2opt, obs, pts, im_intrinsics, baseline, edges_idx)
    
    @staticmethod
    def init_context(config: Any) -> dict:
        return {
            "kernel": Huber(delta=0.1),
            "solver": PINV(),
            "strategy": TrustRegion(radius=1e3),
            "corrector": FastTriggs(Huber(delta=0.1)),
            "vectorize": config.vectorize,
            "device": config.device,
            "use_fancy_cov": config.use_fancy_cov,
            "constraint": config.constraint,
        }

    @staticmethod
    def _optimize(context: dict, graph_data: GraphInput) -> tuple[dict, GraphOutput]:
        with Timer.CPUTimingContext("TwoframePGO"), Timer.GPUTimingContext("TwoframePGO", torch.cuda.current_stream()):
            match context["constraint"]:
                case "none":
                    graph = PoseGraph(graph_data, context["use_fancy_cov"]).to(context["device"])
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


class PoseGraph(torch.nn.Module):
    def __init__(self, graph_data: GraphInput, use_fancy_cov: bool) -> None:
        super().__init__()
        self.use_fancy_cov        = use_fancy_cov
        self.frame2opt: FrameNode = graph_data.frame2opt
        
        self.pose2opt       = pp.Parameter(pp.SE3(self.frame2opt.data["pose"]))
        self.edges_index    = graph_data.edges_index
        
        # ICP-based residual
        self.pts     = graph_data.points
        self.obs     = graph_data.observations
        
        self.register_buffer("K", graph_data.images_intrinsic)
        self.register_buffer("baseline", graph_data.baseline)
        self.register_buffer("pos_Tw" , self.pts.data["pos_Tw"])
        self.register_buffer("cov_Tw" , self.pts.data["cov_Tw"])
        self.register_buffer("kp2"    , self.obs.data["pixel2_uv"])
        
        N = self.obs.data["pixel2_uv_cov"].size(0)
        self.register_buffer("cov_kp2" , torch.empty((N, 2, 2)))
        self.cov_kp2[:, 0, 0] = self.obs.data["pixel2_uv_cov"][:, 0]
        self.cov_kp2[:, 1, 1] = self.obs.data["pixel2_uv_cov"][:, 1]
        self.cov_kp2[:, 0, 1] = self.obs.data["pixel2_uv_cov"][:, 2]
        self.cov_kp2[:, 1, 0] = self.obs.data["pixel2_uv_cov"][:, 2]

    def forward(self) -> torch.Tensor:
        return point2pixel_NED(self.pose2opt.Inv().Act(self.pos_Tw), self.K) - self.kp2
    
    @torch.no_grad()
    @torch.inference_mode()
    def covariance_array(self) -> torch.Tensor:
        if not self.use_fancy_cov:
            return self.cov_kp2
        
        N = self.cov_Tw.size(0)
        R_cw: torch.Tensor = self.pose2opt.rotation().matrix().to(self.cov_Tw)
        cov_Tc  = R_cw @ self.cov_Tw @ R_cw.transpose(-1, -2)

        mean_Tc = self.pose2opt.Inv().Act(self.pos_Tw)
        
        # Since projection is non-linear, we perform first-order approximation with Jacobian.
        J = torch.empty((N, 2, 3), device=self.cov_Tw.device, dtype=self.cov_Tw.dtype)
        J[:, 0, 0] = self.K[0, 0] / mean_Tc[:, 0]
        J[:, 0, 1] = 0.
        J[:, 0, 2] = -(self.K[0, 0] * mean_Tc[:, 1]) / mean_Tc[:, 0].square()
        J[:, 1, 0] = 0.
        J[:, 1, 1] = self.K[1, 1] / mean_Tc[:, 0]
        J[:, 1, 2] = -(self.K[1, 1] * mean_Tc[:, 2]) / mean_Tc[:, 0].square()
        
        cov_kp1 = torch.bmm(torch.bmm(J, cov_Tc), J.transpose(-1, -2))
        return cov_kp1 + self.cov_kp2

    @torch.no_grad()
    @torch.inference_mode()
    def write_back(self) -> GraphOutput:
        self.frame2opt.data["pose"].copy_(self.pose2opt.data)
        return GraphOutput(frame2opt=self.frame2opt)


class PoseGraph_withDepthConstraint(PoseGraph):
    def __init__(self, graph_data: GraphInput, use_network_dcov: bool) -> None:
        super().__init__(graph_data, False)
        self.register_buffer("kp2_depth", graph_data.observations.data["pixel2_d"])
        if use_network_dcov:
            self.register_buffer("kp2_depth_cov", graph_data.observations.data["pixel2_d_cov"].squeeze(-1))
        else:
            self.register_buffer("kp2_depth_cov", graph_data.observations.data["obs2_covTc"][:, 0, 0])
        
        N = self.cov_kp2.size(0)
        self.register_buffer("cov", torch.zeros((N, 3, 3)))
        self.cov[:, :2, :2] = self.cov_kp2
        self.cov[:, 2, 2]   = self.kp2_depth_cov
    
    def forward(self) -> torch.Tensor:
        depth_err  = self.pose2opt.Inv().Act(self.pos_Tw)[:, 0:1] - self.kp2_depth
        reproj_err = point2pixel_NED(self.pose2opt.Inv().Act(self.pos_Tw), self.K) - self.kp2
        return torch.cat((reproj_err, depth_err), dim=-1)

    def covariance_array(self) -> torch.Tensor:
        return self.cov


class PoseGraph_withDisparityConstraint(PoseGraph):
    def __init__(self, graph_data: GraphInput) -> None:
        super().__init__(graph_data, False)
        self.register_buffer("kp2_disparity", graph_data.observations.data["pixel2_disp"])
        self.register_buffer("kp2_sigma_disparity", graph_data.observations.data["pixel2_disp_cov"])
        
        N = self.cov_kp2.size(0)
        self.register_buffer("cov", torch.zeros((N, 3, 3)))
        self.cov[:, :2, :2] = self.cov_kp2
        self.cov[:, 2, 2]   = self.kp2_sigma_disparity.squeeze(-1)
    
    def forward(self) -> torch.Tensor:
        depth_err  = 1/self.pose2opt.Inv().Act(self.pos_Tw)[:, 0:1] * (self.K[0, 0] * self.baseline) - self.kp2_disparity
        reproj_err = point2pixel_NED(self.pose2opt.Inv().Act(self.pos_Tw), self.K) - self.kp2
        return torch.cat((reproj_err, depth_err), dim=-1)

    def covariance_array(self) -> torch.Tensor:
        return self.cov