import torch
import pypose as pp

from Module.Map import TensorMap, BatchFrame, BatchObservation, BatchPoints
from .Interface import IOptimizer

from typing import Any, TypedDict

from pypose.optim import LM
from pypose.optim.corrector import FastTriggs
from pypose.optim.kernel import Huber
from pypose.optim.scheduler import StopOnPlateau
from pypose.optim.solver import PINV
from pypose.optim.strategy import TrustRegion

class PyposeContext(TypedDict):
    kernel: Huber
    solver: PINV
    strategy: TrustRegion
    corrector: FastTriggs
    vectorize: bool
    device: str

# Define arguments

class PyposeGraphInput:
    def __init__(self,
        frame2opt: BatchFrame,
        edges: list[tuple[BatchObservation, BatchPoints, torch.Tensor]],
        device: str
    ):
        self.frame2opt: BatchFrame = frame2opt
        self.edges: list[tuple[BatchObservation, BatchPoints, torch.Tensor]] = edges
        self.device: str = device
    
    def move_self_to_local(self):
        local_frame = self.frame2opt.apply(lambda x: x.clone(), inplace=False)
        local_edges = [
            (
                obs.apply(lambda x: x.clone(), inplace=False),
                pts.apply(lambda x: x.clone(), inplace=False),
                K.clone()
            ) for obs, pts, K in self.edges
        ]
        return PyposeGraphInput(local_frame, local_edges, self.device)


class PyposeGraphOutput:
    def __init__(self, frame2opt: BatchFrame):
        self.frame2opt: BatchFrame = frame2opt
    
    def move_self_to_local(self):
        local_frame = self.frame2opt.apply(lambda x: x.clone(), inplace=False)
        return PyposeGraphOutput(local_frame)


class PyPoseTwoFramePGO(IOptimizer[PyposeGraphInput, PyposeContext, PyposeGraphOutput]):
    """
    Implements a simple two-frame pose graph optimization using Pypose library.
    """
    def _get_graph_data(self, global_map: TensorMap, frame_idx: list[int]) -> PyposeGraphInput:
        frame2opt = global_map.frames[frame_idx].unique_contraction()
        edges = []
        for frame_id in range(len(frame2opt)):
            edges.append((
                global_map.get_frame_observes(frame2opt[frame_id]),
                global_map.get_frame_points(frame2opt[frame_id]),
                frame2opt[frame_id].K
            ))
        return PyposeGraphInput(frame2opt, edges, "cpu")
    
    @staticmethod
    def init_context(config: Any) -> PyposeContext:
        return {
            "kernel": Huber(delta=0.1),
            "solver": PINV(),
            "strategy": TrustRegion(radius=1e3),
            "corrector": FastTriggs(Huber(delta=0.1)),
            "vectorize": config.vectorize,
            "device": config.device
        }

    @staticmethod
    def _optimize(context: PyposeContext, graph_data: PyposeGraphInput) -> tuple[PyposeContext, PyposeGraphOutput]:        
        graph = PoseGraph(graph_data)
        
        optimizer = LM(graph.double(), solver=context["solver"],
                        strategy=context["strategy"], kernel=context["kernel"],
                        corrector=context["corrector"],
                        min=1e-6, vectorize=context["vectorize"])
        scheduler = StopOnPlateau(optimizer, steps=10, patience=2, decreasing=1e-5, verbose=False)
        
        while scheduler.continual():
            weight = torch.block_diag(*torch.pinverse(graph.covariance().to(context["device"])).double())
            loss = optimizer.step(input=(), weight=weight)
            scheduler.step(loss)
        
        return context, graph.write_back()

    @staticmethod
    def _write_map(result: PyposeGraphOutput | None, global_map: TensorMap) -> None:
        if result is None: return
        global_map.frames.update(result.frame2opt, global_map.frames.Scatter.POSE)


class PoseGraph(torch.nn.Module):
    def __init__(self, args: PyposeGraphInput) -> None:
        super().__init__()
        self.device = args.device
        self.frame2opt: BatchFrame = args.frame2opt
        self.edges: list[PointObservationEdge] = []
        
        for obs, pts, K in args.edges:
            self.edges.append(PointObservationEdge(obs, pts, K).to(self.device))
        
        self.pose2opt = pp.Parameter(self.frame2opt.pose)

    def forward(self) -> torch.Tensor:
        losses: list[torch.Tensor] = []
        for idx, edge in enumerate(self.edges):
            losses.append(edge.loss(self.pose2opt[idx]))    #type: ignore
        return torch.cat(losses, dim=0)

    @torch.no_grad()
    @torch.inference_mode()
    def covariance(self) -> torch.Tensor:
        cov_blocks: list[torch.Tensor] = []
        for idx, edge in enumerate(self.edges):
            cov_blocks.append(edge.cov_blocks(self.pose2opt[idx])) #type: ignore
        
        cov_blocks_cat: torch.Tensor = torch.cat(cov_blocks, dim=0)
        return cov_blocks_cat

    @torch.no_grad()
    @torch.inference_mode()
    def write_back(self) -> PyposeGraphOutput:
        self.frame2opt.pose.copy_(self.pose2opt.data)
        return PyposeGraphOutput(frame2opt=self.frame2opt)


class PointObservationEdge(torch.nn.Module):
    def __init__(self, obs: BatchObservation, pts: BatchPoints, K: torch.Tensor):
        super().__init__()
        assert obs.is_batched and pts.is_batched
        self.pts = pts
        self.obs = obs
        
        points_Tc = obs.project_Tc(K)
        points_Tw = pts.position
        self.register_buffer("points_Tc", points_Tc.double())
        self.register_buffer("points_Tw", points_Tw.double())
        self.register_buffer("obs_covTc", obs.cov_Tc)
        self.register_buffer("pts_covTw", pts.cov_Tw)

    def loss(self, frame_pose: pp.Parameter) -> torch.Tensor:
        return frame_pose.Act(self.points_Tc) - self.points_Tw
    
    @torch.no_grad()
    @torch.inference_mode()
    def cov_blocks(self, frame_pose: pp.Parameter) -> torch.Tensor:
        R = frame_pose.rotation().matrix()
        return (R @ self.obs_covTc @ R.T) + self.pts_covTw
