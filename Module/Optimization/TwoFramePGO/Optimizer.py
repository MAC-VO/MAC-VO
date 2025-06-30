import torch
from types import SimpleNamespace
import pypose as pp

from pypose.optim import LM
from pypose.optim.corrector import FastTriggs
from pypose.optim.kernel import Huber
from pypose.optim.scheduler import StopOnPlateau
from pypose.optim.solver import PINV
from pypose.optim.strategy import TrustRegion

from Module.Map import VisualMap
from Utility.Timer import Timer
from Utility.Math  import NormalizeQuat

from ..Interface import IOptimizer
from ..PyposeOptimizers import LM_analytic, AnalyticModule, FactorGraph
from .Graphs import GraphInput, GraphOutput
from .Graphs import ICP_TwoframePGO, Reproj_TwoFramePGO, ReprojDisp_TwoFramePGO
from .Graphs import Analytic_ICP_TwoframePGO, Analytic_Reproj_TwoFramePGO, Analytic_ReprojDisp_TwoFramePGO


class TwoFrame_PGO(IOptimizer[GraphInput, dict, GraphOutput]):
    @torch.no_grad()
    def get_graph_data(self, global_map: VisualMap, frame_idx: torch.Tensor,
                       observations: torch.Tensor | None = None, edges: torch.Tensor | None = None) -> GraphInput:
        frame2opt = global_map.frames[frame_idx]

        obs = global_map.get_frame2match(frame2opt)
        pts = global_map.get_match2point(obs)
        im_intrinsics = frame2opt.data["K"][0]

        lengths = global_map.frame2match.ranges[frame2opt.index, :, 1].flatten()
        lengths = lengths[lengths >= 0]
        edges_idx = torch.repeat_interleave(torch.arange(lengths.size(0)), lengths.long())
        init_motion = pp.SE3(frame2opt.data["pose"])
        baseline = frame2opt.data["baseline"]
        return GraphInput(frame_idx, frame_idx - 1, init_motion, baseline, obs, pts, im_intrinsics, edges_idx, "cpu")

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "graph_type": lambda s: s in {"icp", "reproj", "disp"},
            "device": lambda v: isinstance(v, str) and (v == "cpu" or "cuda" in v),
            "vectorize": lambda b: isinstance(b, bool),
            "parallel": lambda b: isinstance(b, bool),
            "autodiff": lambda b: isinstance(b, bool)
        })

    @staticmethod
    def init_context(config) -> dict:
        match (config.autodiff, config.graph_type):
            case (True, "icp"):
                PoseGraphClass = ICP_TwoframePGO
            case (True, "reproj"):
                PoseGraphClass = Reproj_TwoFramePGO
            case (True, "disp"):
                PoseGraphClass = ReprojDisp_TwoFramePGO
            case (False, "icp"):
                PoseGraphClass = Analytic_ICP_TwoframePGO
            case (False, "reproj"):
                PoseGraphClass = Analytic_Reproj_TwoFramePGO
            case (False, "disp"):
                PoseGraphClass = Analytic_ReprojDisp_TwoFramePGO
            case _:
                raise ValueError(f"Graph type of {config.graph_type} is not supported")

        return {
            "optimizer_cfg": {
                "kernel"   : Huber(delta=0.1),
                "solver"   : PINV(),
                "strategy" : TrustRegion(radius=1e3),
                "corrector": FastTriggs(Huber(delta=0.1)),
                "vectorize": config.vectorize,
            },
            "device": config.device,

            "pose_graph_class": PoseGraphClass
        }

    @staticmethod
    def _optimize(context: dict, graph_data: GraphInput) -> tuple[dict, GraphOutput]:
        with Timer.CPUTimingContext("TwoframePGO"), Timer.GPUTimingContext("TwoframePGO", torch.cuda.current_stream()):
            graph: FactorGraph = context["pose_graph_class"](graph_data)\
                .to(device=torch.device(context["device"]), dtype=torch.double)
            assert isinstance(graph, FactorGraph)

            if isinstance(graph, AnalyticModule):
                optimizer = LM_analytic(graph, min=1e-6, **context["optimizer_cfg"])
            else:
                optimizer = LM(graph, min=1e-6, **context["optimizer_cfg"])

            scheduler = StopOnPlateau(optimizer, steps=10, patience=2, decreasing=1e-5, verbose=False)

            while scheduler.continual():
                weight = torch.block_diag(*(
                    torch.pinverse(graph.covariance_array().to(context["device"]).double())
                ))
                loss = optimizer.step(input=(), weight=weight)
                scheduler.step(loss)

        return context, graph.write_back()

    def write_graph_data(self, result: GraphOutput | None, global_map: VisualMap) -> None:
        if result is None: return
        
        to_pose     = pp.SE3(result.motion[0].data.double().cpu())
        global_map.frames.data["pose"][result.frame_idx] = to_pose.float()


class Local_TwoFrame_PGO(TwoFrame_PGO):
    """
    Simple two-frame PGO in visual-odometry (MAC-VO) under Local frame. May lead to better optimization
    due to more numerical stability (especially in large-scene with 1000+ meters size)
    """
    def get_graph_data(self, global_map: VisualMap, frame_idx: torch.Tensor,
                       observations: torch.Tensor | None = None, edges: torch.Tensor | None = None) -> GraphInput:
        global_graph_data = super().get_graph_data(global_map, frame_idx, observations, edges)
        self.T_o2w_idx = frame_idx - 1

        T_o2w = pp.SE3(global_map.frames.data["pose"][frame_idx - 1])
        T_w2o = T_o2w.Inv()
        return self.world_to_optim(global_graph_data, T_w2o)

    def write_graph_data(self, result: GraphOutput | None, global_map: VisualMap) -> None:
        if result is None: return

        T_o2w = pp.SE3(global_map.frames.data["pose"][self.T_o2w_idx])
        super().write_graph_data(self.optim_to_world(result, T_o2w), global_map)

    def world_to_optim(self, data: GraphInput, T_w2o: pp.LieTensor) -> GraphInput:
        """Transform the optimization graph data into local reference frame (i.e. the reference frame is the pose of previous key frame)
        """
        # Same for below:
        # c = camera to optimize, o = optimization frame, w = world (global) frame
        T_c2w = pp.LieTensor(data.init_motion, ltype=pp.SE3_type)
        T_c2o = T_w2o @ T_c2w
        R_w2o = T_w2o.rotation().matrix().to(data.points.data["cov_Tw"])

        data.init_motion = T_c2o
        data.points.data["pos_Tw"]  = pp.Act(pp.SE3(T_w2o.to(data.points.data["pos_Tw"])), data.points.data["pos_Tw"])
        data.points.data["cov_Tw"]  = R_w2o @ data.points.data["cov_Tw"] @ R_w2o.transpose(-1, -2)
        return data

    def optim_to_world(self, data: GraphOutput, T_o2w: pp.LieTensor) -> GraphOutput:
        """Transform the optimization result under local reference frame (w.r.t. previous KF) to the global frame.
        """
        T_c2o = data.motion
        data.motion = NormalizeQuat(T_o2w @ pp.SE3(T_c2o.to(T_o2w)))
        return data


class Empty_TwoFrame_PGO(TwoFrame_PGO):
    """
    A 'no-op' variant of the Two-frame PGO optimizer. Helpful in debugging process.
    """
    @staticmethod
    def _optimize(context: dict, graph_data: GraphInput) -> tuple[dict, GraphOutput]:
        return context, GraphOutput(motion=graph_data.init_motion,
                                    frame_idx=graph_data.frame_idx,
                                    from_idx=graph_data.from_idx)
