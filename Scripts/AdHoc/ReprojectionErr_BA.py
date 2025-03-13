import torch
import pypose as pp

from pypose.optim import LM
from pypose.optim.corrector import FastTriggs
from pypose.optim.kernel import Huber
from pypose.optim.scheduler import StopOnPlateau
from pypose.optim.solver import PINV
from pypose.optim.strategy import TrustRegion

from DataLoader import SequenceBase, StereoFrame
from Module     import IFrontend, ICovariance2to3
from Module.KeypointSelector import GridSelector

from Utility.Config import build_dynamic_config
from Utility.Point import filterPointsInRange, pixel2point_NED, point2pixel_NED
from Utility.Visualize import fig_plt
from Utility.PrettyPrint import Logger

import argparse


class ReprojPGO(torch.nn.Module):
    def __init__(self,
                kp_A_3d: torch.Tensor, pose: pp.LieTensor,
                kp_A: torch.Tensor, kp_B: torch.Tensor, 
                kp_A_depth: torch.Tensor, kp_B_depth: torch.Tensor,
                kp_A_covd : torch.Tensor, kp_B_covd : torch.Tensor,
                kp_B_covuv: torch.Tensor, K: torch.Tensor):
        super().__init__()
        self.register_buffer("pose_frameA", pp.identity_SE3())
        self.pose_frameB = pp.Parameter(pose)
        # self.pts_Tw      = torch.nn.Parameter(kp_A_3d)
        self.register_buffer("pts_Tw", kp_A_3d)
        
        self.register_buffer("kp_A_d", kp_A_depth)
        self.register_buffer("kp_B_d", kp_B_depth)
        self.register_buffer("kp_A_covd", kp_A_covd)
        self.register_buffer("kp_B_covd", kp_B_covd)
        self.register_buffer("kp_A"  , kp_A)
        self.register_buffer("kp_B"  , kp_B)
        self.register_buffer("K", K)
        self.counter = 0
        self.init_matched_covariance(kp_B_covuv)
    
    def init_matched_covariance(self, kp_B_covuv: torch.Tensor) -> None:
        N = kp_B_covuv.size(1)
        self.register_buffer("kp_Bcov" , torch.empty((N, 2, 2)))
        self.register_buffer("kp_B_covuv", kp_B_covuv)
        self.kp_Bcov[:, 0, 0] = kp_B_covuv[0]
        self.kp_Bcov[:, 1, 1] = kp_B_covuv[1]
        self.kp_Bcov[:, 0, 1] = kp_B_covuv[2]
        self.kp_Bcov[:, 1, 0] = kp_B_covuv[2]

    def forward(self) -> torch.Tensor:
        reproj_frameB = point2pixel_NED(self.pose_frameB.Inv().Act(self.pts_Tw), self.K) - self.kp_B
        depth_frameB  = (self.pose_frameB.Inv().Act(self.pts_Tw)[:, 0] - self.kp_B_d).abs()
        
        loss_frameB = torch.cat((reproj_frameB, depth_frameB.T), dim=1)
        return loss_frameB

    @torch.no_grad()
    def covariance(self) -> torch.Tensor:
        kp_B_cov = torch.zeros((3, 3)).unsqueeze(0).repeat(self.kp_B.size(0), 1, 1)
        kp_B_cov[:, :2, :2] = self.kp_Bcov
        kp_B_cov[:, 2, 2]   = self.kp_B_covd
        return kp_B_cov


class PureReprojPGO(ReprojPGO):
    """
    Optimize the Pose based on reprojection error 
    """
    def __init__(self, kp_A_3d: torch.Tensor, pose: pp.LieTensor,
                 kp_A: torch.Tensor, kp_B: torch.Tensor, kp_A_depth: torch.Tensor, kp_B_depth: torch.Tensor, 
                 kp_A_covd: torch.Tensor, kp_B_covd: torch.Tensor, kp_B_covuv: torch.Tensor, K: torch.Tensor):
        super().__init__(kp_A_3d, pose, kp_A, kp_B, kp_A_depth, kp_B_depth, kp_A_covd, kp_B_covd, kp_B_covuv, K)
    
    def forward(self) -> torch.Tensor:
        reproj_frameB = point2pixel_NED(self.pose_frameB.Inv().Act(self.pts_Tw), self.K) - self.kp_B
        return reproj_frameB
    
    def covariance(self) -> torch.Tensor:
        return self.kp_Bcov


class ReprojBA(ReprojPGO):
    def __init__(self, kp_A_3d: torch.Tensor, pose: pp.LieTensor,
                 kp_A: torch.Tensor, kp_B: torch.Tensor, kp_A_depth: torch.Tensor, kp_B_depth: torch.Tensor, 
                 kp_A_covd: torch.Tensor, kp_B_covd: torch.Tensor, kp_B_covuv: torch.Tensor, K: torch.Tensor):
        super().__init__(kp_A_3d, pose, kp_A, kp_B, kp_A_depth, kp_B_depth, kp_A_covd, kp_B_covd, kp_B_covuv, K)
  
        self.pts_Tw      = torch.nn.Parameter(kp_A_3d)

    def forward(self) -> torch.Tensor:
        reproj_frameA = point2pixel_NED(self.pose_frameA.Inv().Act(self.pts_Tw), self.K) - self.kp_A
        reproj_frameB = point2pixel_NED(self.pose_frameB.Inv().Act(self.pts_Tw), self.K) - self.kp_B
        depth_frameA  = (self.pose_frameA.Inv().Act(self.pts_Tw)[:, 0] - self.kp_A_d).abs()
        depth_frameB  = (self.pose_frameB.Inv().Act(self.pts_Tw)[:, 0] - self.kp_B_d).abs()
        # depth_frameB  = torch.zeros_like(depth_frameA)
        
        loss_frameA = torch.cat((reproj_frameA, depth_frameA.T), dim=1)
        loss_frameB = torch.cat((reproj_frameB, depth_frameB.T), dim=1)
        return torch.cat((loss_frameA, loss_frameB), dim=0)
    
    @torch.no_grad()
    def covariance(self) -> torch.Tensor:
        kp_A_cov = (torch.eye(3) * 0.01).unsqueeze(0).repeat(self.kp_A.size(0), 1, 1)
        kp_A_cov[:, 2, 2] = self.kp_A_covd
        
        kp_B_cov = torch.zeros((3, 3)).unsqueeze(0).repeat(self.kp_B.size(0), 1, 1)
        kp_B_cov[:, :2, :2] = self.kp_Bcov
        kp_B_cov[:, 2, 2]   = self.kp_B_covd
        
        return torch.cat((kp_A_cov, kp_B_cov), dim=0)
        
class ReprojBA2(ReprojBA):
    def __init__(self, kp_A_3d: torch.Tensor, pose: pp.LieTensor,
                 kp_A: torch.Tensor, kp_B: torch.Tensor, kp_A_depth: torch.Tensor, kp_B_depth: torch.Tensor, 
                 kp_A_covd: torch.Tensor, kp_B_covd: torch.Tensor, kp_B_covuv: torch.Tensor, K: torch.Tensor):
        super().__init__(kp_A_3d, pose, kp_A, kp_B, kp_A_depth, kp_B_depth, kp_A_covd, kp_B_covd, kp_B_covuv, K)
  

    def forward(self) -> torch.Tensor:
        reproj_frameA = point2pixel_NED(self.pose_frameA.Inv().Act(self.pts_Tw), self.K) - self.kp_A
        reproj_frameB = point2pixel_NED(self.pose_frameB.Inv().Act(self.pts_Tw), self.K) - self.kp_B
        depth_frameA  = (self.pose_frameA.Inv().Act(self.pts_Tw)[:, 0] - self.kp_A_d).abs()
        depth_frameB  = torch.zeros_like(depth_frameA)
        
        loss_frameA = torch.cat((reproj_frameA, depth_frameA.T), dim=1)
        loss_frameB = torch.cat((reproj_frameB, depth_frameB.T), dim=1)
        return torch.cat((loss_frameA, loss_frameB), dim=0)


class PureReprojBA(PureReprojPGO):
    def __init__(self, kp_A_3d: torch.Tensor, pose: pp.LieTensor,
                 kp_A: torch.Tensor, kp_B: torch.Tensor, kp_A_depth: torch.Tensor, kp_B_depth: torch.Tensor, 
                 kp_A_covd: torch.Tensor, kp_B_covd: torch.Tensor, kp_B_covuv: torch.Tensor, K: torch.Tensor):
        super().__init__(kp_A_3d, pose, kp_A, kp_B, kp_A_depth, kp_B_depth, kp_A_covd, kp_B_covd, kp_B_covuv, K)
  
        self.pts_Tw      = torch.nn.Parameter(kp_A_3d)

    def forward(self) -> torch.Tensor:
        reproj_frameA = point2pixel_NED(self.pose_frameA.Inv().Act(self.pts_Tw), self.K) - self.kp_A
        reproj_frameB = point2pixel_NED(self.pose_frameB.Inv().Act(self.pts_Tw), self.K) - self.kp_B
        return torch.cat((reproj_frameA, reproj_frameB), dim=0)
    
    def covariance(self) -> torch.Tensor:
        kp_Acov = torch.zeros_like(self.kp_Bcov)
        kp_Acov[:, 0, 0] = 0.15
        kp_Acov[:, 1, 1] = 0.15
        return torch.cat((kp_Acov, self.kp_Bcov), dim=0)


def parse_args():
    parser = argparse.ArgumentParser(description="Reprojection Error Bundle Adjustment")
    parser.add_argument('--use_gtpose', action='store_true', help='Use ground truth data')
    parser.add_argument('--use_gtpoint', action='store_true', help='Use ground truth data')
    parser.add_argument('--use_gtdcov', action='store_true', help='Use ground truth data')
    parser.add_argument('--class_type', type=str, choices=['ReprojPGO', 'ReprojBA', 'PureReprojBA', 'PureReprojPGO', "ReprojBA2"], default='ReprojPGO', help='Type of class to use')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    data_cfg, _ = build_dynamic_config({
        "type": "TartanAirv2_NoIMU",
        "args": {
            # "root": "~/data/tartanair_v2_test/Data_hard/P000",
            # "root": "/home/yuheng/data/P000",
            "root": "/data2/datasets/yuhengq/tartanair_v2_test/Data_hard/P000",
            "compressed": True,
            "gtDepth": True,
            "gtPose" : True,
            "gtFlow" : True
        }
    })
    net_cfg, _ = build_dynamic_config({
        "type": "FlowFormerCovFrontend",
        "args": {
            "device": "cpu",
            "weight": "./Model/MACVO_FrontendCov.pth",
            "dtype": "fp32",
            "max_flow": -1,
            "enforce_positive_disparity": False
        }
    })
    kp_cfg, _ = build_dynamic_config({
        "mask_width": 32,
        "device"    : "cpu"
    })
    cov2to3_cfg, _ = build_dynamic_config({
        "type": "MatchCovariance",
        "args": {
            "kernel_size": 7,
            "match_cov_default": 0.25,
            "min_flow_cov": 0.25,
            "min_depth_cov": 0.01,
            "device": "cpu"
        }
    })
    fig_plt.set_fn_mode(fig_plt.plot_reprojerr, "image")

    sequence       = SequenceBase[StereoFrame].instantiate(data_cfg.type, data_cfg.args)
    frontend       = IFrontend.instantiate(net_cfg.type, net_cfg.args)
    kpselect       = GridSelector(kp_cfg)
    cov_2to3       = ICovariance2to3.instantiate(cov2to3_cfg.type, cov2to3_cfg.args)

    frameA, frameB = sequence[0], sequence[1]
    frameA_depth, _          = frontend.estimate(None         , frameA.stereo)
    frameB_depth, frame_AtoB = frontend.estimate(frameA.stereo, frameB.stereo)

    kp_A         = kpselect.select_point(frameA.stereo, 50, frameA_depth, frameB_depth, frame_AtoB)
    kp_B         = kp_A + frontend.retrieve_pixels(kp_A, frame_AtoB.flow).T
    inbound_mask = filterPointsInRange(kp_B, 
        (32, frameA.stereo.width - 32), (32, frameA.stereo.height - 32)
    )
    kp_A, kp_B   = kp_A[inbound_mask], kp_B[inbound_mask]

    assert frameA.stereo.gt_depth is not None
    assert frameB.stereo.gt_depth is not None

    kp_A_d   = frontend.retrieve_pixels(kp_A, frameA_depth.depth)
    kp_A_gtD = frontend.retrieve_pixels(kp_A, frameA.stereo.gt_depth.cpu())
    kp_A_covd  = frontend.retrieve_pixels(kp_A, frameA_depth.cov)
    kp_A_covuv = frontend.retrieve_pixels(kp_A, frame_AtoB.cov)
    kp_A_3d  = pixel2point_NED(kp_A, kp_A_d[0], frameA.stereo.frame_K) # z, x, y
    kp_A_3dGT = pixel2point_NED(kp_A, kp_A_gtD[0], frameA.stereo.frame_K)

    assert (kp_A_covuv is not None) and (kp_A_covd is not None)
    kp_A_cov3d = cov_2to3.estimate(frameA.stereo, kp_A, frameA_depth, kp_A_covd.unsqueeze(0), kp_A_covuv.T)

    kp_A_covdGT = (kp_A_d - kp_A_gtD)**2

    kp_B_covuv = frontend.retrieve_pixels(kp_B, frame_AtoB.cov)
    kp_B_d   = frontend.retrieve_pixels(kp_B, frameB_depth.depth)
    kp_B_covd  = frontend.retrieve_pixels(kp_B, frameB_depth.cov)
    kp_B_gtD = frontend.retrieve_pixels(kp_B, frameB.stereo.gt_depth.cpu())
    kp_B_3dGT = pixel2point_NED(kp_B, kp_B_gtD[0], frameA.stereo.frame_K)
    kp_B_covdGT = (kp_B_d - kp_B_gtD)**2
    assert (kp_B_covuv is not None) and (kp_B_covd is not None) and (frameA.gt_pose is not None)

    gt_transform = (frameA.gt_pose.Inv() @ frameB.gt_pose)
    pose_noise =  pp.se3([0.5, 0.5, 0.5, 0.1, 0.1, 0.1])
    est_transform = pose_noise.Exp() @ gt_transform

    init_kp_A_3d = kp_A_3dGT if args.use_gtpoint else kp_A_3d
    init_transform  = pp.SE3(gt_transform if args.use_gtpose else est_transform)
    kp_A_covd_graph = kp_A_covdGT if args.use_gtdcov else kp_A_covd
    kp_B_covd_graph = kp_B_covdGT if args.use_gtdcov else kp_B_covd

    if args.class_type == 'ReprojPGO':
        graph_class = ReprojPGO
    elif args.class_type == 'ReprojBA':
        graph_class = ReprojBA
    elif args.class_type == 'ReprojBA2':
        graph_class = ReprojBA2
    elif args.class_type == 'PureReprojBA':
        graph_class = PureReprojBA
    elif args.class_type == 'PureReprojPGO':
        graph_class = PureReprojPGO
    else:
        raise Exception(f"Can't find the class type {args.class_type}")

    graph = graph_class(
        init_kp_A_3d,
        init_transform,
        kp_A, kp_B,
        kp_A_d, kp_B_d,
        kp_A_covd_graph, kp_B_covd_graph,
        kp_B_covuv, frameA.stereo.frame_K
    ).cpu()
    context = {
        "kernel": Huber(delta=.1),
        "solver": PINV(),
        "strategy": TrustRegion(radius=1e3),
        "corrector": FastTriggs(Huber(delta=.1)),
        "vectorize": True,
        "device": "cpu"
    }
    optimizer = LM(graph.double(), solver=context["solver"],
                    strategy=context["strategy"], kernel=context["kernel"],
                    corrector=context["corrector"],
                    min=1e-6, vectorize=context["vectorize"])
    scheduler = StopOnPlateau(optimizer, steps=10, patience=5, decreasing=1e-5, verbose=True)

    while scheduler.continual():
        weight = torch.block_diag(*(
            torch.pinverse(graph.covariance().to(context["device"]).double())
        ))
        loss = optimizer.step(input=(), weight=weight)
        # loss = optimizer.step(input=())
        scheduler.step(loss)
        
        with torch.no_grad():
            fig_plt.plot_reprojerr(
                "ReprojErr", 
                point2pixel_NED(graph.pose_frameB.Inv().Act(graph.pts_Tw), graph.K),
                graph.kp_B, graph.kp_B_covuv.T, frameB
            )
            
    assert (frameA.gt_pose is not None) and (frameB.gt_pose is not None)
    est_se3 = graph.pose_frameB
    Logger.write("info", f"Before Optimization\n"
                 f"\trotation error: {(gt_transform.Inv().double() @ est_transform.double()).rotation().Log().norm().item()}\n"   #type: ignore
                 f"\ttranslation err: {(gt_transform.Inv().double() @ est_transform.double()).translation().norm()}")    #type: ignore
    
    Logger.write("info", f"After Optimization\n"
                 f"\trotation error: {(gt_transform.Inv().double() @ est_se3.double()).rotation().Log().norm().item()}\n"   #type: ignore
                 f"\ttranslation err: {(gt_transform.Inv().double() @ est_se3.double()).translation().norm()}")    #type: ignore
    
    Logger.write("info", f"Before Optimization, depth error: {(init_kp_A_3d[:, 0].double() - kp_A_gtD).abs().mean().item()}, {(graph.pose_frameB.Inv().Act(init_kp_A_3d.double())[:, 0] - kp_B_gtD).abs().mean().item()}")
    
    Logger.write("info", f"After Optimization, depth error: {(graph.pts_Tw[:, 0] - kp_A_gtD).abs().mean().item()}, {((graph.pose_frameB.Inv().Act(graph.pts_Tw))[:, 0] - kp_B_gtD).abs().mean().item()}")
