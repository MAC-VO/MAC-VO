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


data_cfg, _ = build_dynamic_config({
    "type": "TartanAirv2_NoIMU",
    "args": {
        "root": "/data2/datasets/yuhengq/tartanair_v2_test/Data_hard/P000",
        "compressed": True,
        "gtDepth": False,
        "gtPose" : True,
        "gtFlow" : False
    }
})
net_cfg, _ = build_dynamic_config({
    "type": "FlowFormerCovFrontend",
    "args": {
        "device": "cuda",
        "weight": "./Model/MACVO_FrontendCov.pth",
        "dtype": "fp32",
        "max_flow": -1,
        "enforce_positive_disparity": False
    }
})
kp_cfg, _ = build_dynamic_config({
    "mask_width": 32,
    "device"    : "cuda"
})
cov2to3_cfg, _ = build_dynamic_config({
    "type": "MatchCovariance",
    "args": {
        "kernel_size": 7,
        "match_cov_default": 0.25,
        "min_flow_cov": 0.25,
        "min_depth_cov": 0.01,
        "device": "cuda"
    }
})
cov3to2_cfg, _ = build_dynamic_config({
    "type": "Gaussian_3to2",
    "args": None
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

kp_A_d   = frontend.retrieve_pixels(kp_A, frameA_depth.depth)
kp_A_σd  = frontend.retrieve_pixels(kp_A, frameA_depth.cov)
kp_A_σuv = frontend.retrieve_pixels(kp_A, frame_AtoB.cov)
kp_A_3d  = pixel2point_NED(kp_A, kp_A_d[0], frameA.stereo.frame_K)
# kp_A_3d  = pp.pixel2point(kp_A, kp_A_d[0], frameA.stereo.frame_K)
assert (kp_A_σuv is not None) and (kp_A_σd is not None)
kp_A_σ3d = cov_2to3.estimate(frameA.stereo, kp_A, frameA_depth, kp_A_σd.unsqueeze(0), kp_A_σuv.T)
 

kp_B_σuv = frontend.retrieve_pixels(kp_B, frame_AtoB.cov)
assert kp_B_σuv is not None


class ReprojErrorGraph(torch.nn.Module):
    def __init__(self, kp_A_3d: torch.Tensor, kp_A_σ3d: torch.Tensor, kp_B: torch.Tensor, kp_B_σuv: torch.Tensor, K: torch.Tensor):
        super().__init__()
        self.pose2opt = pp.Parameter(pp.identity_SE3())
        self.register_buffer("pts_Tw", kp_A_3d )
        self.register_buffer("cov_Tw", kp_A_σ3d)
        self.register_buffer("kp_B"  , kp_B)
        self.register_buffer("K", K)
        self.counter = 0
        
        N = kp_B_σuv.size(1)
        self.register_buffer("kp_Bσ" , torch.empty((N, 2, 2)))
        self.register_buffer("kp_B_σuv", kp_B_σuv)
        self.kp_Bσ[:, 0, 0] = kp_B_σuv[0]
        self.kp_Bσ[:, 1, 1] = kp_B_σuv[1]
        self.kp_Bσ[:, 0, 1] = kp_B_σuv[2]
        self.kp_Bσ[:, 1, 0] = kp_B_σuv[2]
    
    def forward(self) -> torch.Tensor:
        return point2pixel_NED(self.pose2opt.Inv().Act(self.pts_Tw), self.K) - self.kp_B

    @torch.no_grad()
    def covariance(self) -> torch.Tensor:
        return self.kp_Bσ


graph = ReprojErrorGraph(kp_A_3d, kp_A_σ3d, kp_B, kp_B_σuv, frameA.stereo.frame_K).cuda()
context = {
    "kernel": Huber(delta=0.1),
    "solver": PINV(),
    "strategy": TrustRegion(radius=1e3),
    "corrector": FastTriggs(Huber(delta=0.1)),
    "vectorize": True,
    "device": "cuda"
}
optimizer = LM(graph.double(), solver=context["solver"],
                strategy=context["strategy"], kernel=context["kernel"],
                corrector=context["corrector"],
                min=1e-6, vectorize=context["vectorize"])
scheduler = StopOnPlateau(optimizer, steps=10, patience=2, decreasing=1e-5, verbose=True)

while scheduler.continual():
    weight = torch.block_diag(*(
        torch.pinverse(graph.covariance().to(context["device"]).double())
    ))
    loss = optimizer.step(input=(), weight=weight)
    scheduler.step(loss)
    
    with torch.no_grad():
        fig_plt.plot_reprojerr(
            "ReprojErr", 
            point2pixel_NED(graph.pose2opt.Inv().Act(graph.pts_Tw), graph.K),
            graph.kp_B,
            graph.kp_B_σuv.T,
            frameB
        )

assert (frameA.gt_pose is not None) and (frameB.gt_pose is not None)
gt_se3  = (frameA.gt_pose.Inv() @ frameB.gt_pose).Log()
est_se3 = graph.pose2opt.Log().cpu()
print((gt_se3 - est_se3).norm())    #type: ignore
