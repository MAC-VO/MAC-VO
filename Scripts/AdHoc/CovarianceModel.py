"""
Plots Gaussian distribution for some confidence interval and compare the 
predicted distribution (range) with simulation.
"""
import torch
import pypose as pp

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from DataLoader import StereoData
from Module import ICovariance2to3
from Module.Frontend.StereoDepth import IStereoDepth
from Utility.Plot import plot_gaussian_conf
from Utility.Point import pixel2point_NED


def simulate_cov_model(
    ax: Axes,
    model: ICovariance2to3, 
    bl: float, K: torch.Tensor, disparity: float, u: float, v: float, cov_disp: float, cov_u: float, cov_v: float,
    n_sample: int
):
    pixel_u = u + torch.randn((n_sample,)) * cov_u
    pixel_v = v + torch.randn((n_sample,)) * cov_v
    pixel_disp  = (disparity + torch.randn((n_sample,)) * cov_disp).abs()
    
    pixel_depth = (bl * K[0][0]) / pixel_disp
    points = pixel2point_NED(torch.stack([pixel_u, pixel_v], dim=1), pixel_depth, K)
    
    # Estimate depth covariance
    disparity_2 = disparity ** 2
    error_rate_2 = cov_disp / disparity_2
    depth_cov = (((bl * K[0][0].item()) ** 2) * (error_rate_2 / disparity_2))
    # End
    
    
    avg_depth = ((bl * K[0][0].item()) / disparity)
    avg_point = pixel2point_NED(torch.tensor([[u, v]]), torch.tensor([avg_depth]), K)
    fake_depth_map = torch.ones((1, 640, 640))
    fake_depth_cov_map = torch.ones((1, 640, 640))
    fake_depth_map *= avg_depth
    fake_depth_cov_map *= depth_cov
    
    cov = model.estimate(
        StereoData(
            T_BS=pp.identity_SE3(1),
            K=K,
            baseline=torch.tensor([bl]),
            time_ns=[0],
            width=640, height=640,
            
            imageL=torch.empty((1, 3, 640, 640), dtype=torch.float32),
            imageR=torch.empty((1, 3, 640, 640), dtype=torch.float32),
            gt_flow=None, flow_mask=None, gt_depth=None
        ),
        kp=torch.tensor([[u, v]]),
        depth_est=IStereoDepth.Output(depth=fake_depth_map, cov=fake_depth_cov_map),
        depth_cov=torch.tensor([depth_cov]),
        flow_cov=torch.tensor([[cov_u, cov_v]])
    )
    print(cov)
    
    # Plotting
    ax.scatter(points[..., 0].numpy(), points[..., 1].numpy(), s=0.5, marker='o', alpha=0.5)
    plot_gaussian_conf(avg_point[0, :2], cov[0, :2, :2], confidence=0.95)(ax)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(left=0.0, right=20.)
    ax.set_ylim(top=10.0, bottom=-10.0)

    # ax.grid(True, linestyle="--")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Gaussian Confidence Ellipse")


# visualize_gmap_cov_at_frame("./Results/MACVO@v2_H000/08_06_212210/tensor_map.pth", 0, "output.png")

cov_model = ICovariance2to3.instantiate(
    "MatchCovariance",
    device="cpu",
    diag=False,
    normalize="None",
    kernel_size=31,
    match_cov_default=0.25,
    min_depth_cov=0.05,
    min_flow_cov=0.25,
    use_depth_patch=False
)

bl = 0.25
K  = torch.tensor([[320., 0., 320.], [0., 320., 320.], [0., 0., 1.]])

fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
simulate_cov_model(
    ax, 
    cov_model, bl, K,
    disparity=10.0, u=128.0, v=450.0, 
    cov_disp=2.0, cov_u=5.0, cov_v=3.0,
    n_sample=500
)
simulate_cov_model(
    ax, 
    cov_model, bl, K,
    disparity=10.0, u=600.0, v=130.0, 
    cov_disp=1.0, cov_u=10.0, cov_v=10.0,
    n_sample=500
)

simulate_cov_model(
    ax, 
    cov_model, bl, K,
    disparity=10.0, u=400.0, v=200.0, 
    cov_disp=3.0, cov_u=10.0, cov_v=10.0,
    n_sample=500
)

fig.tight_layout()
fig.savefig("output.png")
