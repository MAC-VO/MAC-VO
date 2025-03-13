import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import Utility.Plot as Plot
from Utility.Sandbox import Sandbox
from Utility.Trajectory import Trajectory, PlotableTrajectory

EXP2COLOR = {
    # These are all alias of MAC-VO
    "2pgoFF_eDeM_eMKP_interp": [149/255, 17/255, 32/255],
    "TartanAir_MKP_interp": [149/255, 17/255, 32/255],
    "2pgoFF_Cov_MKP_interp": [149/255, 17/255, 32/255],
    
    "dpvo": [253/255, 181/255, 21/255],
    "DPVO": [253/255, 181/255, 21/255],
    
    "DROID_SLAM": [0, 150/255, 71/255],
    
    "tartanvo": [74/255, 61/255, 152/255],
    "TartanVO": [74/255, 61/255, 152/255],
    
    "ORB_SLAM3": [0/255, 123/255, 192/255],
    "iSLAM_VO": [0/255, 123/255, 192/255],
    
    "mast3r": [185/255, 53/255, 189/255],
}
EXP2KWARGS = {
    "2pgoFF_Cov_MKP_interp": dict(linewidth=2),
    "TartanAir_MKP_interp": dict(linewidth=2),
    "dpvo": dict(linewidth=2),
    "DPVO": dict(linewidth=2),
    "TartanVO": dict(linewidth=2),
    "iSLAM_VO": dict(linewidth=2),
    "DROID_SLAM": dict(linewidth=2),
    "mast3r": dict(linewidth=2),
}
EXP2DISPLAY = {
    "2pgoFF_eDeM_eMKP_interp": "Ours",
    "TartanAir_MKP_interp": "Ours",
    "2pgoFF_Cov_MKP_interp": "Ours",
    "dpvo": "DPVO",
    "DPVO": "DPVO",
    "DROID_SLAM": "DROID-SLAM",
    "tartanvo": "TartanVO Stereo",
    "TartanVO": "TartanVO Stereo",
    "iSLAM_VO": "iSLAM VO",
    "ORB_SLAM3": "ORB-SLAM 3",
    "mast3r": "MASt3R-SLAM"
}
EXAMPTION_ALPHA = {"Ours", "TartanAir_MKP_interp"}
MIN_ALPHA = 0.3
MAX_DIST = .1


def get_color(odom_name: str):
    global EXP2COLOR
    if odom_name in EXP2COLOR:
        return EXP2COLOR[odom_name]
    EXP2COLOR[odom_name] = Plot.getColor("-", len(EXP2COLOR) * 3, 0)
    return EXP2COLOR[odom_name]


def plot_runs(filename: str, runs: list[tuple[PlotableTrajectory | None, PlotableTrajectory]]):
    trajectories: list[PlotableTrajectory] = []
    for run in runs:
        gt_traj, est_traj = run

        odom_name = est_traj.name.split("@")[0]
        est_traj.plot_kwargs |= {"color": EXP2COLOR[odom_name]}
        est_traj.plot_kwargs |= EXP2KWARGS[odom_name]
        est_traj.name = odom_name
        
        trajectories.append(est_traj)

    for run in runs:
        gt_traj, _ = runs[0]
        if gt_traj is not None: break
    else:
        raise Exception("No Groundtruth Trajectory provided!")
    
    gt_traj.plot_kwargs |= {"color": "gray", "linewidth": 4, "linestyle": "--"}
    gt_traj.name = "Ground Truth"
    
    aligned_trajectories = []
    for traj in trajectories:
        gt_cropped = gt_traj.apply(lambda x: x.crop(to_idx=traj.data.length))
        traj_aligned = traj.apply(lambda x: x.align_scale(gt_cropped.data).align_origin(gt_cropped.data))
        aligned_trajectories.append(traj_aligned)

    PlotTrajectory(gt_traj, aligned_trajectories, Path(filename))


def PlotTrajectory(gt_traj: PlotableTrajectory, trajs: list[PlotableTrajectory], file_name: Path):
    fig = plt.figure(figsize=(6, 6), dpi=500)
    ax = fig.add_subplot(1, 1, 1)
    ax.axis("off")
    
    Plot.plot_Trajectory(ax, 0, 1, gt_traj)
    gt_position = gt_traj.data.poses.translation().numpy()[..., :2]
    max_distance = -1.
    for idx, traj in enumerate(trajs):
        traj_position = traj.data.poses.translation().numpy()[..., :2]
        target_size = min(traj_position.shape[0], gt_position.shape[0])
        traj_distance = np.linalg.norm(traj_position[:target_size] - gt_position[:target_size], axis=1)
        
        if target_size < traj.data.length: trajs[idx] = traj.apply(lambda x: x.crop(to_idx=target_size))
        max_distance = max(max_distance, traj_distance.max())
    max_distance = min(max_distance, MAX_DIST * (gt_position[..., 0].max() - gt_position[..., 0].min()))
    
    for traj in trajs:
        traj.name = EXP2DISPLAY[traj.name]
        traj_position = traj.data.poses.translation().numpy()[..., :2]
        target_size = min(traj_position.shape[0], gt_position.shape[0])
        traj_distance = np.linalg.norm(traj_position[:target_size] - gt_position[:target_size], axis=1)
        alpha_value = MIN_ALPHA + (1 - MIN_ALPHA) * (max_distance - traj_distance).clip(0, None) / max_distance
        
        if traj.name in EXAMPTION_ALPHA:
            traj.plot_kwargs |= dict(zorder=100)
            Plot.plot_Trajectory(ax, 0, 1, traj)
        else:
            traj.plot_kwargs.pop("alpha", None)
            Plot.plot_LinewithAlpha(
                traj_position[..., 0],
                traj_position[..., 1],
                # color=traj.plot_kwargs["color"],
                alpha=alpha_value[:-1],
                label=traj.name,
                **traj.plot_kwargs
            )(ax)
    
    ax.legend(frameon=False)
    ax.set_aspect("equal", adjustable="datalim")

    plt.tight_layout()
    plt.savefig(str(file_name))
    plt.close()


if __name__ == "__main__":
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument("--spaces", type=str, nargs="+", default=[])
    args.add_argument("--save_as", type=str)
    args = args.parse_args()

    instances = [Trajectory.from_sandbox_mayberef(Sandbox(Path(space))) for space in args.spaces]
    plot_runs(args.save_as, instances)

# Example Script
#
# python -m Scripts.AdHoc.PlotBeautifulCompare --spaces \
#   /data2/datasets/yutianch/Archive/MACVO_Results/MACVO_TartanAirv2/05_31_183121/TartanAir_MKP_interp\@v2_H000/05_31_183122/ \
#   /data2/datasets/yutianch/Archive/MACVO_Results/DPVO_TartanAirv2/05_15_135508/dpvo\@v2_H000/05_15_135508/ \
#   --save_as output.png
#
