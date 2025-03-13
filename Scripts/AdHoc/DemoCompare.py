import argparse
import rerun as rr
import numpy as np
import pypose as pp
from pathlib import Path

from Module.Map import VisualMap
from Utility.Sandbox import Sandbox
from Utility.Trajectory import Trajectory, Plotable
from Utility.Visualize import rr_plt


COLOR_MAP = {
    "DPVO": (253, 181, 21, 100),
    "DROID-SLAM": (0, 98, 23, 100),
    "DPVO": (191, 128, 0, 100),
    "TartanVO": (74, 61, 152, 100)
}


def VisualizeComparison(macvo_space: Path, others: dict[str, Path]):
    macvo_box = Sandbox.load(macvo_space)
    macvo_map = VisualMap.deserialize(np.load(macvo_box.path("tensor_map.pth")))
    
    gt_traj, macvo_traj = Trajectory.from_sandbox(macvo_box)
    macvo_traj = macvo_traj.apply(lambda x: x.align_origin(gt_traj.data))
    
    other_trajs: dict[str, Plotable[Trajectory]] = {
        key: Plotable(
            Trajectory.from_SE3_numpy(macvo_box.path("poses.npy")).align_origin(gt_traj.data),
            name=macvo_box.config.Project if hasattr(Sandbox.config, "Project") else "Est_Traj",
            plot_kwargs=dict()
        )
        for key in others
    }
    
    assert macvo_traj is not None
    
    for key in {"DPVO", "DROID-SLAM"}:
        if key not in other_trajs: continue
        other_trajs[key] = other_trajs[key].apply(
            lambda traj: traj.align_scale(gt_traj.data)
        )

    for frame_idx in range(macvo_traj.data.length):
        if frame_idx < 2: continue
        rr.set_time_sequence("frame_idx", frame_idx)
        
        map_points = macvo_map.get_match2point(
            macvo_map.get_frame2match(macvo_map.frames[[frame_idx]])
        )
        
        gt_path: pp.LieTensor = gt_traj.data.poses[:frame_idx]          #type: ignore
        macvo_path: pp.LieTensor = macvo_traj.data.poses[:frame_idx]    #type: ignore
        
        rr_plt.log_trajectory("/world/ground_truth", gt_path, colors=(150, 150, 150), radii=0.05)
        rr_plt.log_trajectory("/world/macvo", macvo_path, colors=(149, 17, 32)  , radii=0.05)
        rr_plt.log_points    ("/point_cloud", 
                              map_points.data["pos_Tw"], map_points.data["color"], map_points.data["cov_Tw"], 
                              "sphere")
        rr_plt.log_camera    ("/world/macvo/cam", macvo_map.frames.data["pose"][frame_idx], macvo_map.frames.data["K"][frame_idx])
        rr.log("/cinema_cam", rr.Transform3D(translation=macvo_traj.data.poses[frame_idx].translation()))    #type: ignore
        
        for key, est_traj in other_trajs.items():
            if est_traj is None: continue
            rr_plt.log_trajectory(f"/world/{key}", est_traj.data.poses[:frame_idx], colors=COLOR_MAP[key], radii=0.05)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--macvo_space", type=str, required=True,
                        help="Path to the space (sandbox) stores the result from MAC-VO")
    parser.add_argument("--other_spaces", type=str, nargs="+", required=True,
                        help="Paths to the spaces (sandbox) stores the result from other odometry systems (baselines)")
    parser.add_argument("--other_types", type=str, nargs="+", required=True, choices=["DPVO", "DROID-SLAM", "TartanVO"],
                        help="Names of baseline models (for visualization and coloring)")
    args = parser.parse_args()
    assert len(args.other_spaces) == len(args.other_types)
    return args

if __name__ == "__main__":
    args = get_args()
    rr_plt.default_mode = "rerun"
    rr_plt.init_connect("Comparison Demo")
    VisualizeComparison(
        Path(args.macvo_space),
        {k: Path(v) for k, v in zip(args.other_types, args.other_spaces)}
    )
    rr.rerun_shutdown()

# Exmple Usage
#   python -m Scripts.AdHoc.DemoCompare \
#       --macvo_space /data2/datasets/yutianch/Archive/MACVO_Results/MACVO_TartanAirv2/05_31_183121/TartanAir_MKP_interp\@v2_H000/05_31_183122/ \
#       --other_spaces /data2/datasets/yutianch/Archive/MACVO_Results/DPVO_TartanAirv2/05_15_135508/dpvo\@v2_H000/05_15_135508/ \
#       --other_types DPVO
