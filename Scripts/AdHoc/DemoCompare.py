import torch
import argparse
import rerun as rr
import pypose as pp
from pathlib import Path

from Utility.Sandbox import Sandbox
from Module.Map import TensorMap
from Utility.Visualizer import RerunVisualizer
from Utility.Trajectory import Trajectory, Plotable


COLOR_MAP = {
    "DPVO": (253, 181, 21, 100),
    "DROID-SLAM": (0, 98, 23, 100),
    "DPVO": (191, 128, 0, 100),
    "TartanVO": (74, 61, 152, 100)
}


def VisualizeComparison(macvo_space: Path, others: dict[str, Path]):
    macvo_box = Sandbox.load(macvo_space)
    macvo_map: TensorMap  = torch.load(macvo_box.path("tensor_map.pth"))
    
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
        
        map_points = macvo_map.get_frame_points(macvo_map.frames[frame_idx])
        RerunVisualizer.visualizePoints("Map", map_points.position.numpy(), map_points.color.numpy(), radii=0.04)
        
        
        gt_path: pp.LieTensor = gt_traj.data.poses[:frame_idx]          #type: ignore
        macvo_path: pp.LieTensor = macvo_traj.data.poses[:frame_idx]    #type: ignore
        RerunVisualizer.visualizePath("GroundTruth", gt_path, colors=(150, 150, 150), radii=0.05)
        RerunVisualizer.visualizePath("MAC-VO", macvo_path, colors=(149, 17, 32), radii=0.05)
        rr.log("/CinemaCam", rr.Transform3D(translation=macvo_traj.data.poses[frame_idx].translation()))    #type: ignore
        RerunVisualizer.visualizeFrameAt(macvo_map, frame_idx)
        
        for key, est_traj in other_trajs.items():
            if est_traj is None: continue
            RerunVisualizer.visualizePath(key, est_traj.data.poses[:frame_idx], colors=COLOR_MAP[key], radii=0.05)      #type: ignore


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
    
    RerunVisualizer.setup("ComparisonDemo", True, useRR=True)
    VisualizeComparison(
        Path(args.macvo_space),
        {k: Path(v) for k, v in zip(args.other_types, args.other_spaces)}
    )
    RerunVisualizer.close()

# Exmple Usage
#   python -m Scripts.AdHoc.DemoCompare \
#       --macvo_space /data2/datasets/yutianch/Archive/MACVO_Results/MACVO_TartanAirv2/05_31_183121/TartanAir_MKP_interp\@v2_H000/05_31_183122/ \
#       --other_spaces /data2/datasets/yutianch/Archive/MACVO_Results/DPVO_TartanAirv2/05_15_135508/dpvo\@v2_H000/05_15_135508/ \
#       --other_types DPVO
#