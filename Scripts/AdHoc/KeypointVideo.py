import torch
import rerun as rr
import pypose as pp
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from Module.Map import TensorMap
from Utility.Visualizer import RerunVisualizer
from Utility.Sandbox import Sandbox
from Utility.PrettyPrint import Logger


def VisualizeMap(project_name: str, rrd_filename: Path, global_map: TensorMap, from_frame: int | None, to_frame: int | None) -> None:
    RerunVisualizer.setup(project_name, interact_mode=True, useRR=True)
    
    rr.send_blueprint(rr.blueprint.Blueprint(
        rr.blueprint.Horizontal(
            rr.blueprint.Spatial3DView(origin="/map"),
            rr.blueprint.Spatial3DView(origin="/map/Framet", contents="/map/**")
            
        )
    ))
    
    batch_points = global_map.get_frame_points(global_map.frames[from_frame:to_frame])
    point_3d = batch_points.position
    point_color = batch_points.color
    point_cov = batch_points.cov_Tw.det()
    
    point_mask = point_cov < point_cov[:].quantile(0.9)
    point_3d = point_3d[point_mask]
    point_color = point_color[point_mask]
    point_cov = point_cov[point_mask]
    
    cov_det_normalized = Normalize()(point_cov)
    colormap = plt.cm.plasma    #type: ignore
    cov_color = colormap(cov_det_normalized)[..., :3]
    
    # frame_pos = pp.SE3(global_map.frames[from_frame:to_frame].pose)
    
    for idx in range(len(global_map.frames)):
        if idx > 1:
            frame_pos = pp.SE3(global_map.frames.pose.tensor[idx-2:idx]).translation().cpu().numpy()
            from_pos_np = frame_pos[:-1]
            to_pos_np = frame_pos[1:]
            rr.log("/map/Trajectory", rr.LineStrips3D(
                np.stack([from_pos_np, to_pos_np], axis=1), radii=[0.02], colors=[[149, 17, 32]]),
            )
        
        RerunVisualizer.visualizeFrameAt(global_map, idx)
        RerunVisualizer.visualizeRecentPoints(global_map, index=idx)
    
    RerunVisualizer.visualizePoints("Points", point_3d.numpy(), colors=point_color.numpy(), radii=0.01)
    RerunVisualizer.visualizePoints("PointsCov", point_3d.numpy(), colors=cov_color, radii=0.01)
    RerunVisualizer.close()


def VisualizePair(project_name: str, rrd_filename: Path, global_map: TensorMap, from_frame: int | None, to_frame: int | None) -> None:
    RerunVisualizer.setup(project_name, interact_mode=True, useRR=True)
    from_frame = from_frame if from_frame else 0
    to_frame = to_frame if to_frame else (len(global_map.frames) - 10)
    
    RerunVisualizer.visualizeTrajectory(global_map, timeless=True, radii=0.005, color=(149, 17, 32))
    for frame_idx in range(from_frame, to_frame, 10):
        rr.set_time_sequence("frame_idx", frame_idx)
        
        frame_t, frame_t1 = global_map.frames[frame_idx], global_map.frames[frame_idx + 10]
        
        # Visualize frame t
        K = frame_t.K.squeeze(0)
        RerunVisualizer.visualizeAnnotatedCamera("/map/Framet1", K, frame_t1.squeeze().pose, "Camera t+1")
        RerunVisualizer.visualizeAnnotatedCamera("/map/Framet", K, frame_t.squeeze().pose, "Camera t")
        
        idx_t  = global_map.get_frame_points(frame_t).point_idx
        idx_t1 = global_map.get_frame_points(frame_t1).point_idx
        assert idx_t is not None and idx_t1 is not None
        idx_both = torch.tensor(np.intersect1d(idx_t.numpy(), idx_t1.numpy()))
        points = global_map.points[idx_both]
        points = points[:50]
        
        RerunVisualizer.visualizePoints("/map/points", points.position.numpy(), points.color.numpy(), radii=0.01)
        
        rr.log(
            "/map/Matching",
            rr.LineStrips3D(
                torch.cat(
                    [
                        torch.stack([frame_t.pose.translation().repeat(len(points), 1), points.position], dim=1),
                        torch.stack([frame_t1.pose.translation().repeat(len(points), 1), points.position], dim=1),
                    ], dim=0
                ).numpy(),
                radii=[0.001],
                colors=[[0, 0, 255]],
        ))
    
    
    RerunVisualizer.close()


def Visualize(box: Sandbox, from_frame: int | None, to_frame: int | None) -> None:
    if not box.is_finished:
        Logger.write("info", f"Cannot visualize {box.folder} since this run did not finish.")
        return
    if not box.path("tensor_map.pth").exists():
        Logger.write("info", f"Cannot visualize {box.folder} since no tensor_map.pth is found.")
        return

    global_map = torch.load(box.path("tensor_map.pth"))
    Logger.write("info", f"Visualizing: {box.folder}, {global_map} => tensor_map_vis.rrd")

    if hasattr(box.config, "Project"):
        project_name = box.config.Project
    else:
        project_name = "TensorMap"
    # VisualizeMap(project_name, box.path("tensor_map_vis.rrd"), global_map, from_frame, to_frame)
    VisualizePair(project_name, box.path("tensor_map_vis.rrd"), global_map, from_frame, to_frame)


if __name__ == "__main__":
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument("--spaces", type=str, nargs="+", default=[])
    args.add_argument("--recursive", action="store_true", help="Find and evaluate on leaf sandboxes only.")
    args.add_argument("--from_frame", type=int, default=None)
    args.add_argument("--to_frame", type=int, default=None)
    args = args.parse_args()
    
    if args.recursive:
        spaces_nested = [Sandbox.load(space).get_leaves() for space in args.spaces]
        spaces = [box for group in spaces_nested for box in group]
    else:
        spaces = [Sandbox.load(space) for space in args.spaces]
    
    for space in spaces:
        Visualize(space, args.from_frame, args.to_frame)
