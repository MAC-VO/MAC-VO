import torch
import pypose as pp
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from Module.Map import TensorMap
from Utility.Visualizer import RerunVisualizer
from Utility.Space import Sandbox
from Utility.PrettyPrint import Logger, ColoredTqdm


def VisualizeMap(project_name: str, rrd_filename: Path, global_map: TensorMap, from_frame: int | None, to_frame: int | None) -> None:
    RerunVisualizer.setup(project_name, interact_mode=False, save_rrd_path=rrd_filename, useRR=True, show_plots=False)
    
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
    
    frame_pos = pp.SE3(global_map.frames[from_frame:to_frame].pose)
        
    RerunVisualizer.visualizePoints("Points", point_3d.numpy(), colors=point_color.numpy(), radii=0.05)
    RerunVisualizer.visualizePoints("PointsCov", point_3d.numpy(), colors=cov_color, radii=0.05)
    RerunVisualizer.visualizePath("Trajectory", frame_pos, (149, 17, 32), radii=0.1)
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
    VisualizeMap(project_name, box.path("tensor_map_vis.rrd"), global_map, from_frame, to_frame)


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
    
    for space in ColoredTqdm(spaces):
        Visualize(space, args.from_frame, args.to_frame)
