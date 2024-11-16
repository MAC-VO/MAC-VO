import torch
import pypose as pp
import rerun as rr
from pathlib import Path

from Utility.Visualizer import RerunVisualizer
from Utility.Sandbox import Sandbox
from Utility.Config import load_config
from Utility.PrettyPrint import ColoredTqdm
from DataLoader import GenericSequence
from Module.Map import TensorMap

def visualize_run(seq: GenericSequence | None, space: Sandbox, start: int, to: int):
    RerunVisualizer.setup("Demo", True, useRR=True)
    
    global_map: TensorMap = torch.load(space.path("tensor_map.pth"))
    
    for frame_idx in ColoredTqdm(range(start, to)):
        rr.set_time_sequence("frame_idx", frame_idx)
        
        curr_frame = global_map.frames[frame_idx]
        curr_pose = curr_frame.pose
        curr_points = global_map.get_frame_points(curr_frame)
        curr_obs = global_map.get_frame_observes(curr_frame)
        
        if frame_idx > 1:
            RerunVisualizer.visualizeTrajectory(global_map, from_frame=frame_idx-2, to_frame=frame_idx, radii=0.02, color=(149, 17, 32))
        
        
        RerunVisualizer.visualizeAnnotatedCamera("/map/Frame", curr_frame.K[0], curr_pose)
        
        # For a smoother camera (cinema camera)
        damped_rotation = pp.so3(pp.SO3(global_map.frames[max(0, frame_idx-50): frame_idx].pose.rotation()).Log().mean(dim=0)).Exp().tensor()
        rr.log("/CinemaCam", rr.Transform3D(translation=curr_pose.translation().squeeze(), rotation=damped_rotation))
        #
        
        if seq is not None:
            raw_frame = seq[frame_idx]
            RerunVisualizer.visualizeImageOnCamera(raw_frame.imageL[0].permute(1, 2, 0))
        # RerunVisualizer.visualizeImageOnCamera(torch.tensor(depth_maps[frame_idx]).log10(), "depth", apply_cmap=True, vmax=3.)
        rr.log(
            "/map/Frame/cam/keypoints",
            rr.Points2D(curr_obs.pixel_uv.numpy(), colors=(149, 17, 32), radii=2.),
        )
        
        # Covariance
        covs = curr_points.cov_Tw
        eigen_val, eigen_vec = torch.linalg.eig(covs)
        eigen_val, eigen_vec = eigen_val.real, eigen_vec.real

        delta = curr_points.position.repeat(1, 3, 1).reshape(-1, 3)
        eigen_vec_Tw = eigen_vec.transpose(-1, -2).reshape(-1, 3)
        eigen_val = eigen_val.unsqueeze(-1).repeat(1, 1, 3).reshape(-1, 3)
        eigen_vec_Tw = eigen_vec_Tw * eigen_val.sqrt()
        eigen_vec_Tw_a = delta + .1 * eigen_vec_Tw
        eigen_vec_Tw_b = delta - .1 * eigen_vec_Tw
        rr.log(
            "/map/PointsVar",
            rr.LineStrips3D(
                torch.stack([eigen_vec_Tw_a, eigen_vec_Tw_b], dim=1).numpy(),
                radii=[0.003],
                colors=curr_points.color.unsqueeze(0).repeat(3, 1, 1).reshape(-1, 3)
            ),
        )
        RerunVisualizer.visualizePoints("/map/Points", curr_points.position.numpy(), curr_points.color.numpy(), radii=0.02)

    RerunVisualizer.close()

if __name__ == "__main__":
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument("--space", type=str)
    args.add_argument("--data", type=str, default=None)
    args.add_argument("--from_frame", type=int, default=None)
    args.add_argument("--to_frame", type=int, default=None)
    args = args.parse_args()


    if args.data is None:
        visualize_run(None, Sandbox.load(args.space), args.from_frame, args.to_frame)
    else:
        datacfg, datacfg_dict = load_config(Path(args.data))    
        visualize_run(GenericSequence.instantiate(**datacfg_dict), Sandbox.load(args.space), args.from_frame, args.to_frame)
