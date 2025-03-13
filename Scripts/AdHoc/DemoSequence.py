import torch
import pypose as pp
import rerun as rr
import numpy as np
from pathlib import Path

from DataLoader import SequenceBase, StereoFrame
from Module.Map import VisualMap

from Utility.Sandbox import Sandbox
from Utility.Config import load_config
from Utility.PrettyPrint import ColoredTqdm
from Utility.Visualize import rr_plt

def visualize_run(seq: SequenceBase[StereoFrame] | None, space: Sandbox, start: int, to: int):
    rr_plt.init_connect("Demo")
    
    # global_map: TensorMap = torch.load()
    macvo_map = VisualMap.deserialize(np.load(space.path("tensor_map.pth")))
    
    for frame_idx in ColoredTqdm(range(start, to)):
        rr.set_time_sequence("frame_idx", frame_idx)
        
        curr_frame  = macvo_map.frames[torch.tensor([[frame_idx]], dtype=torch.long)]
        curr_pose   = curr_frame.data["pose"]
        curr_obs    = macvo_map.get_frame2match(curr_frame)
        curr_pts    = macvo_map.get_match2point(curr_obs)
        
        if frame_idx > 1:
            rr_plt.log_trajectory("/world", macvo_map.frames.data["pose"][frame_idx - 2:frame_idx], radii=0.02, color=(149, 17, 32))
        rr_plt.log_camera("/world/frame", curr_pose, curr_frame.data["K"][0])
        
        if seq is not None:
            raw_frame = seq[frame_idx].stereo.imageL[0].permute(1, 2, 0)
            rr_plt.log_image("/world/frame", raw_frame)
        rr.log(
            "/world/frame",
            rr.Points2D(curr_obs.data["pixel1_uv"].numpy(), colors=(149, 17, 32), radii=2.)
        )
        
        rr_plt.log_points(
            "/world/point_cloud",
            curr_pts.data["pos_Tw"], curr_pts.data["color"], curr_pts.data["cov_Tw"],
            cov_mode='axis'
        )
        # For a smoother camera (cinema camera)
        damped_rotation = pp.so3(pp.SO3(macvo_map.frames[max(0, frame_idx-50): frame_idx].data["pose"][3:]).Log().mean(dim=0)).Exp().tensor()
        rr.log("/CinemaCam", rr.Transform3D(translation=curr_pose.translation().squeeze(), rotation=damped_rotation))
        #
    rr.rerun_shutdown()

if __name__ == "__main__":
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument("--space", type=str)
    args.add_argument("--data", type=str, default=None)
    args.add_argument("--from_frame", type=int, default=None)
    args.add_argument("--to_frame", type=int, default=None)
    args = args.parse_args()
    rr_plt.default_mode = "rerun"
    rr_plt.init_connect("DemoSequence")


    if args.data is None:
        visualize_run(None, Sandbox.load(args.space), args.from_frame, args.to_frame)
    else:
        datacfg, _ = load_config(Path(args.data))    
        visualize_run(SequenceBase.instantiate(datacfg.type, datacfg.args), Sandbox.load(args.space), args.from_frame, args.to_frame)
