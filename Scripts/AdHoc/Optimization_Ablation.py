import pypose as pp
from pathlib import Path

from Odometry.MACVO import MACVO
from DataLoader import SequenceBase, StereoFrame
from Utility.Config import load_config
from Utility.Visualize import fig_plt
from Utility.PrettyPrint import Logger



def run_frame(cfg: Path, frame1: StereoFrame, frame2: StereoFrame) -> dict[str, float]:
    assert frame1.gt_pose is not None
    assert frame2.gt_pose is not None
    
    odometry = MACVO[StereoFrame].from_config(load_config(cfg)[0])
    odometry.run(frame1)
    odometry.run(frame2)
    odometry.terminate()
    poses = pp.SE3(odometry.get_map().frames.data["pose"][:])
    avg_depth = odometry.get_map().points.data["pos_Tw"][:, 0].mean().item()
    del odometry
    
    est_motion = pp.SE3(pp.SE3(poses[0]).Inv() @ poses[1])
    gt_motion  = pp.SE3(frame1.gt_pose.Inv() @ frame2.gt_pose)
    
    t_rel = (est_motion.translation() - gt_motion.translation()).norm().item()
    r_rel = (est_motion.rotation().Log() - gt_motion.rotation().Log()).norm().item()
    return {"t_rel": t_rel, "r_rel": r_rel, "avg_d": avg_depth}


if __name__ == "__main__":
    # Example Usage
    #
    # python -m Scripts.AdHoc.Optimization_Ablation --odom ./Config/Experiment/MACVO/Optimizers/ReprojBA_wDepth.yaml
    #
    import argparse
    
    fig_plt.default_mode = 'image'
    parser = argparse.ArgumentParser()
    parser.add_argument("--odom", type=str, required=True)
    args = parser.parse_args()
    
    data = SequenceBase[StereoFrame].instantiate("TartanAir_NoIMU", {
        "root": "/project/learningvo/tartanair_v1_5/westerndesert/Data/P000/",
        "compressed": True,
        "gtDepth": True,
        "gtPose" : True,
        "gtFlow" : True
    })
    
    # Near Scene (frame 522)
    Logger.write("info", f"Near Scene Error: {run_frame(Path(args.odom), data[522], data[523])}")
    
    # Far  Scene (frame 250)
    Logger.write("info", f"Far  Scene Error: {run_frame(Path(args.odom), data[250], data[251])}")
