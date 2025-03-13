from pathlib import Path

from DataLoader import SequenceBase, StereoFrame
from Odometry.BaselineDPVO import DeepPatchVO
from Utility.Config import load_config
from Utility.Sandbox import Sandbox


if __name__ == "__main__":
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument("--odom", type=str, default="Config/Experiment/Baseline/DPVO/DPVO.yaml")
    args.add_argument("--data", type=str, default="Config/Sequence/TartanAir_abandonfac_001.yaml")
    args.add_argument(
        "--to",
        type=int,
        default=-1,
        help="Crop sequence to frame# when ran. Set to -1 (default) if wish to run whole sequence",
    )
    args.add_argument(
        "--resultRoot", type=str, default="/data2/datasets/yutianch/Results"
    )
    args = args.parse_args()

    odomcfg, odomcfg_dict = load_config(Path(args.odom))
    odomcfg, odomcfg_dict = odomcfg.Odometry, odomcfg_dict["Odometry"]
    datacfg, datacfg_dict = load_config(Path(args.data))
    project_name = odomcfg.name + "@" + datacfg.name

    # Initialize data source
    sequence = SequenceBase[StereoFrame].instantiate(datacfg.type, datacfg.args).clip(0, args.to).preload()
    frame0 = sequence[0]
    
    odometry = DeepPatchVO(
        **vars(odomcfg.args), width=frame0.stereo.width, height=frame0.stereo.height
    )
    exp_space = Sandbox.create(Path(args.resultRoot), project_name)
    exp_space.config = {
        "Project": project_name,
        "Odometry": odomcfg_dict,
        "Data": {"args": datacfg_dict, "end_idx": args.to},
    }
    odometry.receive_frames(sequence, exp_space)
