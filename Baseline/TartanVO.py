import argparse
from pathlib import Path

from DataLoader import SequenceBase, StereoFrame
from Odometry.BaselineTartanVO import TartanVO
from Evaluation.EvalSeq import EvaluateSequences
from Utility.Config import load_config, asNamespace
from Utility.PrettyPrint import print_as_table
from Utility.Sandbox import Sandbox


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--odom", type=str, default="Config/Experiment/Baseline/TartanVO/TartanVOStereo.yaml")
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
    args.add_argument(
        "--useRR",
        action="store_true",
        help="use RerunVisualizer to generate <config.Project>.rrd file for visualization.",
    )
    args = args.parse_args()

    odomcfg, odomcfg_dict = load_config(Path(args.odom))
    odomcfg, odomcfg_dict = odomcfg.Odometry, odomcfg_dict["Odometry"]
    datacfg, datacfg_dict = load_config(Path(args.data))
    project_name = odomcfg.name + "@" + datacfg.name

    exp_space = Sandbox.create(Path(args.resultRoot), project_name)
    exp_space.config = {
        "Project": project_name,
        "Odometry": odomcfg_dict,
        "Data": {"args": datacfg_dict, "end_idx": args.to},
    }
    exp_space.set_autoremove()

    # Initialize data source
    sequence = SequenceBase[StereoFrame].instantiate(**vars(datacfg)).clip(0, args.to)
    # Initialize modules for VO
    system = TartanVO.from_config(asNamespace(exp_space.config).Odometry, sequence)
    
    system.receive_frames(sequence, exp_space)
    
    header, result = EvaluateSequences([str(exp_space.folder)], correct_scale=False)
    print_as_table(header, result)
