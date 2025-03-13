import argparse
from pathlib import Path

from DataLoader import SequenceBase, StereoFrame, IDataTransform
from Evaluation.EvalSeq import EvaluateSequences
from Odometry.MACVO import MACVO
from Utility.Config import build_dynamic_config, load_config
from Utility.PrettyPrint import Logger, print_as_table
from Utility.Sandbox import Sandbox


def execute_experiment(name, cfg, cfg_dict, root_box: Sandbox) -> str:
    # Execute an experiment, and return the directory of result sandbox
    exp_space = root_box.new_child(name)
    exp_space.config = cfg_dict
    
    sequence = SequenceBase[StereoFrame].instantiate(**vars(cfg.Data))\
            .preload()\
            .transform(actions=[
                IDataTransform.instantiate(tcfg.type, tcfg.args)
                for tcfg in cfg.Preprocess
            ])
    system = MACVO[StereoFrame].from_config(cfg)
    system.receive_frames(sequence, exp_space)
    
    return str(exp_space.folder)


if __name__ == "__main__":
    import numpy as np
    args = argparse.ArgumentParser()
    args.add_argument("--odom", type=str, default=None, required=True)
    args.add_argument("--data", type=str, default=None, required=True)
    args.add_argument("--resultRoot", type=str, default="Results")
    args = args.parse_args()

    Logger.write("info", f"Using configuration from: \n\todom={args.odom}\n\tdata={args.data}")
    cfg, cfg_dict = load_config(Path(args.odom))
    data_cfg, data_dict = load_config(Path(args.data))
    odometry_cfg = cfg_dict["Odometry"]
    resolutions  = [(int(size), int(size)) for size in np.linspace(160, 640, num=5)]

    run_configs = [
        {
            "Project": f"{odometry_cfg['name']}@{data_dict['name']}_{resolution[0]}x{resolution[1]}",
            "Data": data_dict,
            "Odometry": odometry_cfg,
            "Preprocess": cfg_dict["Preprocess"] + [{
                "type": "SmartResizeFrame",
                "args": {"height": resolution[0], "width": resolution[1], "interp": "nearest"}
            }]
        }
        for resolution in resolutions
    ]

    root_box = Sandbox.create(
        Path(args.resultRoot), Path(args.odom).name.split(".")[0]
    )
    spaces = []

    for run_cfg_template in run_configs:
        cfg, cfg_dict = build_dynamic_config(run_cfg_template)
        Logger.write("info", cfg_dict)
        spaces.append(execute_experiment(cfg.Project, cfg, cfg_dict, root_box))

    Logger.write(
        "info",
        "Finished experiment group, the results are stored in"
        + "\n"
        + " ".join(spaces),
    )
    with root_box.open("runs.txt", "w") as f:
        f.write("\n".join(spaces))

    eval_header, eval_results = EvaluateSequences(spaces, correct_scale=False)
    print_as_table(eval_header, eval_results)
