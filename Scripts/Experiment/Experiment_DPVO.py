import argparse
from pathlib import Path

from DataLoader import SequenceBase, StereoFrame, smart_transform
from Evaluation.EvalSeq import EvaluateSequences
from Odometry.BaselineDPVO import DeepPatchVO
from Utility.Config import build_dynamic_config, load_config
from Utility.PrettyPrint import Logger, print_as_table
from Utility.Sandbox import Sandbox


def execute_experiment(name, cfg, cfg_dict, root_box: Sandbox) -> str:
    # Execute an experiment, and return the spaceID
    exp_space = root_box.new_child(name)
    exp_space.config = cfg_dict
    
    sequence = smart_transform(
        SequenceBase[StereoFrame].instantiate(cfg.Data.type, cfg.Data.args),
        cfg.Preprocess
    ).preload()
    system = DeepPatchVO.from_config(cfg, sequence)
    system.receive_frames(sequence, exp_space)
    
    return str(exp_space.folder)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", type=str, required=True)
    args.add_argument(
        "--resultRoot", type=str, default="Results"
    )
    args = args.parse_args()

    Logger.write("info", f"Using configuration from: {args.config}")
    cfg, cfg_dict = load_config(Path(args.config))
    odometry_cfg = cfg_dict["Odometry"]
    data_cfgs = cfg_dict["Datas"]

    run_configs = [
        {
            "Project": odometry_cfg["name"] + "@" + data_cfg["name"],
            "Data": data_cfg,
            "Odometry": odometry_cfg,
            "Preprocess": cfg_dict["Preprocess"]
        }
        for data_cfg in data_cfgs
    ]

    root_box = Sandbox.create(
        Path(args.resultRoot), Path(args.config).name.split(".")[0]
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

    eval_header, eval_results = EvaluateSequences(spaces, correct_scale=True)
    print_as_table(eval_header, eval_results)
