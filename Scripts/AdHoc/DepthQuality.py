import argparse
from pathlib import Path

from DataLoader import SequenceBase
from Module.Frontend.StereoDepth import IStereoDepth
from Utility.Config import build_dynamic_config, load_config


parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="./Config/Sequence/TartanAir_seaside_000.yaml")
args = parser.parse_args()    

datacfg, _ = load_config(Path(args.data))
depth_cfg, _ = build_dynamic_config({   # For simplicity of editing config
    "type": "FlowFormerCovDepth",
    "args": {
        "weight": "./Model/MACVO_FrontendCov.pth",
        "device": "cuda",
        "cov_mode": "Est"
}})

sequence = SequenceBase.instantiate(**vars(datacfg))
module = IStereoDepth.instantiate(depth_cfg.type, depth_cfg.args)

for frame in sequence:
    assert frame.gtDepth is not None
    est_output = module.estimate(frame)
    diff_depth = (est_output.depth - frame.gtDepth).abs().median()
    print(diff_depth)
