import argparse
from pathlib import Path

from DataLoader import SequenceBase, StereoFrame
from Module import IMatcher
from Utility.Config import build_dynamic_config, load_config


parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="./Config/Sequence/TartanAir_seaside_000.yaml")
args = parser.parse_args()    

datacfg, _ = load_config(Path(args.data))
flow_cfg, _ = build_dynamic_config({   # For simplicity of editing config
    "type": "FlowFormerCovMatcher",
    "args": {
        "weight": "./Model/MACVO_FrontendCov.pth",
        "device": "cuda",
        "cov_mode": "Est"
}})
old_flow_cfg, _ = build_dynamic_config({   # For simplicity of editing config
    "type": "FlowFormerMatcher",
    "args": {
        "weight": "./Model/MACVO_FrontendCov.pth",
        "device": "cuda",
        "cov_mode": "Est"
}})


sequence = SequenceBase[StereoFrame].instantiate(datacfg.type, datacfg.args).clip(0, 10)
module = IMatcher.instantiate(flow_cfg.type, flow_cfg.args)

prev_frame = None
for frame in sequence:
    if prev_frame is None:
        prev_frame = frame
        continue
    
    assert frame.stereo.gt_flow is not None
    output_1 = module.estimate(prev_frame.stereo, frame.stereo)
    output_2 = module.estimate(frame.stereo, prev_frame.stereo)
    
    diff_flow = (output_1.flow.cpu() - frame.stereo.gt_flow).abs().median()
    diff_fwd_rev = (output_1.flow - (-1 * output_2.flow)).abs().median()
    prev_frame = frame
    print(diff_flow, "|| fwd<->rev", diff_fwd_rev)
