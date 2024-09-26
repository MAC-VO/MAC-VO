import argparse
from pathlib import Path

from DataLoader import GenericSequence
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


sequence = GenericSequence.instantiate(**vars(datacfg)).clip(0, 10)
module = IMatcher.instantiate(flow_cfg.type, flow_cfg.args)

prev_frame = None
for frame in sequence:
    if prev_frame is None:
        prev_frame = frame
        continue
    
    assert frame.gtFlow is not None
    flow, flow_cov = module(prev_frame, frame)
    flow2, flow_cov2 = module(frame, prev_frame)
    
    diff_flow = (flow.cpu() - frame.gtFlow).abs().median()
    diff_fwd_rev = (flow - (-1 * flow2)).abs().median()
    prev_frame = frame
    print(diff_flow, "|| fwd<->rev", diff_fwd_rev)
