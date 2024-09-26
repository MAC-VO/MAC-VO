from pathlib import Path

import matplotlib.pyplot as plt
import torch

from DataLoader import GenericSequence
from Module.Frontend.Matching import IMatcher
from Utility.Config import build_dynamic_config, LoadFrom
from Utility.Plot import getColor
from Utility.Utils import cropToMultiple
from Utility.PrettyPrint import ColoredTqdm


def main(seq_cfg, match_cfgs):
    sequence = GenericSequence.instantiate(**vars(seq_cfg)).preload()

    for match_cfg in match_cfgs:
        match_est = IMatcher.instantiate(match_cfg.type, match_cfg.args)
        mid_errs = []
        low_q_errs = []
        high_q_errs = []
        
        low_est_q_errs = []

        for idx in ColoredTqdm(range(1, len(sequence))):
            prev_frame = sequence[idx - 1]
            est_match, flow_cov_map = match_est.estimate(prev_frame, sequence[idx])

            assert prev_frame.gtFlow is not None
            ref_match = cropToMultiple(prev_frame.gtFlow, [64, 64], [2, 3])

            err_flow = torch.abs(est_match - ref_match)

            low_q_errs.append(err_flow.quantile(0.25).item())
            mid_errs.append(err_flow.median().item())
            high_q_errs.append(err_flow.quantile(0.75).item())
            
            if flow_cov_map is not None:
                low_est_q_errs.append(flow_cov_map.median().sqrt().item())

        _ = plt.figure(figsize=(10, 3), dpi=300)
        plt.plot(low_q_errs, color=getColor("-", 4, 0), label="1st quarter Error")
        plt.plot(mid_errs, color=getColor("-", 6, 0), label="Median Match Error")
        plt.plot(low_est_q_errs, color=getColor("-", 2, 0), label="Est Median quarter Error")
        plt.grid(True, "major", linestyle="--")
        plt.title(match_cfg.args.weight.split("/")[-1])
        plt.legend()
        plt.savefig(f"./Match_Err{match_cfg.args.weight.split('/')[-1]}.png")
        plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seqcfg",
        type=str,
        default="./Config/Run/Sequence/TartanAir_abandonedfac_000.yaml",
    )
    args = parser.parse_args()
    
    seq_cfg, _ = build_dynamic_config(LoadFrom(Path(args.seqcfg)))
    match_cfgs, _ = build_dynamic_config([
            {
                "type": "FlowFormerMatcher",
                "args": {
                    "weight": "./Model/MACVO_FrontendCov.pth",
                    "device": "cuda",
                    "cov_mode": "Est"
                }
            },
        ]
    )
    main(seq_cfg, match_cfgs)