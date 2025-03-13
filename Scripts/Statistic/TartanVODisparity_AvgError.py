from pathlib import Path

import matplotlib.pyplot as plt
import torch

from DataLoader import SequenceBase, StereoFrame
from Module.Frontend.StereoDepth import IStereoDepth
from Utility.Config import load_config
from Utility.Plot import getColor
from Utility.Utils import cropToMultiple


def main(args):
    sequence_cfg, _ = load_config(Path(args.seqcfg))
    depth_cfg, _ = load_config(Path(args.depthcfg))

    sequence = SequenceBase.instantiate(**vars(sequence_cfg))
    depth_est = IStereoDepth.instantiate(depth_cfg.type, depth_cfg.args)

    avg_err_percentage = []
    avg_errs = []

    frame: StereoFrame
    for frame in sequence:
        if frame.frame_idx % 3 != 0:
            continue
        est_output = depth_est.estimate(frame.stereo)
        
        assert est_output.depth is not None
        assert frame.stereo.gt_depth is not None
        
        ref_depth = cropToMultiple(frame.stereo.gt_depth, [64, 64], [2, 3])

        est_disparity = (frame.stereo.fx * frame.stereo.frame_baseline) / est_output.depth
        ref_disparity = (frame.stereo.fx * frame.stereo.frame_baseline) / ref_depth

        masking = est_disparity >= 1

        err_disparity = torch.abs(est_disparity[masking] - ref_disparity[masking])
        err_percentage = err_disparity / torch.abs(est_disparity[masking])

        avg_err_percentage.append(err_percentage.mean().item())
        avg_errs.append((est_disparity[masking] - ref_disparity[masking]).mean().item())
        print(
            f"{frame.frame_idx} | Average: ",
            err_disparity.mean().item(),
            " Percent ",
            err_percentage.mean().item(),
        )

    _ = plt.figure(figsize=(10, 5), dpi=300)
    plt.plot(
        avg_err_percentage, color=getColor("-", 4, 0), label="Disp error rate (gamma)"
    )
    plt.grid(True, "major", linestyle="--")
    plt.axhline(
        y=0.2, color="orange", linestyle="--", linewidth=2, label="Approximation Limit"
    )
    plt.title(args.seqcfg.split("/")[-1])
    plt.legend()
    plt.savefig("./Results/ErrorRate.png")
    plt.close()

    _ = plt.figure(figsize=(10, 5), dpi=300)
    plt.plot(avg_errs, color=getColor("-", 4, 0), label="Average Disp Error")
    plt.grid(True, "major", linestyle="--")
    plt.ylim(-5, 5)
    plt.axhline(
        y=0., color="orange", linestyle="--", linewidth=2, label="Approximation Limit"
    )
    plt.title(args.seqcfg.split("/")[-1])
    plt.legend()
    plt.savefig("./Results/ErrorMean.png")
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seqcfg",
        type=str,
        default="./Config/Run/Sequence/TartanAir_abandonedfac_000.yaml",
    )
    parser.add_argument(
        "--depthcfg", type=str, default="./Config/Run/Component/DE_TartanVO.yaml"
    )
    args = parser.parse_args()
    main(args)
