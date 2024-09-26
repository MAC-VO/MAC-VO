from pathlib import Path

from DataLoader import GenericSequence
from Module.Frontend.StereoDepth import IStereoDepth
from Utility.Config import build_dynamic_config, load_config
from Utility.Visualizer import PLTVisualizer, RerunVisualizer


if __name__ == "__main__":
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument("--data", type=str, default="./Config/Sequence/TartanAir_seaside_000.yaml")
    args = args.parse_args()
    datacfg, _ = load_config(Path(args.data))

    depthcfg, _ = build_dynamic_config(
        {
            "type": "TartanVODepth",
            "args": {
                "weight": "./Model/TartanVO_depth_cov.pth",
                "device": "cuda",
                "cov_mode": "Est",
            },
        }
    )
    gtdepthcfg, _ = build_dynamic_config({"type": "GTDepth", "args": {}})

    seq = GenericSequence.instantiate(**vars(datacfg)).clip(0, 20)
    depth_estimator = IStereoDepth.instantiate(depthcfg.type, depthcfg.args)
    gt_estimator = IStereoDepth.instantiate(gtdepthcfg.type, gtdepthcfg.args)

    RerunVisualizer.setup("visualizer", False, Path("Results", "output.rrd"), True)
    PLTVisualizer.setup(state=PLTVisualizer.State.SAVE_FILE, save_path=Path("."), dpi=600)

    for idx in range(262, 300):
        frameA, frameB = seq[idx], seq[idx + 1]
        est_depth, est_cov = depth_estimator.estimate(frameA)
        gt_depth , _       = gt_estimator.estimate(frameA)
        
        assert est_depth is not None
        assert gt_depth is not None
        assert est_cov is not None
        
        cov_gt = (gt_depth[0, 0, 16:464, :] - est_depth).square()
        PLTVisualizer.visualize_depth_grid(
            "./results", 
            frameA.imageL[0].permute(1, 2, 0), frameA.imageR[0].permute(1, 2, 0),
            gt_depth[0, 0], est_depth[0, 0],
            cov_gt[0, 0], est_cov[0, 0]
        )
