from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from DataLoader import GenericSequence
from Module.Frontend.Matching import IMatcher
from Utility.Config import build_dynamic_config, load_config
from Utility.Visualizer import PLTVisualizer


def sparsification_plot(cov, mse):
    cov, mse = cov.flatten(), mse.flatten()
    sorted_indices = np.argsort(cov)[::-1]
    sorted_indices_mse = np.argsort(mse)[::-1]
    est, oracle = [], []
    for i in range(0, len(cov), len(cov) // 100):
        remain_indices = sorted_indices[i:]
        remain_indices_mse = sorted_indices_mse[i:]
        if remain_indices.shape[0] == 0:
            est.append(0)
            oracle.append(0)
            break
        est.append(np.mean(mse[remain_indices]))
        oracle.append(np.mean(mse[remain_indices_mse]))
    est, oracle = np.array(est), np.array(oracle)
    oracle = (oracle - np.min(oracle)) / (np.max(oracle) - np.min(oracle))
    factor = oracle[0] / est[0]
    est = est * factor
    cc = spearmanr(est, oracle)[0]
    auc = np.sum(est) / est.shape[0]
    dauc = np.sum(est) / np.sum(oracle)
    return est, oracle, cc, auc, dauc


if __name__ == "__main__":
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument("--data", type=str, default="./Config/Sequence/TartanAir_seaside_000.yaml")
    args = args.parse_args()
    
    datacfg, _ = load_config(Path(args.data))
    matchcfg, _ = build_dynamic_config(
        {
            "type": "FlowFormerMatcher",
            "args": {
                "weight": "./Model/MACVO_FrontendCov.pth",
                "device": "cuda",
                "cov_mode": "Est",
            },
        }
    )
    gtmatchcfg, _ = build_dynamic_config({"type": "GTMatcher", "args": {}})

    seq = GenericSequence.instantiate(**vars(datacfg))
    match_est = IMatcher.instantiate(matchcfg.type, matchcfg.args)
    match_gt = IMatcher.instantiate(gtmatchcfg.type, gtmatchcfg.args)

    PLTVisualizer.setup(state=PLTVisualizer.State.SAVE_FILE, save_path=Path("."), dpi=600)

    for idx in [0]:
        frameA, frameB = seq[idx], seq[idx + 1]
        est_flow, est_flowcov = match_est.estimate(frameA, frameB)
        gt_flow, _ = match_gt.estimate(frameA, frameB)
        flow_est = est_flow  # 2, 480, 640
        
        assert frameA.gtFlow is not None
        assert est_flowcov is not None
        
        flow_gt = frameA.gtFlow[0]
        cov_est = est_flowcov
        cov_gt = (flow_gt - flow_est).square()

        est, oracle, _, _, _ = sparsification_plot(
            np.linalg.norm(cov_est, axis=0), np.linalg.norm(cov_gt, axis=0)
        )
                
        PLTVisualizer.visualize_flow_grid(
            "result",
            frameA.imageL[0].permute(1, 2, 0), frameB.imageL[0].permute(1, 2, 0),
            flow_est, flow_gt, 
            cov_est.sum(dim=0), cov_gt.sum(dim=0),
            est, oracle
        )
