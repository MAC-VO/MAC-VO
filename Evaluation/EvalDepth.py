import torch
from types import SimpleNamespace

from DataLoader import SequenceBase, StereoFrame, ScaleFrame, NoTransform
from Module.Frontend.StereoDepth import IStereoDepth
from Utility.PrettyPrint import ColoredTqdm, Logger
from Utility.Math import MahalanobisDist
from Utility.Datatypes import DepthPerformance, DepthCovPerformance
from Utility.Extensions import GridRecorder


@torch.inference_mode()
def evaluate_depth(depth: IStereoDepth, seq: SequenceBase[StereoFrame], max_depth: float = 80.) -> DepthPerformance:
    results : list[DepthPerformance] = []
    frame   : StereoFrame
    for frame in ColoredTqdm(seq, desc="Evaluating DepthModel"):
        assert frame.stereo.gt_depth is not None, "To evaluate depth quality, must use trajectory with gtDepth"
        
        est_output   = depth.estimate(frame.stereo)
        gt_depth     = frame.stereo.gt_depth.to(est_output.depth.device)
        
        error        = (est_output.depth - gt_depth).abs()
        mask         = est_output.depth < max_depth
        
        # if est_output.mask is not None:
        #     mask &= est_output.mask
        
        try:
            results.append(DepthPerformance(
                masked_err=error[mask].float().nanmean().item(),
                err_25    =error[mask].float().nanquantile(0.25).item(),
                err_50    =error[mask].float().nanmedian().item(),
                err_75    =error[mask].float().nanquantile(0.75).item(),
            ))
        except Exception as e:
            Logger.show_exception()
            Logger.write("warn", f"Failed to evaluate depth performance at frame {frame.frame_idx} - reason:\n{e}")
    
    return DepthPerformance.median(results)


@torch.inference_mode()
def evaluate_depthcov(depth: IStereoDepth, seq: SequenceBase[StereoFrame], max_depth: float = 80.) -> DepthCovPerformance:
    assert depth.provide_cov, f"Cannot evaluate covariance for {depth} since no cov is provided by the module."
    
    cov_performance_recorder = GridRecorder((0., 50., .5), (0., 50., .5))
    results: list[DepthCovPerformance] = []
    frame: StereoFrame
    for frame in ColoredTqdm(seq, desc="Evaluate DepthCov"):
        assert frame.stereo.gt_depth is not None, "To evaluate depth cov quality, must use sequence with ground truth depth."
        
        est_out = depth.estimate(frame.stereo)
        assert est_out.cov is not None, "IStereoDepth implementation did not provide cov estimation as promised (via provide_cov API)"
        
        gt_depth    = frame.stereo.gt_depth.to(est_out.depth.device)
        
        error       = est_out.depth - gt_depth
        error2      = error.square()
        mask        = est_out.depth < max_depth
        
        B, H, W     = est_out.depth.size(0), est_out.depth.size(2), est_out.depth.size(3)
        est_cov_mat = est_out.cov.permute(0, 2, 3, 1).diag_embed()  # [B x H x W x 1 x 1]
        error_mat   = error.permute(0, 2, 3, 1)                     # [B x H x W x 1]
        
        est_cov_mat = est_cov_mat.flatten(end_dim=2)                # [B*H*W x 1 x 1]
        error_mat   = error_mat.flatten(end_dim=2)                  # [B*H*W x 1]
        
        likelihood  = est_cov_mat.det().log().view(-1, 1, 1) + MahalanobisDist(error_mat, torch.zeros_like(error_mat), est_cov_mat).square()
        likelihood  = likelihood.reshape(B, H, W, 1).permute(0, 3, 1, 2)

        uncertainty = est_cov_mat.det()
        q25_mask    = (uncertainty < uncertainty.nanquantile(0.25)).resize_as_(likelihood)
        q50_mask    = (uncertainty < uncertainty.nanquantile(0.5)).resize_as_(likelihood)
        q75_mask    = (uncertainty < uncertainty.nanquantile(0.75)).resize_as_(likelihood)
        
        results.append(DepthCovPerformance(
            masked_nll        = likelihood[mask].nanmean().item(),
            q25_nll           = likelihood[q25_mask].nanmean().item(),
            q50_nll           = likelihood[q50_mask].nanmean().item(),
            q75_nll           = likelihood[q75_mask].nanmean().item(),
        ))
        cov_performance_recorder.store(error2.detach().cpu().numpy(), est_out.cov.detach().cpu().numpy())
    
    mean_cov = DepthCovPerformance.mean(results)
    cov_performance_recorder.plot_figure("Error^2", "Estimated Covariance", "Log").savefig(
        "depth_cov_accuracy.png"
    )
    return mean_cov



if __name__ == "__main__":
    import argparse
    from Utility.Config import load_config
    from pathlib import Path
    
    args = argparse.ArgumentParser()
    args.add_argument("--data", type=str, nargs="+", default=[])
    args.add_argument("--depth_estimator", type=str, required=True)
    args.add_argument("--scale_image", type=float, default=1.)
    args.add_argument("--max_depth", type=float, default=80.)
    args = args.parse_args()
    
    depth_cfg, _ = load_config(Path(args.depth_estimator))
    depth_estimator = IStereoDepth.instantiate(depth_cfg.type, depth_cfg.args)
    
    if args.scale_image != 1.0:
        scale = args.scale_image
        assert isinstance(scale, (int, float))
        transform_fn = ScaleFrame(SimpleNamespace(scale_u=scale, scale_v=scale, interpolate="nearest"))
    else:
        transform_fn = NoTransform(SimpleNamespace())
    
    
    for data_path in args.data:
        data_cfg, _ = load_config(Path(data_path))
        seq         = SequenceBase[StereoFrame].instantiate(data_cfg.type, data_cfg.args).preload()
        seq         = seq.transform(transform_fn)
        
        print(evaluate_depth(depth_estimator, seq, max_depth=args.max_depth))
        if depth_estimator.provide_cov: print(evaluate_depthcov(depth_estimator, seq, max_depth=args.max_depth))
        else: print(f"Skipped cov evaluation since module does not estimate covariance.")
