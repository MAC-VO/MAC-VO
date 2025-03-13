import torch

from DataLoader import SequenceBase, StereoFrame, ScaleFrame, NoTransform
from Module.Frontend.Matching import IMatcher
from Utility.PrettyPrint import ColoredTqdm, Logger
from Utility.Math import MahalanobisDist
from Utility.Plot import plot_flow_performance
from Utility.Visualize import fig_plt
from Utility.Datatypes import FlowPerformance, FlowCovPerformance
from Utility.Extensions import GridRecorder


@torch.inference_mode()
def evaluate_flow(matcher: IMatcher, seq: SequenceBase[StereoFrame], max_flow: float, huge_epe_warn: float | None = None, use_gt_mask: bool=False) -> FlowPerformance:
    prev_frame: StereoFrame | None = None
    results   : list[FlowPerformance]  = []
    
    frame: StereoFrame
    for frame in ColoredTqdm(seq, desc="Evaluating FlowModel"):
        if prev_frame is None:
            prev_frame = frame
            continue
        assert prev_frame.stereo.gt_flow is not None, "To evaluate flow quality, must use sequence with ground truth flow."
        
        match_out   = matcher.estimate(prev_frame.stereo, frame.stereo)
        est_flow    = match_out.flow
        gt_flow     = prev_frame.stereo.gt_flow.to(est_flow.device)
        
        error       = (est_flow - gt_flow).square_()
        epe         = torch.sum(error, dim=1, keepdim=True).sqrt()
        
        if use_gt_mask:
            mask        = prev_frame.stereo.flow_mask
            assert mask is not None
        else:
            mask        = est_flow < max_flow
            mask        = torch.logical_and(mask[:, :1], mask[:, 1:])
            if match_out.mask is not None:
                mask &= match_out.mask
        
        results.append(FlowPerformance(
            masked_epe= epe[mask].float().nanmean().item(),
            epe       = epe.float().nanmean().item(),
            px1       = (epe[mask] < 1).float().nanmean().item(),
            px3       = (epe[mask] < 3).float().nanmean().item(),
            px5       = (epe[mask] < 5).float().nanmean().item()
        ))
        
        if huge_epe_warn is not None and results[-1].masked_epe > huge_epe_warn:
            Logger.write("warn", f"Flow {prev_frame.frame_idx}->{frame.frame_idx} huge masked epe (> {huge_epe_warn}): epe={results[-1].masked_epe}")
            fig_plt.plot_imatcher("EvalFlow", match_out, prev_frame, frame)
        prev_frame = frame
    
    plot_flow_performance(results, f"Flow_error_distribution.png")
    
    return FlowPerformance.mean(results)


@torch.inference_mode()
def evaluate_flowcov(matcher: IMatcher, seq: SequenceBase[StereoFrame], max_flow: float, use_gt_mask: bool=False) -> FlowCovPerformance:
    assert matcher.provide_cov, f"Cannot evaluate covariance for {matcher} since no cov is provided by the module."
    
    prev_frame: StereoFrame | None = None
    cov_u_recorder = GridRecorder((0., 25., .25), (0., 25., .25))
    cov_v_recorder = GridRecorder((0., 25., .25), (0., 25., .25))
    results: list[FlowCovPerformance] = []
    
    frame: StereoFrame
    for frame in ColoredTqdm(seq, desc="Evaluate FlowCov"):
        if prev_frame is None:
            prev_frame = frame
            continue
        assert prev_frame.stereo.gt_flow is not None, "To evaluate flow quality, must use sequence with ground truth flow."
        
        est_out = matcher.estimate(prev_frame.stereo, frame.stereo)
        est_flow, est_cov = est_out.flow, est_out.cov
        assert est_cov is not None
        est_cov = est_cov[:, :2]    #FIXME: we ignore the model's prediction on off-diagonal matching uncertainty here.
        
        gt_flow           = prev_frame.stereo.gt_flow.to(est_flow.device)
        
        error       = est_flow - gt_flow
        error2      = error.square()
        
        if use_gt_mask:
            mask = prev_frame.stereo.flow_mask
        else:
            mask        = est_flow < max_flow
            mask        = torch.logical_and(mask[:, :1], mask[:, 1:])
            if est_out.mask is not None: mask &= est_out.mask
        
        B, H, W     = est_flow.size(0), est_cov.size(2), est_cov.size(3)
        est_cov_mat = est_cov.permute(0, 2, 3, 1).diag_embed()  # [B x H x W x 2 x 2]
        error_mat   = error.permute(0, 2, 3, 1)                 # [B x H x W x 2]
        
        est_cov_mat = est_cov_mat.flatten(end_dim=2)            # [B*H*W x 2 x 2]
        error_mat   = error_mat.flatten(end_dim=2)              # [B*H*W x 2]
        
        likelihood  = est_cov_mat.det().log().view(-1, 1, 1) + MahalanobisDist(error_mat, torch.zeros_like(error_mat), est_cov_mat).square()
        likelihood  = likelihood.reshape(B, H, W, 1).permute(0, 3, 1, 2)
        
        if est_out.mask is not None:
            mask &= est_out.mask

        uncertainty = est_cov_mat.det()
        q25_mask    = (uncertainty < uncertainty.nanquantile(0.25)).resize_as_(likelihood)
        q50_mask    = (uncertainty < uncertainty.nanquantile(0.5)).resize_as_(likelihood)
        q75_mask    = (uncertainty < uncertainty.nanquantile(0.75)).resize_as_(likelihood)
        
        
        results.append(FlowCovPerformance(
            masked_nll        = likelihood[mask].nanmean().item(),
            q25_nll           = likelihood[q25_mask].nanmean().item(),
            q50_nll           = likelihood[q50_mask].nanmean().item(),
            q75_nll           = likelihood[q75_mask].nanmean().item(),
        ))
        
        cov_u_recorder.store(
            error2[:, 0].detach().cpu().numpy(),
            est_cov[:, 0].detach().cpu().numpy()
        )
        cov_v_recorder.store(
            error2[:, 1].detach().cpu().numpy(),
            est_cov[:, 1].detach().cpu().numpy()
        )
        
        prev_frame = frame
    
    mean_cov = FlowCovPerformance.mean(results)
    cov_u_recorder.plot_figure("Error_u^2", "Estimated Covariance (u)", 'Log').savefig("flow_u_cov_accuracy.png")
    cov_v_recorder.plot_figure("Error_v^2", "Estimated Covariance (v)", 'Log').savefig("flow_v_cov_accuracy.png")
    
    return mean_cov


if __name__ == "__main__":
    import argparse
    from Utility.Config import load_config
    from types import SimpleNamespace
    from pathlib import Path
    
    torch.set_float32_matmul_precision('medium')
    
    args = argparse.ArgumentParser()
    args.add_argument("--data", type=str, nargs="+", default=[])
    args.add_argument("--config", type=str, required=True)
    args.add_argument("--scale_image", type=float, default=1.)
    args.add_argument("--max_flow", type=float, default=128.)
    args.add_argument("--gt_mask", action="store_true")
    args = args.parse_args()
    
    match_cfg, _ = load_config(Path(args.config))
    matcher = IMatcher.instantiate(match_cfg.type, match_cfg.args)
    
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
        
        print(evaluate_flow(matcher, seq, max_flow=args.max_flow, huge_epe_warn=None, use_gt_mask=args.gt_mask))
        
        if matcher.provide_cov: print(evaluate_flowcov(matcher, seq, max_flow=args.max_flow, use_gt_mask=args.gt_mask))
        else: print(f"Skipped cov evaluation since module does not estimate covariance.")
