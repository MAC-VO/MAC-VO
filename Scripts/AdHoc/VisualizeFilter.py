import torch
from pathlib import Path
from types import SimpleNamespace

from DataLoader import GenericSequence, SourceDataFrame
from Module.Map import BatchObservation
from Module import Covariance, IKeypointSelector, IObservationFilter, IFrontend
from Utility.Config import build_dynamic_config, load_config
from Utility.Point import filterPointsInRange
from Utility.Visualizer import PLTVisualizer
from Utility.PrettyPrint import ColoredTqdm

PLTVisualizer.setup(PLTVisualizer.State.SAVE_FILE, save_path=Path("./Results"), dpi=600)


gtmatchcfg, _ = build_dynamic_config({"type": "GTMatcher", "args": {}})
frontendcfg, _ = build_dynamic_config({
    # Frontend config here
})

kpcfg, _ = build_dynamic_config(
    {
        "type": "CovAwareSelector",
        "args": {"mask_width": 32, "max_depth": "auto", "kernel_size": 7, "device": "cuda",
                 "max_depth_cov": 250., "max_match_cov": 100.},
    }
)
outliercfg, _ = build_dynamic_config(
    {
        "type": "FilterCompose",
        "args": {
            "verbose": True,
            "filter_args": [
                {"type": "CovarianceSanityFilter", "args": {}},
                {"type": "SimpleDepthFilter", "args": {"min_depth": 0.05, "max_depth": "auto"}},
            ]
        },
    }
)


if __name__ == "__main__":
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument("--data", type=str, default="./Config/Sequence/TartanAir_seaside_000.yaml")
    args = args.parse_args()    
    datacfg, _ = load_config(Path(args.data))

    seq = GenericSequence[SourceDataFrame].instantiate(**vars(datacfg))
    kp_select = IKeypointSelector.instantiate(kpcfg.type, kpcfg.args)
    frontend   = IFrontend.instantiate(frontendcfg.type, frontendcfg.args)
    obs_filter = IObservationFilter.instantiate(outliercfg.type, outliercfg.args)

    covmodel1 = Covariance.IObservationCov.instantiate(
        "GaussianMixtureCovariance", SimpleNamespace(kernel_size=17, match_cov_default=0.25,
        min_flow_cov=0.25, min_depth_cov=0.05)
    )

    for idx in ColoredTqdm(range(2588, 2589)):
        frameA = seq[idx]
        frameB = seq[idx + 1]
        assert frameA is not None and frameB is not None
        
        est_depthA, est_depthcovA, _, _                  = frontend.estimate(None  , frameA)
        est_depth , est_depthcov , est_flow, est_flowcov = frontend.estimate(frameA, frameB)
        
        kps = kp_select.select_point(
            frameA,
            200,
            est_depthA,
            est_depthcovA,
            est_flowcov,
        )
        kp2 = kps + frontend.retrieve_pixels(kps, est_flow).T
        kpfilter = filterPointsInRange(
            kp2, (32, frameA.meta.width - 32), (32, frameA.meta.height - 32)
        )
        kp2 = kp2[kpfilter]

        kpm_cov     = frontend.retrieve_pixels(kps[kpfilter], est_flowcov)
        kpd         = frontend.retrieve_pixels(kp2, est_depth)
        kpd_cov     = frontend.retrieve_pixels(kp2, est_depthcov)
        
        assert kpm_cov is not None
        assert kpd_cov is not None
        assert est_depth is not None

        kpcovs1 = covmodel1.estimate(
            frameB,
            kp2,
            depth_map=est_depth,
            depth_cov_map=est_depthcov,
            depth_cov=kpd_cov,
            flow_cov=kpm_cov
        )

        obs = BatchObservation(
            torch.zeros((kp2.size(0)), dtype=torch.long), torch.zeros((kp2.size(0)), dtype=torch.long),
            # For simplicity - 
            # we don't need graph here, only want to have some observation to plot
            # so put some dummy value above
            kp2, kpd, kpcovs1, kpm_cov, kpd_cov
        )
        obs_filter.set_meta(frameB.meta)
        outlier_filter = obs_filter.filter(obs)

        PLTVisualizer.visualize_Obs(
            f"Keypoints", 
            frameA.imageL, frameB.imageL,
            obs,
            est_depthcov,
            est_flowcov,
            outlier_filter
        )
        if est_flow is not None:
            PLTVisualizer.visualize_flow(f"flow", est_flow)
        
        PLTVisualizer.visualize_stereo("Stereo", frameB.imageL, frameB.imageR)
        PLTVisualizer.visualize_depth("depth", est_depth)
        PLTVisualizer.visualize_covTc("covTc", obs)
