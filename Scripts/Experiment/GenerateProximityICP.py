"""Generate geometry-selected, bidirectionally validated proximity ICP factors."""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from DataLoader import SequenceBase, StereoFrame, smart_transform
from Module.Optimization.CovisibilityCache import build_covisibility_cache
from Module.Optimization.CovisibilitySelector import (
    CovisibilitySelector,
    CovisibilitySelectorConfig,
)
from Module.Optimization.FactorArchive import (
    load_factor_archive,
    save_factor_archive,
)
from Module.Optimization.ProximityICP import (
    ProximityValidationConfig,
    generate_proximity_archive,
)
from Odometry.WindowMACVO import WindowMACVO
from Scripts.Experiment.GenerateLongRangeICP import (
    file_sha256,
    optional_index,
)
from Utility.Sandbox import Sandbox


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space", type=Path, required=True)
    parser.add_argument("--cache", default="covisibility_cache")
    parser.add_argument("--output", default="proximity_factors.npz")
    parser.add_argument("--keyframe-stride", type=int, default=5)
    parser.add_argument("--proxy-points", type=int, default=1024)
    parser.add_argument("--min-temporal-gap", type=int, default=15)
    parser.add_argument("--min-directional-overlap", type=float, default=0.15)
    parser.add_argument("--min-mean-overlap", type=float, default=0.25)
    parser.add_argument("--min-depth-consistency", type=float, default=0.30)
    parser.add_argument("--min-median-motion-px", type=float, default=8.0)
    parser.add_argument("--max-median-motion-px", type=float, default=240.0)
    parser.add_argument("--candidate-nms-frames", type=int, default=5)
    parser.add_argument("--max-candidates-per-target", type=int, default=1)
    parser.add_argument("--max-forward-backward-error-px", type=float, default=2.0)
    parser.add_argument("--min-forward-backward-inliers", type=int, default=30)
    parser.add_argument("--min-forward-backward-ratio", type=float, default=0.50)
    parser.add_argument("--min-depth-valid-ratio", type=float, default=0.50)
    parser.add_argument("--grid-rows", type=int, default=4)
    parser.add_argument("--grid-cols", type=int, default=6)
    parser.add_argument("--min-occupied-grid-cells", type=int, default=6)
    parser.add_argument("--mahalanobis-threshold", type=float, default=3.5)
    parser.add_argument("--min-mahalanobis-inliers", type=int, default=30)
    parser.add_argument("--min-mahalanobis-inlier-ratio", type=float, default=0.50)
    return parser


def _directory_bytes(path):
    return sum(item.stat().st_size for item in Path(path).rglob("*") if item.is_file())


def main(argv=None):
    args = build_parser().parse_args(argv)
    space = Sandbox.load(args.space)
    cfg = space.config
    if not hasattr(cfg, "Preprocess"):
        raise ValueError("Saved result config does not contain Preprocess")
    source_path = space.path("global_factors.npz")
    source = load_factor_archive(source_path)
    sequence = smart_transform(
        SequenceBase[StereoFrame].instantiate(
            cfg.Data.args.type, cfg.Data.args.args
        ).clip(
            optional_index(cfg.Data.start_idx),
            optional_index(cfg.Data.end_idx),
        ),
        cfg.Preprocess,
    )
    system = WindowMACVO.from_config(cfg)
    cache_path = space.path(args.cache)
    cache_preexisted = cache_path.exists()
    cache_start = time.perf_counter()
    cache = build_covisibility_cache(
        sequence,
        source,
        cache_path,
        system.Frontend,
        stride=args.keyframe_stride,
        proxy_points=args.proxy_points,
    )
    cache_seconds = time.perf_counter() - cache_start
    intrinsics = {
        int(frame_id): sequence[int(frame_id)].stereo.frame_K.cpu().double()
        for frame_id in cache.frame_ids
    }
    selector_config = CovisibilitySelectorConfig(
        min_temporal_gap=args.min_temporal_gap,
        min_directional_overlap=args.min_directional_overlap,
        min_mean_overlap=args.min_mean_overlap,
        min_depth_consistency=args.min_depth_consistency,
        min_median_motion_px=args.min_median_motion_px,
        max_median_motion_px=args.max_median_motion_px,
        candidate_nms_frames=args.candidate_nms_frames,
        max_candidates_per_target=args.max_candidates_per_target,
    )
    validation_config = ProximityValidationConfig(
        max_forward_backward_error_px=args.max_forward_backward_error_px,
        min_forward_backward_inliers=args.min_forward_backward_inliers,
        min_forward_backward_ratio=args.min_forward_backward_ratio,
        min_depth_valid_ratio=args.min_depth_valid_ratio,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        min_occupied_grid_cells=args.min_occupied_grid_cells,
        mahalanobis_threshold=args.mahalanobis_threshold,
        min_mahalanobis_inliers=args.min_mahalanobis_inliers,
        min_mahalanobis_inlier_ratio=args.min_mahalanobis_inlier_ratio,
    )
    selector = CovisibilitySelector(
        cache=cache,
        poses=source.initial_sensor_poses,
        intrinsics=intrinsics,
        config=selector_config,
    )
    result = generate_proximity_archive(
        sequence=sequence,
        source_archive=source,
        cache=cache,
        frontend=system.Frontend,
        covisibility_selector=selector,
        keypoint_selector=system.KeypointSelector,
        covariance_model=system.ObsCovModel,
        num_point=system.num_point,
        edge_width=system.edge_width,
        match_cov_default=system.match_cov_default,
        device=system.device,
        validation_config=validation_config,
    )
    output_path = space.path(args.output)
    save_factor_archive(output_path, result.archive)
    (space.path("covisibility_candidates.json")).write_text(
        json.dumps(result.candidate_records, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    diagnostics = result.diagnostics | {
        "source_factor_sha256": file_sha256(source_path),
        "proximity_factor_sha256": file_sha256(output_path),
        "cache": cache_path.name,
        "cache_bytes": _directory_bytes(cache_path),
        "cache_seconds": cache_seconds,
        "pass_a_frontend_calls": 0 if cache_preexisted else len(cache.frame_ids),
        "pass_b_frontend_calls": result.diagnostics["frontend_calls"],
        "validation_records": result.validation_records,
        "selector_config": vars(selector_config),
        "validation_config": vars(validation_config),
    }
    space.path("proximity_generation.json").write_text(
        json.dumps(diagnostics, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(output_path)
    return 0 if result.diagnostics["status"] == "generated" else 2


if __name__ == "__main__":
    raise SystemExit(main())
