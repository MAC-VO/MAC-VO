"""Geometry-driven historical frame selection from cached depth proxies."""

from dataclasses import dataclass
import math

import numpy as np
import pypose as pp
import torch

from Utility.Point import pixel2point_NED, point2pixel_NED


@dataclass(frozen=True)
class CovisibilitySelectorConfig:
    min_temporal_gap: int = 15
    min_directional_overlap: float = 0.15
    min_mean_overlap: float = 0.25
    min_depth_consistency: float = 0.30
    min_median_motion_px: float = 8.0
    max_median_motion_px: float = 240.0
    candidate_nms_frames: int = 5
    max_candidates_per_target: int = 1

    def __post_init__(self):
        if self.min_temporal_gap < 0 or self.candidate_nms_frames < 0:
            raise ValueError("temporal selector settings must be non-negative")
        for name in (
            "min_directional_overlap",
            "min_mean_overlap",
            "min_depth_consistency",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be within [0, 1]")
        if (
            not math.isfinite(self.min_median_motion_px)
            or not math.isfinite(self.max_median_motion_px)
            or self.min_median_motion_px < 0
            or self.max_median_motion_px < self.min_median_motion_px
        ):
            raise ValueError("invalid covisibility motion bounds")
        if self.max_candidates_per_target != 1:
            raise ValueError("first proximity version supports one candidate")


@dataclass(frozen=True)
class CovisibilityMetrics:
    source: int
    target: int
    overlap_forward: float
    overlap_backward: float
    mean_overlap: float
    depth_consistency: float
    median_motion_px: float
    score: float
    status: str
    reason: str | None

    def as_dict(self):
        return {
            "source": self.source,
            "target": self.target,
            "overlap_forward": self.overlap_forward,
            "overlap_backward": self.overlap_backward,
            "mean_overlap": self.mean_overlap,
            "depth_consistency": self.depth_consistency,
            "median_motion_px": self.median_motion_px,
            "score": self.score,
            "status": self.status,
            "reason": self.reason,
        }


def _cache_index(cache, frame_id):
    matches = np.flatnonzero(cache.frame_ids == frame_id)
    if len(matches) != 1:
        raise KeyError(f"Frame {frame_id} is not in the covisibility cache")
    return int(matches[0])


def _directional_projection(
    source,
    target,
    cache,
    poses,
    intrinsics,
):
    source_index = _cache_index(cache, source)
    target_index = _cache_index(cache, target)
    uv_source = torch.from_numpy(
        np.asarray(cache.proxy_uv, dtype=np.float64)
    )
    source_depth = torch.from_numpy(
        np.asarray(cache.proxy_depth[source_index], dtype=np.float64)
    )
    source_cov = torch.from_numpy(
        np.asarray(cache.proxy_depth_cov[source_index], dtype=np.float64)
    )
    valid_source = (
        torch.isfinite(source_depth)
        & torch.isfinite(source_cov)
        & (source_depth > 0)
        & (source_cov >= 0)
    )
    if not bool(valid_source.any()):
        return 0.0, 0.0, torch.empty(0, dtype=torch.float64)
    uv_source = uv_source[valid_source]
    source_depth = source_depth[valid_source]
    source_cov = source_cov[valid_source]
    K_source = torch.as_tensor(intrinsics[source]).double()
    K_target = torch.as_tensor(intrinsics[target]).double()
    points_source = pixel2point_NED(
        uv_source, source_depth, K_source
    ).double()
    relative = pp.SE3(torch.as_tensor(poses[target]).double()).Inv() @ pp.SE3(
        torch.as_tensor(poses[source]).double()
    )
    points_target = relative.Act(points_source)
    projected_uv = point2pixel_NED(points_target, K_target).double()
    height = int(cache.metadata["height"])
    width = int(cache.metadata["width"])
    in_bounds = (
        torch.isfinite(projected_uv).all(-1)
        & torch.isfinite(points_target).all(-1)
        & (points_target[:, 0] > 0)
        & (projected_uv[:, 0] >= 0)
        & (projected_uv[:, 0] < width)
        & (projected_uv[:, 1] >= 0)
        & (projected_uv[:, 1] < height)
    )
    overlap = float(in_bounds.double().mean())
    if not bool(in_bounds.any()):
        return overlap, 0.0, torch.empty(0, dtype=torch.float64)

    uv_valid = projected_uv[in_bounds]
    source_uv_valid = uv_source[in_bounds]
    target_points_valid = points_target[in_bounds]
    source_cov_valid = source_cov[in_bounds]
    rounded = uv_valid.round().long()
    rounded[:, 0].clamp_(0, width - 1)
    rounded[:, 1].clamp_(0, height - 1)
    target_depth = torch.from_numpy(
        np.asarray(cache.depth[target_index], dtype=np.float64)
    )[rounded[:, 1], rounded[:, 0]]
    target_cov = torch.from_numpy(
        np.asarray(cache.depth_cov[target_index], dtype=np.float64)
    )[rounded[:, 1], rounded[:, 0]]
    valid_target = (
        torch.isfinite(target_depth)
        & torch.isfinite(target_cov)
        & (target_depth > 0)
        & (target_cov >= 0)
    )
    consistent = torch.zeros(len(uv_valid), dtype=torch.bool)
    if bool(valid_target.any()):
        error = (
            target_points_valid[valid_target, 0]
            - target_depth[valid_target]
        ).abs()
        sigma = torch.sqrt(
            source_cov_valid[valid_target] + target_cov[valid_target]
        ).clamp_min(1e-9)
        consistent[valid_target] = error / sigma <= 3.0
    depth_consistency = float(consistent.double().mean())
    motion = (uv_valid - source_uv_valid).norm(dim=-1)
    return overlap, depth_consistency, motion


def _rejected(source, target, reason):
    return CovisibilityMetrics(
        source=source,
        target=target,
        overlap_forward=0.0,
        overlap_backward=0.0,
        mean_overlap=0.0,
        depth_consistency=0.0,
        median_motion_px=0.0,
        score=1e9,
        status="rejected",
        reason=reason,
    )


def score_covisibility_pair(
    source,
    target,
    cache,
    poses,
    intrinsics,
    config,
):
    if target - source < config.min_temporal_gap:
        return _rejected(source, target, "temporal_gap_too_small")
    forward, forward_depth, forward_motion = _directional_projection(
        source, target, cache, poses, intrinsics
    )
    backward, backward_depth, backward_motion = _directional_projection(
        target, source, cache, poses, intrinsics
    )
    mean_overlap = 0.5 * (forward + backward)
    depth_consistency = 0.5 * (forward_depth + backward_depth)
    motions = torch.cat([forward_motion, backward_motion])
    median_motion = 0.0 if len(motions) == 0 else float(motions.median())
    diagonal = math.hypot(
        float(cache.metadata["height"]),
        float(cache.metadata["width"]),
    )
    score = (
        (1.0 - mean_overlap)
        + 0.5 * (1.0 - depth_consistency)
        + 0.1 * median_motion / diagonal
    )
    reason = None
    if forward < config.min_directional_overlap:
        reason = "forward_overlap_too_low"
    elif backward < config.min_directional_overlap:
        reason = "backward_overlap_too_low"
    elif mean_overlap < config.min_mean_overlap:
        reason = "mean_overlap_too_low"
    elif depth_consistency < config.min_depth_consistency:
        reason = "depth_consistency_too_low"
    elif median_motion < config.min_median_motion_px:
        reason = "motion_too_small"
    elif median_motion > config.max_median_motion_px:
        reason = "motion_too_large"
    elif not math.isfinite(score):
        reason = "nonfinite_score"
    return CovisibilityMetrics(
        source=source,
        target=target,
        overlap_forward=forward,
        overlap_backward=backward,
        mean_overlap=mean_overlap,
        depth_consistency=depth_consistency,
        median_motion_px=median_motion,
        score=score if math.isfinite(score) else 1e9,
        status="eligible" if reason is None else "rejected",
        reason=reason,
    )


def pair_suppressed(source, target, accepted_pairs, radius):
    return any(
        abs(source - old_source) <= radius
        and abs(target - old_target) <= radius
        for old_source, old_target in accepted_pairs
    )


class CovisibilitySelector:
    def __init__(self, *, cache, poses, intrinsics, config):
        self.cache = cache
        self.poses = poses
        self.intrinsics = intrinsics
        self.config = config

    def select(self, *, target, history, accepted_pairs):
        evaluated = [
            score_covisibility_pair(
                source,
                target,
                self.cache,
                self.poses,
                self.intrinsics,
                self.config,
            )
            for source in sorted(history)
        ]
        eligible = [
            record
            for record in evaluated
            if record.status == "eligible"
            and not pair_suppressed(
                record.source,
                record.target,
                accepted_pairs,
                self.config.candidate_nms_frames,
            )
        ]
        eligible.sort(key=lambda record: (record.score, record.source))
        return (eligible[0] if eligible else None), evaluated
