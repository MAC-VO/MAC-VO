"""Shared deterministic construction of covariance-aware ICP match factors."""

from dataclasses import dataclass

import torch

from Module.Optimization.WindowICP import Edge
from Utility.Point import filterPointsInRange, pixel2point_NED


@dataclass(frozen=True)
class MatchEdgeBuildResult:
    edge: Edge | None
    reason: str | None
    selected_points: int
    inbound_points: int


@dataclass(frozen=True)
class CorrespondenceEdgeBuildResult:
    edge: Edge | None
    uv_a: torch.Tensor
    uv_b: torch.Tensor
    reason: str | None


@torch.inference_mode()
def build_edge_from_correspondences(
    *,
    a,
    b,
    uv_a,
    uv_b,
    stereo_a,
    stereo_b,
    depth_a,
    depth_b,
    match,
    frontend,
    covariance_model,
    min_num_point,
    match_cov_default,
    device,
):
    d_a = frontend.retrieve_pixels(uv_a, depth_a.depth).squeeze(0)
    d_b = frontend.retrieve_pixels(uv_b, depth_b.depth).squeeze(0)
    depth_cov_a = frontend.retrieve_pixels(uv_a, depth_a.cov).squeeze(0)
    depth_cov_b = frontend.retrieve_pixels(uv_b, depth_b.cov).squeeze(0)
    uv_cov_a = torch.full(
        (len(uv_a), 3), match_cov_default, device=device
    )
    uv_cov_a[:, 2] = 0
    uv_cov_b = frontend.retrieve_pixels(uv_a, match.cov).T.clone()
    cov_a = covariance_model.estimate(
        stereo_a, uv_a, depth_a, depth_cov_a, uv_cov_a
    ).cpu().double()
    cov_b = covariance_model.estimate(
        stereo_b, uv_b, depth_b, depth_cov_b, uv_cov_b
    ).cpu().double()
    points_a = pixel2point_NED(
        uv_a.cpu(), d_a.cpu(), stereo_a.frame_K.cpu()
    ).double()
    points_b = pixel2point_NED(
        uv_b.cpu(), d_b.cpu(), stereo_b.frame_K.cpu()
    ).double()
    d_a = d_a.cpu()
    d_b = d_b.cpu()
    valid = (
        torch.isfinite(points_a).all(-1)
        & torch.isfinite(points_b).all(-1)
        & (d_a > 0)
        & (d_b > 0)
        & torch.isfinite(cov_a).all(dim=(-2, -1))
        & torch.isfinite(cov_b).all(dim=(-2, -1))
    )
    ids = torch.nonzero(valid).flatten()
    if len(ids):
        positive_definite = (
            torch.linalg.cholesky_ex(cov_a[ids]).info == 0
        ) & (torch.linalg.cholesky_ex(cov_b[ids]).info == 0)
        ids = ids[positive_definite]
    kept_uv_a = uv_a.detach().cpu()[ids]
    kept_uv_b = uv_b.detach().cpu()[ids]
    if len(ids) < min_num_point:
        return CorrespondenceEdgeBuildResult(
            None,
            kept_uv_a,
            kept_uv_b,
            "too_few_valid_observations",
        )
    return CorrespondenceEdgeBuildResult(
        Edge(
            a,
            b,
            points_a[ids],
            points_b[ids],
            cov_a[ids],
            cov_b[ids],
        ),
        kept_uv_a,
        kept_uv_b,
        None,
    )


@torch.inference_mode()
def build_match_edge(
    *,
    a,
    b,
    stereo_a,
    stereo_b,
    depth_a,
    depth_b,
    match,
    frontend,
    selector,
    covariance_model,
    num_point,
    min_num_point,
    edge_width,
    match_cov_default,
    device,
):
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(1000003 + a * 1009 + b)
        uv_a = selector.select_point(
            stereo_a, num_point, depth_a, depth_b, match
        )
    selected_points = len(uv_a)
    uv_b = uv_a + frontend.retrieve_pixels(uv_a, match.flow).T
    inbound = torch.isfinite(uv_b).all(-1) & filterPointsInRange(
        uv_b,
        (edge_width, stereo_b.width - edge_width),
        (edge_width, stereo_b.height - edge_width),
    )
    uv_a, uv_b = uv_a[inbound], uv_b[inbound]
    inbound_points = int(inbound.sum())
    if len(uv_a) < min_num_point:
        return MatchEdgeBuildResult(
            None,
            "too_few_inbound_points",
            selected_points,
            inbound_points,
        )

    built = build_edge_from_correspondences(
        a=a,
        b=b,
        uv_a=uv_a,
        uv_b=uv_b,
        stereo_a=stereo_a,
        stereo_b=stereo_b,
        depth_a=depth_a,
        depth_b=depth_b,
        match=match,
        frontend=frontend,
        covariance_model=covariance_model,
        min_num_point=min_num_point,
        match_cov_default=match_cov_default,
        device=device,
    )
    return MatchEdgeBuildResult(
        built.edge,
        built.reason,
        selected_points,
        inbound_points,
    )
