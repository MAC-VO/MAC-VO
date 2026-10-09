"""Sparse fixed-scale SE(3) geometry for loop verification."""

from dataclasses import dataclass
import math

import pypose as pp
import torch
import torch.nn.functional as F

from Module.Optimization.PoseGraph import PoseGraphFactor
from Utility.Point import pixel2point_NED, point2pixel_NED


@dataclass(frozen=True)
class SparseGeometryConfig:
    min_mutual_matches: int = 40
    min_valid_3d: int = 30
    min_ransac_inliers: int = 25
    min_ransac_ratio: float = 0.35
    grid_rows: int = 4
    grid_cols: int = 6
    min_grid_cells: int = 6
    ransac_iterations: int = 256
    mahalanobis_threshold: float = 3.5
    max_median_reprojection_px: float = 3.0
    max_p90_reprojection_px: float = 6.0
    min_geometry_ratio: float = 1e-3
    orb_pixel_variance: float = 2.25


@dataclass(frozen=True)
class SparseLoopResult:
    source: int
    target: int
    measurement: torch.Tensor | None
    inlier_mask: torch.Tensor
    metrics: dict
    reason: str | None


@dataclass(frozen=True)
class SE3RansacResult:
    measurement: torch.Tensor | None
    mask: torch.Tensor
    inliers: int
    ratio: float
    median_mahalanobis: float | None
    p90_mahalanobis: float | None
    singular_values: torch.Tensor | None
    reason: str | None


def _failure(count, reason, singular_values=None):
    return SE3RansacResult(
        None,
        torch.zeros(count, dtype=torch.bool),
        0,
        0.0,
        None,
        None,
        singular_values,
        reason,
    )


def _loop_failure(source, target, count, reason, metrics):
    return SparseLoopResult(
        source,
        target,
        None,
        torch.zeros(count, dtype=torch.bool),
        dict(metrics),
        reason,
    )


def _sample_bilinear(pixel_uv, value_map):
    height, width = value_map.shape[-2:]
    x = pixel_uv[:, 0] * 2 / max(width - 1, 1) - 1
    y = pixel_uv[:, 1] * 2 / max(height - 1, 1) - 1
    grid = torch.stack((x, y), dim=-1).view(1, -1, 1, 2)
    sampled = F.grid_sample(
        value_map,
        grid.to(device=value_map.device, dtype=value_map.dtype),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled[0, :, :, 0].T


def occupied_grid_cells(pixel_uv, width, height, rows, cols):
    if len(pixel_uv) == 0:
        return 0
    column = torch.clamp((pixel_uv[:, 0] / width * cols).long(), 0, cols - 1)
    row = torch.clamp((pixel_uv[:, 1] / height * rows).long(), 0, rows - 1)
    return int(torch.unique(row * cols + column).numel())


def _geometry_singular_values(points):
    centered = points - points.mean(dim=0)
    return torch.linalg.svdvals(centered)


def _geometry_valid(points_a, points_b, minimum_ratio):
    singular_a = _geometry_singular_values(points_a)
    singular_b = _geometry_singular_values(points_b)
    singular = torch.minimum(singular_a, singular_b)
    denominator = singular[0].clamp_min(1e-15)
    return bool(singular[1] / denominator >= minimum_ratio), singular


def _weighted_kabsch(points_a, points_b, covariance):
    weights = covariance.diagonal(dim1=-2, dim2=-1).sum(-1)
    weights = weights.clamp_min(1e-15).reciprocal()
    weights = weights / weights.sum()
    center_a = (weights[:, None] * points_a).sum(dim=0)
    center_b = (weights[:, None] * points_b).sum(dim=0)
    a = points_a - center_a
    b = points_b - center_b
    cross = (weights[:, None] * b).T @ a
    U, _, Vh = torch.linalg.svd(cross)
    rotation = Vh.T @ U.T
    if torch.linalg.det(rotation) < 0:
        Vh = Vh.clone()
        Vh[-1] *= -1
        rotation = Vh.T @ U.T
    translation = center_a - rotation @ center_b
    matrix = torch.eye(4, dtype=torch.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return pp.from_matrix(matrix, pp.SE3_type).tensor()


def _mahalanobis(measurement, points_a, points_b, cov_a, cov_b):
    transform = pp.SE3(measurement)
    rotation = transform.rotation().matrix()
    residual = points_a - transform.Act(points_b)
    covariance = cov_a + rotation @ cov_b @ rotation.T
    covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
    cholesky, info = torch.linalg.cholesky_ex(covariance)
    if int(info.max()) != 0:
        raise ValueError("invalid combined covariance")
    white = torch.linalg.solve_triangular(
        cholesky, residual[..., None], upper=False
    ).squeeze(-1)
    return white.norm(dim=-1)


@torch.no_grad()
def estimate_se3_ransac(
    points_a,
    points_b,
    cov_a,
    cov_b,
    *,
    iterations,
    mahalanobis_threshold,
    min_geometry_ratio,
    seed,
):
    points_a = torch.as_tensor(points_a).detach().cpu().double()
    points_b = torch.as_tensor(points_b).detach().cpu().double()
    cov_a = torch.as_tensor(cov_a).detach().cpu().double()
    cov_b = torch.as_tensor(cov_b).detach().cpu().double()
    count = len(points_a)
    if (
        points_a.shape != (count, 3)
        or points_b.shape != (count, 3)
        or cov_a.shape != (count, 3, 3)
        or cov_b.shape != (count, 3, 3)
    ):
        raise ValueError("sparse SE3 inputs have incompatible shapes")
    if count < 3:
        return _failure(count, "too_few_correspondences")
    if not all(torch.isfinite(value).all() for value in (
        points_a, points_b, cov_a, cov_b
    )):
        return _failure(count, "nonfinite_input")
    if (
        int(torch.linalg.cholesky_ex(cov_a).info.max()) != 0
        or int(torch.linalg.cholesky_ex(cov_b).info.max()) != 0
    ):
        return _failure(count, "invalid_covariance")
    geometry_valid, singular = _geometry_valid(
        points_a, points_b, min_geometry_ratio
    )
    if not geometry_valid:
        return _failure(count, "degenerate_geometry", singular)

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    best_measurement = None
    best_mask = None
    best_count = -1
    best_median = float("inf")
    valid_samples = 0
    covariance_sum = cov_a + cov_b
    for _ in range(int(iterations)):
        sample = torch.randperm(count, generator=generator)[:3]
        sample_valid, _ = _geometry_valid(
            points_a[sample], points_b[sample], min_geometry_ratio
        )
        if not sample_valid:
            continue
        valid_samples += 1
        try:
            measurement = _weighted_kabsch(
                points_a[sample], points_b[sample], covariance_sum[sample]
            )
            distance = _mahalanobis(
                measurement, points_a, points_b, cov_a, cov_b
            )
        except (RuntimeError, ValueError):
            continue
        mask = distance <= mahalanobis_threshold
        inliers = int(mask.sum())
        median = (
            float(distance[mask].median()) if inliers else float("inf")
        )
        if inliers > best_count or (inliers == best_count and median < best_median):
            best_measurement = measurement
            best_mask = mask
            best_count = inliers
            best_median = median
    if valid_samples == 0:
        return _failure(count, "degenerate_geometry", singular)
    if best_measurement is None or best_mask is None or best_count < 3:
        return _failure(count, "no_consensus", singular)

    for _ in range(8):
        measurement = _weighted_kabsch(
            points_a[best_mask],
            points_b[best_mask],
            (cov_a + cov_b)[best_mask],
        )
        distance = _mahalanobis(
            measurement, points_a, points_b, cov_a, cov_b
        )
        updated_mask = distance <= mahalanobis_threshold
        if int(updated_mask.sum()) < 3:
            return _failure(count, "no_consensus", singular)
        if torch.equal(updated_mask, best_mask):
            break
        best_mask = updated_mask

    measurement = _weighted_kabsch(
        points_a[best_mask],
        points_b[best_mask],
        (cov_a + cov_b)[best_mask],
    )
    distance = _mahalanobis(
        measurement, points_a, points_b, cov_a, cov_b
    )

    final_valid, singular = _geometry_valid(
        points_a[best_mask], points_b[best_mask], min_geometry_ratio
    )
    if not final_valid:
        return _failure(count, "degenerate_geometry", singular)
    inlier_distance = distance[best_mask]
    inliers = int(best_mask.sum())
    return SE3RansacResult(
        measurement=measurement,
        mask=best_mask,
        inliers=inliers,
        ratio=inliers / count,
        median_mahalanobis=float(inlier_distance.median()),
        p90_mahalanobis=float(torch.quantile(inlier_distance, 0.9)),
        singular_values=singular,
        reason=None,
    )


def _reprojection_errors(measurement, points_a, points_b, uv_a, uv_b, K_a, K_b):
    transform = pp.SE3(measurement)
    predicted_a = point2pixel_NED(transform.Act(points_b), K_a.double())
    predicted_b = point2pixel_NED(transform.Inv().Act(points_a), K_b.double())
    return (predicted_a - uv_a).norm(dim=-1), (predicted_b - uv_b).norm(dim=-1)


@torch.inference_mode()
def validate_sparse_loop(
    *,
    source,
    target,
    matches,
    source_stereo,
    target_stereo,
    source_depth,
    target_depth,
    covariance_model,
    config,
    seed,
):
    matches = torch.tensor(matches, dtype=torch.float64)
    if matches.ndim != 2 or matches.shape[1] != 5:
        raise ValueError("ORB sparse matches must have shape Nx5")
    count = len(matches)
    metrics = {"mutual_matches": count}
    if count < config.min_mutual_matches:
        return _loop_failure(
            source, target, count, "insufficient_mutual_matches", metrics
        )
    uv_a = matches[:, :2]
    uv_b = matches[:, 2:4]
    inbound = (
        torch.isfinite(uv_a).all(-1)
        & torch.isfinite(uv_b).all(-1)
        & (uv_a[:, 0] >= 0)
        & (uv_a[:, 0] <= source_stereo.width - 1)
        & (uv_a[:, 1] >= 0)
        & (uv_a[:, 1] <= source_stereo.height - 1)
        & (uv_b[:, 0] >= 0)
        & (uv_b[:, 0] <= target_stereo.width - 1)
        & (uv_b[:, 1] >= 0)
        & (uv_b[:, 1] <= target_stereo.height - 1)
    )
    uv_a, uv_b = uv_a[inbound], uv_b[inbound]
    metrics["inbound_matches"] = int(inbound.sum())
    grid_cells = occupied_grid_cells(
        uv_a, source_stereo.width, source_stereo.height,
        config.grid_rows, config.grid_cols,
    )
    metrics["initial_grid_cells"] = grid_cells
    if grid_cells < config.min_grid_cells:
        return _loop_failure(
            source, target, count, "insufficient_grid_coverage", metrics
        )
    if source_depth.cov is None or target_depth.cov is None:
        return _loop_failure(
            source, target, count, "missing_depth_covariance", metrics
        )

    depth_a = _sample_bilinear(uv_a, source_depth.depth).squeeze(-1).cpu().double()
    depth_b = _sample_bilinear(uv_b, target_depth.depth).squeeze(-1).cpu().double()
    depth_cov_a = _sample_bilinear(
        uv_a, source_depth.cov
    ).squeeze(-1).cpu().double()
    depth_cov_b = _sample_bilinear(
        uv_b, target_depth.cov
    ).squeeze(-1).cpu().double()
    valid_depth = (
        torch.isfinite(depth_a)
        & torch.isfinite(depth_b)
        & torch.isfinite(depth_cov_a)
        & torch.isfinite(depth_cov_b)
        & (depth_a > 0)
        & (depth_b > 0)
        & (depth_cov_a > 0)
        & (depth_cov_b > 0)
    )
    uv_a, uv_b = uv_a[valid_depth], uv_b[valid_depth]
    depth_a, depth_b = depth_a[valid_depth], depth_b[valid_depth]
    depth_cov_a = depth_cov_a[valid_depth]
    depth_cov_b = depth_cov_b[valid_depth]
    metrics["valid_depth_pairs"] = int(valid_depth.sum())
    if len(uv_a) < config.min_valid_3d:
        return _loop_failure(
            source, target, count, "insufficient_valid_depth", metrics
        )

    device = source_depth.depth.device
    uv_a_device = uv_a.to(device=device, dtype=source_depth.depth.dtype)
    uv_b_device = uv_b.to(device=target_depth.depth.device, dtype=target_depth.depth.dtype)
    pixel_cov_a = torch.zeros(len(uv_a), 3, device=device)
    pixel_cov_b = torch.zeros(len(uv_b), 3, device=target_depth.depth.device)
    pixel_cov_a[:, :2] = config.orb_pixel_variance
    pixel_cov_b[:, :2] = config.orb_pixel_variance
    cov_a = covariance_model.estimate(
        source_stereo,
        uv_a_device,
        source_depth,
        depth_cov_a.to(device),
        pixel_cov_a,
    ).detach().cpu().double()
    cov_b = covariance_model.estimate(
        target_stereo,
        uv_b_device,
        target_depth,
        depth_cov_b.to(target_depth.depth.device),
        pixel_cov_b,
    ).detach().cpu().double()
    points_a = pixel2point_NED(
        uv_a, depth_a.cpu(), source_stereo.frame_K.cpu()
    ).double()
    points_b = pixel2point_NED(
        uv_b, depth_b.cpu(), target_stereo.frame_K.cpu()
    ).double()
    valid_3d = (
        torch.isfinite(points_a).all(-1)
        & torch.isfinite(points_b).all(-1)
        & torch.isfinite(cov_a).all(dim=(-2, -1))
        & torch.isfinite(cov_b).all(dim=(-2, -1))
    )
    ids = torch.nonzero(valid_3d).flatten()
    if len(ids):
        positive = (
            torch.linalg.cholesky_ex(cov_a[ids]).info == 0
        ) & (torch.linalg.cholesky_ex(cov_b[ids]).info == 0)
        ids = ids[positive]
    metrics["valid_3d_pairs"] = len(ids)
    if len(ids) < config.min_valid_3d:
        return _loop_failure(
            source, target, count, "insufficient_valid_covariance", metrics
        )
    uv_a, uv_b = uv_a[ids], uv_b[ids]
    points_a, points_b = points_a[ids], points_b[ids]
    cov_a, cov_b = cov_a[ids], cov_b[ids]

    ransac = estimate_se3_ransac(
        points_a,
        points_b,
        cov_a,
        cov_b,
        iterations=config.ransac_iterations,
        mahalanobis_threshold=config.mahalanobis_threshold,
        min_geometry_ratio=config.min_geometry_ratio,
        seed=seed,
    )
    metrics.update({
        "ransac_inliers": ransac.inliers,
        "ransac_ratio": ransac.ratio,
        "mahalanobis_median": ransac.median_mahalanobis,
        "mahalanobis_p90": ransac.p90_mahalanobis,
        "geometry_singular_values": (
            None if ransac.singular_values is None
            else ransac.singular_values.tolist()
        ),
    })
    if ransac.measurement is None:
        return _loop_failure(source, target, count, ransac.reason, metrics)
    if (
        ransac.inliers < config.min_ransac_inliers
        or ransac.ratio < config.min_ransac_ratio
    ):
        return _loop_failure(
            source, target, count, "insufficient_ransac_inliers", metrics
        )

    inlier_uv_a, inlier_uv_b = uv_a[ransac.mask], uv_b[ransac.mask]
    error_a, error_b = _reprojection_errors(
        ransac.measurement,
        points_a[ransac.mask],
        points_b[ransac.mask],
        inlier_uv_a,
        inlier_uv_b,
        source_stereo.frame_K.cpu(),
        target_stereo.frame_K.cpu(),
    )
    metrics.update({
        "reprojection_forward_median_px": float(error_a.median()),
        "reprojection_backward_median_px": float(error_b.median()),
        "reprojection_forward_p90_px": float(torch.quantile(error_a, 0.9)),
        "reprojection_backward_p90_px": float(torch.quantile(error_b, 0.9)),
    })
    if (
        max(
            metrics["reprojection_forward_median_px"],
            metrics["reprojection_backward_median_px"],
        ) > config.max_median_reprojection_px
        or max(
            metrics["reprojection_forward_p90_px"],
            metrics["reprojection_backward_p90_px"],
        ) > config.max_p90_reprojection_px
    ):
        return _loop_failure(
            source, target, count, "excessive_reprojection_error", metrics
        )
    final_grid = occupied_grid_cells(
        inlier_uv_a,
        source_stereo.width,
        source_stereo.height,
        config.grid_rows,
        config.grid_cols,
    )
    metrics["final_grid_cells"] = final_grid
    if final_grid < config.min_grid_cells:
        return _loop_failure(
            source, target, count, "insufficient_final_grid_coverage", metrics
        )
    return SparseLoopResult(
        source,
        target,
        ransac.measurement,
        ransac.mask,
        metrics,
        None,
    )


def conservative_sparse_factor(
    source,
    target,
    measurement,
    inliers,
    inlier_ratio,
    *,
    translation_sigma_m,
    rotation_sigma_deg,
    information_scale=1.0,
):
    if translation_sigma_m <= 0 or rotation_sigma_deg <= 0:
        raise ValueError("sparse factor sigmas must be positive")
    if not math.isfinite(information_scale) or information_scale <= 0:
        raise ValueError("sparse factor information scale must be positive")
    rotation_sigma = math.radians(rotation_sigma_deg)
    information = information_scale * torch.diag(torch.tensor(
        [1 / translation_sigma_m**2] * 3 + [1 / rotation_sigma**2] * 3,
        dtype=torch.float64,
    ))
    return PoseGraphFactor(
        int(source),
        int(target),
        torch.as_tensor(measurement).double(),
        information,
        "loop",
        max(0.0, min(1.0, float(inlier_ratio))),
        int(inliers),
    )
