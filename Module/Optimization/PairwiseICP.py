"""Compress covariance-aware 3D correspondences into relative SE3 factors."""

from dataclasses import dataclass
import math

import pypose as pp
import torch

from Module.Optimization.PoseGraph import PoseGraphFactor
from Module.Optimization.WindowICP import (
    Edge,
    factor_system,
    optimize_window,
    whiten,
)


@dataclass(frozen=True)
class PairwiseCompressionResult:
    factor: PoseGraphFactor | None
    status: str
    reason: str | None
    initial_cost: float | None
    final_cost: float | None
    condition_number: float | None


def weighted_kabsch(points_a, points_b, covariance):
    points_a = torch.as_tensor(points_a).double()
    points_b = torch.as_tensor(points_b).double()
    covariance = torch.as_tensor(covariance).double()
    weights = covariance.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(1e-12).reciprocal()
    weights = weights / weights.sum()
    center_a = (weights[:, None] * points_a).sum(0)
    center_b = (weights[:, None] * points_b).sum(0)
    a = points_a - center_a
    b = points_b - center_b
    cross = (weights[:, None] * b).T @ a
    U, singular, Vh = torch.linalg.svd(cross)
    if singular[-2] < 1e-10 * singular[0].clamp_min(1e-12):
        raise ValueError("pairwise ICP geometry is degenerate")
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


def _information_at_solution(
    remapped,
    poses,
    huber_delta,
    min_eigenvalue,
    max_eigenvalue,
    max_condition,
):
    residual, covariance, jacobian = factor_system(
        poses, [0, 1], [remapped]
    )
    white_residual, white_jacobian = whiten(
        residual, covariance, jacobian
    )
    weight = torch.clamp(
        huber_delta / white_residual.norm(dim=-1).clamp_min(1e-15),
        max=1.0,
    ).sqrt()
    A = (white_jacobian * weight[:, None, None]).reshape(-1, 6)
    information = A.T @ A
    information = 0.5 * (information + information.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(information)
    raw_min = float(eigenvalues.min())
    raw_max = float(eigenvalues.max())
    if raw_min <= 0 or raw_max / raw_min > max_condition:
        raise ValueError("pairwise ICP information is degenerate")
    clamped = eigenvalues.clamp(min_eigenvalue, max_eigenvalue)
    information = eigenvectors @ torch.diag(clamped) @ eigenvectors.T
    return information, float(clamped.max() / clamped.min())


def linearize_edge_to_pose_factor(
    edge,
    poses,
    *,
    kind,
    huber_delta=3.0,
    min_eigenvalue=1e-6,
    max_eigenvalue=1e6,
    max_condition=1e8,
):
    try:
        pose = pp.SE3(torch.as_tensor(poses).double())
        measurement = (pose[edge.a].Inv() @ pose[edge.b]).tensor()
        remapped = Edge(
            0, 1,
            edge.points_a, edge.points_b,
            edge.cov_a, edge.cov_b,
        )
        local = torch.stack([
            pp.identity_SE3(dtype=torch.float64).tensor(), measurement
        ])
        information, condition = _information_at_solution(
            remapped,
            local,
            huber_delta,
            min_eigenvalue,
            max_eigenvalue,
            max_condition,
        )
        factor = PoseGraphFactor(
            edge.a,
            edge.b,
            measurement,
            information,
            kind,
            1.0,
            len(edge.points_a),
        )
        return PairwiseCompressionResult(
            factor, "compressed", None, 0.0, 0.0, condition
        )
    except Exception as error:
        return PairwiseCompressionResult(
            None, "rejected", str(error), None, None, None
        )


def compress_edge_to_pose_factor(
    edge,
    *,
    kind,
    max_iters=10,
    huber_delta=3.0,
    min_eigenvalue=1e-6,
    max_eigenvalue=1e6,
    max_condition=1e8,
):
    try:
        initial = weighted_kabsch(
            edge.points_a, edge.points_b, edge.cov_a + edge.cov_b
        )
        remapped = Edge(
            0, 1,
            edge.points_a, edge.points_b,
            edge.cov_a, edge.cov_b,
        )
        poses = torch.stack([
            pp.identity_SE3(dtype=torch.float64).tensor(), initial
        ])
        result = optimize_window(
            poses, [0, 1], [remapped],
            max_iters=max_iters, huber_delta=huber_delta,
        )
        information, condition = _information_at_solution(
            remapped,
            result.poses,
            huber_delta,
            min_eigenvalue,
            max_eigenvalue,
            max_condition,
        )
        factor = PoseGraphFactor(
            a=edge.a,
            b=edge.b,
            measurement=result.poses[1],
            information=information,
            kind=kind,
            confidence=1.0,
            observation_count=len(edge.points_a),
        )
        return PairwiseCompressionResult(
            factor=factor,
            status="compressed",
            reason=None,
            initial_cost=float(result.diagnostics["initial_cost"]),
            final_cost=float(result.diagnostics["final_cost"]),
            condition_number=condition,
        )
    except Exception as error:
        return PairwiseCompressionResult(
            factor=None,
            status="rejected",
            reason=str(error),
            initial_cost=None,
            final_cost=None,
            condition_number=None,
        )
