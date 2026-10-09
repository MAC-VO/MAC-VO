"""Block-sparse global pose-only ICP using retained WindowICP factors."""
from dataclasses import dataclass
import math
import time
import warnings

import numpy as np
import pypose as pp
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import MatrixRankWarning, spsolve
import torch

from Module.Optimization.WindowICP import (
    Edge,
    edge_gap_counts,
    huber_cost,
    normalized_pose,
    skew,
)
from Utility.Timer import Timer


@dataclass
class GlobalPoseResult:
    poses: torch.Tensor
    diagnostics: dict


def _edge_linearization(pose, edge):
    qa = pose[edge.a].Act(edge.points_a)
    qb = pose[edge.b].Act(edge.points_b)
    Ra = pose[edge.a].rotation().matrix()
    Rb = pose[edge.b].rotation().matrix()
    covariance = Ra @ edge.cov_a @ Ra.T + Rb @ edge.cov_b @ Rb.T
    covariance = (covariance + covariance.transpose(-1, -2)) * 0.5
    residual = qa - qb

    L = torch.linalg.cholesky(covariance)
    rw = torch.linalg.solve_triangular(
        L, residual[..., None], upper=False
    ).squeeze(-1)

    eye = torch.eye(3, dtype=qa.dtype, device=qa.device)
    Ja = torch.cat((eye.expand(len(qa), 3, 3), -skew(qa)), dim=-1)
    Jb = torch.cat((-eye.expand(len(qb), 3, 3), skew(qb)), dim=-1)
    Jaw = torch.linalg.solve_triangular(L, Ja, upper=False)
    Jbw = torch.linalg.solve_triangular(L, Jb, upper=False)
    return rw, Jaw, Jbw


def global_cost(poses, edges, huber_delta):
    pose = pp.SE3(poses)
    return sum(
        float(huber_cost(_edge_linearization(pose, edge)[0], huber_delta))
        for edge in edges
    )


def accumulate_blocks(poses, edges, huber_delta):
    pose = pp.SE3(poses)
    blocks = {}
    gradient = torch.zeros((len(poses) - 1, 6), dtype=torch.float64)
    for edge in edges:
        rw, Ja, Jb = _edge_linearization(pose, edge)
        weight = torch.clamp(
            huber_delta / rw.norm(dim=-1).clamp_min(1e-15), max=1.0
        ).sqrt()
        Aa = (Ja * weight[:, None, None]).reshape(-1, 6)
        Ab = (Jb * weight[:, None, None]).reshape(-1, 6)
        error = (rw * weight[:, None]).reshape(-1)

        local_blocks = {
            (edge.a, edge.a): Aa.T @ Aa,
            (edge.a, edge.b): Aa.T @ Ab,
            (edge.b, edge.a): Ab.T @ Aa,
            (edge.b, edge.b): Ab.T @ Ab,
        }
        for (a, b), value in local_blocks.items():
            if a == 0 or b == 0:
                continue
            key = (a - 1, b - 1)
            if key in blocks:
                blocks[key] += value
            else:
                blocks[key] = value.clone()

        if edge.a:
            gradient[edge.a - 1] += Aa.T @ error
        if edge.b:
            gradient[edge.b - 1] += Ab.T @ error
    return blocks, gradient


def build_sparse_system(blocks, pose_variables):
    rows, cols, values = [], [], []
    for (block_i, block_j), block in sorted(blocks.items()):
        base_i, base_j = 6 * block_i, 6 * block_j
        for i in range(6):
            for j in range(6):
                rows.append(base_i + i)
                cols.append(base_j + j)
                values.append(float(block[i, j]))
    size = pose_variables * 6
    return coo_matrix((values, (rows, cols)), shape=(size, size)).tocsc()


def _validate_inputs(poses, edges, max_iters, huber_delta, damping):
    if poses.ndim != 2 or poses.shape[1] != 7 or len(poses) < 2:
        raise ValueError("Global poses must be a finite Nx7 tensor with N >= 2")
    if not torch.isfinite(poses).all():
        raise ValueError("Global poses must be finite")
    if not torch.allclose(
        poses[:, 3:].norm(dim=-1),
        torch.ones(len(poses), dtype=poses.dtype),
        atol=1e-4,
    ):
        raise ValueError("Global poses must have unit quaternions")
    if not edges:
        raise ValueError("Global pose graph requires at least one edge")
    if max_iters < 1 or not math.isfinite(huber_delta) or huber_delta <= 0:
        raise ValueError("Invalid global solver settings")
    if not math.isfinite(damping) or damping <= 0:
        raise ValueError("Invalid global damping")

    connected = {0}
    for edge in edges:
        if edge.a < 0 or edge.b >= len(poses) or edge.a >= edge.b:
            raise ValueError(
                f"Global edge ({edge.a}, {edge.b}) is outside the pose graph"
            )
    for _ in range(len(poses)):
        for edge in edges:
            if edge.a in connected or edge.b in connected:
                connected.update((edge.a, edge.b))
    if connected != set(range(len(poses))):
        missing = sorted(set(range(len(poses))) - connected)
        raise ValueError(f"Global pose graph must be connected; missing poses {missing}")


@Timer.cpu_timeit("GlobalPoseICP")
@torch.no_grad()
def optimize_global_pose_graph(
    poses,
    edges,
    max_iters=5,
    huber_delta=3.0,
    damping=1e-3,
):
    start = time.perf_counter()
    original = poses.detach().to(device="cpu", dtype=torch.float64).clone()
    edges = list(edges)
    base_diagnostics = {
        "poses": len(original),
        "edges": len(edges),
        "edge_gaps": edge_gap_counts(edges),
        "observations": sum(len(edge.points_a) for edge in edges),
    }
    try:
        _validate_inputs(original, edges, max_iters, huber_delta, damping)
        original_anchor = original[0].clone()
        anchor = pp.SE3(normalized_pose(original[0]))
        local = normalized_pose((anchor.Inv() @ pp.SE3(original)).tensor())
        local[0] = pp.identity_SE3(dtype=torch.float64).tensor()

        initial_cost = cost = global_cost(local, edges, huber_delta)
        accepted = rejected = iterations = 0
        for iteration in range(max_iters):
            iterations = iteration + 1
            blocks, gradient = accumulate_blocks(local, edges, huber_delta)
            if not torch.isfinite(gradient).all():
                raise ValueError("Global gradient is non-finite")
            if gradient.abs().max() < 1e-10:
                break

            system = build_sparse_system(blocks, len(local) - 1)
            if not np.isfinite(system.data).all():
                raise ValueError("Global sparse Hessian is non-finite")
            diagonal = np.maximum(system.diagonal(), 1e-9)
            improved = False
            for _ in range(8):
                damped = system + diags(damping * diagonal)
                with warnings.catch_warnings():
                    warnings.simplefilter("error", MatrixRankWarning)
                    step = spsolve(
                        damped, -gradient.reshape(-1).numpy()
                    )
                step = np.asarray(step, dtype=np.float64)
                if not np.isfinite(step).all():
                    damping *= 10
                    rejected += 1
                    continue

                delta = torch.zeros(len(local), 6, dtype=torch.float64)
                delta[1:] = torch.from_numpy(step.reshape(-1, 6))
                trial = normalized_pose(
                    (pp.se3(delta).Exp() @ pp.SE3(local)).tensor()
                )
                trial[0] = local[0]
                trial_cost = global_cost(trial, edges, huber_delta)
                if math.isfinite(trial_cost) and trial_cost < cost:
                    decrease = cost - trial_cost
                    local, cost = trial, trial_cost
                    damping = max(damping * 0.3, 1e-9)
                    accepted += 1
                    improved = True
                    break
                damping *= 10
                rejected += 1

            if not improved:
                break
            if np.linalg.norm(step) < 1e-8 or decrease < 1e-8 * max(1.0, cost):
                break

        output = normalized_pose((anchor @ pp.SE3(local)).tensor())
        output[0] = original_anchor
        diagnostics = base_diagnostics | {
            "status": "refined",
            "initial_cost": initial_cost,
            "final_cost": cost,
            "iterations": iterations,
            "accepted_steps": accepted,
            "rejected_steps": rejected,
            "anchor_preserved": bool(torch.equal(output[0], original[0])),
            "seconds": time.perf_counter() - start,
        }
        return GlobalPoseResult(output, diagnostics)
    except Exception as error:
        return GlobalPoseResult(
            original,
            base_diagnostics | {
                "status": "not_refined",
                "reason": str(error),
                "initial_cost": None,
                "final_cost": None,
                "iterations": 0,
                "accepted_steps": 0,
                "rejected_steps": 0,
                "anchor_preserved": True,
                "seconds": time.perf_counter() - start,
            },
        )
