"""Small pose-only sliding-window ICP. All optimization runs on CPU float64."""
from dataclasses import dataclass
import math
import time

import pypose as pp
import torch


@dataclass(frozen=True)
class Edge:
    a: int
    b: int
    points_a: torch.Tensor
    points_b: torch.Tensor
    cov_a: torch.Tensor
    cov_b: torch.Tensor

    def __post_init__(self):
        if (
            not isinstance(self.a, int)
            or not isinstance(self.b, int)
            or self.a < 0
            or self.b <= self.a
        ):
            raise ValueError(
                "Edge endpoints must be non-negative and strictly forward"
            )
        n = len(self.points_a)
        for name, shape in (
            ("points_a", (n, 3)), ("points_b", (n, 3)),
            ("cov_a", (n, 3, 3)), ("cov_b", (n, 3, 3)),
        ):
            value = getattr(self, name).detach().to(device="cpu", dtype=torch.float64).clone()
            if n == 0 or tuple(value.shape) != shape or not torch.isfinite(value).all():
                raise ValueError("Edge observations must have valid shapes and finite nonempty values")
            object.__setattr__(self, name, value)
        for cov in (self.cov_a, self.cov_b):
            if not torch.allclose(cov, cov.transpose(-1, -2), atol=1e-9):
                raise ValueError("Covariance must be symmetric")
            if torch.any(torch.linalg.cholesky_ex(cov).info != 0):
                raise ValueError("Covariance must be positive definite")


def edge_tensor_bytes(edge):
    return sum(
        value.numel() * value.element_size()
        for value in (edge.points_a, edge.points_b, edge.cov_a, edge.cov_b)
    )


def edge_gap_counts(edges):
    counts = {}
    for edge in edges:
        key = str(edge.b - edge.a)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


class EdgeWindow:
    def __init__(self, size=5):
        if not isinstance(size, int) or not 2 <= size <= 5:
            raise ValueError("Window size must be between 2 and 5")
        self.size = size
        self._edges = {}

    @property
    def edges(self):
        return [self._edges[key] for key in sorted(self._edges)]

    def add(self, edge):
        self._edges[(edge.a, edge.b)] = edge

    def advance(self, current):
        first = current - self.size + 1
        evicted = [
            edge for _, edge in sorted(self._edges.items())
            if not (first <= edge.a < edge.b <= current)
        ]
        self._edges = {k: e for k, e in self._edges.items() if first <= e.a < e.b <= current}
        return evicted


def skew(v):
    zero = torch.zeros_like(v[..., 0])
    x, y, z = v.unbind(-1)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*v.shape[:-1], 3, 3)


def factor_system(poses, frame_ids, edges):
    """Return residuals, full covariance, and left-SE3 Jacobians (oldest fixed).

    r = T_a p_a - T_b p_b
    J_a = [I, -skew(T_a p_a)], J_b = [-I, skew(T_b p_b)].
    The Jacobian does not differentiate covariance, matching iteratively
    reweighted ICP. Trial steps are checked against the recomputed objective.
    """
    index = {f: i for i, f in enumerate(frame_ids)}
    pose = pp.SE3(poses)
    eye = torch.eye(3, dtype=poses.dtype, device=poses.device)
    residuals, covariances, jacobians = [], [], []
    for edge in edges:
        a, b = index[edge.a], index[edge.b]
        qa, qb = pose[a].Act(edge.points_a), pose[b].Act(edge.points_b)
        Ra, Rb = pose[a].rotation().matrix(), pose[b].rotation().matrix()
        covariance = Ra @ edge.cov_a @ Ra.T + Rb @ edge.cov_b @ Rb.T
        J = torch.zeros(len(qa), 3, 6 * (len(frame_ids) - 1), dtype=poses.dtype, device=poses.device)
        if a:
            J[..., (a-1)*6:a*6] = torch.cat((eye.expand(len(qa), 3, 3), -skew(qa)), -1)
        if b:
            J[..., (b-1)*6:b*6] = torch.cat((-eye.expand(len(qb), 3, 3), skew(qb)), -1)
        residuals.append(qa-qb)
        covariances.append((covariance + covariance.transpose(-1, -2)) * .5)
        jacobians.append(J)
    return torch.cat(residuals), torch.cat(covariances), torch.cat(jacobians)


def whiten(residual, covariance, jacobian):
    L = torch.linalg.cholesky(covariance)
    rw = torch.linalg.solve_triangular(L, residual[..., None], upper=False).squeeze(-1)
    Jw = torch.linalg.solve_triangular(L, jacobian, upper=False)
    return rw, Jw


def huber_cost(r, delta):
    norm = r.norm(dim=-1)
    return torch.where(norm <= delta, norm.square(), 2*delta*norm-delta**2).sum()

def normalized_pose(poses):
    result = poses.clone()
    result[..., 3:] /= result[..., 3:].norm(dim=-1, keepdim=True)
    return result


@dataclass
class WindowResult:
    poses: torch.Tensor
    diagnostics: dict


@torch.no_grad()
def optimize_window(poses, frame_ids, edges, max_iters=10, huber_delta=3.0, damping=1e-3):
    start = time.perf_counter()
    poses = poses.detach().to(device="cpu", dtype=torch.float64).clone()
    frame_ids = list(frame_ids)
    edges = list(edges)
    if len(frame_ids) < 2 or len(frame_ids) > 5 or frame_ids != sorted(set(frame_ids)):
        raise ValueError("Provide 2–5 distinct, ordered frame IDs")
    if poses.shape != (len(frame_ids), 7) or not torch.isfinite(poses).all():
        raise ValueError("Poses must be finite Nx7")
    if not torch.allclose(poses[:, 3:].norm(dim=-1), torch.ones(len(poses), dtype=poses.dtype), atol=1e-4):
        raise ValueError("Poses must have unit quaternions")
    if max_iters < 1 or not math.isfinite(huber_delta) or huber_delta <= 0 or damping <= 0:
        raise ValueError("Invalid solver settings")
    connected = {frame_ids[0]}
    for _ in frame_ids:
        for edge in edges:
            if edge.a not in frame_ids or edge.b not in frame_ids:
                raise ValueError("Edge outside current window")
            if edge.a in connected or edge.b in connected:
                connected.update((edge.a, edge.b))
    if connected != set(frame_ids):
        raise ValueError("Window must be connected to its fixed anchor")

    # Solve near the origin for stable translational/rotational coupling.
    original_anchor = poses[0].clone()
    poses = normalized_pose(poses)
    anchor = pp.SE3(poses[0])
    local = normalized_pose((anchor.Inv() @ pp.SE3(poses)).tensor())
    local[0] = pp.identity_SE3(dtype=torch.float64).tensor()
    r, covariance, J = factor_system(local, frame_ids, edges)
    rw, Jw = whiten(r, covariance, J)
    initial_cost = cost = float(huber_cost(rw, huber_delta))
    accepted, rejected, iterations = 0, 0, 0
    for iteration in range(max_iters):
        iterations = iteration + 1
        weights = torch.clamp(huber_delta / rw.norm(dim=-1).clamp_min(1e-15), max=1.).sqrt()
        A = (Jw * weights[:, None, None]).reshape(-1, Jw.shape[-1])
        e = (rw * weights[:, None]).reshape(-1)
        H, gradient = A.T @ A, A.T @ e
        if gradient.abs().max() < 1e-10:
            break
        diagonal = H.diagonal().clamp_min(1e-9)
        improved = False
        for _ in range(8):
            step, info = torch.linalg.solve_ex(H + damping * torch.diag(diagonal), -gradient)
            if int(info) or not torch.isfinite(step).all():
                damping *= 10
                rejected += 1
                continue
            delta = torch.zeros(len(local), 6, dtype=torch.float64)
            delta[1:] = step.reshape(-1, 6)
            trial = normalized_pose((pp.se3(delta).Exp() @ pp.SE3(local)).tensor())
            trial[0] = local[0]
            tr, tc, tj = factor_system(trial, frame_ids, edges)
            twr, twj = whiten(tr, tc, tj)
            trial_cost = float(huber_cost(twr, huber_delta))
            if math.isfinite(trial_cost) and trial_cost < cost:
                decrease = cost - trial_cost
                local, rw, Jw, cost = trial, twr, twj, trial_cost
                damping = max(damping * .3, 1e-9)
                accepted += 1
                improved = True
                break
            damping *= 10
            rejected += 1
        if not improved or step.norm() < 1e-8 or decrease < 1e-8 * max(1., cost):
            break

    output = normalized_pose((anchor @ pp.SE3(local)).tensor())
    output[0] = original_anchor  # exact gauge preservation
    return WindowResult(output, {
        "frames": frame_ids, "anchor": frame_ids[0], "active_poses": len(frame_ids),
        "edges": len(edges), "adjacent_edges": sum(e.b-e.a == 1 for e in edges),
        "skip_edges": sum(e.b-e.a == 2 for e in edges),
        "edge_gaps": edge_gap_counts(edges),
        "observations": sum(len(e.points_a) for e in edges),
        "initial_cost": initial_cost, "final_cost": cost, "iterations": iterations,
        "accepted_steps": accepted, "rejected_steps": rejected,
        "seconds": time.perf_counter()-start,
    })
