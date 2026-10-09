"""Compact relative-pose graph data structures and serialization."""

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import time
import uuid
import warnings

import numpy as np
import pypose as pp
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import MatrixRankWarning, spsolve
import torch

from Module.Optimization.WindowICP import normalized_pose


POSE_FACTOR_KINDS = frozenset({"adjacent", "skip2", "loop"})


@dataclass(frozen=True)
class PoseGraphFactor:
    a: int
    b: int
    measurement: torch.Tensor
    information: torch.Tensor
    kind: str
    confidence: float
    observation_count: int

    def __post_init__(self):
        measurement = normalized_pose(
            torch.as_tensor(self.measurement).detach().cpu().double().clone()
        )
        information = torch.as_tensor(
            self.information
        ).detach().cpu().double().clone()
        if self.a < 0 or self.b <= self.a:
            raise ValueError("pose factor endpoints must be strictly forward")
        if measurement.shape != (7,) or not torch.isfinite(measurement).all():
            raise ValueError("pose factor measurement must be finite SE3")
        if information.shape != (6, 6) or not torch.isfinite(information).all():
            raise ValueError("pose factor information must be finite 6x6")
        if not torch.allclose(information, information.T, atol=1e-9):
            raise ValueError("pose factor information must be symmetric")
        if torch.linalg.cholesky_ex(information).info.item() != 0:
            raise ValueError("pose factor information must be positive definite")
        if self.kind not in POSE_FACTOR_KINDS:
            raise ValueError("unknown pose factor kind")
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("pose factor confidence must be within [0, 1]")
        if self.observation_count < 1:
            raise ValueError("pose factor observation count must be positive")
        object.__setattr__(self, "measurement", measurement)
        object.__setattr__(self, "information", information)


@dataclass(frozen=True)
class PoseGraphArchive:
    poses: torch.Tensor
    time_ns: np.ndarray
    factors: tuple[PoseGraphFactor, ...]
    metadata: dict

    def __post_init__(self):
        poses = normalized_pose(
            torch.as_tensor(self.poses).detach().cpu().double().clone()
        )
        times = np.asarray(self.time_ns, dtype=np.int64).copy()
        factors = tuple(self.factors)
        if poses.ndim != 2 or poses.shape[1] != 7 or not torch.isfinite(poses).all():
            raise ValueError("pose graph poses must be finite Nx7")
        if times.shape != (len(poses),):
            raise ValueError("pose graph timestamps must match poses")
        if any(factor.b >= len(poses) for factor in factors):
            raise ValueError("pose graph factor endpoint exceeds poses")
        keys = [(factor.a, factor.b, factor.kind) for factor in factors]
        if len(keys) != len(set(keys)):
            raise ValueError("pose graph contains duplicate factors")
        object.__setattr__(self, "poses", poses)
        object.__setattr__(self, "time_ns", times)
        object.__setattr__(self, "factors", factors)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class PoseGraphResult:
    poses: torch.Tensor
    switches: list[float]
    diagnostics: dict


def effective_information(factor, skip_information_cap=0.5):
    scale = factor.confidence
    if factor.kind == "skip2":
        scale *= skip_information_cap
    return factor.information * scale


def factor_residuals_from_endpoints(pose_a, pose_b, measurement):
    predicted = pose_a.Inv() @ pose_b
    return (measurement.Inv() @ predicted).Log().tensor()


def pose_graph_factor_system(poses, factors, epsilon=1e-6):
    pose = pp.SE3(poses)
    index_a = torch.tensor([factor.a for factor in factors], dtype=torch.long)
    index_b = torch.tensor([factor.b for factor in factors], dtype=torch.long)
    pose_a = pose[index_a]
    pose_b = pose[index_b]
    measurement = pp.SE3(torch.stack([
        factor.measurement for factor in factors
    ]))
    residual = factor_residuals_from_endpoints(
        pose_a, pose_b, measurement
    )
    jacobian_a = torch.empty(len(factors), 6, 6, dtype=torch.float64)
    jacobian_b = torch.empty_like(jacobian_a)
    for column in range(6):
        delta = torch.zeros(len(factors), 6, dtype=torch.float64)
        delta[:, column] = epsilon
        plus = pp.se3(delta).Exp()
        minus = pp.se3(-delta).Exp()
        jacobian_a[:, :, column] = (
            factor_residuals_from_endpoints(
                plus @ pose_a, pose_b, measurement
            )
            - factor_residuals_from_endpoints(
                minus @ pose_a, pose_b, measurement
            )
        ) / (2 * epsilon)
        jacobian_b[:, :, column] = (
            factor_residuals_from_endpoints(
                pose_a, plus @ pose_b, measurement
            )
            - factor_residuals_from_endpoints(
                pose_a, minus @ pose_b, measurement
            )
        ) / (2 * epsilon)
    return residual, jacobian_a, jacobian_b


def build_pose_graph_sparse(blocks, pose_variables):
    rows, columns, values = [], [], []
    for (block_i, block_j), block in sorted(blocks.items()):
        base_i, base_j = block_i * 6, block_j * 6
        for row in range(6):
            for column in range(6):
                rows.append(base_i + row)
                columns.append(base_j + column)
                values.append(float(block[row, column]))
    size = pose_variables * 6
    return coo_matrix(
        (values, (rows, columns)), shape=(size, size)
    ).tocsc()


def _huber_cost(norm, delta):
    return torch.where(
        norm <= delta,
        norm.square(),
        2 * delta * norm - delta**2,
    )


def _linearize_pose_graph(
    poses,
    factors,
    *,
    huber_delta,
    skip_information_cap,
    switch_prior,
):
    residual, jacobian_a, jacobian_b = pose_graph_factor_system(
        poses, factors
    )
    blocks = {}
    gradient = torch.zeros((len(poses) - 1, 6), dtype=torch.float64)
    switches = []
    total_cost = 0.0
    for index, factor in enumerate(factors):
        information = effective_information(
            factor, skip_information_cap
        )
        L = torch.linalg.cholesky(information)
        rw = L.T @ residual[index]
        Jaw = L.T @ jacobian_a[index]
        Jbw = L.T @ jacobian_b[index]
        chi2 = float(rw.square().sum())
        switch = (
            switch_prior / (switch_prior + chi2)
            if factor.kind == "loop"
            else 1.0
        )
        switches.append(switch)
        robust_norm = rw.norm()
        huber_weight = min(
            1.0,
            math.sqrt(huber_delta / max(float(robust_norm), 1e-15)),
        )
        scale = switch * huber_weight
        Aa = Jaw * scale
        Ab = Jbw * scale
        error = rw * scale
        total_cost += float(switch**2 * _huber_cost(
            robust_norm, huber_delta
        ))
        if factor.kind == "loop":
            total_cost += switch_prior * (1.0 - switch) ** 2
        for a, b, value in (
            (factor.a, factor.a, Aa.T @ Aa),
            (factor.a, factor.b, Aa.T @ Ab),
            (factor.b, factor.a, Ab.T @ Aa),
            (factor.b, factor.b, Ab.T @ Ab),
        ):
            if a == 0 or b == 0:
                continue
            key = (a - 1, b - 1)
            blocks[key] = blocks.get(
                key, torch.zeros(6, 6, dtype=torch.float64)
            ) + value
        if factor.a:
            gradient[factor.a - 1] += Aa.T @ error
        if factor.b:
            gradient[factor.b - 1] += Ab.T @ error
    return blocks, gradient, switches, total_cost


def _pose_graph_cost(
    poses,
    factors,
    *,
    huber_delta,
    skip_information_cap,
    switch_prior,
):
    residual, _, _ = pose_graph_factor_system(poses, factors)
    switches = []
    cost = 0.0
    for index, factor in enumerate(factors):
        information = effective_information(
            factor, skip_information_cap
        )
        rw = torch.linalg.cholesky(information).T @ residual[index]
        chi2 = float(rw.square().sum())
        switch = (
            switch_prior / (switch_prior + chi2)
            if factor.kind == "loop"
            else 1.0
        )
        switches.append(switch)
        cost += float(switch**2 * _huber_cost(rw.norm(), huber_delta))
        if factor.kind == "loop":
            cost += switch_prior * (1.0 - switch) ** 2
    return cost, switches


@torch.no_grad()
def optimize_pose_graph(
    poses,
    factors,
    *,
    max_iters=10,
    huber_delta=3.0,
    skip_information_cap=0.5,
    switch_prior=1.0,
    damping=1e-3,
):
    start = time.perf_counter()
    original = normalized_pose(
        torch.as_tensor(poses).detach().cpu().double().clone()
    )
    factors = tuple(factors)
    base = {
        "poses": len(original),
        "factors": len(factors),
        "loop_factors": sum(factor.kind == "loop" for factor in factors),
    }
    try:
        if len(original) < 2 or not factors:
            raise ValueError("pose graph requires poses and factors")
        connected = {0}
        for _ in range(len(original)):
            for factor in factors:
                if factor.b >= len(original):
                    raise ValueError("pose graph factor endpoint outside poses")
                if factor.a in connected or factor.b in connected:
                    connected.update((factor.a, factor.b))
        if connected != set(range(len(original))):
            raise ValueError("pose graph must be connected to pose zero")
        anchor_pose = pp.SE3(original[0])
        original_anchor = original[0].clone()
        local = normalized_pose(
            (anchor_pose.Inv() @ pp.SE3(original)).tensor()
        )
        local[0] = pp.identity_SE3(dtype=torch.float64).tensor()
        initial_cost, switches = _pose_graph_cost(
            local,
            factors,
            huber_delta=huber_delta,
            skip_information_cap=skip_information_cap,
            switch_prior=switch_prior,
        )
        cost = initial_cost
        accepted = rejected = iterations = 0
        for iteration in range(max_iters):
            iterations = iteration + 1
            blocks, gradient, switches, _ = _linearize_pose_graph(
                local,
                factors,
                huber_delta=huber_delta,
                skip_information_cap=skip_information_cap,
                switch_prior=switch_prior,
            )
            if gradient.abs().max() < 1e-10:
                break
            system = build_pose_graph_sparse(blocks, len(local) - 1)
            diagonal = np.maximum(system.diagonal(), 1e-9)
            improved = False
            for _ in range(8):
                with warnings.catch_warnings():
                    warnings.simplefilter("error", MatrixRankWarning)
                    step = spsolve(
                        system + diags(damping * diagonal),
                        -gradient.reshape(-1).numpy(),
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
                trial_cost, trial_switches = _pose_graph_cost(
                    trial,
                    factors,
                    huber_delta=huber_delta,
                    skip_information_cap=skip_information_cap,
                    switch_prior=switch_prior,
                )
                if math.isfinite(trial_cost) and trial_cost < cost:
                    decrease = cost - trial_cost
                    local = trial
                    cost = trial_cost
                    switches = trial_switches
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
        output = normalized_pose(
            (anchor_pose @ pp.SE3(local)).tensor()
        )
        output[0] = original_anchor
        loop_switches = [
            {"a": factor.a, "b": factor.b, "switch": switch}
            for factor, switch in zip(factors, switches)
            if factor.kind == "loop"
        ]
        return PoseGraphResult(output, switches, base | {
            "status": "refined",
            "initial_cost": initial_cost,
            "final_cost": cost,
            "iterations": iterations,
            "accepted_steps": accepted,
            "rejected_steps": rejected,
            "anchor_preserved": bool(torch.equal(output[0], original[0])),
            "loop_switches": loop_switches,
            "seconds": time.perf_counter() - start,
        })
    except Exception as error:
        return PoseGraphResult(original, [1.0] * len(factors), base | {
            "status": "not_refined",
            "reason": str(error),
            "initial_cost": None,
            "final_cost": None,
            "iterations": 0,
            "accepted_steps": 0,
            "rejected_steps": 0,
            "anchor_preserved": True,
            "seconds": time.perf_counter() - start,
        })


def save_pose_graph_archive(path, archive):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    factors = archive.factors
    payload = {
        "poses": archive.poses.numpy(),
        "time_ns": archive.time_ns,
        "a": np.asarray([factor.a for factor in factors], dtype=np.int64),
        "b": np.asarray([factor.b for factor in factors], dtype=np.int64),
        "measurement": np.asarray(
            [factor.measurement.numpy() for factor in factors], dtype=np.float64
        ).reshape(-1, 7),
        "information": np.asarray(
            [factor.information.numpy() for factor in factors], dtype=np.float64
        ).reshape(-1, 6, 6),
        "kind": np.asarray([factor.kind for factor in factors], dtype="<U16"),
        "confidence": np.asarray(
            [factor.confidence for factor in factors], dtype=np.float64
        ),
        "observation_count": np.asarray(
            [factor.observation_count for factor in factors], dtype=np.int64
        ),
        "metadata_json": np.array(
            json.dumps(archive.metadata, sort_keys=True, allow_nan=False)
        ),
    }
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_pose_graph_archive(path):
    with np.load(path, allow_pickle=False) as data:
        payload = {key: data[key] for key in data.files}
    count = len(payload["a"])
    if any(len(payload[key]) != count for key in (
        "b", "measurement", "information", "kind",
        "confidence", "observation_count",
    )):
        raise ValueError("pose graph factor arrays have different lengths")
    factors = tuple(
        PoseGraphFactor(
            int(payload["a"][index]),
            int(payload["b"][index]),
            torch.from_numpy(payload["measurement"][index].copy()),
            torch.from_numpy(payload["information"][index].copy()),
            str(payload["kind"][index]),
            float(payload["confidence"][index]),
            int(payload["observation_count"][index]),
        )
        for index in range(count)
    )
    metadata = json.loads(str(payload["metadata_json"].item()))
    return PoseGraphArchive(
        torch.from_numpy(payload["poses"].copy()),
        payload["time_ns"].copy(),
        factors,
        metadata,
    )
