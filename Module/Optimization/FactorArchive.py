"""Validated, reusable serialization for global ICP pose-graph factors."""

from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import uuid

import numpy as np
import pypose as pp
import torch

from Module.Optimization.WindowICP import Edge


SCHEMA_VERSION = 1
KNOWN_EDGE_KINDS = frozenset({
    "adjacent", "skip2", "gap5", "gap10", "proximity"
})
_ARCHIVE_FIELDS = frozenset({
    "schema_version",
    "edge_a",
    "edge_b",
    "edge_offsets",
    "points_a",
    "points_b",
    "cov_a",
    "cov_b",
    "initial_sensor_poses",
    "time_ns",
    "T_BS",
    "edge_kind",
    "metadata_json",
})


def _float64_numpy(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.ascontiguousarray(value, dtype=np.float64)


def trajectory_identity(poses, time_ns, T_BS):
    digest = hashlib.sha256()
    digest.update(_float64_numpy(poses).tobytes())
    digest.update(np.ascontiguousarray(time_ns, dtype=np.int64).tobytes())
    digest.update(_float64_numpy(T_BS).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class FactorArchive:
    initial_sensor_poses: torch.Tensor
    time_ns: np.ndarray
    T_BS: torch.Tensor
    edges: tuple[Edge, ...]
    edge_kinds: tuple[str, ...]
    metadata: dict

    def __post_init__(self):
        poses = torch.as_tensor(self.initial_sensor_poses).detach().cpu().double().clone()
        extrinsic = torch.as_tensor(self.T_BS).detach().cpu().double().clone()
        times = np.ascontiguousarray(self.time_ns, dtype=np.int64).copy()
        edges = tuple(self.edges)
        edge_kinds = tuple(str(kind) for kind in self.edge_kinds)
        metadata = dict(self.metadata)

        if poses.ndim != 2 or poses.shape[1] != 7 or not torch.isfinite(poses).all():
            raise ValueError("initial_sensor_poses must be finite Nx7")
        pose_norms = poses[:, 3:].norm(dim=-1)
        if not torch.allclose(
            pose_norms,
            torch.ones(len(poses), dtype=torch.float64),
            atol=1e-4,
        ):
            raise ValueError(
                "initial_sensor_poses must contain unit quaternions"
            )
        if times.shape != (len(poses),):
            raise ValueError("time_ns must match the pose count")
        if extrinsic.shape != (7,) or not torch.isfinite(extrinsic).all():
            raise ValueError("T_BS must be a finite SE3 vector")
        if not torch.allclose(
            extrinsic[3:].norm(),
            torch.tensor(1.0, dtype=torch.float64),
            atol=1e-4,
        ):
            raise ValueError("T_BS must contain a unit quaternion")
        if len(edges) != len(edge_kinds):
            raise ValueError("edge_kinds must match edges")
        if any(kind not in KNOWN_EDGE_KINDS for kind in edge_kinds):
            raise ValueError("unknown edge kind")
        keys = [(edge.a, edge.b) for edge in edges]
        if len(keys) != len(set(keys)):
            raise ValueError("factor archive contains duplicate edge keys")
        if any(edge.b >= len(poses) for edge in edges):
            raise ValueError("factor endpoint exceeds pose count")

        source_id = trajectory_identity(poses, times, extrinsic)
        if metadata.get("source_id", source_id) != source_id:
            raise ValueError("factor archive source identity does not match poses")
        metadata["source_id"] = source_id

        object.__setattr__(self, "initial_sensor_poses", poses)
        object.__setattr__(self, "time_ns", times)
        object.__setattr__(self, "T_BS", extrinsic)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "edge_kinds", edge_kinds)
        object.__setattr__(self, "metadata", metadata)


def sensor_to_body_trajectory(poses, time_ns, T_BS):
    sensor = pp.SE3(torch.as_tensor(poses).detach().cpu().double())
    extrinsic = pp.SE3(torch.as_tensor(T_BS).detach().cpu().double())
    body = (extrinsic @ sensor @ extrinsic.Inv()).tensor().cpu().numpy()
    return np.concatenate(
        [np.asarray(time_ns, dtype=np.int64)[:, None], body], axis=1
    )


def _stack_observations(edges, name, shape):
    if not edges:
        return np.empty((0,) + shape, dtype=np.float64)
    return np.concatenate([
        getattr(edge, name).numpy() for edge in edges
    ], axis=0)


def save_factor_archive(path, archive):
    if not isinstance(archive, FactorArchive):
        raise TypeError("archive must be a FactorArchive")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lengths = np.asarray(
        [len(edge.points_a) for edge in archive.edges], dtype=np.int64
    )
    offsets = np.concatenate([
        np.zeros(1, dtype=np.int64), np.cumsum(lengths, dtype=np.int64)
    ])
    payload = {
        "schema_version": np.array(SCHEMA_VERSION, dtype=np.int64),
        "edge_a": np.asarray([edge.a for edge in archive.edges], dtype=np.int64),
        "edge_b": np.asarray([edge.b for edge in archive.edges], dtype=np.int64),
        "edge_offsets": offsets,
        "points_a": _stack_observations(archive.edges, "points_a", (3,)),
        "points_b": _stack_observations(archive.edges, "points_b", (3,)),
        "cov_a": _stack_observations(archive.edges, "cov_a", (3, 3)),
        "cov_b": _stack_observations(archive.edges, "cov_b", (3, 3)),
        "initial_sensor_poses": archive.initial_sensor_poses.numpy(),
        "time_ns": archive.time_ns,
        "T_BS": archive.T_BS.numpy(),
        "edge_kind": np.asarray(archive.edge_kinds, dtype="<U16"),
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


def _validate_serialized_arrays(payload):
    offsets = np.asarray(payload["edge_offsets"], dtype=np.int64)
    edge_a = np.asarray(payload["edge_a"], dtype=np.int64)
    edge_b = np.asarray(payload["edge_b"], dtype=np.int64)
    edge_kind = np.asarray(payload["edge_kind"])
    observation_count = len(payload["points_a"])
    if offsets.ndim != 1 or len(offsets) != len(edge_a) + 1:
        raise ValueError("edge offset count is invalid")
    if len(edge_a) != len(edge_b) or len(edge_a) != len(edge_kind):
        raise ValueError("edge arrays must have matching lengths")
    if offsets[0] != 0 or offsets[-1] != observation_count:
        raise ValueError("edge offsets must span all observations")
    if np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("edge offsets must be nondecreasing")
    for name, shape in (
        ("points_a", (observation_count, 3)),
        ("points_b", (observation_count, 3)),
        ("cov_a", (observation_count, 3, 3)),
        ("cov_b", (observation_count, 3, 3)),
    ):
        if np.asarray(payload[name]).shape != shape:
            raise ValueError(f"{name} has invalid shape")
    if np.any(offsets[1:] == offsets[:-1]):
        raise ValueError("edge offsets must describe nonempty observations")


def load_factor_archive(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as stored:
        if "schema_version" not in stored.files:
            raise ValueError("factor archive is missing schema_version")
        version = int(np.asarray(stored["schema_version"]).item())
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported factor archive schema_version {version}"
            )
        fields = frozenset(stored.files)
        if fields != _ARCHIVE_FIELDS:
            missing = sorted(_ARCHIVE_FIELDS - fields)
            extra = sorted(fields - _ARCHIVE_FIELDS)
            raise ValueError(
                f"factor archive fields differ; missing={missing}, extra={extra}"
            )
        payload = {name: stored[name] for name in stored.files}

    _validate_serialized_arrays(payload)
    offsets = np.asarray(payload["edge_offsets"], dtype=np.int64)
    edges = []
    for index, (a, b) in enumerate(zip(payload["edge_a"], payload["edge_b"])):
        start, stop = int(offsets[index]), int(offsets[index + 1])
        edges.append(Edge(
            int(a),
            int(b),
            torch.from_numpy(np.asarray(payload["points_a"][start:stop]).copy()),
            torch.from_numpy(np.asarray(payload["points_b"][start:stop]).copy()),
            torch.from_numpy(np.asarray(payload["cov_a"][start:stop]).copy()),
            torch.from_numpy(np.asarray(payload["cov_b"][start:stop]).copy()),
        ))
    try:
        metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("metadata_json is invalid") from error
    if not isinstance(metadata, dict):
        raise ValueError("metadata_json must contain an object")
    return FactorArchive(
        initial_sensor_poses=torch.from_numpy(
            np.asarray(payload["initial_sensor_poses"]).copy()
        ),
        time_ns=np.asarray(payload["time_ns"]).copy(),
        T_BS=torch.from_numpy(np.asarray(payload["T_BS"]).copy()),
        edges=tuple(edges),
        edge_kinds=tuple(str(kind) for kind in payload["edge_kind"].tolist()),
        metadata=metadata,
    )


def filter_factor_archive(archive, kinds):
    kinds = frozenset(kinds)
    unknown = kinds - KNOWN_EDGE_KINDS
    if unknown:
        raise ValueError(f"unknown edge kinds requested: {sorted(unknown)}")
    selected = [
        (edge, kind)
        for edge, kind in zip(archive.edges, archive.edge_kinds)
        if kind in kinds
    ]
    return replace(
        archive,
        edges=tuple(edge for edge, _ in selected),
        edge_kinds=tuple(kind for _, kind in selected),
    )


def merge_factor_archives(left, right):
    if left.metadata.get("source_id") != right.metadata.get("source_id"):
        raise ValueError("factor archives have different source identities")
    if not torch.equal(
        left.initial_sensor_poses, right.initial_sensor_poses
    ) or not np.array_equal(left.time_ns, right.time_ns) or not torch.equal(
        left.T_BS, right.T_BS
    ):
        raise ValueError("factor archives do not share the same source state")
    left_keys = {(edge.a, edge.b) for edge in left.edges}
    right_keys = {(edge.a, edge.b) for edge in right.edges}
    duplicate = sorted(left_keys & right_keys)
    if duplicate:
        raise ValueError(f"factor archives contain duplicate edges: {duplicate}")
    return replace(
        left,
        edges=left.edges + right.edges,
        edge_kinds=left.edge_kinds + right.edge_kinds,
    )
