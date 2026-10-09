"""Atomic depth and sparse-proxy cache for covisibility keyframes."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import uuid

import numpy as np
import torch

from Module.Frontend.StereoDepth import IStereoDepth
from Module.Optimization.FactorArchive import trajectory_identity


_CACHE_FILES = frozenset({
    "metadata.json",
    "frame_ids.npy",
    "time_ns.npy",
    "depth.npy",
    "depth_cov.npy",
    "proxy_uv.npy",
    "proxy_depth.npy",
    "proxy_depth_cov.npy",
})


@dataclass(frozen=True)
class CovisibilityDepthCache:
    root: Path
    frame_ids: np.ndarray
    time_ns: np.ndarray
    depth: np.ndarray
    depth_cov: np.ndarray
    proxy_uv: np.ndarray
    proxy_depth: np.ndarray
    proxy_depth_cov: np.ndarray
    metadata: dict

    def depth_output(self, frame_id):
        matches = np.flatnonzero(self.frame_ids == frame_id)
        if len(matches) != 1:
            raise KeyError(f"Frame {frame_id} is not present in covisibility cache")
        index = int(matches[0])
        return IStereoDepth.Output(
            depth=torch.from_numpy(
                np.asarray(self.depth[index], dtype=np.float32).copy()
            )[None, None],
            cov=torch.from_numpy(
                np.asarray(self.depth_cov[index], dtype=np.float32).copy()
            )[None, None],
        )


def covisibility_keyframes(frame_count, stride=5):
    if not isinstance(frame_count, int) or frame_count < 1:
        raise ValueError("frame_count must be a positive integer")
    if not isinstance(stride, int) or stride < 1:
        raise ValueError("stride must be a positive integer")
    return list(range(0, frame_count, stride))


def deterministic_proxy_uv(width, height, count):
    if width < 1 or height < 1 or count < 1:
        raise ValueError("width, height, and proxy count must be positive")
    columns = max(1, round((count * width / height) ** 0.5))
    rows = max(1, int(np.ceil(count / columns)))
    u = np.linspace(0, width - 1, columns, dtype=np.float32)
    v = np.linspace(0, height - 1, rows, dtype=np.float32)
    grid = np.stack(np.meshgrid(u, v), axis=-1).reshape(-1, 2)
    return grid[:count]


def _source_id(source_archive):
    return trajectory_identity(
        source_archive.initial_sensor_poses,
        source_archive.time_ns,
        source_archive.T_BS,
    )


def _validate_frame(frame, frame_id, source_archive):
    actual = int(frame.stereo.frame_ns)
    expected = int(source_archive.time_ns[frame_id])
    if actual != expected:
        raise ValueError(
            f"sequence timestamp mismatch at frame {frame_id}: "
            f"expected {expected}, got {actual}"
        )


def load_covisibility_cache(root, source_archive=None):
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"covisibility cache directory is missing: {root}")
    existing = frozenset(path.name for path in root.iterdir() if path.is_file())
    missing = sorted(_CACHE_FILES - existing)
    if missing:
        raise ValueError(f"covisibility cache is missing files: {missing}")
    try:
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("covisibility cache metadata is invalid") from error
    frame_ids = np.load(root / "frame_ids.npy", allow_pickle=False)
    time_ns = np.load(root / "time_ns.npy", allow_pickle=False)
    depth = np.load(root / "depth.npy", mmap_mode="r", allow_pickle=False)
    depth_cov = np.load(
        root / "depth_cov.npy", mmap_mode="r", allow_pickle=False
    )
    proxy_uv = np.load(root / "proxy_uv.npy", allow_pickle=False)
    proxy_depth = np.load(root / "proxy_depth.npy", allow_pickle=False)
    proxy_depth_cov = np.load(
        root / "proxy_depth_cov.npy", allow_pickle=False
    )
    if frame_ids.ndim != 1 or time_ns.shape != frame_ids.shape:
        raise ValueError("covisibility cache frame and timestamp shapes differ")
    if depth.dtype != np.float32 or depth_cov.dtype != np.float32:
        raise ValueError("covisibility cache depth arrays must be float32")
    if depth.ndim != 3 or depth_cov.shape != depth.shape:
        raise ValueError("covisibility cache depth shapes are invalid")
    count, height, width = depth.shape
    if count != len(frame_ids):
        raise ValueError("covisibility cache depth count is invalid")
    if proxy_uv.ndim != 2 or proxy_uv.shape[1] != 2:
        raise ValueError("covisibility proxy UV shape is invalid")
    expected_proxy_shape = (count, len(proxy_uv))
    if proxy_depth.shape != expected_proxy_shape or proxy_depth_cov.shape != expected_proxy_shape:
        raise ValueError("covisibility proxy depth shapes are invalid")
    if metadata.get("height") != height or metadata.get("width") != width:
        raise ValueError("covisibility cache metadata shape is invalid")
    if source_archive is not None:
        if metadata.get("source_id") != _source_id(source_archive):
            raise ValueError("covisibility cache source identity mismatch")
        expected_times = source_archive.time_ns[frame_ids.astype(np.int64)]
        if not np.array_equal(time_ns, expected_times):
            raise ValueError("covisibility cache source timestamps mismatch")
    return CovisibilityDepthCache(
        root=root,
        frame_ids=frame_ids,
        time_ns=time_ns,
        depth=depth,
        depth_cov=depth_cov,
        proxy_uv=proxy_uv,
        proxy_depth=proxy_depth,
        proxy_depth_cov=proxy_depth_cov,
        metadata=metadata,
    )


def build_covisibility_cache(
    sequence,
    source_archive,
    destination,
    frontend,
    *,
    stride=5,
    proxy_points=1024,
):
    destination = Path(destination)
    if destination.exists():
        cache = load_covisibility_cache(destination, source_archive)
        if cache.metadata.get("stride") != stride or cache.metadata.get(
            "proxy_points"
        ) != proxy_points:
            raise ValueError("existing covisibility cache settings differ")
        return cache
    if len(sequence) != len(source_archive.initial_sensor_poses):
        raise ValueError("sequence and source archive frame counts differ")
    frame_ids = np.asarray(
        covisibility_keyframes(len(sequence), stride), dtype=np.int64
    )
    first_frame = sequence[int(frame_ids[0])]
    _validate_frame(first_frame, int(frame_ids[0]), source_archive)
    first_depth, _, _ = frontend.estimate_bidirectional(
        first_frame.stereo, first_frame.stereo
    )
    if first_depth.cov is None:
        raise ValueError("covisibility cache requires depth covariance")
    first_map = first_depth.depth.detach().cpu().float().numpy()[0, 0]
    first_cov = first_depth.cov.detach().cpu().float().numpy()[0, 0]
    if first_map.ndim != 2 or first_cov.shape != first_map.shape:
        raise ValueError("frontend depth output shape is invalid")
    height, width = first_map.shape
    proxy_uv = deterministic_proxy_uv(width, height, proxy_points)
    proxy_index = np.rint(proxy_uv).astype(np.int64)
    proxy_index[:, 0] = np.clip(proxy_index[:, 0], 0, width - 1)
    proxy_index[:, 1] = np.clip(proxy_index[:, 1], 0, height - 1)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f"{destination.name}.tmp-{uuid.uuid4().hex}"
    )
    temporary.mkdir()
    try:
        shape = (len(frame_ids), height, width)
        depth_map = np.lib.format.open_memmap(
            temporary / "depth.npy", mode="w+", dtype=np.float32, shape=shape
        )
        depth_cov = np.lib.format.open_memmap(
            temporary / "depth_cov.npy", mode="w+", dtype=np.float32, shape=shape
        )
        proxy_depth = np.full(
            (len(frame_ids), len(proxy_uv)), np.nan, dtype=np.float32
        )
        proxy_depth_cov = np.full_like(proxy_depth, np.nan)
        times = np.empty(len(frame_ids), dtype=np.int64)

        for cache_index, frame_id in enumerate(frame_ids):
            frame_id = int(frame_id)
            frame = first_frame if cache_index == 0 else sequence[frame_id]
            _validate_frame(frame, frame_id, source_archive)
            if cache_index == 0:
                current_map, current_cov = first_map, first_cov
            else:
                output, _, _ = frontend.estimate_bidirectional(
                    frame.stereo, frame.stereo
                )
                if output.cov is None:
                    raise ValueError("covisibility cache requires depth covariance")
                current_map = output.depth.detach().cpu().float().numpy()[0, 0]
                current_cov = output.cov.detach().cpu().float().numpy()[0, 0]
            if current_map.shape != (height, width) or current_cov.shape != (height, width):
                raise ValueError("scheduled depth map shape changed")
            depth_map[cache_index] = current_map
            depth_cov[cache_index] = current_cov
            proxy_depth[cache_index] = current_map[
                proxy_index[:, 1], proxy_index[:, 0]
            ]
            proxy_depth_cov[cache_index] = current_cov[
                proxy_index[:, 1], proxy_index[:, 0]
            ]
            times[cache_index] = int(frame.stereo.frame_ns)

        depth_map.flush()
        depth_cov.flush()
        np.save(temporary / "frame_ids.npy", frame_ids)
        np.save(temporary / "time_ns.npy", times)
        np.save(temporary / "proxy_uv.npy", proxy_uv)
        np.save(temporary / "proxy_depth.npy", proxy_depth)
        np.save(temporary / "proxy_depth_cov.npy", proxy_depth_cov)
        metadata = {
            "schema_version": 1,
            "source_id": _source_id(source_archive),
            "stride": stride,
            "proxy_points": proxy_points,
            "height": height,
            "width": width,
            "keyframes": len(frame_ids),
        }
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8"
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return load_covisibility_cache(destination, source_archive)
