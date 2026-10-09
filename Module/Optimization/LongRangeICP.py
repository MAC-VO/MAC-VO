"""Offline scheduling and generation of sparse long-range ICP factors."""

from collections import Counter
from dataclasses import dataclass
import time

from Module.Optimization.FactorArchive import FactorArchive
from Module.Optimization.MatchICP import build_match_edge


@dataclass(frozen=True)
class LongRangeGenerationResult:
    archive: FactorArchive
    diagnostics: dict


def long_range_targets(frame_count, stride=5, gaps=(5, 10)):
    if not isinstance(frame_count, int) or frame_count < 0:
        raise ValueError("frame_count must be a non-negative integer")
    if not isinstance(stride, int) or stride < 1:
        raise ValueError("stride must be a positive integer")
    if any(not isinstance(gap, int) or gap < 1 for gap in gaps):
        raise ValueError("gaps must contain positive integers")
    return [
        (
            target,
            tuple(target - gap for gap in gaps if target - gap >= 0),
        )
        for target in range(stride, frame_count, stride)
    ]


def _frame_time_ns(frame):
    return int(frame.stereo.frame_ns)


def _validate_frame_timestamp(frame, index, source_archive):
    actual = _frame_time_ns(frame)
    expected = int(source_archive.time_ns[index])
    if actual != expected:
        raise ValueError(
            f"sequence timestamp mismatch at frame {index}: "
            f"expected {expected}, got {actual}"
        )


def _counter_dict(counters):
    return {
        kind: dict(sorted(counter.items()))
        for kind, counter in counters.items()
    }


def generate_long_range_archive(
    sequence,
    source_archive,
    *,
    frontend,
    selector,
    covariance_model,
    num_point,
    min_num_point,
    edge_width,
    match_cov_default,
    device,
    edge_builder=build_match_edge,
    stride=5,
    gaps=(5, 10),
    metadata=None,
):
    start = time.perf_counter()
    frame_count = len(sequence)
    if frame_count != len(source_archive.initial_sensor_poses):
        raise ValueError(
            "sequence frame count must match the source factor archive"
        )
    if frame_count == 0:
        raise ValueError("long-range generation requires at least one frame")
    schedule = long_range_targets(frame_count, stride=stride, gaps=gaps)
    frame0 = sequence[0]
    _validate_frame_timestamp(frame0, 0, source_archive)

    frame_cache = {0: frame0}
    depth_cache = {}
    frontend_calls = 0
    max_cached_frames = 1
    max_cached_depths = 0
    edges = []
    edge_kinds = []
    attempted = {"gap5": 0, "gap10": 0}
    accepted = {"gap5": 0, "gap10": 0}
    rejected = {"gap5": Counter(), "gap10": Counter()}
    inference_failures = []

    depth0, _, _ = frontend.estimate_window(
        None, frame0.stereo, frame0.stereo
    )
    frontend_calls += 1
    depth_cache[0] = depth0
    max_cached_depths = 1

    for target, sources in schedule:
        current = sequence[target]
        _validate_frame_timestamp(current, target, source_archive)
        for source in sources:
            if source not in frame_cache:
                frame_cache[source] = sequence[source]
                _validate_frame_timestamp(
                    frame_cache[source], source, source_archive
                )

        try:
            if len(sources) == 1:
                depth_t, gap5_match, _ = frontend.estimate_window(
                    None,
                    frame_cache[sources[0]].stereo,
                    current.stereo,
                )
                matches = [(sources[0], "gap5", gap5_match)]
            else:
                depth_t, gap5_match, gap10_match = frontend.estimate_window(
                    frame_cache[sources[1]].stereo,
                    frame_cache[sources[0]].stereo,
                    current.stereo,
                )
                matches = [
                    (sources[0], "gap5", gap5_match),
                    (sources[1], "gap10", gap10_match),
                ]
            frontend_calls += 1
        except Exception as error:
            frontend_calls += 1
            inference_failures.append({
                "target": target,
                "reason": str(error),
            })
            frame_cache[target] = current
            keep = {target, target - stride}
            frame_cache = {
                index: value
                for index, value in frame_cache.items()
                if index in keep
            }
            depth_cache = {
                index: value
                for index, value in depth_cache.items()
                if index in keep
            }
            max_cached_frames = max(max_cached_frames, len(frame_cache))
            max_cached_depths = max(max_cached_depths, len(depth_cache))
            continue

        depth_cache = {
            index: value
            for index, value in depth_cache.items()
            if index in sources
        }
        frame_cache = {
            index: value
            for index, value in frame_cache.items()
            if index in sources
        }
        depth_cache[target] = depth_t
        frame_cache[target] = current
        max_cached_frames = max(max_cached_frames, len(frame_cache))
        max_cached_depths = max(max_cached_depths, len(depth_cache))

        for source, kind, match in matches:
            attempted[kind] += 1
            if source not in depth_cache:
                rejected[kind]["missing_source_depth"] += 1
                continue
            if match is None:
                rejected[kind]["missing_match"] += 1
                continue
            built = edge_builder(
                a=source,
                b=target,
                stereo_a=frame_cache[source].stereo,
                stereo_b=current.stereo,
                depth_a=depth_cache[source],
                depth_b=depth_t,
                match=match,
                frontend=frontend,
                selector=selector,
                covariance_model=covariance_model,
                num_point=num_point,
                min_num_point=min_num_point,
                edge_width=edge_width,
                match_cov_default=match_cov_default,
                device=device,
            )
            if built.edge is None:
                rejected[kind][built.reason or "unknown"] += 1
            else:
                edges.append(built.edge)
                edge_kinds.append(kind)
                accepted[kind] += 1

        keep = {target, target - stride}
        frame_cache = {
            index: value
            for index, value in frame_cache.items()
            if index in keep
        }
        depth_cache = {
            index: value
            for index, value in depth_cache.items()
            if index in keep
        }

    diagnostics = {
        "status": "generated" if edges else "no_edges",
        "frames": frame_count,
        "stride": stride,
        "gaps": list(gaps),
        "targets": len(schedule),
        "frontend_calls": frontend_calls,
        "max_cached_frames": max_cached_frames,
        "max_cached_depths": max_cached_depths,
        "attempted": attempted,
        "accepted": accepted,
        "rejected": _counter_dict(rejected),
        "inference_failures": inference_failures,
        "edges": len(edges),
        "observations": sum(len(edge.points_a) for edge in edges),
        "seconds": time.perf_counter() - start,
    }
    archive_metadata = dict(source_archive.metadata)
    archive_metadata.update({
        "generator": "offline_long_range_icp",
        "stride": stride,
        "gaps": list(gaps),
        "num_point": num_point,
        "min_num_point": min_num_point,
        "edge_width": edge_width,
        "match_cov_default": match_cov_default,
        "generation": diagnostics,
    })
    if metadata:
        archive_metadata.update(metadata)
    archive = FactorArchive(
        initial_sensor_poses=source_archive.initial_sensor_poses,
        time_ns=source_archive.time_ns,
        T_BS=source_archive.T_BS,
        edges=tuple(edges),
        edge_kinds=tuple(edge_kinds),
        metadata=archive_metadata,
    )
    return LongRangeGenerationResult(archive, diagnostics)
