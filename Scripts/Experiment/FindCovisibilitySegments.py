"""Find difficult fixed-length trajectory windows containing GT revisit pairs."""

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys

import numpy as np
import pypose as pp
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class CovisibilitySegment:
    start: int
    end: int
    revisit_pairs: int
    rte_rmse: float
    roe_rmse: float
    difficulty: float


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space", type=Path, required=True)
    parser.add_argument("--window", type=int, default=120)
    parser.add_argument("--min-temporal-gap", type=int, default=30)
    parser.add_argument("--max-distance", type=float, default=2.0)
    parser.add_argument("--max-angle-deg", type=float, default=30.0)
    parser.add_argument("--min-revisit-pairs", type=int, default=3)
    parser.add_argument(
        "--exclude-range",
        nargs=2,
        type=int,
        action="append",
        default=[],
        metavar=("START", "END"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def rotation_angle_deg(first, second):
    first = pp.SO3(torch.as_tensor(first).double())
    second = pp.SO3(torch.as_tensor(second).double())
    return float((first.Inv() @ second).Log().tensor().norm() * 180 / math.pi)


def is_revisit_pair(
    poses,
    i,
    j,
    *,
    min_gap,
    max_distance,
    max_angle_deg,
):
    if j - i < min_gap:
        return False
    poses = torch.as_tensor(poses).double()
    distance = float((poses[j, :3] - poses[i, :3]).norm())
    angle = rotation_angle_deg(poses[i, 3:], poses[j, 3:])
    return distance <= max_distance and angle <= max_angle_deg


def _revisit_matrix(
    gt,
    min_temporal_gap,
    max_distance,
    max_angle_deg,
):
    gt = torch.as_tensor(gt).double()
    count = len(gt)
    result = np.zeros((count, count), dtype=np.int32)
    rotations = pp.SO3(gt[:, 3:])
    for i in range(count):
        first_j = i + min_temporal_gap
        if first_j >= count:
            continue
        js = torch.arange(first_j, count)
        distance = (gt[js, :3] - gt[i, :3]).norm(dim=-1)
        relative = rotations[i].Inv() @ rotations[js]
        angle = relative.Log().tensor().norm(dim=-1) * 180 / math.pi
        valid = (distance <= max_distance) & (angle <= max_angle_deg)
        valid_js = js[valid].cpu().numpy()
        result[i, valid_js] = 1
    return result


def _rectangle_sum(prefix, start, end):
    total = prefix[end - 1, end - 1]
    if start:
        total -= prefix[start - 1, end - 1]
        total -= prefix[end - 1, start - 1]
        total += prefix[start - 1, start - 1]
    return int(total)


def _motion_errors(gt, estimate):
    gt_pose = pp.SE3(torch.as_tensor(gt).double())
    est_pose = pp.SE3(torch.as_tensor(estimate).double())
    gt_motion = gt_pose[:-1].Inv() @ gt_pose[1:]
    est_motion = est_pose[:-1].Inv() @ est_pose[1:]
    error = est_motion.Inv() @ gt_motion
    rte = error.translation().norm(dim=-1).cpu().numpy()
    roe = (
        error.rotation().Log().tensor().norm(dim=-1).cpu().numpy()
        * 180
        / math.pi
    )
    return rte, roe


def find_segments(
    gt,
    estimate,
    *,
    window=120,
    min_temporal_gap=30,
    max_distance=2.0,
    max_angle_deg=30.0,
    min_pairs=3,
    exclude_ranges=(),
):
    gt = torch.as_tensor(gt).double()
    estimate = torch.as_tensor(estimate).double()
    if gt.shape != estimate.shape or gt.ndim != 2 or gt.shape[1] != 7:
        raise ValueError("GT and estimate must be matching Nx7 poses")
    if window < 2 or window > len(gt):
        raise ValueError("window must fit the trajectory")
    revisit = _revisit_matrix(
        gt, min_temporal_gap, max_distance, max_angle_deg
    )
    prefix = revisit.cumsum(axis=0).cumsum(axis=1)
    rte, roe = _motion_errors(gt, estimate)
    candidates = []
    for start in range(len(gt) - window + 1):
        end = start + window
        if any(start < excluded_end and excluded_start < end
               for excluded_start, excluded_end in exclude_ranges):
            continue
        pairs = _rectangle_sum(prefix, start, end)
        if pairs < min_pairs:
            continue
        motion_slice = slice(start, end - 1)
        rte_rmse = float(np.sqrt(np.mean(np.square(rte[motion_slice]))))
        roe_rmse = float(np.sqrt(np.mean(np.square(roe[motion_slice]))))
        candidates.append(CovisibilitySegment(
            start=start,
            end=end,
            revisit_pairs=pairs,
            rte_rmse=rte_rmse,
            roe_rmse=roe_rmse,
            difficulty=0.0,
        ))
    if not candidates:
        return []
    rte_scale = max(float(np.median([item.rte_rmse for item in candidates])), 1e-12)
    roe_scale = max(float(np.median([item.roe_rmse for item in candidates])), 1e-12)
    ranked = [
        CovisibilitySegment(
            start=item.start,
            end=item.end,
            revisit_pairs=item.revisit_pairs,
            rte_rmse=item.rte_rmse,
            roe_rmse=item.roe_rmse,
            difficulty=item.rte_rmse / rte_scale + item.roe_rmse / roe_scale,
        )
        for item in candidates
    ]
    return sorted(ranked, key=lambda item: (-item.difficulty, item.start))


def write_segment_results(output_root, segments):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    payload = [asdict(segment) for segment in segments]
    (output_root / "covisibility_segments.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    columns = [
        "start", "end", "revisit_pairs", "rte_rmse", "roe_rmse", "difficulty"
    ]
    with (output_root / "covisibility_segments.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(payload)
    selected = (
        {"status": "selected", **payload[0]}
        if payload
        else {"status": "no_segment"}
    )
    (output_root / "selected_segment.json").write_text(
        json.dumps(selected, indent=2, allow_nan=False), encoding="utf-8"
    )
    return selected


def main(argv=None):
    args = build_parser().parse_args(argv)
    estimated = np.load(args.space / "poses.npy")
    reference = np.load(args.space / "ref_poses.npy")
    if estimated.shape != reference.shape or not np.array_equal(
        estimated[:, 0], reference[:, 0]
    ):
        raise ValueError("estimated and reference trajectory timestamps differ")
    segments = find_segments(
        reference[:, 1:],
        estimated[:, 1:],
        window=args.window,
        min_temporal_gap=args.min_temporal_gap,
        max_distance=args.max_distance,
        max_angle_deg=args.max_angle_deg,
        min_pairs=args.min_revisit_pairs,
        exclude_ranges=args.exclude_range,
    )
    selected = write_segment_results(args.output_root, segments)
    print(json.dumps(selected, indent=2, allow_nan=False))
    return 0 if segments else 2


if __name__ == "__main__":
    raise SystemExit(main())
