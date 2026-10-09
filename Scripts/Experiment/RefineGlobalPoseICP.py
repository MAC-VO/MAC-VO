"""Refine saved global ICP factors without rerunning network inference."""

import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np
import pypose as pp
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Evaluation.MetricsSeq import (
    evaluateATE,
    evaluateROE,
    evaluateRPE,
    evaluateRTE,
)
from Module.Optimization.FactorArchive import (
    filter_factor_archive,
    load_factor_archive,
    merge_factor_archives,
    sensor_to_body_trajectory,
)
from Module.Optimization.GlobalPoseICP import optimize_global_pose_graph
from Utility.Trajectory import Trajectory


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space", type=Path, required=True)
    parser.add_argument("--long-factors", type=Path)
    parser.add_argument(
        "--edge-kinds",
        nargs="+",
        choices=("gap5", "gap10", "proximity"),
    )
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--huber", type=float, default=3.0)
    parser.add_argument("--output-tag", default="short")
    return parser


def refine_archives(
    short_archive,
    long_archive=None,
    *,
    edge_kinds=None,
    iterations=5,
    huber=3.0,
):
    selected = long_archive
    if selected is not None and edge_kinds is not None:
        selected = filter_factor_archive(selected, edge_kinds)
    merged = (
        short_archive
        if selected is None
        else merge_factor_archives(short_archive, selected)
    )
    return optimize_global_pose_graph(
        merged.initial_sensor_poses,
        merged.edges,
        max_iters=iterations,
        huber_delta=huber,
    )


def _trajectory_from_timed_array(data):
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 8 or not np.isfinite(data).all():
        raise ValueError("trajectory must be a finite Nx8 array")
    return Trajectory(
        pp.SE3(torch.from_numpy(data[:, 1:].copy())),
        torch.from_numpy(data[:, 0].copy()),
        torch.zeros(len(data), dtype=torch.bool),
    )


def evaluate_timed_trajectory(space, timed_poses):
    reference_path = Path(space, "ref_poses.npy")
    if not reference_path.is_file():
        return {
            "status": "unavailable",
            "reason": "ref_poses.npy is missing",
        }
    gt_traj = _trajectory_from_timed_array(np.load(reference_path))
    est_traj = _trajectory_from_timed_array(timed_poses)
    est_traj = est_traj.align_origin(gt_traj)
    gt_traj.time = gt_traj.time - gt_traj.time[0]
    est_traj.time = est_traj.time - est_traj.time[0]
    est_traj = est_traj.align_time(gt_traj.time)
    ate = evaluateATE(gt_traj.as_evo, est_traj.as_evo, correct_scale=False)
    rte = evaluateRTE(gt_traj.as_evo, est_traj.as_evo, correct_scale=False)
    roe = evaluateROE(gt_traj.as_evo, est_traj.as_evo, correct_scale=False)
    rpe = evaluateRPE(gt_traj.as_evo, est_traj.as_evo, correct_scale=False)
    return {
        "status": "evaluated",
        "RMSE_ATE": float(ate.stats["rmse"]),
        "RMSE_RTE": float(rte.stats["rmse"]),
        "RMSE_ROE": float(roe.stats["rmse"]),
        "RMSE_RPE": float(rpe.stats["rmse"]),
    }


def evaluate_pose_file(space, pose_path):
    return evaluate_timed_trajectory(space, np.load(pose_path))


def write_refinement_outputs(space, tag, result, archive, metrics):
    if not re.fullmatch(r"[a-z0-9_]+", tag):
        raise ValueError(
            "output tag must contain lowercase letters, digits, or underscores"
        )
    space = Path(space)
    space.mkdir(parents=True, exist_ok=True)
    diagnostics_path = space / f"global_{tag}_diagnostics.json"
    metrics_path = space / f"global_{tag}_metrics.json"
    diagnostics_path.write_text(
        json.dumps(result.diagnostics, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    metrics_path.write_text(
        json.dumps(metrics, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    if result.diagnostics.get("status") != "refined":
        return None
    trajectory = sensor_to_body_trajectory(
        result.poses, archive.time_ns, archive.T_BS
    )
    pose_path = space / f"poses_global_{tag}.npy"
    np.save(pose_path, trajectory)
    return pose_path


def _resolve_long_path(space, path):
    if path is None:
        return None
    return path if path.is_absolute() else Path(space, path)


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.iterations < 1:
        raise ValueError("iterations must be positive")
    if not np.isfinite(args.huber) or args.huber <= 0:
        raise ValueError("huber must be finite and positive")
    if args.long_factors is not None and args.output_tag == "short":
        raise ValueError("the short output tag cannot include long factors")
    if args.long_factors is None and args.edge_kinds is not None:
        raise ValueError("edge kinds require a long factor archive")

    space = args.space.resolve()
    short = load_factor_archive(space / "global_factors.npz")
    long_path = _resolve_long_path(space, args.long_factors)
    long = None if long_path is None else load_factor_archive(long_path)
    result = refine_archives(
        short,
        long,
        edge_kinds=None if args.edge_kinds is None else set(args.edge_kinds),
        iterations=args.iterations,
        huber=args.huber,
    )
    if result.diagnostics.get("status") == "refined":
        timed = sensor_to_body_trajectory(
            result.poses, short.time_ns, short.T_BS
        )
        metrics = evaluate_timed_trajectory(space, timed)
    else:
        metrics = {
            "status": "unavailable",
            "reason": "global solver did not refine the trajectory",
        }
    pose_path = write_refinement_outputs(
        space, args.output_tag, result, short, metrics
    )
    print(json.dumps({
        "pose": None if pose_path is None else str(pose_path),
        "diagnostics": result.diagnostics,
        "metrics": metrics,
    }, allow_nan=False))
    return 0 if pose_path is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
