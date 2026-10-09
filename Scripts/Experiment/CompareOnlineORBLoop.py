"""Run strict online WindowICP versus online ORB-loop comparisons."""

import argparse
import copy
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import uuid

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Evaluation.EvalSeq import EvaluateSequences
from Scripts.Experiment.CompareWindowICP import run_and_tee
from Utility.Config import load_config


MODE_CONFIGS = {
    "window_skip": ROOT / "Config/Experiment/MACVO/MACVO_Fast_WindowICP.yaml",
    "window_pose_graph": ROOT / "Config/Experiment/MACVO/MACVO_Fast_WindowICP_ORBLoop.yaml",
    "window_orb_loop": ROOT / "Config/Experiment/MACVO/MACVO_Fast_WindowICP_ORBLoop.yaml",
    "window_orb_loop_sparse": (
        ROOT / "Config/Experiment/MACVO/"
        "MACVO_Fast_WindowICP_ORBLoop_Sparse.yaml"
    ),
    "window_orb_loop_sparse_t030": (
        ROOT / "Config/Experiment/MACVO/"
        "MACVO_Fast_WindowICP_ORBLoop_Sparse_t030.yaml"
    ),
    "window_orb_loop_no_skip": (
        ROOT / "Config/Experiment/MACVO/"
        "MACVO_Fast_WindowICP_ORBLoop_NoSkip.yaml"
    ),
}
ONLINE_DIAGNOSTIC_MODES = frozenset({
    "window_pose_graph",
    "window_orb_loop",
    "window_orb_loop_sparse",
    "window_orb_loop_sparse_t030",
    "window_orb_loop_no_skip",
})


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", default="V203")
    parser.add_argument("--seq-from", type=int, default=0)
    parser.add_argument("--seq-to", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--modes", nargs="+", choices=tuple(MODE_CONFIGS),
        default=[
            "window_skip", "window_pose_graph", "window_orb_loop",
        ],
    )
    parser.add_argument(
        "--result-root", type=Path,
        default=ROOT / "Results/OnlineORBLoop_comparison",
    )
    return parser


def online_invariants(diagnostics):
    frontend = diagnostics.get("frontend", {})
    checks = {
        "one_model": frontend.get("wrapped_instances_created") == 1,
        "one_cuda_graph": frontend.get("wrapped_graphs_captured") == 1,
        "concurrency_one": frontend.get("max_concurrent_calls") == 1,
        "vram_below_6gib": diagnostics.get(
            "cuda_max_memory_reserved", 1 << 62
        ) < 6 * 1024**3,
    }
    return {"passed": all(checks.values()), "checks": checks}


def uses_online_diagnostics(mode):
    return mode in ONLINE_DIAGNOSTIC_MODES


def loop_switch_summary(diagnostics, threshold=0.01, long_gap=100):
    results = diagnostics.get("pose_graph_results", [])
    records = next((
        result.get("loop_switches", [])
        for result in reversed(results)
        if "loop_switches" in result
    ), [])
    switches = [float(record["switch"]) for record in records]
    return {
        "max_loop_switch": max(switches, default=0.0),
        "effective_loops_001": sum(
            switch > threshold for switch in switches
        ),
        "effective_long_loops_001": sum(
            float(record["switch"]) > threshold
            and int(record["b"]) - int(record["a"]) > long_gap
            for record in records
        ),
    }


def sparse_loop_summary(diagnostics):
    hypothesis = diagnostics.get("hypothesis_records", [])
    loop_records = diagnostics.get("loop_records", [])
    ratios = [
        float(item["metrics"]["ransac_ratio"])
        for item in loop_records
        if item.get("metrics", {}).get("ransac_ratio") is not None
    ]
    reprojection = [
        float(item["metrics"]["reprojection_forward_median_px"])
        for item in loop_records
        if item.get("metrics", {}).get("reprojection_forward_median_px") is not None
    ]
    information = [
        float(value)
        for item in loop_records
        for value in item.get("information_eigenvalues", [])
    ]
    return {
        "tentative_hypotheses": sum(
            item.get("state") == "tentative" for item in hypothesis
        ),
        "confirmed_hypotheses": sum(
            item.get("state") == "confirmed" for item in hypothesis
        ),
        "strong_hypotheses": sum(
            item.get("state") == "strong" for item in hypothesis
        ),
        "emitted_sparse_factors": sum(
            item.get("status") == "accepted_sparse" for item in loop_records
        ),
        "median_sparse_ransac_ratio": (
            float(np.median(ratios)) if ratios else 0.0
        ),
        "median_sparse_reprojection_px": (
            float(np.median(reprojection)) if reprojection else 0.0
        ),
        "max_sparse_information_eigenvalue": max(information, default=0.0),
    }


def write_rows(folder, rows):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    (folder / "metrics.json").write_text(
        json.dumps(rows, indent=2, allow_nan=False), encoding="utf-8"
    )
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (folder / "metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.seq_from < 0 or (
        args.seq_to is not None and args.seq_to <= args.seq_from
    ):
        raise ValueError("Expected 0 <= seq-from < seq-to")
    run_root = args.result_root.resolve() / (
        datetime.now().strftime("%Y%m%d_%H%M%S")
        + "_" + uuid.uuid4().hex[:6]
    )
    run_root.mkdir(parents=True, exist_ok=False)
    data_path = ROOT / f"Config/Sequence/EuRoC_{args.sequence}_local.yaml"
    rows = []
    reference_times = None
    for mode in args.modes:
        mode_root = run_root / mode
        mode_root.mkdir()
        _, config = load_config(MODE_CONFIGS[mode])
        config = copy.deepcopy(config)
        if mode == "window_pose_graph":
            config["Odometry"]["args"]["online_loop"]["enabled"] = False
            config["Odometry"]["name"] = "MACVO-Fast-WindowICP5-PoseGraph"
        config_path = mode_root / "input.yaml"
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        command = [
            sys.executable, str(ROOT / "MACVO.py"),
            "--odom", str(config_path),
            "--data", str(data_path),
            "--resultRoot", str(mode_root / "outputs"),
            "--seed", str(args.seed),
            "--seq_from", str(args.seq_from),
            "--noeval", "--timing",
        ]
        if args.seq_to is not None:
            command.extend(["--seq_to", str(args.seq_to)])
        env = os.environ.copy()
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        env.setdefault("OMP_NUM_THREADS", "4")
        env.setdefault("MKL_NUM_THREADS", "4")
        run_and_tee(command, ROOT, env, mode_root / "run.log")
        provenance_paths = list(
            (mode_root / "outputs").glob("*/*/run_provenance.json")
        )
        if len(provenance_paths) != 1:
            raise RuntimeError(f"Expected one completed result for {mode}")
        provenance = json.loads(provenance_paths[0].read_text())
        if provenance.get("status") != "complete":
            raise RuntimeError(f"Incomplete result for {mode}")
        space = provenance_paths[0].parent
        poses = np.load(space / "poses.npy")
        if reference_times is None:
            reference_times = poses[:, 0]
        elif not np.array_equal(reference_times, poses[:, 0]):
            raise RuntimeError("online comparison timestamps differ")
        header, results = EvaluateSequences([str(space)], correct_scale=False)
        metric_values = next(row for row in results if row[0] != "Average")
        elapsed = json.loads((space / "elapsed_time.json").read_text())
        runtime = np.asarray(
            elapsed["CPU_ElapsedTime"]["Odom_Runtime"], dtype=float
        )
        row = {
            "sequence": args.sequence,
            "mode": mode,
            "frames": len(poses),
            "space": str(space),
            "git_commit": provenance.get("git_commit"),
            "seed": args.seed,
            "runtime_mean_ms": float(runtime.mean()),
            "runtime_median_ms": float(np.median(runtime)),
            "runtime_p95_ms": float(np.percentile(runtime, 95)),
            **dict(zip(header[1:], metric_values[1:])),
        }
        window = json.loads((space / "window_diagnostics.json").read_text())
        row["unrefined_windows"] = sum(
            item["status"] != "refined" for item in window["windows"]
        )
        if uses_online_diagnostics(mode):
            diagnostics = json.loads(
                (space / "online_loop_diagnostics.json").read_text()
            )
            invariant = online_invariants(diagnostics)
            row.update({
                "online_invariants_passed": invariant["passed"],
                "normal_frontend_calls": diagnostics["frontend"]["tracking_calls"],
                "loop_frontend_calls": diagnostics["frontend"]["loop_calls"],
                "accepted_loops": sum(
                    item.get("status") in {"accepted", "accepted_sparse"}
                    for item in diagnostics["loop_records"]
                ),
                "pose_graph_writebacks": diagnostics["pose_graph_writebacks"],
                "peak_vram_bytes": diagnostics["cuda_max_memory_reserved"],
                "pose_factors": diagnostics["pose_factors"],
                **loop_switch_summary(diagnostics),
                **sparse_loop_summary(diagnostics),
            })
            (run_root / "online_invariants.json").write_text(
                json.dumps(invariant, indent=2, allow_nan=False),
                encoding="utf-8",
            )
            if not invariant["passed"]:
                raise RuntimeError(f"Online invariants failed: {invariant}")
        rows.append(row)
        write_rows(run_root, rows)
        print(
            f"{mode}: ATE={row['RMSE_ATE']:.6f}, "
            f"runtime={row['runtime_mean_ms']:.1f} ms",
            flush=True,
        )
    print(f"Comparison results: {run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
