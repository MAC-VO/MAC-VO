"""Run and verify strict same-source offline long-range ICP comparisons."""

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Module.Optimization.FactorArchive import load_factor_archive
from Scripts.Experiment.GenerateLongRangeICP import file_sha256, optional_index
from Scripts.Experiment.RefineGlobalPoseICP import evaluate_pose_file
from Utility.Sandbox import Sandbox


MODES = ("before_global", "short", "gap5", "gap5_10")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", default="V203")
    parser.add_argument("--seq-from", type=int, default=1095)
    parser.add_argument("--seq-to", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--space", type=Path)
    parser.add_argument(
        "--result-root",
        type=Path,
        default=ROOT / "Results/LongRangeICP_comparison",
    )
    parser.add_argument(
        "--odom-config",
        type=Path,
        default=ROOT / "Config/Experiment/MACVO/MACVO_Fast_WindowICP_Global.yaml",
    )
    parser.add_argument("--verify-space", type=Path)
    return parser


def _mode_row(rows, mode):
    matches = [row for row in rows if row.get("mode") == mode]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one comparison row for {mode}")
    return matches[0]


def stage_one_passed(rows):
    try:
        short = _mode_row(rows, "short")
        long = _mode_row(rows, "gap5_10")
        short_ate = float(short["RMSE_ATE"])
        long_ate = float(long["RMSE_ATE"])
        initial_cost = float(long["initial_cost"])
        final_cost = float(long["final_cost"])
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        np.isfinite([short_ate, long_ate, initial_cost, final_cost]).all()
        and long.get("status") == "evaluated"
        and long.get("global_status") == "refined"
        and long.get("anchor_preserved") is True
        and final_cost < initial_cost
        and long_ate < short_ate
    )


def write_comparison_artifacts(folder, rows):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    (folder / "long_range_metrics.json").write_text(
        json.dumps(rows, indent=2, allow_nan=False), encoding="utf-8"
    )
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (folder / "long_range_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    short_ate = float(_mode_row(rows, "short")["RMSE_ATE"])
    long_ate = float(_mode_row(rows, "gap5_10")["RMSE_ATE"])
    gate = {
        "passed": stage_one_passed(rows),
        "short_ate": short_ate,
        "gap5_10_ate": long_ate,
        "absolute_change": long_ate - short_ate,
        "percent_change": (long_ate - short_ate) / short_ate * 100,
    }
    (folder / "stage_gate.json").write_text(
        json.dumps(gate, indent=2, allow_nan=False), encoding="utf-8"
    )
    return gate


def discover_result_space(root):
    root = Path(root)
    direct = root / "run_provenance.json"
    provenance_files = [direct] if direct.is_file() else sorted(
        root.rglob("run_provenance.json")
    )
    if len(provenance_files) != 1:
        raise RuntimeError(
            f"Expected exactly one result run under {root}, found "
            f"{len(provenance_files)}"
        )
    payload = json.loads(provenance_files[0].read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise RuntimeError(
            f"Result run is not complete: {provenance_files[0].parent}"
        )
    return provenance_files[0].parent


def source_run_metadata(space):
    space = Path(space)
    config = Sandbox.load(space).config
    provenance = json.loads(
        (space / "run_provenance.json").read_text(encoding="utf-8")
    )
    return {
        "sequence": config.Data.args.name,
        "frame_from": optional_index(config.Data.start_idx),
        "frame_to": optional_index(config.Data.end_idx),
        "seed": provenance.get("seed"),
    }


def _manifest_path(root):
    root = Path(root)
    direct = root / "comparison_manifest.json"
    candidates = [direct] if direct.is_file() else sorted(
        root.rglob("comparison_manifest.json")
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one comparison manifest under {root}, found "
            f"{len(candidates)}"
        )
    return candidates[0]


def verify_comparison_space(root):
    manifest_path = _manifest_path(root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    space = Path(manifest["source_space"])
    short_path = space / "global_factors.npz"
    long_path = space / "long_factors_gap5_10.npz"
    if file_sha256(short_path) != manifest["source_factor_sha256"]:
        raise RuntimeError("source factor SHA-256 mismatch")
    if file_sha256(long_path) != manifest["long_factor_sha256"]:
        raise RuntimeError("long factor SHA-256 mismatch")
    if file_sha256(space / "poses.npy") != manifest["source_pose_sha256"]:
        raise RuntimeError("source poses.npy was modified")
    short = load_factor_archive(short_path)
    long = load_factor_archive(long_path)
    if short.metadata["source_id"] != long.metadata["source_id"]:
        raise RuntimeError("short and long archives have different source IDs")

    before = np.load(space / "poses_before_global.npy")
    anchor_preserved = True
    for mode in ("short", "gap5", "gap5_10"):
        candidate = np.load(space / f"poses_global_{mode}.npy")
        if not np.isfinite(candidate).all():
            raise RuntimeError(f"{mode} trajectory is non-finite")
        if not np.array_equal(candidate[:, 0], before[:, 0]):
            raise RuntimeError(f"{mode} trajectory timestamps differ")
        anchor_preserved = anchor_preserved and np.array_equal(
            candidate[0], before[0]
        )
    if not anchor_preserved:
        raise RuntimeError("a tagged trajectory changed the fixed anchor")
    return {
        "status": "verified",
        "source_space": str(space),
        "source_id": short.metadata["source_id"],
        "anchor_preserved": True,
        "poses": len(before),
        "short_edges": len(short.edges),
        "long_edges": len(long.edges),
    }


def _run(command, env=None):
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def _new_run_root(result_root):
    name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    root = Path(result_root).resolve() / name
    root.mkdir(parents=True, exist_ok=False)
    return root


def _run_online_source(args, run_root):
    source_root = run_root / "source"
    data_config = ROOT / f"Config/Sequence/EuRoC_{args.sequence}_local.yaml"
    command = [
        sys.executable,
        str(ROOT / "MACVO.py"),
        "--odom",
        str(args.odom_config.resolve()),
        "--data",
        str(data_config),
        "--resultRoot",
        str(source_root),
        "--seed",
        str(args.seed),
        "--seq_from",
        str(args.seq_from),
        "--noeval",
        "--timing",
    ]
    if args.seq_to is not None:
        command.extend(["--seq_to", str(args.seq_to)])
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("MKL_NUM_THREADS", "4")
    _run(command, env=env)
    return discover_result_space(source_root)


def _run_offline_commands(space):
    generator = ROOT / "Scripts/Experiment/GenerateLongRangeICP.py"
    refiner = ROOT / "Scripts/Experiment/RefineGlobalPoseICP.py"
    _run([sys.executable, str(generator), "--space", str(space)])
    _run([
        sys.executable, str(refiner), "--space", str(space),
        "--output-tag", "short",
    ])
    _run([
        sys.executable, str(refiner), "--space", str(space),
        "--long-factors", "long_factors_gap5_10.npz",
        "--edge-kinds", "gap5", "--output-tag", "gap5",
    ])
    _run([
        sys.executable, str(refiner), "--space", str(space),
        "--long-factors", "long_factors_gap5_10.npz",
        "--edge-kinds", "gap5", "gap10", "--output-tag", "gap5_10",
    ])


def _build_rows(space, manifest):
    common = {
        "sequence": manifest["sequence"],
        "frame_from": manifest["frame_from"],
        "frame_to": manifest["frame_to"],
        "seed": manifest["seed"],
        "space": str(space),
        "source_factor_sha256": manifest["source_factor_sha256"],
        "long_factor_sha256": manifest["long_factor_sha256"],
        "source_pose_sha256": manifest["source_pose_sha256"],
    }
    rows = [common | {
        "mode": "before_global",
        **evaluate_pose_file(space, space / "poses_before_global.npy"),
    }]
    for mode in ("short", "gap5", "gap5_10"):
        diagnostics = json.loads(
            (space / f"global_{mode}_diagnostics.json").read_text()
        )
        metrics = json.loads(
            (space / f"global_{mode}_metrics.json").read_text()
        )
        rows.append(common | {
            "mode": mode,
            "global_status": diagnostics.get("status"),
            "anchor_preserved": diagnostics.get("anchor_preserved"),
            "edges": diagnostics.get("edges"),
            "observations": diagnostics.get("observations"),
            "initial_cost": diagnostics.get("initial_cost"),
            "final_cost": diagnostics.get("final_cost"),
            "solver_seconds": diagnostics.get("seconds"),
            **metrics,
        })
    return rows


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.verify_space is not None:
        print(json.dumps(
            verify_comparison_space(args.verify_space),
            indent=2,
            allow_nan=False,
        ))
        return 0
    if args.seq_from < 0 or (
        args.seq_to is not None and args.seq_to <= args.seq_from
    ):
        raise ValueError("Expected 0 <= seq-from < seq-to")

    run_root = _new_run_root(args.result_root)
    space = (
        discover_result_space(args.space)
        if args.space is not None
        else _run_online_source(args, run_root)
    )
    required = [
        "poses.npy",
        "poses_before_global.npy",
        "global_factors.npz",
        "ref_poses.npy",
    ]
    missing = [name for name in required if not (space / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Source result is missing artifacts: {missing}")
    source_pose_sha256 = file_sha256(space / "poses.npy")
    _run_offline_commands(space)
    manifest = source_run_metadata(space) | {
        "source_space": str(space.resolve()),
        "source_factor_sha256": file_sha256(space / "global_factors.npz"),
        "long_factor_sha256": file_sha256(
            space / "long_factors_gap5_10.npz"
        ),
        "source_pose_sha256": source_pose_sha256,
    }
    (run_root / "comparison_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8"
    )
    rows = _build_rows(space, manifest)
    gate = write_comparison_artifacts(run_root, rows)
    verification = verify_comparison_space(run_root)
    (run_root / "verification.json").write_text(
        json.dumps(verification, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"Comparison results: {run_root}")
    print(json.dumps(gate, indent=2, allow_nan=False))
    if not gate["passed"]:
        print(
            "Stage 1 did not lower ATE; full V203 is not authorized by "
            "the design gate."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
