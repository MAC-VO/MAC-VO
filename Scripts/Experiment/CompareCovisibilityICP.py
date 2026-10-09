"""Compare dynamic covisibility proximity factors with fixed temporal factors."""

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

from Module.Optimization.FactorArchive import (
    filter_factor_archive,
    load_factor_archive,
)
from Scripts.Experiment.CompareLongRangeICP import (
    discover_result_space,
    source_run_metadata,
)
from Scripts.Experiment.GenerateLongRangeICP import file_sha256
from Scripts.Experiment.RefineGlobalPoseICP import evaluate_pose_file


MODES = ("before_global", "short", "gap5", "proximity")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", default="V203")
    parser.add_argument("--seq-from", type=int)
    parser.add_argument("--seq-to", type=int)
    parser.add_argument("--segment-json", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--space", type=Path)
    parser.add_argument(
        "--result-root",
        type=Path,
        default=ROOT / "Results/CovisibilityICP_comparison",
    )
    parser.add_argument(
        "--odom-config",
        type=Path,
        default=ROOT / "Config/Experiment/MACVO/MACVO_Fast_WindowICP_Global.yaml",
    )
    parser.add_argument("--verify-space", type=Path)
    return parser


def resolve_sequence_bounds(args):
    if args.segment_json is not None:
        if args.seq_from is not None or args.seq_to is not None:
            raise ValueError("segment-json conflicts with explicit sequence bounds")
        payload = json.loads(args.segment_json.read_text(encoding="utf-8"))
        start, end = payload.get("start"), payload.get("end")
        if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start:
            raise ValueError("segment JSON must contain valid integer start/end")
        return start, end, file_sha256(args.segment_json)
    start = 1095 if args.seq_from is None else args.seq_from
    end = args.seq_to
    if start < 0 or (end is not None and end <= start):
        raise ValueError("Expected 0 <= seq-from < seq-to")
    return start, end, None


def _mode(rows, name):
    matches = [row for row in rows if row.get("mode") == name]
    if len(matches) != 1:
        raise ValueError(f"Expected one {name} comparison row")
    return matches[0]


def dynamic_covisibility_gate(rows):
    try:
        short = _mode(rows, "short")
        gap5 = _mode(rows, "gap5")
        proximity = _mode(rows, "proximity")
        short_ate = float(short["RMSE_ATE"])
        gap5_ate = float(gap5["RMSE_ATE"])
        proximity_ate = float(proximity["RMSE_ATE"])
        initial_cost = float(proximity["initial_cost"])
        final_cost = float(proximity["final_cost"])
        gap5_budget = int(gap5["added_edges"])
        proximity_edges = int(proximity["added_edges"])
    except (KeyError, TypeError, ValueError):
        return {
            "basic_passed": False,
            "selector_superiority": False,
            "reason": "missing or invalid comparison values",
        }
    finite = np.isfinite([
        short_ate, gap5_ate, proximity_ate, initial_cost, final_cost
    ]).all()
    basic = bool(
        finite
        and proximity.get("status") == "evaluated"
        and proximity.get("global_status") == "refined"
        and proximity.get("anchor_preserved") is True
        and final_cost < initial_cost
        and proximity_ate < short_ate
    )
    superior = bool(
        basic
        and proximity_ate <= gap5_ate
        and proximity_edges <= gap5_budget
    )
    return {
        "basic_passed": basic,
        "selector_superiority": superior,
        "short_ate": short_ate,
        "gap5_ate": gap5_ate,
        "proximity_ate": proximity_ate,
        "proximity_vs_short_percent": (proximity_ate / short_ate - 1) * 100,
        "proximity_vs_gap5_percent": (proximity_ate / gap5_ate - 1) * 100,
        "gap5_edge_budget": gap5_budget,
        "proximity_edges": proximity_edges,
    }


def write_comparison_artifacts(folder, rows):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    (folder / "dynamic_covisibility_metrics.json").write_text(
        json.dumps(rows, indent=2, allow_nan=False), encoding="utf-8"
    )
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (folder / "dynamic_covisibility_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    gate = dynamic_covisibility_gate(rows)
    (folder / "dynamic_covisibility_gate.json").write_text(
        json.dumps(gate, indent=2, allow_nan=False), encoding="utf-8"
    )
    return gate


def _run(command, *, allowed=(0,), env=None):
    result = subprocess.run(command, cwd=ROOT, env=env)
    if result.returncode not in allowed:
        raise subprocess.CalledProcessError(result.returncode, command)
    return result.returncode


def _new_run_root(result_root):
    name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    root = Path(result_root).resolve() / name
    root.mkdir(parents=True, exist_ok=False)
    return root


def _run_online_source(args, run_root, start, end):
    source_root = run_root / "source"
    data_config = ROOT / f"Config/Sequence/EuRoC_{args.sequence}_local.yaml"
    command = [
        sys.executable, str(ROOT / "MACVO.py"),
        "--odom", str(args.odom_config.resolve()),
        "--data", str(data_config),
        "--resultRoot", str(source_root),
        "--seed", str(args.seed),
        "--seq_from", str(start),
        "--noeval", "--timing",
    ]
    if end is not None:
        command.extend(["--seq_to", str(end)])
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("MKL_NUM_THREADS", "4")
    _run(command, env=env)
    return discover_result_space(source_root)


def _ensure_fixed_factors(space):
    path = space / "long_factors_gap5_10.npz"
    if path.is_file():
        load_factor_archive(path)
        return path
    _run([
        sys.executable,
        str(ROOT / "Scripts/Experiment/GenerateLongRangeICP.py"),
        "--space", str(space),
    ], allowed=(0, 2))
    if not path.is_file():
        raise RuntimeError("fixed gap factor generator did not write its archive")
    return path


def _generate_proximity(space):
    returncode = _run([
        sys.executable,
        str(ROOT / "Scripts/Experiment/GenerateProximityICP.py"),
        "--space", str(space),
    ], allowed=(0, 2))
    path = space / "proximity_factors.npz"
    if not path.is_file():
        raise RuntimeError("proximity generator did not write its archive")
    return path, returncode == 0


def _refine(space, tag, factor_path=None, kind=None):
    command = [
        sys.executable,
        str(ROOT / "Scripts/Experiment/RefineGlobalPoseICP.py"),
        "--space", str(space),
        "--output-tag", tag,
    ]
    if factor_path is not None:
        command.extend([
            "--long-factors", factor_path.name,
            "--edge-kinds", kind,
        ])
    _run(command)


def _build_rows(space, manifest, fixed_archive, proximity_archive, proximity_generated):
    common = {
        "sequence": manifest["sequence"],
        "frame_from": manifest["frame_from"],
        "frame_to": manifest["frame_to"],
        "seed": manifest["seed"],
        "space": str(space),
        "source_factor_sha256": manifest["source_factor_sha256"],
        "source_pose_sha256": manifest["source_pose_sha256"],
    }
    rows = [common | {
        "mode": "before_global",
        "added_edges": 0,
        **evaluate_pose_file(space, space / "poses_before_global.npy"),
    }]
    added = {
        "short": 0,
        "gap5": len(filter_factor_archive(fixed_archive, {"gap5"}).edges),
        "proximity": len(proximity_archive.edges),
    }
    for mode in ("short", "gap5"):
        diagnostics = json.loads(
            (space / f"global_{mode}_diagnostics.json").read_text()
        )
        metrics = json.loads(
            (space / f"global_{mode}_metrics.json").read_text()
        )
        rows.append(common | {
            "mode": mode,
            "added_edges": added[mode],
            "global_status": diagnostics.get("status"),
            "anchor_preserved": diagnostics.get("anchor_preserved"),
            "edges": diagnostics.get("edges"),
            "observations": diagnostics.get("observations"),
            "initial_cost": diagnostics.get("initial_cost"),
            "final_cost": diagnostics.get("final_cost"),
            "solver_seconds": diagnostics.get("seconds"),
            **metrics,
        })
    if proximity_generated:
        diagnostics = json.loads(
            (space / "global_proximity_diagnostics.json").read_text()
        )
        metrics = json.loads(
            (space / "global_proximity_metrics.json").read_text()
        )
        proximity_row = {
            "global_status": diagnostics.get("status"),
            "anchor_preserved": diagnostics.get("anchor_preserved"),
            "edges": diagnostics.get("edges"),
            "observations": diagnostics.get("observations"),
            "initial_cost": diagnostics.get("initial_cost"),
            "final_cost": diagnostics.get("final_cost"),
            "solver_seconds": diagnostics.get("seconds"),
            **metrics,
        }
    else:
        proximity_row = {
            "status": "unavailable",
            "global_status": "no_proximity_edges",
            "anchor_preserved": None,
            "RMSE_ATE": None,
            "RMSE_RTE": None,
            "RMSE_ROE": None,
            "RMSE_RPE": None,
            "initial_cost": None,
            "final_cost": None,
        }
    rows.append(common | {
        "mode": "proximity",
        "added_edges": added["proximity"],
        **proximity_row,
    })
    return rows


def verify_comparison_space(root):
    manifests = sorted(Path(root).rglob("comparison_manifest.json"))
    if len(manifests) != 1:
        raise RuntimeError("Expected exactly one comparison manifest")
    manifest = json.loads(manifests[0].read_text())
    space = Path(manifest["source_space"])
    for filename, key in (
        ("global_factors.npz", "source_factor_sha256"),
        ("long_factors_gap5_10.npz", "fixed_factor_sha256"),
        ("proximity_factors.npz", "proximity_factor_sha256"),
        ("poses.npy", "source_pose_sha256"),
    ):
        if file_sha256(space / filename) != manifest[key]:
            raise RuntimeError(f"SHA-256 mismatch for {filename}")
    source = load_factor_archive(space / "global_factors.npz")
    proximity = load_factor_archive(space / "proximity_factors.npz")
    if source.metadata["source_id"] != proximity.metadata["source_id"]:
        raise RuntimeError("proximity factors use a different source trajectory")
    before = np.load(space / "poses_before_global.npy")
    modes = ["short", "gap5"]
    if manifest["proximity_generated"]:
        modes.append("proximity")
    for mode in modes:
        candidate = np.load(space / f"poses_global_{mode}.npy")
        if not np.array_equal(candidate[:, 0], before[:, 0]):
            raise RuntimeError(f"{mode} timestamps differ")
        if not np.array_equal(candidate[0], before[0]):
            raise RuntimeError(f"{mode} changed the fixed anchor")
    return {
        "status": "verified",
        "source_space": str(space),
        "poses": len(before),
        "proximity_edges": len(proximity.edges),
        "anchor_preserved": True,
    }


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.verify_space is not None:
        print(json.dumps(
            verify_comparison_space(args.verify_space), indent=2,
            allow_nan=False))
        return 0
    run_root = _new_run_root(args.result_root)
    segment_sha256 = None
    if args.space is None:
        start, end, segment_sha256 = resolve_sequence_bounds(args)
        space = _run_online_source(args, run_root, start, end)
    else:
        space = discover_result_space(args.space)
    required = [
        "poses.npy", "poses_before_global.npy", "global_factors.npz",
        "ref_poses.npy",
    ]
    missing = [name for name in required if not (space / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Source result is missing artifacts: {missing}")
    source_pose_sha256 = file_sha256(space / "poses.npy")
    fixed_path = _ensure_fixed_factors(space)
    proximity_path, proximity_generated = _generate_proximity(space)
    _refine(space, "short")
    _refine(space, "gap5", fixed_path, "gap5")
    if proximity_generated:
        _refine(space, "proximity", proximity_path, "proximity")
    source_metadata = source_run_metadata(space)
    manifest = source_metadata | {
        "source_space": str(space.resolve()),
        "source_factor_sha256": file_sha256(space / "global_factors.npz"),
        "fixed_factor_sha256": file_sha256(fixed_path),
        "proximity_factor_sha256": file_sha256(proximity_path),
        "source_pose_sha256": source_pose_sha256,
        "proximity_generated": proximity_generated,
        "segment_json_sha256": segment_sha256,
    }
    (run_root / "comparison_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8"
    )
    fixed_archive = load_factor_archive(fixed_path)
    proximity_archive = load_factor_archive(proximity_path)
    rows = _build_rows(
        space, manifest, fixed_archive, proximity_archive, proximity_generated
    )
    gate = write_comparison_artifacts(run_root, rows)
    verification = verify_comparison_space(run_root)
    (run_root / "verification.json").write_text(
        json.dumps(verification, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"Comparison results: {run_root}")
    print(json.dumps(gate, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
