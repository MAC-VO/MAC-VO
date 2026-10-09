"""Sequential, fresh-run comparisons; never select a result by 'latest'."""
import argparse
import copy
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Utility.Config import load_config
from Evaluation.EvalSeq import EvaluateSequences


MODE_CONFIGS = {
    "twoframe": ROOT / "Config/Experiment/MACVO/MACVO_Fast_ICP_local.yaml",
    "window_adjacent": ROOT / "Config/Experiment/MACVO/MACVO_Fast_WindowICP.yaml",
    "window_skip": ROOT / "Config/Experiment/MACVO/MACVO_Fast_WindowICP.yaml",
    "window_global": ROOT / "Config/Experiment/MACVO/MACVO_Fast_WindowICP_Global.yaml",
}


def append_window_diagnostics(row, space, mode):
    if mode == "twoframe":
        return row
    payload = json.loads((space / "window_diagnostics.json").read_text())
    diagnostics = payload["windows"]
    if any(
        record["cached_frames"] > 5 or record["cached_edges"] > 7
        for record in diagnostics
    ):
        raise RuntimeError("Window cache exceeded its bound")
    row["unrefined_windows"] = sum(
        record["status"] != "refined" for record in diagnostics
    )
    row["refine_mean_ms"] = float(
        np.mean([record.get("seconds", 0) * 1000 for record in diagnostics])
    )
    row["skip_mean_ms"] = float(
        np.mean([record["skip_seconds"] * 1000 for record in diagnostics])
    )
    if mode == "window_global":
        global_record = payload["global_refinement"]
        row.update({
            "inactive_edges": payload["inactive_edges"],
            "retained_tensor_bytes": payload["retained_tensor_bytes"],
            "global_status": global_record["status"],
            "global_seconds": global_record.get("seconds"),
            "global_initial_cost": global_record.get("initial_cost"),
            "global_final_cost": global_record.get("final_cost"),
        })
    return row


def run_and_tee(command, cwd, env, log_path, terminal=None):
    """Stream a child's raw stdout to the terminal and its per-run log."""
    terminal = sys.stdout.buffer if terminal is None else terminal
    with Path(log_path).open("wb") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
        )
        assert process.stdout is not None
        with process.stdout:
            while True:
                chunk = process.stdout.read(8192)
                if not chunk:
                    break
                terminal.write(chunk)
                terminal.flush()
                log.write(chunk)
                log.flush()
        returncode = process.wait()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", nargs="+", default=["MH01"])
    parser.add_argument("--seq_from", type=int, default=0)
    parser.add_argument("--seq_to", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--modes", nargs="+", choices=list(MODE_CONFIGS),
                        default=["twoframe", "window_adjacent", "window_skip"])
    parser.add_argument("--resultRoot", type=Path, default=ROOT / "Results/WindowICP_comparison")
    args = parser.parse_args()
    if args.seq_from < 0 or (args.seq_to is not None and args.seq_to <= args.seq_from):
        parser.error("Expected 0 <= seq_from < seq_to")
    run_dir = args.resultRoot.resolve() / (datetime.now().strftime("%Y%m%d_%H%M%S")+"_"+uuid.uuid4().hex[:6])
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"Comparison results: {run_dir}", flush=True)
    rows = []
    for seq in args.sequences:
        data_path = ROOT / f"Config/Sequence/EuRoC_{seq}_local.yaml"
        data, _ = load_config(data_path)
        if not Path(data.args.root, "cam0/sensor.yaml").is_file():
            raise FileNotFoundError(f"Dataset not accessible: {data.args.root}")
        reference_times = None
        for mode in args.modes:
            folder = run_dir / seq / mode
            folder.mkdir(parents=True, exist_ok=False)
            _, config = load_config(MODE_CONFIGS[mode])
            config = copy.deepcopy(config)
            if mode == "window_adjacent":
                config["Odometry"]["args"]["skip_matching"] = False
                config["Odometry"]["name"] = "MACVO-Fast-WindowICP5-Adjacent"
            config_path = folder / "input.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            command = [sys.executable, str(ROOT/"MACVO.py"), "--odom", str(config_path),
                       "--data", str(data_path), "--resultRoot", str(folder/"outputs"),
                       "--seed", str(args.seed), "--seq_from", str(args.seq_from), "--noeval", "--timing"]
            if args.seq_to is not None:
                command += ["--seq_to", str(args.seq_to)]
            env = os.environ.copy()
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            env.setdefault("OMP_NUM_THREADS", "4")
            env.setdefault("MKL_NUM_THREADS", "4")
            print(f"Running {seq} {mode} ... log: {folder/'run.log'}", flush=True)
            run_and_tee(command, ROOT, env, folder/"run.log")
            spaces = list((folder/"outputs").glob("*/*/run_provenance.json"))
            if len(spaces) != 1:
                raise RuntimeError(f"Expected exactly one run in {folder}")
            provenance = json.loads(spaces[0].read_text())
            if provenance["status"] != "complete":
                raise RuntimeError(f"Incomplete run: {spaces[0].parent}")
            space = spaces[0].parent
            poses = np.load(space/"poses.npy")
            if reference_times is None:
                reference_times = poses[:, 0]
            elif not np.array_equal(reference_times, poses[:, 0]):
                raise RuntimeError("Comparison frame timestamps differ")
            header, result = EvaluateSequences([str(space)], correct_scale=False)
            metrics = next(row for row in result if row[0] != "Average")
            if any(value is None for value in metrics[1:]):
                raise RuntimeError(f"Evaluation failed: {space}")
            elapsed = json.loads((space/"elapsed_time.json").read_text())
            timer = elapsed["CPU_ElapsedTime"]["Odom_Runtime"]
            row = {"sequence": seq, "mode": mode, "frames": len(poses),
                   "space": str(space), "git_commit": provenance["git_commit"],
                   "seed": args.seed, "runtime_mean_ms": float(np.mean(timer)),
                   **dict(zip(header[1:], metrics[1:]))}
            append_window_diagnostics(row, space, mode)
            rows.append(row)
            (run_dir/"metrics.json").write_text(json.dumps(rows, indent=2, allow_nan=False))
            columns = list(dict.fromkeys(key for row in rows for key in row))
            with (run_dir/"metrics.csv").open("w") as out:
                writer = csv.DictWriter(out, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
            print(f"Finished {seq} {mode}: ATE={row['RMSE_ATE']:.6f} m, "
                  f"RTE={row['RMSE_RTE']:.6f} m/frame, ROE={row['RMSE_ROE']:.6f} deg/frame", flush=True)
    print(f"All requested comparisons complete: {run_dir/'metrics.csv'}", flush=True)


if __name__ == "__main__":
    main()
