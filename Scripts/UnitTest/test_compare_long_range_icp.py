import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pypose as pp
import torch
import yaml

from Module.Optimization.FactorArchive import (
    FactorArchive,
    save_factor_archive,
    sensor_to_body_trajectory,
)
from Module.Optimization.WindowICP import Edge
from Scripts.Experiment.CompareLongRangeICP import (
    build_parser,
    discover_result_space,
    stage_one_passed,
    source_run_metadata,
    verify_comparison_space,
    write_comparison_artifacts,
)
from Scripts.Experiment.GenerateLongRangeICP import file_sha256


def comparison_rows(long_ate=0.49):
    common = {
        "source_factor_sha256": "same",
        "source_pose_sha256": "pose",
        "status": "evaluated",
    }
    return [
        common | {"mode": "before_global", "RMSE_ATE": 0.55},
        common | {
            "mode": "short",
            "RMSE_ATE": 0.50,
            "global_status": "refined",
            "anchor_preserved": True,
            "initial_cost": 10.0,
            "final_cost": 9.0,
        },
        common | {
            "mode": "gap5",
            "RMSE_ATE": 0.495,
            "global_status": "refined",
            "anchor_preserved": True,
            "initial_cost": 12.0,
            "final_cost": 10.0,
        },
        common | {
            "mode": "gap5_10",
            "RMSE_ATE": long_ate,
            "global_status": "refined",
            "anchor_preserved": True,
            "initial_cost": 14.0,
            "final_cost": 11.0,
        },
    ]


def archive_fixture():
    poses = pp.identity_SE3(11, dtype=torch.float64).tensor()
    times = np.arange(11, dtype=np.int64) + 1000
    extrinsic = pp.identity_SE3(dtype=torch.float64).tensor()
    covariance = torch.eye(3, dtype=torch.float64)[None] * 0.001
    points = torch.tensor([[3.0, 0.0, 0.0]], dtype=torch.float64)
    short_edge = Edge(0, 1, points, points.clone(), covariance, covariance.clone())
    long_edge = Edge(0, 5, points, points.clone(), covariance, covariance.clone())
    common = {
        "initial_sensor_poses": poses,
        "time_ns": times,
        "T_BS": extrinsic,
    }
    return (
        FactorArchive(
            **common,
            edges=(short_edge,),
            edge_kinds=("adjacent",),
            metadata={},
        ),
        FactorArchive(
            **common,
            edges=(long_edge,),
            edge_kinds=("gap5",),
            metadata={},
        ),
    )


class CompareLongRangeICPTests(unittest.TestCase):
    def test_stage_gate_requires_gap5_10_ate_below_short(self):
        self.assertTrue(stage_one_passed(comparison_rows(0.49)))
        self.assertFalse(stage_one_passed(comparison_rows(0.51)))

    def test_stage_gate_requires_valid_solver_result(self):
        rows = comparison_rows(0.49)
        rows[-1]["anchor_preserved"] = False
        self.assertFalse(stage_one_passed(rows))
        rows = comparison_rows(0.49)
        rows[-1]["final_cost"] = rows[-1]["initial_cost"]
        self.assertFalse(stage_one_passed(rows))

    def test_write_comparison_artifacts_keeps_json_csv_and_gate_consistent(self):
        rows = comparison_rows(0.49)
        with tempfile.TemporaryDirectory() as directory:
            gate = write_comparison_artifacts(directory, rows)
            saved_rows = json.loads(
                Path(directory, "long_range_metrics.json").read_text()
            )
            with Path(directory, "long_range_metrics.csv").open() as stream:
                csv_rows = list(csv.DictReader(stream))
            saved_gate = json.loads(
                Path(directory, "stage_gate.json").read_text()
            )

        self.assertEqual(saved_rows, rows)
        self.assertEqual(len(csv_rows), len(rows))
        self.assertEqual(saved_gate, gate)
        self.assertTrue(saved_gate["passed"])

    def test_discover_result_space_rejects_incomplete_or_multiple_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            failed = root / "failed"
            failed.mkdir()
            (failed / "run_provenance.json").write_text('{"status":"failed"}')
            with self.assertRaisesRegex(RuntimeError, "complete"):
                discover_result_space(root)
            (failed / "run_provenance.json").write_text('{"status":"complete"}')
            second = root / "second"
            second.mkdir()
            (second / "run_provenance.json").write_text('{"status":"complete"}')
            with self.assertRaisesRegex(RuntimeError, "exactly one"):
                discover_result_space(root)

    def test_verify_comparison_space_checks_hashes_timestamps_and_anchor(self):
        short, long = archive_fixture()
        trajectory = sensor_to_body_trajectory(
            short.initial_sensor_poses, short.time_ns, short.T_BS
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            space = root / "leaf"
            space.mkdir()
            save_factor_archive(space / "global_factors.npz", short)
            save_factor_archive(space / "long_factors_gap5_10.npz", long)
            np.save(space / "poses.npy", trajectory)
            np.save(space / "poses_before_global.npy", trajectory)
            for mode in ("short", "gap5", "gap5_10"):
                np.save(space / f"poses_global_{mode}.npy", trajectory)
            manifest = {
                "source_space": str(space),
                "source_factor_sha256": file_sha256(space / "global_factors.npz"),
                "long_factor_sha256": file_sha256(space / "long_factors_gap5_10.npz"),
                "source_pose_sha256": file_sha256(space / "poses.npy"),
            }
            (root / "comparison_manifest.json").write_text(json.dumps(manifest))

            result = verify_comparison_space(root)

        self.assertEqual(result["status"], "verified")
        self.assertTrue(result["anchor_preserved"])

    def test_cli_accepts_verification_mode(self):
        args = build_parser().parse_args(["--verify-space", "/tmp/result"])

        self.assertEqual(args.verify_space, Path("/tmp/result"))

    def test_source_metadata_comes_from_saved_run_not_cli_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            space = Path(directory)
            (space / "config.yaml").write_text(yaml.safe_dump({
                "Project": "MACVO@V203",
                "Data": {
                    "args": {"name": "V203", "type": "EuRoC", "args": {}},
                    "start_idx": 1095,
                    "end_idx": 1215,
                },
            }))
            (space / "run_provenance.json").write_text(json.dumps({
                "status": "complete",
                "seed": 7,
            }))

            metadata = source_run_metadata(space)

        self.assertEqual(metadata, {
            "sequence": "V203",
            "frame_from": 1095,
            "frame_to": 1215,
            "seed": 7,
        })


if __name__ == "__main__":
    unittest.main()
