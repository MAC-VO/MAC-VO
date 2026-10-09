from pathlib import Path
import tempfile
import unittest

import numpy as np
import pypose as pp
import torch

from Module.Optimization.FactorArchive import (
    FactorArchive,
    sensor_to_body_trajectory,
)
from Module.Optimization.GlobalPoseICP import (
    GlobalPoseResult,
    optimize_global_pose_graph,
)
from Module.Optimization.WindowICP import Edge
from Scripts.Experiment.RefineGlobalPoseICP import (
    build_parser,
    evaluate_pose_file,
    refine_archives,
    write_refinement_outputs,
)


def archives(count=11):
    tangent = torch.zeros(count, 6, dtype=torch.float64)
    tangent[:, 0] = torch.arange(count, dtype=torch.float64) * 0.08
    truth = pp.se3(tangent).Exp()
    noise = torch.zeros(count, 6, dtype=torch.float64)
    noise[1:, 1] = 0.01
    initial = (pp.se3(noise).Exp() @ truth).tensor()
    world = torch.tensor(
        [[3.0, 0.0, 0.0], [4.0, 0.4, -0.2], [5.0, -0.3, 0.5]],
        dtype=torch.float64,
    )
    covariance = torch.eye(3, dtype=torch.float64).repeat(3, 1, 1) * 0.001

    def edge(a, b):
        return Edge(
            a,
            b,
            truth[a].Inv().Act(world),
            truth[b].Inv().Act(world),
            covariance,
            covariance,
        )

    short_edges = tuple(edge(index - 1, index) for index in range(1, count))
    long_specs = [(0, 5, "gap5")]
    if count > 10:
        long_specs.append((0, 10, "gap10"))
    long_edges = tuple(edge(a, b) for a, b, _ in long_specs)
    common = {
        "initial_sensor_poses": initial,
        "time_ns": np.arange(count, dtype=np.int64) + 1000,
        "T_BS": pp.identity_SE3(dtype=torch.float64).tensor(),
    }
    short = FactorArchive(
        **common,
        edges=short_edges,
        edge_kinds=("adjacent",) * len(short_edges),
        metadata={"fixture": True},
    )
    long = FactorArchive(
        **common,
        edges=long_edges,
        edge_kinds=tuple(kind for _, _, kind in long_specs),
        metadata={"fixture": True},
    )
    return short, long


class OfflineGlobalRefinementTests(unittest.TestCase):
    def test_short_only_refinement_uses_saved_initial_sensor_poses(self):
        short, _ = archives()

        result = refine_archives(
            short, None, edge_kinds=None, iterations=10, huber=3.0
        )
        expected = optimize_global_pose_graph(
            short.initial_sensor_poses,
            short.edges,
            max_iters=10,
            huber_delta=3.0,
        )

        self.assertTrue(torch.allclose(
            result.poses, expected.poses, atol=1e-12, rtol=0
        ))

    def test_gap5_filter_excludes_gap10_edges(self):
        short, long = archives()

        result = refine_archives(
            short, long, edge_kinds={"gap5"}, iterations=5, huber=3.0
        )

        self.assertEqual(result.diagnostics["edge_gaps"].get("10", 0), 0)
        self.assertEqual(result.diagnostics["edge_gaps"]["5"], 1)

    def test_tagged_outputs_do_not_overwrite_source_poses(self):
        short, _ = archives()
        result = refine_archives(short, iterations=5, huber=3.0)
        metrics = {"status": "evaluated", "RMSE_ATE": 0.1}

        with tempfile.TemporaryDirectory() as directory:
            space = Path(directory)
            source_path = space / "poses.npy"
            source_path.write_bytes(b"original")
            write_refinement_outputs(
                space, "gap5", result, short, metrics
            )

            self.assertEqual(source_path.read_bytes(), b"original")
            self.assertTrue((space / "poses_global_gap5.npy").is_file())
            self.assertTrue((space / "global_gap5_diagnostics.json").is_file())
            self.assertTrue((space / "global_gap5_metrics.json").is_file())

    def test_evaluate_pose_file_reports_zero_for_identical_trajectory(self):
        short, _ = archives(count=6)
        trajectory = sensor_to_body_trajectory(
            short.initial_sensor_poses, short.time_ns, short.T_BS
        )

        with tempfile.TemporaryDirectory() as directory:
            space = Path(directory)
            np.save(space / "ref_poses.npy", trajectory)
            np.save(space / "candidate.npy", trajectory)
            metrics = evaluate_pose_file(space, space / "candidate.npy")

        self.assertEqual(metrics["status"], "evaluated")
        self.assertLess(metrics["RMSE_ATE"], 1e-7)
        self.assertLess(metrics["RMSE_RTE"], 1e-7)

    def test_evaluate_pose_file_reports_missing_reference(self):
        short, _ = archives(count=6)
        trajectory = sensor_to_body_trajectory(
            short.initial_sensor_poses, short.time_ns, short.T_BS
        )

        with tempfile.TemporaryDirectory() as directory:
            space = Path(directory)
            np.save(space / "candidate.npy", trajectory)
            metrics = evaluate_pose_file(space, space / "candidate.npy")

        self.assertEqual(metrics["status"], "unavailable")
        self.assertIn("ref_poses.npy", metrics["reason"])

    def test_invalid_output_tag_is_rejected(self):
        short, _ = archives()
        result = GlobalPoseResult(
            short.initial_sensor_poses,
            {"status": "refined"},
        )

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "output tag"):
                write_refinement_outputs(
                    directory, "../bad", result, short, {}
                )

    def test_failed_solver_writes_diagnostics_but_not_pose(self):
        short, _ = archives()
        result = GlobalPoseResult(
            short.initial_sensor_poses,
            {"status": "not_refined", "reason": "singular"},
        )

        with tempfile.TemporaryDirectory() as directory:
            space = Path(directory)
            write_refinement_outputs(
                space,
                "short",
                result,
                short,
                {"status": "unavailable", "reason": "solver failed"},
            )

            self.assertFalse((space / "poses_global_short.npy").exists())
            self.assertTrue((space / "global_short_diagnostics.json").is_file())

    def test_cli_defaults_to_short_refinement(self):
        args = build_parser().parse_args(["--space", "/tmp/result"])

        self.assertEqual(args.output_tag, "short")
        self.assertEqual(args.iterations, 5)
        self.assertEqual(args.huber, 3.0)

    def test_cli_accepts_proximity_edge_kind(self):
        args = build_parser().parse_args([
            "--space", "/tmp/result",
            "--long-factors", "proximity_factors.npz",
            "--edge-kinds", "proximity",
            "--output-tag", "proximity",
        ])

        self.assertEqual(args.edge_kinds, ["proximity"])


if __name__ == "__main__":
    unittest.main()
