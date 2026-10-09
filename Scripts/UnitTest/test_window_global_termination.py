import unittest
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pypose as pp
import torch

from Module.Map import FrameNode, MatchObs, PointNode, VisualMap
from Module.Optimization.FactorArchive import load_factor_archive
from Module.Optimization.GlobalPoseICP import GlobalPoseResult
from Module.Optimization.WindowICP import Edge, EdgeWindow
from Odometry.WindowMACVO import WindowMACVO


def make_edge(a, b):
    points = torch.tensor([[3.0, 0.0, 0.0]], dtype=torch.float64)
    covariance = torch.eye(3, dtype=torch.float64).unsqueeze(0) * 0.001
    return Edge(a, b, points, points.clone(), covariance, covariance.clone())


def make_system():
    system = WindowMACVO.__new__(WindowMACVO)
    system.graph = VisualMap()
    for i in range(3):
        pose = pp.identity_SE3(1)
        pose[0, 0] = float(i)
        system.graph.frames.push(
            FrameNode.init(
                {
                    "pose": pose,
                    "K": torch.eye(3)[None],
                    "baseline": torch.tensor([0.1]),
                    "T_BS": pp.identity_SE3(1),
                    "need_interp": torch.tensor([False]),
                    "time_ns": torch.tensor([i]),
                }
            )
        )

    points = torch.tensor(
        [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]]
    )
    covariance = torch.eye(3, dtype=torch.float64).repeat(3, 1, 1) * 0.001
    point_ids = system.graph.points.push(
        PointNode.init(
            {
                "pos_Tw": points.clone(),
                "cov_Tw": covariance.clone(),
                "color": torch.zeros(3, 3, dtype=torch.uint8),
            }
        )
    )
    fields = {
        key: torch.zeros((3,) + tuple(value.shape[1:]), dtype=value.dtype)
        for key, value in system.graph.match.data.items()
    }
    match_ids = system.graph.match.push(MatchObs.init(fields))
    system.graph.match2point.set(match_ids, point_ids)
    system.graph.match2frame1.set(match_ids, torch.arange(3))
    for i in range(3):
        system.graph.frame2match.add(
            torch.tensor([i]), torch.tensor([i]), torch.tensor([1])
        )

    system.global_refine = True
    system.global_iterations = 5
    system.global_huber_delta = 3.0
    system.inactive_edges = {(0, 1): make_edge(0, 1)}
    system.edge_window = EdgeWindow(5)
    system.edge_window.add(make_edge(1, 2))
    system.global_record = {"status": "pending"}
    system.retained_tensor_bytes = 0
    system.inactive_edge_count = 0
    return system, points, covariance


class WindowGlobalTerminationTests(unittest.TestCase):
    def test_termination_snapshots_interpolated_poses_before_global_writeback(self):
        system, _, _ = make_system()
        interpolated = system.graph.frames.data["pose"].tensor.double().clone()
        refined = interpolated.clone()
        refined[1, 0] += 0.2
        system.terminated = False
        system.Optimizer = SimpleNamespace(optimize_res=None, terminate=Mock())
        system.MapRefiner = SimpleNamespace(elaborate_map=Mock())
        system.frame_cache = {}

        with patch.object(system, "_run_global_refinement") as refine:
            refine.side_effect = lambda before: system.graph.frames.data[
                "pose"
            ].tensor.copy_(refined.float())
            system.terminate()

        self.assertTrue(torch.equal(
            system._factor_snapshot.initial_sensor_poses, interpolated
        ))

    def test_save_diagnostics_writes_reusable_artifacts(self):
        system, _, _ = make_system()
        system.skip_matching = True
        system.window_huber_delta = 3.0
        system.window_records = []
        poses = system.graph.frames.data["pose"].tensor.double().clone()
        system._capture_factor_snapshot(poses)

        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "run_provenance.json").write_text(
                '{"git_commit":"abc"}', encoding="utf-8"
            )
            system.save_window_diagnostics(directory)
            before = np.load(Path(directory, "poses_before_global.npy"))
            archive = load_factor_archive(Path(directory, "global_factors.npz"))
            payload = json.loads(
                Path(directory, "global_refinement.json").read_text()
            )

        self.assertEqual(before.shape, (3, 8))
        self.assertTrue(torch.equal(archive.initial_sensor_poses, poses))
        self.assertEqual(payload["artifacts"]["status"], "saved")
        self.assertEqual(payload["artifacts"]["git_commit"], "abc")
        self.assertIsNone(system._factor_snapshot)

    def test_serialization_failure_is_reported_and_keeps_snapshot(self):
        system, _, _ = make_system()
        system.skip_matching = True
        system.window_huber_delta = 3.0
        system.window_records = []
        system._capture_factor_snapshot(
            system.graph.frames.data["pose"].tensor.double()
        )

        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "Odometry.WindowMACVO.save_factor_archive",
                side_effect=OSError("disk full"),
            ):
                system.save_window_diagnostics(directory)
            payload = json.loads(
                Path(directory, "global_refinement.json").read_text()
            )

            self.assertTrue(Path(directory, "window_diagnostics.json").is_file())
            self.assertFalse(Path(directory, "global_factors.npz").exists())
        self.assertEqual(payload["artifacts"]["status"], "failed")
        self.assertIn("disk full", payload["artifacts"]["reason"])
        self.assertIsNotNone(system._factor_snapshot)

    def test_disabled_refinement_writes_no_factor_artifacts(self):
        system, _, _ = make_system()
        system.global_refine = False
        system.skip_matching = True
        system.window_huber_delta = 3.0
        system.window_records = []

        with tempfile.TemporaryDirectory() as directory:
            system.save_window_diagnostics(directory)

            self.assertFalse(Path(directory, "poses_before_global.npy").exists())
            self.assertFalse(Path(directory, "global_factors.npz").exists())

    def test_saved_diagnostics_include_global_summary(self):
        system, _, _ = make_system()
        system.skip_matching = True
        system.window_huber_delta = 3.0
        system.window_records = []
        system.inactive_edge_count = 1
        system.retained_tensor_bytes = 384
        system.global_record = {"status": "refined", "seconds": 0.25}

        with tempfile.TemporaryDirectory() as directory:
            system.save_window_diagnostics(directory)
            payload = json.loads(
                Path(directory, "window_diagnostics.json").read_text()
            )

        self.assertTrue(payload["global_refine"])
        self.assertEqual(payload["inactive_edges"], 1)
        self.assertEqual(payload["retained_tensor_bytes"], 384)
        self.assertEqual(payload["global_refinement"]["status"], "refined")

    def test_successful_result_updates_poses_points_and_covariances(self):
        system, points, covariance = make_system()
        before = system.graph.frames.data["pose"].tensor.double().clone()
        delta = torch.zeros(3, 6, dtype=torch.float64)
        delta[1, 0] = 0.1
        delta[2, 1] = -0.2
        target = (pp.se3(delta).Exp() @ pp.SE3(before)).tensor()
        target[0] = before[0]
        result = GlobalPoseResult(
            target, {"status": "refined", "anchor_preserved": True}
        )

        with patch(
            "Odometry.WindowMACVO.optimize_global_pose_graph",
            return_value=result,
        ):
            system._run_global_refinement(before)

        self.assertTrue(
            torch.allclose(system.graph.frames.data["pose"].tensor.double(), target)
        )
        for i in range(3):
            correction = pp.SE3(target[i]) @ pp.SE3(before[i]).Inv()
            expected_point = correction.Act(points[i].double())
            expected_covariance = (
                correction.rotation().matrix()
                @ covariance[i]
                @ correction.rotation().matrix().T
            )
            self.assertTrue(
                torch.allclose(
                    system.graph.points.data["pos_Tw"][i].double(),
                    expected_point,
                    atol=1e-6,
                )
            )
            self.assertTrue(
                torch.allclose(
                    system.graph.points.data["cov_Tw"][i], expected_covariance
                )
            )

    def test_failed_result_preserves_interpolated_state(self):
        system, points, covariance = make_system()
        before = system.graph.frames.data["pose"].tensor.double().clone()
        result = GlobalPoseResult(
            before.clone(), {"status": "not_refined", "reason": "singular"}
        )

        with patch(
            "Odometry.WindowMACVO.optimize_global_pose_graph",
            return_value=result,
        ):
            system._run_global_refinement(before)

        self.assertTrue(
            torch.equal(system.graph.frames.data["pose"].tensor.double(), before)
        )
        self.assertTrue(
            torch.equal(system.graph.points.data["pos_Tw"], points)
        )
        self.assertTrue(
            torch.equal(system.graph.points.data["cov_Tw"], covariance)
        )

    def test_disabled_result_does_not_call_solver(self):
        system, points, _ = make_system()
        system.global_refine = False
        before = system.graph.frames.data["pose"].tensor.double().clone()

        with patch("Odometry.WindowMACVO.optimize_global_pose_graph") as solver:
            system._run_global_refinement(before)

        solver.assert_not_called()
        self.assertEqual(system.global_record, {"status": "disabled"})
        self.assertEqual(system.retained_tensor_bytes, 0)
        self.assertTrue(torch.equal(system.graph.points.data["pos_Tw"], points))


if __name__ == "__main__":
    unittest.main()
