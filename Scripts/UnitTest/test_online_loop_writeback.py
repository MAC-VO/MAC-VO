import unittest
from types import SimpleNamespace

import pypose as pp
import torch

from Module.Map import FrameNode, VisualMap
from Module.Optimization.AsyncPoseGraph import PoseGraphAsyncResult
from Module.Optimization.PoseGraph import PoseGraphResult
from Odometry.OnlineLoopWindowMACVO import OnlineLoopWindowMACVO


def system_fixture():
    system = OnlineLoopWindowMACVO.__new__(OnlineLoopWindowMACVO)
    system.graph = VisualMap()
    for index in range(4):
        pose = pp.identity_SE3(1)
        pose[0, 0] = index * 0.1
        system.graph.frames.push(FrameNode.init({
            "pose": pose,
            "K": torch.eye(3)[None],
            "baseline": torch.tensor([0.1]),
            "T_BS": pp.identity_SE3(1),
            "need_interp": torch.tensor([False]),
            "time_ns": torch.tensor([index]),
        }))
    system.prev_keyframe = (None, 3, None)
    system.MotionEstimator = SimpleNamespace(update=lambda pose: None)
    system.pose_graph_writebacks = 0
    system.pose_graph_results = []
    return system


class OnlineLoopWritebackTests(unittest.TestCase):
    def test_prefix_writeback_applies_last_pose_correction_to_suffix(self):
        system = system_fixture()
        old = system.graph.frames.data["pose"].tensor.double().clone()
        optimized = old[:3].clone()
        optimized[1, 1] += 0.1
        optimized[2, 1] += 0.2
        result = PoseGraphResult(
            optimized,
            [],
            {"status": "refined", "final_cost": 1.0},
        )
        asynchronous = PoseGraphAsyncResult(
            graph_version=1,
            frame_count=3,
            status="refined",
            result=result,
            reason=None,
            seconds=0.1,
        )

        accepted = system._apply_pose_graph_result(asynchronous)
        output = system.graph.frames.data["pose"].tensor.double()
        correction = pp.SE3(optimized[2]) @ pp.SE3(old[2]).Inv()
        expected_suffix = correction @ pp.SE3(old[3])

        self.assertTrue(accepted)
        self.assertTrue(torch.allclose(output[:3], optimized, atol=1e-6))
        self.assertTrue(torch.allclose(
            output[3], expected_suffix.tensor(), atol=1e-6
        ))
        self.assertEqual(system.pose_graph_writebacks, 1)

    def test_stale_result_larger_than_live_map_is_rejected(self):
        system = system_fixture()
        before = system.graph.frames.data["pose"].tensor.clone()
        result = PoseGraphResult(
            pp.identity_SE3(5).tensor(), [], {"status": "refined"}
        )
        asynchronous = PoseGraphAsyncResult(1, 5, "refined", result, None, 0.1)

        accepted = system._apply_pose_graph_result(asynchronous)

        self.assertFalse(accepted)
        self.assertTrue(torch.equal(
            system.graph.frames.data["pose"].tensor, before
        ))

    def test_failed_backend_result_preserves_live_map(self):
        system = system_fixture()
        before = system.graph.frames.data["pose"].tensor.clone()
        asynchronous = PoseGraphAsyncResult(
            1, 4, "failed", None, "solver", 0.1
        )

        accepted = system._apply_pose_graph_result(asynchronous)

        self.assertFalse(accepted)
        self.assertTrue(torch.equal(
            system.graph.frames.data["pose"].tensor, before
        ))


if __name__ == "__main__":
    unittest.main()
