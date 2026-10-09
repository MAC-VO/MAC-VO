import unittest

import pypose as pp
import torch

from Module.Optimization.PoseGraph import (
    PoseGraphFactor,
    build_pose_graph_sparse,
    effective_information,
    optimize_pose_graph,
)


def truth_poses(count=10):
    tangent = torch.zeros(count, 6, dtype=torch.float64)
    tangent[:, 0] = torch.arange(count, dtype=torch.float64) * 0.1
    return pp.se3(tangent).Exp().tensor()


def factor(poses, a, b, kind="adjacent", measurement=None, confidence=1.0):
    if measurement is None:
        measurement = (
            pp.SE3(poses[a]).Inv() @ pp.SE3(poses[b])
        ).tensor()
    return PoseGraphFactor(
        a=a,
        b=b,
        measurement=measurement,
        information=torch.eye(6, dtype=torch.float64) * 100,
        kind=kind,
        confidence=confidence,
        observation_count=100,
    )


class PoseGraphTests(unittest.TestCase):
    def test_chain_recovers_all_poses_and_preserves_anchor(self):
        truth = truth_poses()
        factors = tuple(
            factor(truth, index - 1, index)
            for index in range(1, len(truth))
        )
        noise = torch.zeros(len(truth), 6, dtype=torch.float64)
        noise[1:, 1] = torch.linspace(0.01, 0.1, len(truth) - 1)
        initial = (pp.se3(noise).Exp() @ pp.SE3(truth)).tensor()

        result = optimize_pose_graph(initial, factors, max_iters=15)

        self.assertEqual(result.diagnostics["status"], "refined")
        self.assertTrue(torch.equal(result.poses[0], initial[0]))
        error = (
            pp.SE3(result.poses).Inv() @ pp.SE3(truth)
        ).Log().tensor().norm(dim=-1)
        self.assertLess(error.max().item(), 1e-5)

    def test_loop_distributes_correction_across_drifting_chain(self):
        truth = truth_poses()
        biased = truth.clone()
        biased[:, 0] = torch.arange(len(truth), dtype=torch.float64) * 0.11
        factors = [
            factor(biased, index - 1, index)
            for index in range(1, len(truth))
        ]
        factors.append(factor(truth, 0, len(truth) - 1, kind="loop"))

        result = optimize_pose_graph(biased, factors, max_iters=20)

        before = (pp.SE3(biased).Inv() @ pp.SE3(truth)).Log().tensor().norm(dim=-1)
        after = (pp.SE3(result.poses).Inv() @ pp.SE3(truth)).Log().tensor().norm(dim=-1)
        self.assertLess(after[-1].item(), before[-1].item())
        self.assertGreater((result.poses[4] - biased[4]).abs().max().item(), 1e-4)

    def test_false_loop_is_switch_suppressed(self):
        truth = truth_poses()
        factors = [
            factor(truth, index - 1, index)
            for index in range(1, len(truth))
        ]
        wrong = pp.SE3(factor(truth, 0, 9, kind="loop").measurement).tensor().clone()
        wrong[0] += 20
        factors.append(factor(truth, 0, 9, kind="loop", measurement=wrong))

        result = optimize_pose_graph(truth, factors, max_iters=10)

        self.assertLess(result.switches[-1], 0.25)
        self.assertTrue(torch.allclose(result.poses, truth, atol=1e-4))

    def test_diagnostics_report_final_loop_switches_with_endpoints(self):
        truth = truth_poses(4)
        factors = [
            factor(truth, index - 1, index)
            for index in range(1, len(truth))
        ]
        factors.append(factor(truth, 0, 3, kind="loop"))

        result = optimize_pose_graph(truth, factors, max_iters=2)

        self.assertEqual(result.diagnostics["loop_switches"], [{
            "a": 0,
            "b": 3,
            "switch": result.switches[-1],
        }])

    def test_skip_information_cap_scales_information(self):
        truth = truth_poses(3)
        skip = factor(truth, 0, 2, kind="skip2")

        information = effective_information(skip, skip_information_cap=0.5)

        self.assertTrue(torch.equal(information, skip.information * 0.5))

    def test_sparse_system_uses_six_variables_per_non_anchor_pose(self):
        blocks = {
            (0, 0): torch.eye(6, dtype=torch.float64),
            (0, 1): torch.ones(6, 6, dtype=torch.float64),
            (1, 0): torch.ones(6, 6, dtype=torch.float64),
            (1, 1): torch.eye(6, dtype=torch.float64),
        }

        system = build_pose_graph_sparse(blocks, pose_variables=2)

        self.assertEqual(system.shape, (12, 12))
        self.assertLessEqual(system.nnz, 144)


if __name__ == "__main__":
    unittest.main()
