import unittest

import pypose as pp
import torch

from Module.Optimization.GlobalPoseICP import (
    build_sparse_system,
    optimize_global_pose_graph,
)
from Module.Optimization.WindowICP import Edge, optimize_window


def global_fixture(count=12):
    torch.manual_seed(23)
    tangent = torch.zeros(count, 6, dtype=torch.float64)
    tangent[:, 0] = torch.arange(count) * 0.08
    tangent[:, 4] = torch.arange(count) * 0.012
    truth = pp.se3(tangent).Exp()
    world = torch.randn(40, 3, dtype=torch.float64)
    world[:, 0] += 4
    covariance = (
        torch.eye(3, dtype=torch.float64).expand(40, 3, 3).clone() * 0.001
    )
    edges = []
    for b in range(1, count):
        for gap in (1, 2):
            a = b - gap
            if a >= 0:
                edges.append(
                    Edge(
                        a,
                        b,
                        truth[a].Inv().Act(world),
                        truth[b].Inv().Act(world),
                        covariance,
                        covariance,
                    )
                )
    noise = torch.randn(count, 6, dtype=torch.float64) * 0.01
    noise[0] = 0
    initial = pp.se3(noise).Exp() @ truth
    return truth, initial.tensor(), edges


class GlobalPoseICPTests(unittest.TestCase):
    def test_diagnostics_count_long_edge_gaps(self):
        truth, initial, edges = global_fixture()
        covariance = torch.eye(3, dtype=torch.float64)[None] * 0.001
        world = torch.tensor([[4.0, 0.2, -0.1]], dtype=torch.float64)
        edges.append(
            Edge(
                0,
                10,
                truth[0].Inv().Act(world),
                truth[10].Inv().Act(world),
                covariance,
                covariance,
            )
        )

        result = optimize_global_pose_graph(initial, edges, max_iters=15)

        self.assertEqual(result.diagnostics["edge_gaps"]["10"], 1)

    def test_recovers_more_than_five_poses_and_preserves_anchor(self):
        truth, initial, edges = global_fixture()

        result = optimize_global_pose_graph(initial, edges, max_iters=15)

        self.assertEqual(result.diagnostics["status"], "refined")
        self.assertTrue(torch.equal(result.poses[0], initial[0]))
        error = (pp.SE3(result.poses).Inv() @ truth).Log().tensor().norm(dim=-1)
        self.assertLess(error.max().item(), 1e-5)
        self.assertLess(
            result.diagnostics["final_cost"], result.diagnostics["initial_cost"]
        )

    def test_matches_dense_solver_for_five_pose_graph(self):
        _, initial, edges = global_fixture(5)

        sparse = optimize_global_pose_graph(initial, edges, max_iters=10)
        dense = optimize_window(initial, list(range(5)), edges, max_iters=10)

        self.assertEqual(sparse.diagnostics["status"], "refined")
        self.assertTrue(
            torch.allclose(sparse.poses, dense.poses, atol=1e-7, rtol=1e-6)
        )

    def test_disconnected_graph_preserves_input(self):
        _, initial, edges = global_fixture()

        result = optimize_global_pose_graph(initial, edges[:1], max_iters=5)

        self.assertEqual(result.diagnostics["status"], "not_refined")
        self.assertTrue(torch.equal(result.poses, initial))
        self.assertIn("connected", result.diagnostics["reason"])

    def test_sparse_system_has_global_pose_dimension(self):
        blocks = {
            (0, 0): torch.eye(6, dtype=torch.float64),
            (0, 1): torch.ones(6, 6, dtype=torch.float64),
            (1, 0): torch.ones(6, 6, dtype=torch.float64),
            (1, 1): torch.eye(6, dtype=torch.float64) * 2,
        }

        system = build_sparse_system(blocks, pose_variables=2)

        self.assertEqual(system.shape, (12, 12))
        self.assertLessEqual(system.nnz, 4 * 36)


if __name__ == "__main__":
    unittest.main()
