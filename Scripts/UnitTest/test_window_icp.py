import unittest

import torch
import pypose as pp

from Module.Optimization.WindowICP import (
    Edge,
    EdgeWindow,
    edge_tensor_bytes,
    factor_system,
    optimize_window,
)


def fixture():
    torch.manual_seed(7)
    tangent = torch.zeros(5, 6, dtype=torch.float64)
    tangent[:, 0] = torch.arange(5) * .08
    tangent[:, 4] = torch.arange(5) * .015
    truth = pp.se3(tangent).Exp()
    world = torch.randn(30, 3, dtype=torch.float64)
    world[:, 0] += 4
    A = torch.tensor([[.02, 0, 0], [.01, .01, 0], [0, .003, .01]], dtype=torch.float64)
    cov = (A @ A.T).expand(len(world), 3, 3).clone()
    edges = []
    for b in range(1, 5):
        for gap in (1, 2):
            a = b - gap
            if a >= 0:
                edges.append(Edge(a, b, truth[a].Inv().Act(world),
                                  truth[b].Inv().Act(world), cov, cov))
    noise = torch.randn(5, 6, dtype=torch.float64) * .015
    noise[0] = 0
    initial = pp.se3(noise).Exp() @ truth
    return truth, initial, edges


class WindowICPTests(unittest.TestCase):
    def test_edge_accepts_positive_long_gap(self):
        _, _, edges = fixture()
        base = edges[0]

        edge = Edge(
            0, 10,
            base.points_a, base.points_b,
            base.cov_a, base.cov_b,
        )

        self.assertEqual((edge.a, edge.b), (0, 10))

    def test_edge_rejects_non_forward_endpoint(self):
        _, _, edges = fixture()
        base = edges[0]

        with self.assertRaisesRegex(ValueError, "strictly forward"):
            Edge(
                4, 4,
                base.points_a, base.points_b,
                base.cov_a, base.cov_b,
            )

    def test_recovers_five_poses_and_keeps_anchor(self):
        truth, initial, edges = fixture()
        result = optimize_window(initial.tensor(), list(range(5)), edges, max_iters=20)
        self.assertTrue(torch.equal(result.poses[0], initial.tensor()[0]))
        error = (pp.SE3(result.poses).Inv() @ truth).Log().tensor().norm(dim=-1)
        self.assertLess(error.max().item(), 1e-5)
        self.assertLess(result.diagnostics["final_cost"], result.diagnostics["initial_cost"])
        self.assertEqual(result.diagnostics["skip_edges"], 3)

    def test_two_pose_jacobian_matches_finite_difference(self):
        _, poses, edges = fixture()
        r, covariance, J = factor_system(poses.tensor(), list(range(5)), edges)
        for col in (0, 4, 7, 13, 21, 23):
            delta = torch.zeros(5, 6, dtype=torch.float64)
            delta[1 + col // 6, col % 6] = 1e-6
            plus = pp.se3(delta).Exp() @ poses
            minus = pp.se3(-delta).Exp() @ poses
            rp = factor_system(plus.tensor(), list(range(5)), edges)[0]
            rm = factor_system(minus.tensor(), list(range(5)), edges)[0]
            self.assertTrue(torch.allclose((rp - rm) / 2e-6, J[..., col], atol=1e-7))
        self.assertTrue(torch.all(torch.linalg.eigvalsh(covariance) > 0))

    def test_whitening_preserves_full_covariance(self):
        _, poses, edges = fixture()
        r, cov, _ = factor_system(poses.tensor(), list(range(5)), edges)
        white = torch.linalg.solve_triangular(torch.linalg.cholesky(cov), r[..., None], upper=False)
        expected = (r[..., None, :] @ torch.linalg.solve(cov, r[..., None])).sum()
        self.assertAlmostEqual(white.square().sum().item(), expected.item(), places=7)
        diagonal_cost = (r.square() / cov.diagonal(dim1=-2, dim2=-1)).sum()
        self.assertGreater(abs(expected.item() - diagonal_cost.item()), 1.)

    def test_window_eviction_and_edge_replacement(self):
        _, _, edges = fixture()
        window = EdgeWindow(5)
        for edge in edges:
            window.add(edge)
        window.add(edges[-1])
        self.assertEqual(len(window.edges), 7)
        window.advance(5)
        self.assertTrue(all(e.a >= 1 for e in window.edges))
        self.assertEqual(len(window.edges), 5)
        window.advance(9)
        self.assertEqual(len(window.edges), 0)

    def test_window_advance_returns_evicted_edges(self):
        _, _, edges = fixture()
        window = EdgeWindow(5)
        for edge in edges:
            window.add(edge)

        evicted = window.advance(5)

        self.assertEqual([(edge.a, edge.b) for edge in evicted], [(0, 1), (0, 2)])
        self.assertTrue(all(edge.a >= 1 for edge in window.edges))

    def test_edge_tensor_bytes_counts_only_observation_payload(self):
        _, _, edges = fixture()
        edge = edges[0]
        expected = sum(
            value.numel() * value.element_size()
            for value in (edge.points_a, edge.points_b, edge.cov_a, edge.cov_b)
        )

        self.assertEqual(edge_tensor_bytes(edge), expected)

    def test_disconnected_window_rejected_without_mutation(self):
        _, poses, edges = fixture()
        before = poses.tensor().clone()
        with self.assertRaisesRegex(ValueError, "connected"):
            optimize_window(poses.tensor(), list(range(5)), edges[:1])
        self.assertTrue(torch.equal(poses.tensor(), before))

    def test_invalid_observation_rejected(self):
        _, _, edges = fixture()
        edge = edges[0]
        points = edge.points_a.clone()
        points[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            Edge(edge.a, edge.b, points, edge.points_b, edge.cov_a, edge.cov_b)

    def test_repeated_float32_writeback_preserves_unit_quaternions(self):
        torch.manual_seed(15)
        world = torch.randn(12, 3, dtype=torch.float64) + torch.tensor([4., 0., 0.])
        cov = torch.eye(3, dtype=torch.float64).expand(12, 3, 3).clone() * .001
        tangents = torch.zeros(40, 6, dtype=torch.float64)
        tangents[:, 0] = torch.arange(40)*.04
        tangents[:, 4] = torch.arange(40)*.01
        truth = pp.se3(tangents).Exp()
        estimates = [truth[0].tensor().float()]
        window = EdgeWindow(5)
        for b in range(1, 40):
            estimates.append(estimates[-1].clone())
            for gap in (1, 2):
                a = b-gap
                if a >= 0:
                    window.add(Edge(a, b, truth[a].Inv().Act(world),
                                    truth[b].Inv().Act(world), cov, cov))
            window.advance(b)
            ids = list(range(max(0, b-4), b+1))
            output = optimize_window(torch.stack([estimates[i] for i in ids]), ids, window.edges)
            for j, i in enumerate(ids):
                estimates[i] = output.poses[j].float()
        pose = pp.SE3(torch.stack(estimates).double())
        self.assertLess((pose.tensor()[:, 3:].norm(dim=-1)-1).abs().max().item(), 1e-6)
        self.assertLess((pose.Inv() @ truth).Log().tensor().norm(dim=-1).max().item(), 1e-4)


if __name__ == "__main__":
    unittest.main()
