from pathlib import Path
import tempfile
import unittest

import numpy as np
import pypose as pp
import torch

from Module.Optimization.PairwiseICP import (
    compress_edge_to_pose_factor,
    linearize_edge_to_pose_factor,
    weighted_kabsch,
)
from Module.Optimization.PoseGraph import (
    PoseGraphArchive,
    load_pose_graph_archive,
    save_pose_graph_archive,
)
from Module.Optimization.WindowICP import Edge


def fixture(count=80):
    torch.manual_seed(81)
    points_b = torch.randn(count, 3, dtype=torch.float64)
    points_b[:, 0] += 4
    tangent = torch.tensor(
        [0.2, -0.1, 0.05, 0.02, -0.03, 0.04], dtype=torch.float64
    )
    truth = pp.se3(tangent).Exp()
    points_a = truth.Act(points_b)
    covariance = torch.eye(3, dtype=torch.float64).repeat(count, 1, 1) * 0.001
    return truth, Edge(
        0, 15, points_a, points_b, covariance, covariance.clone()
    )


class PairwiseICPTests(unittest.TestCase):
    def test_weighted_kabsch_recovers_known_transform(self):
        truth, edge = fixture()

        estimate = weighted_kabsch(
            edge.points_a, edge.points_b, edge.cov_a + edge.cov_b
        )

        error = (pp.SE3(estimate).Inv() @ truth).Log().tensor().norm()
        self.assertLess(error.item(), 1e-8)

    def test_compression_recovers_measurement_and_spd_information(self):
        truth, edge = fixture()

        result = compress_edge_to_pose_factor(edge, kind="loop")

        self.assertEqual(result.status, "compressed")
        error = (
            pp.SE3(result.factor.measurement).Inv() @ truth
        ).Log().tensor().norm()
        self.assertLess(error.item(), 1e-7)
        self.assertTrue(torch.all(
            torch.linalg.eigvalsh(result.factor.information) > 0
        ))
        self.assertLess(result.final_cost, result.initial_cost + 1e-12)

    def test_compression_does_not_depend_on_global_pose_prediction(self):
        _, edge = fixture()

        first = compress_edge_to_pose_factor(edge, kind="loop")
        second = compress_edge_to_pose_factor(edge, kind="loop")

        self.assertTrue(torch.equal(
            first.factor.measurement, second.factor.measurement
        ))
        self.assertTrue(torch.equal(
            first.factor.information, second.factor.information
        ))

    def test_local_edge_linearization_uses_current_relative_pose(self):
        truth, raw_edge = fixture()
        poses = pp.identity_SE3(16, dtype=torch.float64).tensor()
        poses[15] = truth.tensor()

        result = linearize_edge_to_pose_factor(
            raw_edge, poses, kind="adjacent"
        )

        self.assertEqual(result.status, "compressed")
        self.assertTrue(torch.allclose(
            result.factor.measurement, truth.tensor(), atol=1e-10
        ))
        self.assertTrue(torch.all(
            torch.linalg.eigvalsh(result.factor.information) > 0
        ))

    def test_collinear_geometry_is_rejected(self):
        x = torch.linspace(1, 5, 20, dtype=torch.float64)
        points = torch.stack((x, torch.zeros_like(x), torch.zeros_like(x)), -1)
        covariance = torch.eye(3, dtype=torch.float64).repeat(20, 1, 1) * 0.001
        edge = Edge(0, 15, points, points.clone(), covariance, covariance.clone())

        result = compress_edge_to_pose_factor(edge, kind="loop")

        self.assertEqual(result.status, "rejected")
        self.assertIn("degenerate", result.reason)

    def test_pose_graph_archive_round_trip(self):
        _, edge = fixture()
        factor = compress_edge_to_pose_factor(edge, kind="loop").factor
        poses = pp.identity_SE3(16, dtype=torch.float64).tensor()
        archive = PoseGraphArchive(
            poses=poses,
            time_ns=np.arange(16, dtype=np.int64),
            factors=(factor,),
            metadata={"fixture": True},
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "pose_graph.npz")
            save_pose_graph_archive(path, archive)
            loaded = load_pose_graph_archive(path)

        self.assertTrue(torch.equal(loaded.poses, archive.poses))
        self.assertTrue(torch.equal(
            loaded.factors[0].information, factor.information
        ))
        self.assertEqual(loaded.factors[0].kind, "loop")


if __name__ == "__main__":
    unittest.main()
