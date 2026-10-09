import unittest
from dataclasses import replace
from types import SimpleNamespace

import pypose as pp
import torch

from DataLoader import StereoData
from Module.Frontend.Frontend import IFrontend
from Module.Frontend.Matching import IMatcher
from Module.Frontend.StereoDepth import IStereoDepth
from Module.Optimization.CovisibilitySelector import CovisibilityMetrics
from Module.Optimization.ProximityICP import (
    ProximityValidationConfig,
    depth_output_to_device,
    filter_mahalanobis_inliers,
    validate_proximity_match,
)
from Module.Optimization.WindowICP import Edge


def stereo(time_ns):
    size = 32
    K = torch.eye(3).unsqueeze(0)
    K[0, 0, 0] = 20
    K[0, 1, 1] = 20
    K[0, 0, 2] = 16
    K[0, 1, 2] = 16
    return StereoData(
        T_BS=pp.identity_SE3(1),
        K=K,
        baseline=torch.tensor([0.2]),
        time_ns=[time_ns],
        height=size,
        width=size,
        imageL=torch.zeros(1, 3, size, size),
        imageR=torch.zeros(1, 3, size, size),
    )


def depth(value=4.0):
    values = torch.full((1, 1, 32, 32), float(value))
    return IStereoDepth.Output(
        depth=values,
        cov=torch.full_like(values, 0.1),
    )


def flow(x):
    values = torch.zeros(1, 2, 32, 32)
    values[:, 0] = float(x)
    covariance = torch.zeros(1, 3, 32, 32)
    covariance[:, :2] = 0.25
    return IMatcher.Output(flow=values, cov=covariance)


class DistributedSelector:
    def select_point(self, *_):
        points = []
        for row in (4, 9, 14, 19, 24):
            for column in (4, 8, 12, 16, 20, 24, 27, 29):
                points.append((column, row))
        return torch.tensor(points, dtype=torch.float32)


class ConcentratedSelector:
    def select_point(self, *_):
        return torch.tensor(
            [(4 + index % 3, 4 + index // 3 % 3) for index in range(40)],
            dtype=torch.float32,
        )


class IdentityCovariance:
    def estimate(self, _stereo, uv, *_):
        return torch.eye(3, dtype=torch.float64).repeat(len(uv), 1, 1) * 0.1


def candidate():
    return CovisibilityMetrics(
        source=0,
        target=15,
        overlap_forward=0.6,
        overlap_backward=0.6,
        mean_overlap=0.6,
        depth_consistency=0.7,
        median_motion_px=8.0,
        score=0.3,
        status="eligible",
        reason=None,
    )


def poses():
    result = pp.identity_SE3(16, dtype=torch.float64).tensor()
    result[15, 1] = -0.2
    return result


def validation_inputs(**updates):
    values = {
        "candidate": candidate(),
        "forward": flow(1.0),
        "backward": flow(-1.0),
        "source_frame": SimpleNamespace(stereo=stereo(0)),
        "target_frame": SimpleNamespace(stereo=stereo(15)),
        "source_depth": depth(),
        "target_depth": depth(),
        "poses": poses(),
        "frontend": IFrontend,
        "keypoint_selector": DistributedSelector(),
        "covariance_model": IdentityCovariance(),
        "num_point": 40,
        "edge_width": 1,
        "match_cov_default": 0.25,
        "device": "cpu",
        "config": ProximityValidationConfig(),
    }
    values.update(updates)
    return values


class ProximityICPTests(unittest.TestCase):
    def test_cached_depth_output_moves_all_available_maps_to_device(self):
        source = depth()

        moved = depth_output_to_device(source, "cpu")

        self.assertEqual(moved.depth.device.type, "cpu")
        self.assertEqual(moved.cov.device.type, "cpu")
        self.assertTrue(torch.equal(moved.depth, source.depth))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cached_cpu_depth_can_move_to_cuda_for_gpu_keypoints(self):
        moved = depth_output_to_device(depth(), "cuda")

        self.assertEqual(moved.depth.device.type, "cuda")
        self.assertEqual(moved.cov.device.type, "cuda")

    def test_inverse_flows_pass_forward_backward_validation(self):
        result = validate_proximity_match(**validation_inputs())

        self.assertIsNone(result.reason)
        self.assertIsNotNone(result.record)
        self.assertEqual(result.metrics["forward_backward_inliers"], 40)
        self.assertEqual(result.record.state, "accepted")

    def test_inconsistent_backward_flow_is_rejected(self):
        result = validate_proximity_match(
            **validation_inputs(backward=flow(3.0))
        )

        self.assertEqual(result.reason, "insufficient_forward_backward_inliers")
        self.assertIsNone(result.record)

    def test_concentrated_matches_fail_grid_coverage(self):
        result = validate_proximity_match(
            **validation_inputs(keypoint_selector=ConcentratedSelector())
        )

        self.assertEqual(result.reason, "insufficient_grid_coverage")

    def test_invalid_target_depth_ratio_is_rejected(self):
        result = validate_proximity_match(
            **validation_inputs(target_depth=depth(-1.0))
        )

        self.assertEqual(result.reason, "insufficient_valid_depth")

    def test_mahalanobis_filter_rejects_known_outlier(self):
        points_a = torch.tensor(
            [[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=torch.float64
        )
        points_b = points_a.clone()
        points_b[1, 1] += 10
        covariance = torch.eye(3, dtype=torch.float64).repeat(2, 1, 1) * 0.01
        edge = Edge(0, 1, points_a, points_b, covariance, covariance)
        uv = torch.tensor([[4.0, 4.0], [8.0, 8.0]])

        filtered = filter_mahalanobis_inliers(
            edge,
            uv,
            uv,
            pp.identity_SE3(2, dtype=torch.float64).tensor(),
            threshold=3.5,
        )

        self.assertEqual(len(filtered.edge.points_a), 1)
        self.assertEqual(filtered.inliers, 1)

    def test_confidence_uses_candidate_and_validation_ratios(self):
        result = validate_proximity_match(**validation_inputs())
        expected = 0.25 * (0.6 + 0.7 + 1.0 + 1.0)

        self.assertAlmostEqual(result.record.confidence, expected)

    def test_stricter_grid_requirement_rejects_otherwise_valid_match(self):
        config = replace(
            ProximityValidationConfig(), min_occupied_grid_cells=25
        )

        result = validate_proximity_match(
            **validation_inputs(config=config)
        )

        self.assertEqual(result.reason, "insufficient_grid_coverage")


if __name__ == "__main__":
    unittest.main()
