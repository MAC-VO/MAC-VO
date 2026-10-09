import unittest
from types import SimpleNamespace

import pypose as pp
import torch

from DataLoader import StereoData
from Module.Frontend.Frontend import IFrontend
from Module.Frontend.Matching import IMatcher
from Module.Frontend.StereoDepth import IStereoDepth
from Module.Optimization.MatchICP import (
    build_edge_from_correspondences,
    build_match_edge,
)
from Odometry.WindowMACVO import WindowMACVO


def stereo(value):
    size = 16
    K = torch.eye(3).unsqueeze(0)
    K[0, 0, 0] = 10
    K[0, 1, 1] = 10
    K[0, 0, 2] = 8
    K[0, 1, 2] = 8
    return StereoData(
        T_BS=pp.identity_SE3(1),
        K=K,
        baseline=torch.tensor([0.2]),
        time_ns=[int(value)],
        height=size,
        width=size,
        imageL=torch.full((1, 3, size, size), float(value)),
        imageR=torch.full((1, 3, size, size), float(value + 1)),
    )


def depth(value=4.0):
    depth_map = torch.full((1, 1, 16, 16), float(value))
    covariance = torch.full((1, 1, 16, 16), 0.01)
    return IStereoDepth.Output(depth=depth_map, cov=covariance)


def match(flow_value=0.0):
    flow = torch.full((1, 2, 16, 16), float(flow_value))
    covariance = torch.zeros((1, 3, 16, 16))
    covariance[:, :2] = 0.25
    return IMatcher.Output(flow=flow, cov=covariance)


class FixedSelector:
    def select_point(self, *_):
        return torch.tensor(
            [[3, 3], [5, 4], [7, 6], [9, 8]], dtype=torch.float32
        )


class IdentityCovariance:
    def estimate(self, _stereo, uv, *_):
        return torch.eye(3, dtype=torch.float64).repeat(len(uv), 1, 1) * 0.01


def inputs():
    return {
        "stereo_a": stereo(0),
        "stereo_b": stereo(5),
        "depth_a": depth(4.0),
        "depth_b": depth(4.2),
        "match": match(),
        "frontend": IFrontend,
        "selector": FixedSelector(),
        "covariance_model": IdentityCovariance(),
        "num_point": 4,
        "min_num_point": 2,
        "edge_width": 1,
        "match_cov_default": 0.25,
        "device": "cpu",
    }


class MatchICPTests(unittest.TestCase):
    def test_correspondence_builder_matches_existing_builder(self):
        values = inputs()
        uv_a = values["selector"].select_point()
        uv_b = uv_a + values["frontend"].retrieve_pixels(
            uv_a, values["match"].flow
        ).T

        legacy = build_match_edge(**values, a=0, b=5)
        direct = build_edge_from_correspondences(
            a=0,
            b=5,
            uv_a=uv_a,
            uv_b=uv_b,
            stereo_a=values["stereo_a"],
            stereo_b=values["stereo_b"],
            depth_a=values["depth_a"],
            depth_b=values["depth_b"],
            match=values["match"],
            frontend=values["frontend"],
            covariance_model=values["covariance_model"],
            min_num_point=values["min_num_point"],
            match_cov_default=values["match_cov_default"],
            device=values["device"],
        )

        self.assertIsNotNone(legacy.edge)
        self.assertIsNotNone(direct.edge)
        self.assertTrue(torch.equal(legacy.edge.points_a, direct.edge.points_a))
        self.assertTrue(torch.equal(legacy.edge.cov_b, direct.edge.cov_b))

    def test_match_builder_is_deterministic(self):
        first = build_match_edge(**inputs(), a=0, b=5)
        second = build_match_edge(**inputs(), a=0, b=5)

        self.assertIsNone(first.reason)
        self.assertIsNotNone(first.edge)
        self.assertTrue(torch.equal(first.edge.points_a, second.edge.points_a))
        self.assertTrue(torch.equal(first.edge.cov_b, second.edge.cov_b))

    def test_match_builder_reports_too_few_inbound_points(self):
        values = inputs()
        values["match"] = match(1e6)

        result = build_match_edge(**values, a=0, b=5)

        self.assertIsNone(result.edge)
        self.assertEqual(result.reason, "too_few_inbound_points")
        self.assertEqual(result.inbound_points, 0)

    def test_match_builder_reports_too_few_valid_observations(self):
        values = inputs()
        values["depth_b"] = depth(-1.0)

        result = build_match_edge(**values, a=0, b=5)

        self.assertIsNone(result.edge)
        self.assertEqual(result.reason, "too_few_valid_observations")

    def test_online_skip_wrapper_matches_shared_builder(self):
        values = inputs()
        system = WindowMACVO.__new__(WindowMACVO)
        system.frame_cache = {
            0: (SimpleNamespace(stereo=values["stereo_a"]), values["depth_a"]),
            2: (SimpleNamespace(stereo=values["stereo_b"]), values["depth_b"]),
        }
        system.Frontend = values["frontend"]
        system.KeypointSelector = values["selector"]
        system.ObsCovModel = values["covariance_model"]
        system.num_point = values["num_point"]
        system.min_num_point = values["min_num_point"]
        system.edge_width = values["edge_width"]
        system.match_cov_default = values["match_cov_default"]
        system.device = values["device"]
        expected = build_match_edge(
            **{key: value for key, value in values.items() if key != "match"},
            match=values["match"],
            a=0,
            b=2,
        ).edge

        actual = system._skip_edge_from_match(0, 2, values["match"])

        self.assertIsNotNone(expected)
        self.assertIsNotNone(actual)
        self.assertTrue(torch.equal(actual.points_a, expected.points_a))
        self.assertTrue(torch.equal(actual.cov_b, expected.cov_b))


if __name__ == "__main__":
    unittest.main()
