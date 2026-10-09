import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from Module.Optimization.WindowICP import Edge, EdgeWindow
from Odometry.WindowMACVO import WindowMACVO
from Odometry.MACVO import MACVO
from Utility.Config import load_config


def make_edge(a, b):
    points = torch.tensor([[3.0, 0.0, 0.0]], dtype=torch.float64)
    covariance = torch.eye(3, dtype=torch.float64).unsqueeze(0) * 0.001
    return Edge(a, b, points, points.clone(), covariance, covariance.clone())


class WindowGlobalConfigTests(unittest.TestCase):
    def test_from_config_rejects_invalid_global_type_before_construction(self):
        config, _ = load_config(
            Path("Config/Experiment/MACVO/MACVO_Fast_WindowICP_Global.yaml")
        )
        config.Odometry.args.global_refine = "yes"

        with patch.object(
            MACVO,
            "from_config",
            side_effect=AssertionError("construction should not start"),
        ):
            with self.assertRaisesRegex(ValueError, "Config does not match"):
                WindowMACVO.from_config(config)

    def test_v03_config_enables_global_refinement(self):
        config, _ = load_config(
            Path("Config/Experiment/MACVO/MACVO_Fast_WindowICP_Global.yaml")
        )

        self.assertTrue(config.Odometry.args.global_refine)
        self.assertEqual(config.Odometry.args.global_iterations, 5)
        self.assertEqual(config.Odometry.name, "MACVO-Fast-WindowICP5-Global")
        WindowMACVO.is_valid_config(config.Odometry)

    def test_v02_config_remains_valid_without_global_keys(self):
        config, _ = load_config(
            Path("Config/Experiment/MACVO/MACVO_Fast_WindowICP.yaml")
        )

        WindowMACVO.is_valid_config(config.Odometry)

    def test_disabled_mode_does_not_retain_evicted_edges(self):
        system = WindowMACVO.__new__(WindowMACVO)
        system.global_refine = False
        system.inactive_edges = {}

        system._retain_evicted_edges([make_edge(0, 1)])

        self.assertEqual(system.inactive_edges, {})

    def test_enabled_mode_retains_unique_inactive_and_active_edges(self):
        system = WindowMACVO.__new__(WindowMACVO)
        system.global_refine = True
        system.inactive_edges = {}
        system.edge_window = EdgeWindow(5)
        old = make_edge(0, 1)
        replacement = make_edge(0, 1)
        active = make_edge(1, 2)

        system._retain_evicted_edges([old, replacement])
        system.edge_window.add(active)

        self.assertEqual(
            [(edge.a, edge.b) for edge in system._global_edges()],
            [(0, 1), (1, 2)],
        )
        self.assertIs(system.inactive_edges[(0, 1)], replacement)


if __name__ == "__main__":
    unittest.main()
