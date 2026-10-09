import copy
import json
from pathlib import Path
import tempfile
import unittest

from Scripts.Experiment.CompareOnlineORBLoop import (
    MODE_CONFIGS,
    build_parser,
    loop_switch_summary,
    online_invariants,
    sparse_loop_summary,
    uses_online_diagnostics,
    write_rows,
)
from Utility.Config import load_config
from Odometry.OnlineLoopWindowMACVO import OnlineLoopWindowMACVO


class CompareOnlineORBLoopTests(unittest.TestCase):
    def test_online_invariants_require_single_model_graph_and_concurrency(self):
        diagnostics = {
            "frontend": {
                "wrapped_instances_created": 1,
                "wrapped_graphs_captured": 1,
                "max_concurrent_calls": 1,
            },
            "cuda_max_memory_reserved": 5 * 1024**3,
        }

        result = online_invariants(diagnostics)

        self.assertTrue(result["passed"])
        diagnostics["frontend"]["max_concurrent_calls"] = 2
        self.assertFalse(online_invariants(diagnostics)["passed"])

    def test_write_rows_preserves_mode_order(self):
        rows = [
            {"mode": "window_skip", "RMSE_ATE": 0.2},
            {"mode": "window_pose_graph", "RMSE_ATE": 0.15},
            {"mode": "window_orb_loop", "RMSE_ATE": 0.1},
        ]
        with tempfile.TemporaryDirectory() as directory:
            write_rows(directory, rows)
            saved = json.loads(Path(directory, "metrics.json").read_text())

        self.assertEqual(
            [row["mode"] for row in saved],
            ["window_skip", "window_pose_graph", "window_orb_loop"],
        )

    def test_parser_accepts_complete_sequence(self):
        args = build_parser().parse_args([
            "--sequence", "V203", "--seq-from", "0"
        ])

        self.assertEqual(args.sequence, "V203")
        self.assertIsNone(args.seq_to)
        self.assertEqual(args.modes, [
            "window_skip", "window_pose_graph", "window_orb_loop",
        ])

    def test_loop_switch_summary_counts_only_effective_factors(self):
        diagnostics = {
            "pose_graph_results": [{
                "loop_switches": [
                    {"a": 0, "b": 30, "switch": 0.5},
                    {"a": 100, "b": 500, "switch": 0.02},
                    {"a": 200, "b": 900, "switch": 1e-5},
                ],
            }],
        }

        self.assertEqual(loop_switch_summary(diagnostics), {
            "max_loop_switch": 0.5,
            "effective_loops_001": 2,
            "effective_long_loops_001": 1,
        })

    def test_no_skip_mode_disables_skip_and_keeps_online_loop(self):
        _, config = load_config(MODE_CONFIGS["window_orb_loop_no_skip"])

        self.assertEqual(config["Odometry"]["type"], "OnlineLoopWindowMACVO")
        self.assertFalse(config["Odometry"]["args"]["skip_matching"])
        self.assertTrue(
            config["Odometry"]["args"]["online_loop"]["enabled"]
        )
        self.assertTrue(uses_online_diagnostics("window_orb_loop_no_skip"))

    def test_sparse_mode_uses_sparse_configuration_and_diagnostics(self):
        _, config = load_config(MODE_CONFIGS["window_orb_loop_sparse"])

        self.assertEqual(
            config["Odometry"]["args"]["online_loop"]["validation_mode"],
            "sparse_se3",
        )
        self.assertEqual(
            config["Odometry"]["args"]["pose_graph"]["loop_information_scale"],
            10000.0,
        )
        self.assertTrue(uses_online_diagnostics("window_orb_loop_sparse"))

    def test_t030_sparse_mode_changes_only_name_and_translation_threshold(self):
        _, baseline = load_config(MODE_CONFIGS["window_orb_loop_sparse"])
        _, t030 = load_config(MODE_CONFIGS["window_orb_loop_sparse_t030"])

        self.assertEqual(
            t030["Odometry"]["args"]["online_loop"]["hypothesis"]
            ["max_correction_translation_m"],
            0.30,
        )
        self.assertTrue(
            uses_online_diagnostics("window_orb_loop_sparse_t030")
        )

        normalized = copy.deepcopy(t030)
        normalized["Odometry"]["name"] = baseline["Odometry"]["name"]
        normalized["Odometry"]["args"]["online_loop"]["hypothesis"][
            "max_correction_translation_m"
        ] = 0.25
        self.assertEqual(normalized, baseline)

    def test_legacy_mode_defaults_to_unit_loop_scale(self):
        cfg, _ = load_config(MODE_CONFIGS["window_orb_loop"])

        self.assertEqual(
            getattr(cfg.Odometry.args.pose_graph, "loop_information_scale", 1.0),
            1.0,
        )

    def test_invalid_loop_information_scale_is_rejected(self):
        cfg, _ = load_config(MODE_CONFIGS["window_orb_loop_sparse"])
        for value in (0.0, -1.0, float("nan"), float("inf")):
            cfg.Odometry.args.pose_graph.loop_information_scale = value
            with self.assertRaisesRegex(ValueError, "information scale"):
                OnlineLoopWindowMACVO.is_valid_config(cfg.Odometry)

    def test_legacy_mode_rejects_nonunit_explicit_loop_scale(self):
        cfg, _ = load_config(MODE_CONFIGS["window_orb_loop"])
        cfg.Odometry.args.pose_graph.loop_information_scale = 10000.0

        with self.assertRaisesRegex(ValueError, "sparse_se3"):
            OnlineLoopWindowMACVO.is_valid_config(cfg.Odometry)

    def test_sparse_loop_summary_reports_states_and_quality(self):
        diagnostics = {
            "hypothesis_records": [
                {"state": "tentative"},
                {"state": "confirmed"},
                {"state": "strong"},
            ],
            "loop_records": [
                {
                    "status": "tentative",
                    "metrics": {
                        "ransac_ratio": 0.5,
                        "reprojection_forward_median_px": 2.0,
                    },
                },
                {
                    "status": "accepted_sparse",
                    "information_eigenvalues": [16.0, 32.8],
                },
            ],
        }

        self.assertEqual(sparse_loop_summary(diagnostics), {
            "tentative_hypotheses": 1,
            "confirmed_hypotheses": 1,
            "strong_hypotheses": 1,
            "emitted_sparse_factors": 1,
            "median_sparse_ransac_ratio": 0.5,
            "median_sparse_reprojection_px": 2.0,
            "max_sparse_information_eigenvalue": 32.8,
        })


if __name__ == "__main__":
    unittest.main()
