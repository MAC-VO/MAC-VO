import csv
import json
from pathlib import Path
import tempfile
import unittest

from Scripts.Experiment.CompareCovisibilityICP import (
    build_parser,
    dynamic_covisibility_gate,
    resolve_sequence_bounds,
    write_comparison_artifacts,
)


def rows(proximity_ate=0.136, proximity_edges=20):
    common = {
        "status": "evaluated",
        "source_factor_sha256": "same",
        "source_pose_sha256": "pose",
    }
    return [
        common | {"mode": "before_global", "RMSE_ATE": 0.145},
        common | {
            "mode": "short", "RMSE_ATE": 0.140,
            "global_status": "refined", "anchor_preserved": True,
            "initial_cost": 10.0, "final_cost": 9.0,
            "added_edges": 0,
        },
        common | {
            "mode": "gap5", "RMSE_ATE": 0.137,
            "global_status": "refined", "anchor_preserved": True,
            "initial_cost": 12.0, "final_cost": 10.0,
            "added_edges": 23,
        },
        common | {
            "mode": "proximity", "RMSE_ATE": proximity_ate,
            "global_status": "refined", "anchor_preserved": True,
            "initial_cost": 13.0, "final_cost": 11.0,
            "added_edges": proximity_edges,
        },
    ]


class CompareCovisibilityICPTests(unittest.TestCase):
    def test_basic_gate_requires_proximity_ate_below_short(self):
        gate = dynamic_covisibility_gate(rows())

        self.assertTrue(gate["basic_passed"])

    def test_selector_gate_requires_gap5_accuracy_and_edge_budget(self):
        self.assertTrue(
            dynamic_covisibility_gate(rows())["selector_superiority"]
        )
        self.assertFalse(
            dynamic_covisibility_gate(
                rows(proximity_edges=24)
            )["selector_superiority"]
        )
        self.assertFalse(
            dynamic_covisibility_gate(
                rows(proximity_ate=0.138)
            )["selector_superiority"]
        )

    def test_basic_gate_rejects_invalid_solver_result(self):
        values = rows()
        values[-1]["anchor_preserved"] = False

        self.assertFalse(
            dynamic_covisibility_gate(values)["basic_passed"]
        )

    def test_write_artifacts_preserves_mode_order_and_csv_rows(self):
        values = rows()
        with tempfile.TemporaryDirectory() as directory:
            gate = write_comparison_artifacts(directory, values)
            saved = json.loads(Path(
                directory, "dynamic_covisibility_metrics.json"
            ).read_text())
            with Path(
                directory, "dynamic_covisibility_metrics.csv"
            ).open() as stream:
                csv_rows = list(csv.DictReader(stream))

        self.assertEqual(
            [row["mode"] for row in saved],
            ["before_global", "short", "gap5", "proximity"],
        )
        self.assertEqual(len(csv_rows), 4)
        self.assertEqual(gate["gap5_edge_budget"], 23)

    def test_segment_json_supplies_exact_sequence_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            segment = Path(directory, "selected_segment.json")
            segment.write_text(json.dumps({"start": 400, "end": 520}))
            args = build_parser().parse_args([
                "--sequence", "V203", "--segment-json", str(segment)
            ])

            start, end, digest = resolve_sequence_bounds(args)

        self.assertEqual((start, end), (400, 520))
        self.assertEqual(len(digest), 64)

    def test_segment_json_rejects_conflicting_explicit_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            segment = Path(directory, "selected_segment.json")
            segment.write_text(json.dumps({"start": 400, "end": 520}))
            args = build_parser().parse_args([
                "--segment-json", str(segment), "--seq-from", "401"
            ])

            with self.assertRaisesRegex(ValueError, "conflict"):
                resolve_sequence_bounds(args)


if __name__ == "__main__":
    unittest.main()
