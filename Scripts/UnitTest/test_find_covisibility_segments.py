from pathlib import Path
import tempfile
import unittest

import numpy as np
import pypose as pp
import torch

from Scripts.Experiment.FindCovisibilitySegments import (
    build_parser,
    find_segments,
    is_revisit_pair,
    rotation_angle_deg,
    write_segment_results,
)


class FindCovisibilitySegmentsTests(unittest.TestCase):
    def test_revisit_pair_requires_gap_position_and_view_angle(self):
        values = pp.identity_SE3(50, dtype=torch.float64).tensor()

        self.assertTrue(is_revisit_pair(
            values, 0, 40, min_gap=30,
            max_distance=2.0, max_angle_deg=30.0,
        ))
        self.assertFalse(is_revisit_pair(
            values, 0, 20, min_gap=30,
            max_distance=2.0, max_angle_deg=30.0,
        ))

    def test_rotation_angle_uses_so3_geodesic(self):
        tangent = torch.tensor([0.0, 0.0, np.deg2rad(20.0)], dtype=torch.float64)
        rotated = pp.so3(tangent).Exp().tensor()
        identity = pp.identity_SO3(dtype=torch.float64).tensor()

        self.assertAlmostEqual(
            rotation_angle_deg(identity, rotated), 20.0, places=6
        )

    def test_segment_requires_revisits_and_ranks_by_difficulty(self):
        gt = pp.identity_SE3(8, dtype=torch.float64).tensor()
        estimate = gt.clone()
        estimate[:, 0] = torch.tensor(
            [0, 0, 0, 0, 0, 10, 20, 30], dtype=torch.float64
        )

        segments = find_segments(
            gt,
            estimate,
            window=4,
            min_temporal_gap=2,
            max_distance=2.0,
            max_angle_deg=30.0,
            min_pairs=1,
        )

        self.assertEqual(segments[0].start, 4)
        self.assertEqual(segments[0].end, 8)
        self.assertGreaterEqual(segments[0].difficulty, segments[-1].difficulty)

    def test_window_without_enough_revisits_is_excluded(self):
        gt = pp.identity_SE3(6, dtype=torch.float64).tensor()
        gt[:, 0] = torch.arange(6, dtype=torch.float64) * 10

        segments = find_segments(
            gt,
            gt.clone(),
            window=4,
            min_temporal_gap=2,
            max_distance=2.0,
            max_angle_deg=30.0,
            min_pairs=1,
        )

        self.assertEqual(segments, [])

    def test_excluded_range_removes_every_overlapping_window(self):
        gt = pp.identity_SE3(12, dtype=torch.float64).tensor()

        segments = find_segments(
            gt,
            gt.clone(),
            window=4,
            min_temporal_gap=2,
            max_distance=2.0,
            max_angle_deg=30.0,
            min_pairs=1,
            exclude_ranges=[(4, 8)],
        )

        self.assertTrue(all(
            segment.end <= 4 or segment.start >= 8
            for segment in segments
        ))

    def test_result_writer_uses_exact_half_open_selected_range(self):
        gt = pp.identity_SE3(8, dtype=torch.float64).tensor()
        segments = find_segments(
            gt, gt.clone(), window=4, min_temporal_gap=2,
            max_distance=2.0, max_angle_deg=30.0, min_pairs=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            write_segment_results(directory, segments)
            selected = __import__("json").loads(
                Path(directory, "selected_segment.json").read_text()
            )

        self.assertEqual(selected["end"] - selected["start"], 4)

    def test_cli_defaults_match_approved_search(self):
        args = build_parser().parse_args([
            "--space", "/tmp/full", "--output-root", "/tmp/out"
        ])

        self.assertEqual(args.window, 120)
        self.assertEqual(args.min_temporal_gap, 30)
        self.assertEqual(args.min_revisit_pairs, 3)


if __name__ == "__main__":
    unittest.main()
