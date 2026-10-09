import unittest
from types import SimpleNamespace

import numpy as np
import pypose as pp
import torch

from Module.Optimization.FactorArchive import FactorArchive
from Module.Optimization.LongRangeICP import (
    generate_long_range_archive,
    long_range_targets,
)
from Module.Optimization.MatchICP import MatchEdgeBuildResult
from Module.Optimization.WindowICP import Edge
from Scripts.Experiment.GenerateLongRangeICP import build_parser, optional_index


class FakeSequence:
    def __init__(self, count):
        self.frames = [
            SimpleNamespace(
                stereo=SimpleNamespace(tag=index, frame_ns=1000 + index)
            )
            for index in range(count)
        ]

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        return self.frames[index]


class FakeFrontend:
    def __init__(self, fail_target=None):
        self.calls = []
        self.fail_target = fail_target

    def estimate_window(self, frame_t2, frame_t1, frame_t):
        call = (
            None if frame_t2 is None else frame_t2.tag,
            frame_t1.tag,
            frame_t.tag,
        )
        self.calls.append(call)
        if frame_t.tag == self.fail_target:
            raise RuntimeError(f"inference failed at {frame_t.tag}")
        depth = SimpleNamespace(tag=frame_t.tag)
        adjacent = SimpleNamespace(source=frame_t1.tag, target=frame_t.tag)
        skip = (
            None
            if frame_t2 is None
            else SimpleNamespace(source=frame_t2.tag, target=frame_t.tag)
        )
        return depth, adjacent, skip


def source_archive(count):
    poses = pp.identity_SE3(count, dtype=torch.float64).tensor()
    return FactorArchive(
        initial_sensor_poses=poses,
        time_ns=np.arange(count, dtype=np.int64) + 1000,
        T_BS=pp.identity_SE3(dtype=torch.float64).tensor(),
        edges=(),
        edge_kinds=(),
        metadata={"fixture": True},
    )


def fake_edge_builder(*, a, b, match, **_):
    covariance = torch.eye(3, dtype=torch.float64)[None] * 0.001
    points = torch.tensor([[3.0, 0.0, 0.0]], dtype=torch.float64)
    edge = Edge(a, b, points, points.clone(), covariance, covariance.clone())
    return MatchEdgeBuildResult(edge, None, 4, 4)


def rejecting_edge_builder(**_):
    return MatchEdgeBuildResult(None, "too_few_valid_observations", 4, 4)


def generate(count, frontend=None, edge_builder=fake_edge_builder):
    frontend = FakeFrontend() if frontend is None else frontend
    result = generate_long_range_archive(
        FakeSequence(count),
        source_archive(count),
        frontend=frontend,
        selector=object(),
        covariance_model=object(),
        num_point=200,
        min_num_point=10,
        edge_width=32,
        match_cov_default=0.25,
        device="cpu",
        edge_builder=edge_builder,
    )
    return frontend, result


class LongRangeICPTests(unittest.TestCase):
    def test_schedule_for_twelve_frames(self):
        self.assertEqual(
            long_range_targets(12),
            [(5, (0,)), (10, (5, 0))],
        )

    def test_generator_uses_one_fixed_batch_call_per_target_plus_initialization(self):
        frontend, result = generate(16)

        self.assertEqual(
            frontend.calls,
            [(None, 0, 0), (None, 0, 5), (0, 5, 10), (5, 10, 15)],
        )
        self.assertEqual(result.diagnostics["frontend_calls"], 4)

    def test_depth_and_frame_caches_never_exceed_three_scheduled_frames(self):
        _, result = generate(31)

        self.assertLessEqual(result.diagnostics["max_cached_depths"], 3)
        self.assertLessEqual(result.diagnostics["max_cached_frames"], 3)

    def test_t5_duplicate_slot_is_discarded(self):
        _, result = generate(6)

        self.assertEqual(result.archive.edge_kinds, ("gap5",))
        self.assertEqual([(edge.a, edge.b) for edge in result.archive.edges], [(0, 5)])

    def test_generator_records_counts_by_gap(self):
        _, result = generate(16)

        self.assertEqual(result.diagnostics["attempted"], {"gap5": 3, "gap10": 2})
        self.assertEqual(result.diagnostics["accepted"], {"gap5": 3, "gap10": 2})
        self.assertEqual(result.diagnostics["observations"], 5)

    def test_local_inference_failure_is_recorded_and_later_targets_continue(self):
        frontend = FakeFrontend(fail_target=10)

        _, result = generate(16, frontend=frontend)

        self.assertEqual(len(frontend.calls), 4)
        self.assertEqual(result.diagnostics["inference_failures"][0]["target"], 10)
        self.assertIn("inference failed", result.diagnostics["inference_failures"][0]["reason"])
        self.assertEqual(result.diagnostics["accepted"]["gap10"], 1)
        self.assertEqual(
            result.diagnostics["rejected"]["gap5"]["missing_source_depth"],
            1,
        )

    def test_no_accepted_edges_has_explicit_status(self):
        _, result = generate(11, edge_builder=rejecting_edge_builder)

        self.assertEqual(result.diagnostics["status"], "no_edges")
        self.assertEqual(len(result.archive.edges), 0)
        self.assertEqual(
            result.diagnostics["rejected"]["gap5"][
                "too_few_valid_observations"
            ],
            2,
        )

    def test_timestamp_mismatch_is_rejected_before_frontend_call(self):
        sequence = FakeSequence(6)
        source = source_archive(6)
        source.time_ns[0] += 1
        frontend = FakeFrontend()

        with self.assertRaisesRegex(ValueError, "timestamp"):
            generate_long_range_archive(
                sequence,
                source,
                frontend=frontend,
                selector=object(),
                covariance_model=object(),
                num_point=200,
                min_num_point=10,
                edge_width=32,
                match_cov_default=0.25,
                device="cpu",
                edge_builder=fake_edge_builder,
            )

        self.assertEqual(frontend.calls, [])

    def test_cli_defaults_to_gap5_gap10_archive(self):
        args = build_parser().parse_args(["--space", "/tmp/result"])

        self.assertEqual(args.output, "long_factors_gap5_10.npz")

    def test_saved_null_sequence_bound_becomes_none(self):
        self.assertIsNone(optional_index(SimpleNamespace()))
        self.assertEqual(optional_index(12), 12)


if __name__ == "__main__":
    unittest.main()
