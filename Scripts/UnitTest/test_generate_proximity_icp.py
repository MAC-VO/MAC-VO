import unittest
from types import SimpleNamespace

import numpy as np
import pypose as pp
import torch

from Module.Frontend.StereoDepth import IStereoDepth
from Module.Optimization.CovisibilitySelector import CovisibilityMetrics
from Module.Optimization.FactorArchive import FactorArchive
from Module.Optimization.ProximityFactorStore import ProximityFactorRecord
from Module.Optimization.ProximityICP import (
    ProximityValidationResult,
    generate_proximity_archive,
)
from Module.Optimization.WindowICP import Edge
from Scripts.Experiment.GenerateProximityICP import build_parser


class FakeSequence:
    def __init__(self, count):
        self.frames = [
            SimpleNamespace(stereo=SimpleNamespace(
                tag=index, frame_ns=1000 + index,
            ))
            for index in range(count)
        ]

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        return self.frames[index]


class FakeCache:
    def __init__(self, source):
        self.frame_ids = np.array([0, 5, 10, 15, 20, 25], dtype=np.int64)
        self.metadata = {"source_id": source.metadata["source_id"]}

    def depth_output(self, frame_id):
        values = torch.full((1, 1, 2, 2), float(frame_id))
        return IStereoDepth.Output(depth=values, cov=torch.ones_like(values))


def source_archive(count=26):
    return FactorArchive(
        initial_sensor_poses=pp.identity_SE3(
            count, dtype=torch.float64
        ).tensor(),
        time_ns=np.arange(count, dtype=np.int64) + 1000,
        T_BS=pp.identity_SE3(dtype=torch.float64).tensor(),
        edges=(),
        edge_kinds=(),
        metadata={},
    )


def candidate(source, target):
    return CovisibilityMetrics(
        source=source,
        target=target,
        overlap_forward=0.6,
        overlap_backward=0.6,
        mean_overlap=0.6,
        depth_consistency=0.7,
        median_motion_px=10.0,
        score=0.3,
        status="eligible",
        reason=None,
    )


class FakeSelector:
    def __init__(self):
        self.config = SimpleNamespace(min_temporal_gap=15)
        self.calls = []

    def select(self, *, target, history, accepted_pairs):
        self.calls.append((target, tuple(history), frozenset(accepted_pairs)))
        source = target - 15
        selected = candidate(source, target)
        return selected, [selected]


class FakeFrontend:
    def __init__(self, fail_target=None):
        self.calls = []
        self.fail_target = fail_target

    def estimate_bidirectional(self, source, target):
        self.calls.append((source.tag, target.tag))
        if target.tag == self.fail_target:
            raise RuntimeError("frontend failed")
        return SimpleNamespace(frame_id=source.tag), object(), object()


def make_record(selected):
    covariance = torch.eye(3, dtype=torch.float64)[None] * 0.001
    points = torch.tensor([[3.0, 0.0, 0.0]], dtype=torch.float64)
    return ProximityFactorRecord(
        edge=Edge(
            selected.source, selected.target,
            points, points.clone(), covariance, covariance.clone(),
        ),
        score=selected.score,
        confidence=0.8,
        age=0,
        state="accepted",
        candidate_metrics=selected.as_dict(),
        validation_metrics={"mahalanobis_inlier_ratio": 1.0},
    )


class FakeValidator:
    def __init__(self, reject_pairs=()):
        self.reject_pairs = set(reject_pairs)
        self.calls = []
        self.target_depth_ids = []

    def __call__(self, **kwargs):
        selected = kwargs["candidate"]
        pair = (selected.source, selected.target)
        self.calls.append(pair)
        self.target_depth_ids.append(
            int(kwargs["target_depth"].depth[0, 0, 0, 0])
        )
        if pair in self.reject_pairs:
            return ProximityValidationResult(
                None, "rejected_fixture", {"reason": "rejected_fixture"}
            )
        return ProximityValidationResult(
            make_record(selected), None, {"reason": None}
        )


def generation_inputs(**updates):
    source = source_archive()
    values = {
        "sequence": FakeSequence(26),
        "source_archive": source,
        "cache": FakeCache(source),
        "frontend": FakeFrontend(),
        "covisibility_selector": FakeSelector(),
        "keypoint_selector": object(),
        "covariance_model": object(),
        "num_point": 200,
        "edge_width": 32,
        "match_cov_default": 0.25,
        "device": "cpu",
        "validation_config": object(),
        "validator": FakeValidator(),
    }
    values.update(updates)
    return values


class GenerateProximityICPTests(unittest.TestCase):
    def test_generator_processes_targets_chronologically_and_adds_at_most_one_edge(self):
        validator = FakeValidator()
        result = generate_proximity_archive(**generation_inputs(validator=validator))

        self.assertEqual(result.diagnostics["targets"], [15, 20, 25])
        self.assertEqual(result.diagnostics["accepted_edges"], 3)
        self.assertEqual(
            [(edge.a, edge.b) for edge in result.archive.edges],
            [(0, 15), (5, 20), (10, 25)],
        )
        self.assertEqual(validator.target_depth_ids, [15, 20, 25])

    def test_failed_match_does_not_nms_suppress_future_candidate(self):
        selector = FakeSelector()
        validator = FakeValidator(reject_pairs={(0, 15)})

        result = generate_proximity_archive(**generation_inputs(
            covisibility_selector=selector, validator=validator
        ))

        target20 = next(call for call in selector.calls if call[0] == 20)
        self.assertEqual(target20[2], frozenset())
        self.assertEqual(result.diagnostics["accepted_edges"], 2)

    def test_no_candidate_is_recorded_without_frontend_call(self):
        selector = FakeSelector()
        selector.select = lambda **_: (None, [])
        frontend = FakeFrontend()

        result = generate_proximity_archive(**generation_inputs(
            covisibility_selector=selector, frontend=frontend
        ))

        self.assertEqual(frontend.calls, [])
        self.assertEqual(result.diagnostics["status"], "no_proximity_edges")
        self.assertEqual(result.diagnostics["no_candidate_targets"], 3)

    def test_frontend_failure_is_recorded_and_later_targets_continue(self):
        frontend = FakeFrontend(fail_target=20)

        result = generate_proximity_archive(**generation_inputs(
            frontend=frontend
        ))

        self.assertEqual(len(frontend.calls), 3)
        self.assertEqual(result.diagnostics["inference_failures"][0]["target"], 20)
        self.assertEqual(result.diagnostics["accepted_edges"], 2)

    def test_source_identity_mismatch_is_rejected(self):
        values = generation_inputs()
        values["cache"].metadata["source_id"] = "wrong"

        with self.assertRaisesRegex(ValueError, "source identity"):
            generate_proximity_archive(**values)

    def test_cli_defaults_match_approved_thresholds(self):
        args = build_parser().parse_args(["--space", "/tmp/result"])

        self.assertEqual(args.min_temporal_gap, 15)
        self.assertEqual(args.max_candidates_per_target, 1)
        self.assertEqual(args.mahalanobis_threshold, 3.5)


if __name__ == "__main__":
    unittest.main()
