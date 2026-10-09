import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pypose as pp
import torch

from Module.Frontend.SerializedFrontend import SerializedFrontend
from Module.LoopClosure.ORBBoW import (
    ORBLoopCandidate,
    ORBLoopCandidateBatch,
)
from Module.LoopClosure.LoopHypothesis import (
    HypothesisConfig,
    LoopHypothesisTracker,
    LoopSupport,
)
from Module.LoopClosure.SparseGeometry import (
    SparseGeometryConfig,
    SparseLoopResult,
    conservative_sparse_factor,
)
from Module.Optimization.PairwiseICP import PairwiseCompressionResult
from Module.Optimization.PoseGraph import PoseGraphFactor
from Module.Optimization.WindowICP import Edge
from Odometry.OnlineLoopWindowMACVO import OnlineLoopWindowMACVO
from Odometry.WindowMACVO import WindowMACVO


def edge(a=0, b=1):
    points = torch.tensor([[3.0, 0.0, 0.0]], dtype=torch.float64)
    covariance = torch.eye(3, dtype=torch.float64)[None] * 0.001
    return Edge(a, b, points, points.clone(), covariance, covariance.clone())


def factor(a=0, b=1, kind="adjacent"):
    return PoseGraphFactor(
        a, b, pp.identity_SE3(dtype=torch.float64).tensor(),
        torch.eye(6, dtype=torch.float64), kind, 1.0, 10,
    )


def sparse_candidate(source, target, count=50, score=0.2, rank=0):
    matches = np.zeros((count, 5), dtype=np.float64)
    matches[:, 0] = np.linspace(10, 50, count)
    matches[:, 1] = np.linspace(10, 40, count)
    matches[:, 2:4] = matches[:, :2]
    return ORBLoopCandidate(
        source=source,
        score=score,
        rank=rank,
        raw_knn_matches=count + 20,
        ratio_matches=count + 10,
        matches=matches,
    )


def sparse_validation(
    source, target, translation=0.0, inliers=40, inlier_ratio=0.8
):
    measurement = pp.se3(torch.tensor(
        [translation, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64
    )).Exp().tensor()
    return SparseLoopResult(
        source=source,
        target=target,
        measurement=measurement,
        inlier_mask=torch.ones(inliers, dtype=torch.bool),
        metrics={
            "ransac_inliers": inliers,
            "ransac_ratio": inlier_ratio,
            "final_grid_cells": 10,
            "reprojection_forward_p90_px": 1.0,
            "reprojection_backward_p90_px": 1.0,
        },
        reason=None,
    )


class OnlineLoopWindowMACVOTests(unittest.TestCase):
    def test_window_hook_is_default_noop(self):
        system = WindowMACVO.__new__(WindowMACVO)

        self.assertIsNone(system._after_window_step(None, None, None, None))

    def test_frontend_wrapper_reuses_original_object(self):
        original = SimpleNamespace(model=object(), cuda_graph=object())
        system = OnlineLoopWindowMACVO.__new__(OnlineLoopWindowMACVO)
        system.Frontend = original

        system._wrap_frontend_once()
        first = system.Frontend
        system._wrap_frontend_once()

        self.assertIsInstance(first, SerializedFrontend)
        self.assertIs(system.Frontend, first)
        self.assertIs(first.frontend, original)

    def test_compressed_factor_is_inserted_once_and_increments_version(self):
        system = OnlineLoopWindowMACVO.__new__(OnlineLoopWindowMACVO)
        system.pose_factors = {}
        system.pose_graph_version = 0
        system.compression_records = []
        system.pairwise_iterations = 5
        system.pairwise_huber_delta = 3.0
        system.compress_edge = Mock(return_value=PairwiseCompressionResult(
            factor(0, 1), "compressed", None, 1.0, 0.5, 10.0
        ))
        system.linearize_edge = Mock(return_value=PairwiseCompressionResult(
            factor(0, 1), "compressed", None, 0.5, 0.5, 10.0
        ))
        system.graph = SimpleNamespace(frames=SimpleNamespace(
            data={"pose": SimpleNamespace(tensor=pp.identity_SE3(2).tensor())}
        ))

        first = system._compress_and_store(edge(), "adjacent")
        second = system._compress_and_store(edge(), "adjacent")

        self.assertTrue(first)
        self.assertFalse(second)
        self.assertEqual(system.pose_graph_version, 1)
        self.assertEqual(len(system.pose_factors), 1)
        system.linearize_edge.assert_called_once()
        system.compress_edge.assert_not_called()

    def test_runtime_folder_creates_bounded_online_stores(self):
        system = OnlineLoopWindowMACVO.__new__(OnlineLoopWindowMACVO)
        system.loop_enabled = True
        system.max_pending_targets = 3
        system.loop_provider = None
        system.orb_sidecar = "/missing"
        system.orb_vocabulary = "/missing"
        system.loop_min_temporal_gap = 30
        system.bow_top_k = 3
        system.max_candidate_queue = 8

        with tempfile.TemporaryDirectory() as directory:
            system.set_runtime_folder(Path(directory))

            self.assertTrue(Path(directory, "online_loop_keyframes").is_dir())
            self.assertEqual(system.pending_targets.capacity, 3)
            self.assertFalse(system.loop_provider.enabled)

    def test_orb_response_without_eligible_candidate_releases_target(self):
        system = OnlineLoopWindowMACVO.__new__(OnlineLoopWindowMACVO)
        system.loop_provider = SimpleNamespace(poll=lambda: [
            ORBLoopCandidateBatch(
                1, 40, (ORBLoopCandidate(0, 0.001, 0),)
            )
        ])
        system.min_bow_score = 0.005
        system.existing_loop_pairs = set()
        system.pending_loop_pairs = set()
        system.rejected_loop_pairs = set()
        system.candidate_nms_frames = 10
        system.max_candidates_per_target = 1
        system.candidate_filter_counts = __import__("collections").Counter()
        system.loop_candidates = __import__("collections").deque(maxlen=8)
        system.pending_targets = SimpleNamespace(pop=Mock())

        system._poll_orb_candidates()

        system.pending_targets.pop.assert_called_once_with(40)

    def test_sparse_candidate_with_too_few_matches_skips_gpu_depth(self):
        system = OnlineLoopWindowMACVO.__new__(OnlineLoopWindowMACVO)
        system.sparse_geometry_config = SparseGeometryConfig(
            min_mutual_matches=40
        )
        system.Frontend = SimpleNamespace(estimate_loop_depth=Mock())
        system.sparse_validator = Mock()
        source = SimpleNamespace(stereo=object())
        target = SimpleNamespace(stereo=object(), depth=object())

        result = system._validate_sparse_candidate(
            source, target, sparse_candidate(0, 50, count=20)
        )

        self.assertEqual(result.reason, "insufficient_mutual_matches")
        system.Frontend.estimate_loop_depth.assert_not_called()
        system.sparse_validator.assert_not_called()

    def test_second_sparse_support_stores_one_direct_loop_factor(self):
        system = OnlineLoopWindowMACVO.__new__(OnlineLoopWindowMACVO)
        system.graph = SimpleNamespace(frames=SimpleNamespace(data={
            "pose": SimpleNamespace(tensor=pp.identity_SE3(30).tensor())
        }))
        system.hypothesis_tracker = LoopHypothesisTracker(HypothesisConfig(
            min_supports=2,
            strong_supports=3,
            source_cluster_frames=10,
            target_support_frames=10,
            max_correction_translation_m=0.25,
            max_correction_rotation_deg=10.0,
        ))
        system.sparse_factor_builder = conservative_sparse_factor
        system.sparse_translation_sigma = 0.25
        system.sparse_rotation_sigma_deg = 10.0
        system.loop_information_scale = 10000.0
        system.pose_factors = {}
        system.pose_graph_version = 0
        system.existing_loop_pairs = set()
        system.hypothesis_records = []
        system.loop_records = []
        system.pose_backend = SimpleNamespace(submit=Mock())
        system._pose_graph_snapshot = Mock(return_value="snapshot")
        system.compress_edge = Mock()

        first = system._register_sparse_support(
            sparse_candidate(10, 20, score=0.4, rank=1),
            sparse_validation(10, 20, inliers=50, inlier_ratio=0.9),
        )
        second = system._register_sparse_support(
            sparse_candidate(15, 25, score=0.2, rank=0),
            sparse_validation(15, 25, inliers=40, inlier_ratio=0.8),
        )

        self.assertFalse(first)
        self.assertTrue(second)
        self.assertEqual(len(system.pose_factors), 1)
        stored = next(iter(system.pose_factors.values()))
        self.assertEqual((stored.a, stored.b, stored.kind), (10, 20, "loop"))
        self.assertAlmostEqual(float(stored.information[0, 0]), 160000.0)
        system.compress_edge.assert_not_called()
        system.pose_backend.submit.assert_called_once_with("snapshot")
        accepted = next(
            item for item in system.loop_records
            if item["status"] == "accepted_sparse"
        )
        self.assertEqual(accepted["bow_score"], 0.4)
        self.assertEqual(accepted["rank"], 1)
        self.assertEqual(accepted["metrics"]["ransac_inliers"], 50)
        self.assertEqual(len(accepted["measurement_se3"]), 7)

    def test_sparse_factor_failure_rolls_back_hypothesis_emission(self):
        system = OnlineLoopWindowMACVO.__new__(OnlineLoopWindowMACVO)
        system.graph = SimpleNamespace(frames=SimpleNamespace(data={
            "pose": SimpleNamespace(tensor=pp.identity_SE3(30).tensor())
        }))
        system.hypothesis_tracker = LoopHypothesisTracker(HypothesisConfig())
        system.sparse_factor_builder = Mock(side_effect=RuntimeError("factor failed"))
        system.sparse_translation_sigma = 0.25
        system.sparse_rotation_sigma_deg = 10.0
        system.loop_information_scale = 1.0
        system.pose_factors = {}
        system.pose_graph_version = 0
        system.existing_loop_pairs = set()
        system.hypothesis_records = []
        system.loop_records = []
        system.pose_backend = SimpleNamespace(submit=Mock())
        system._pose_graph_snapshot = Mock(return_value="snapshot")
        system._register_sparse_support(
            sparse_candidate(10, 20), sparse_validation(10, 20)
        )

        with self.assertRaisesRegex(RuntimeError, "factor failed"):
            system._register_sparse_support(
                sparse_candidate(15, 25), sparse_validation(15, 25)
            )

        hypothesis = system.hypothesis_tracker.hypotheses[0]
        self.assertEqual(len(hypothesis.supports), 1)
        self.assertFalse(hypothesis.emitted)
        self.assertEqual(system.pose_factors, {})
        self.assertEqual(system.pose_graph_version, 0)

    def test_sparse_registration_exception_is_recorded_and_continues(self):
        system = OnlineLoopWindowMACVO.__new__(OnlineLoopWindowMACVO)
        candidate = sparse_candidate(0, 50)
        system.loop_candidates = __import__("collections").deque([(50, candidate)])
        system.last_loop_match_frame = -10**9
        system.min_frames_between_loop_matches = 0
        system.pending_loop_pairs = {(0, 50)}
        target = SimpleNamespace(frame_id=50, stereo=SimpleNamespace())
        system.pending_targets = SimpleNamespace(get=lambda _: target, pop=Mock())
        source = SimpleNamespace(stereo=SimpleNamespace(
            imageL=torch.zeros(1), imageR=torch.zeros(1)
        ))
        target.stereo.imageL = torch.zeros(1)
        target.stereo.imageR = torch.zeros(1)
        system.loop_keyframes = SimpleNamespace(
            records={0: object()}, load=Mock(return_value=source)
        )
        system.rejected_loop_pairs = set()
        system.loop_records = []
        system.validation_mode = "sparse_se3"
        system._validate_sparse_candidate = Mock(
            return_value=sparse_validation(0, 50)
        )
        system._register_sparse_support = Mock(
            side_effect=RuntimeError("registration failed")
        )

        result = system._process_one_loop_candidate(50)

        self.assertFalse(result)
        self.assertEqual(system.loop_records[-1]["status"], "validation_failed")
        self.assertIn("registration failed", system.loop_records[-1]["reason"])

    def test_sparse_validation_exception_does_not_stop_odometry(self):
        system = OnlineLoopWindowMACVO.__new__(OnlineLoopWindowMACVO)
        candidate = sparse_candidate(0, 50)
        system.loop_candidates = __import__("collections").deque([(50, candidate)])
        system.last_loop_match_frame = -10**9
        system.min_frames_between_loop_matches = 0
        system.pending_loop_pairs = {(0, 50)}
        target = SimpleNamespace(frame_id=50, stereo=SimpleNamespace())
        system.pending_targets = SimpleNamespace(get=lambda _: target, pop=Mock())
        system.loop_keyframes = SimpleNamespace(
            records={0: object()},
            load=Mock(side_effect=OSError("corrupt image")),
        )
        system.rejected_loop_pairs = set()
        system.loop_records = []
        system.validation_mode = "sparse_se3"

        result = system._process_one_loop_candidate(50)

        self.assertFalse(result)
        self.assertIn((0, 50), system.rejected_loop_pairs)
        self.assertEqual(system.loop_records[-1]["status"], "validation_failed")
        system.pending_targets.pop.assert_called_once_with(50)

    def test_sparse_hypotheses_expire_without_new_candidate(self):
        system = OnlineLoopWindowMACVO.__new__(OnlineLoopWindowMACVO)
        system.validation_mode = "sparse_se3"
        system.hypothesis_tracker = LoopHypothesisTracker(HypothesisConfig(
            target_support_frames=10
        ))
        system.hypothesis_records = []
        identity = pp.identity_SE3(dtype=torch.float64).tensor()
        update = system.hypothesis_tracker.add(LoopSupport(
            source=0,
            target=20,
            measurement=identity,
            correction=identity,
            quality=(1,),
            metrics={},
        ))

        system._expire_sparse_hypotheses(31)

        self.assertEqual(system.hypothesis_records[-1], {
            "hypothesis_id": update.hypothesis_id,
            "target": 31,
            "state": "expired",
        })

    def test_pending_target_is_released_after_last_queued_candidate(self):
        system = OnlineLoopWindowMACVO.__new__(OnlineLoopWindowMACVO)
        system.pending_targets = SimpleNamespace(pop=Mock())
        system.loop_candidates = __import__("collections").deque([
            (50, sparse_candidate(5, 50))
        ])

        system._release_pending_target(50)
        system.pending_targets.pop.assert_not_called()
        system.loop_candidates.clear()
        system._release_pending_target(50)
        system.pending_targets.pop.assert_called_once_with(50)

    def test_sparse_diagnostic_payload_contains_thresholds_and_hypotheses(self):
        system = OnlineLoopWindowMACVO.__new__(OnlineLoopWindowMACVO)
        system.loop_enabled = True
        system.validation_mode = "sparse_se3"
        system.loop_provider = SimpleNamespace(
            status={"status": "ready"}, handshake={"protocol_version": 2}
        )
        system.Frontend = SimpleNamespace(diagnostics={"loop_calls": 2})
        system.pose_graph_version = 1
        system.loop_records = [{"status": "accepted_sparse"}]
        system.hypothesis_records = [{"state": "confirmed"}]
        system.compression_records = []
        system.candidate_filter_counts = __import__("collections").Counter()
        system.expired_targets = 0
        system.pose_backend = SimpleNamespace(diagnostics={"submitted": 1})
        system.pose_graph_writebacks = 0
        system.pose_graph_results = []
        system.sparse_geometry_config = SparseGeometryConfig(
            min_ransac_inliers=25
        )
        system.sparse_translation_sigma = 0.25
        system.sparse_rotation_sigma_deg = 10.0
        system.loop_information_scale = 10000.0
        system.switch_prior = 1.0

        payload = system._online_loop_diagnostics((factor(0, 1, "loop"),))

        self.assertEqual(payload["validation_mode"], "sparse_se3")
        self.assertEqual(payload["protocol_version"], 2)
        self.assertEqual(payload["sparse_geometry"]["min_ransac_inliers"], 25)
        self.assertEqual(payload["hypothesis_records"], [{"state": "confirmed"}])
        self.assertEqual(payload["loop_information_scale"], 10000.0)
        self.assertEqual(payload["switch_prior_base"], 1.0)
        self.assertEqual(payload["switch_prior_effective"], 10000.0)

    def test_effective_switch_prior_scales_with_loop_information(self):
        system = OnlineLoopWindowMACVO.__new__(OnlineLoopWindowMACVO)
        system.switch_prior = 1.0
        system.loop_information_scale = 10000.0

        self.assertEqual(system._effective_switch_prior(), 10000.0)

    def test_pose_graph_solver_receives_effective_switch_prior(self):
        system = OnlineLoopWindowMACVO.__new__(OnlineLoopWindowMACVO)
        system.pose_graph_iterations = 10
        system.pose_graph_huber_delta = 3.0
        system.skip_information_cap = 0.5
        system.switch_prior = 1.0
        system.loop_information_scale = 10000.0

        with patch(
            "Odometry.OnlineLoopWindowMACVO.optimize_pose_graph",
            return_value="solved",
        ) as solve:
            result = system._solve_pose_graph("poses", "factors")

        self.assertEqual(result, "solved")
        solve.assert_called_once_with(
            "poses",
            "factors",
            max_iters=10,
            huber_delta=3.0,
            skip_information_cap=0.5,
            switch_prior=10000.0,
        )


if __name__ == "__main__":
    unittest.main()
