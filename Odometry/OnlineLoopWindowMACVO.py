"""WindowICP with asynchronous ORB retrieval and serialized online loop closure."""

from collections import Counter, deque
import copy
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pypose as pp
import torch

from Module.Frontend.SerializedFrontend import SerializedFrontend
from Module.LoopClosure.ORBBoW import (
    ORBLoopCandidateProvider,
    filter_loop_candidates,
)
from Module.LoopClosure.OnlineLoopStore import (
    LoopKeyframeStore,
    PendingLoopTarget,
    PendingLoopTargetStore,
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
    validate_sparse_loop,
)
from Module.Optimization.AsyncPoseGraph import (
    AsyncPoseGraphBackend,
    PoseGraphSnapshot,
)
from Module.Optimization.CovisibilitySelector import CovisibilityMetrics
from Module.Optimization.PairwiseICP import (
    compress_edge_to_pose_factor,
    linearize_edge_to_pose_factor,
)
from Module.Optimization.PoseGraph import (
    PoseGraphArchive,
    optimize_pose_graph,
    save_pose_graph_archive,
)
from Module.Optimization.ProximityICP import (
    ProximityValidationConfig,
    depth_output_to_device,
    validate_proximity_match,
)
from Odometry.WindowMACVO import WindowMACVO, reanchor_points


class OnlineLoopWindowMACVO(WindowMACVO):
    def __init__(
        self,
        *,
        online_loop,
        pairwise_icp,
        pose_graph,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.loop_enabled = bool(online_loop.enabled)
        self.loop_keyframe_stride = int(online_loop.keyframe_stride)
        self.loop_min_temporal_gap = int(online_loop.min_temporal_gap)
        self.bow_top_k = int(online_loop.bow_top_k)
        self.max_candidates_per_target = int(
            online_loop.max_candidates_per_target
        )
        self.candidate_nms_frames = int(online_loop.candidate_nms_frames)
        self.max_candidate_queue = int(online_loop.max_candidate_queue)
        self.max_pending_targets = int(online_loop.max_pending_targets)
        self.min_bow_score = float(getattr(online_loop, "min_bow_score", 0.02))
        self.min_frames_between_loop_matches = int(
            getattr(online_loop, "min_frames_between_loop_matches", 25)
        )
        self.validation_mode = str(
            getattr(online_loop, "validation_mode", "legacy_dense")
        )
        if self.validation_mode not in {"legacy_dense", "sparse_se3"}:
            raise ValueError("unknown online loop validation mode")
        self.orb_ratio_test = float(
            getattr(online_loop, "orb_ratio_test", 0.80)
        )
        self.orb_max_matches = int(
            getattr(online_loop, "orb_max_matches", 300)
        )
        self.orb_sidecar = str(online_loop.orb_sidecar)
        self.orb_vocabulary = str(online_loop.vocabulary)
        self.pairwise_iterations = int(pairwise_icp.iterations)
        self.pairwise_huber_delta = float(pairwise_icp.huber_delta)
        self.pose_graph_iterations = int(pose_graph.iterations)
        self.pose_graph_huber_delta = float(pose_graph.huber_delta)
        self.skip_information_cap = float(pose_graph.skip_information_cap)
        self.switch_prior = float(pose_graph.switch_prior)
        self.loop_information_scale = float(
            getattr(pose_graph, "loop_information_scale", 1.0)
        )
        if (
            not math.isfinite(self.loop_information_scale)
            or self.loop_information_scale <= 0
        ):
            raise ValueError("loop information scale must be positive")
        if (
            self.validation_mode != "sparse_se3"
            and self.loop_information_scale != 1.0
        ):
            raise ValueError(
                "non-unit loop information scale requires sparse_se3 mode"
            )
        self.compress_edge = compress_edge_to_pose_factor
        self.linearize_edge = linearize_edge_to_pose_factor
        self.pose_factors = {}
        self.pose_graph_version = 0
        self.compression_records = []
        self.loop_records = []
        self.hypothesis_records = []
        self.loop_candidates = deque(maxlen=self.max_candidate_queue)
        self.existing_loop_pairs = set()
        self.pending_loop_pairs = set()
        self.rejected_loop_pairs = set()
        self.candidate_filter_counts = Counter()
        self.expired_targets = 0
        self.last_loop_match_frame = -10**9
        self.pose_graph_writebacks = 0
        self.pose_graph_results = []
        self.runtime_folder = None
        self.loop_keyframes = None
        self.pending_targets = None
        self.loop_provider = None
        self.pose_backend = AsyncPoseGraphBackend(
            solver=self._solve_pose_graph
        )
        sparse_geometry = getattr(online_loop, "sparse_geometry", None)
        hypothesis = getattr(online_loop, "hypothesis", None)
        sparse_factor = getattr(online_loop, "sparse_factor", None)
        self.sparse_geometry_config = SparseGeometryConfig(
            **({} if sparse_geometry is None else vars(sparse_geometry))
        )
        self.hypothesis_tracker = LoopHypothesisTracker(HypothesisConfig(
            **({} if hypothesis is None else vars(hypothesis))
        ))
        self.sparse_translation_sigma = float(
            0.25 if sparse_factor is None
            else sparse_factor.translation_sigma_m
        )
        self.sparse_rotation_sigma_deg = float(
            10.0 if sparse_factor is None
            else sparse_factor.rotation_sigma_deg
        )
        self.sparse_validator = validate_sparse_loop
        self.sparse_factor_builder = conservative_sparse_factor
        self._wrap_frontend_once()

    def _effective_switch_prior(self):
        return self.switch_prior * self.loop_information_scale

    def _solve_pose_graph(self, poses, factors):
        return optimize_pose_graph(
            poses,
            factors,
            max_iters=self.pose_graph_iterations,
            huber_delta=self.pose_graph_huber_delta,
            skip_information_cap=self.skip_information_cap,
            switch_prior=self._effective_switch_prior(),
        )

    @classmethod
    def is_valid_config(cls, config):
        assert config is not None
        base_args = SimpleNamespace(**{
            key: value
            for key, value in vars(config.args).items()
            if key not in {"online_loop", "pairwise_icp", "pose_graph"}
        })
        base = SimpleNamespace(**vars(config))
        base.args = base_args
        WindowMACVO.is_valid_config(base)
        for name in ("online_loop", "pairwise_icp", "pose_graph"):
            if not hasattr(config.args, name):
                raise ValueError(f"Online loop config is missing {name}")
        scale = float(getattr(
            config.args.pose_graph, "loop_information_scale", 1.0
        ))
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("loop information scale must be positive")
        validation_mode = str(getattr(
            config.args.online_loop, "validation_mode", "legacy_dense"
        ))
        if validation_mode != "sparse_se3" and scale != 1.0:
            raise ValueError(
                "non-unit loop information scale requires sparse_se3 mode"
            )

    def _wrap_frontend_once(self):
        if not isinstance(self.Frontend, SerializedFrontend):
            self.Frontend = SerializedFrontend(self.Frontend)

    def set_runtime_folder(self, folder):
        self.runtime_folder = Path(folder)
        self.loop_keyframes = LoopKeyframeStore(
            self.runtime_folder / "online_loop_keyframes"
        )
        self.pending_targets = PendingLoopTargetStore(
            self.max_pending_targets
        )
        if self.loop_provider is None:
            if self.loop_enabled:
                self.loop_provider = ORBLoopCandidateProvider(
                    executable=Path(self.orb_sidecar),
                    vocabulary=Path(self.orb_vocabulary),
                    min_temporal_gap=self.loop_min_temporal_gap,
                    top_k=self.bow_top_k,
                    ratio_test=getattr(self, "orb_ratio_test", 0.80),
                    max_matches=getattr(self, "orb_max_matches", 300),
                    queue_size=self.max_candidate_queue,
                )
            else:
                self.loop_provider = None
        if self.loop_provider is not None and not self.loop_provider.enabled:
            self.loop_enabled = False

    def _compress_and_store(self, edge, kind, confidence=1.0):
        if edge is None:
            return False
        key = (edge.a, edge.b, kind)
        if key in self.pose_factors:
            return False
        if kind == "loop":
            result = self.compress_edge(
                edge,
                kind=kind,
                max_iters=self.pairwise_iterations,
                huber_delta=self.pairwise_huber_delta,
            )
        else:
            result = self.linearize_edge(
                edge,
                self.graph.frames.data["pose"].tensor.double(),
                kind=kind,
                huber_delta=self.pairwise_huber_delta,
            )
        self.compression_records.append({
            "a": edge.a,
            "b": edge.b,
            "kind": kind,
            "status": result.status,
            "reason": result.reason,
            "condition_number": result.condition_number,
        })
        if result.factor is None:
            return False
        self.pose_factors[key] = replace(
            result.factor, confidence=float(confidence)
        )
        self.pose_graph_version += 1
        return True

    def initialize(self, frame0):
        super().initialize(frame0)
        if self.loop_enabled:
            self._publish_loop_keyframe(frame0, self.prev_keyframe[2], 0)

    def _publish_loop_keyframe(self, frame, depth, frame_id):
        if (
            not self.loop_enabled
            or self.loop_keyframes is None
            or self.pending_targets is None
            or frame_id % self.loop_keyframe_stride
        ):
            return
        pose = self.graph.frames.data["pose"][frame_id].double()
        stored = self.loop_keyframes.add(frame_id, frame.stereo, pose)
        expired = self.pending_targets.add(
            PendingLoopTarget(frame_id, frame.stereo, depth)
        )
        if expired is not None:
            self.expired_targets += 1
        try:
            self.loop_provider.submit(
                frame_id=frame_id, left_image_path=stored.left_path
            )
        except Exception as error:
            self.loop_enabled = False
            self.loop_records.append({
                "status": "provider_failed", "reason": str(error)
            })

    def _poll_orb_candidates(self):
        if self.loop_provider is None:
            return
        for response in self.loop_provider.poll():
            raw = [
                candidate
                for candidate in response.candidates
                if candidate.score >= self.min_bow_score
            ]
            selected, counts = filter_loop_candidates(
                target=response.target,
                candidates=raw,
                existing_pairs=self.existing_loop_pairs,
                pending_pairs=self.pending_loop_pairs,
                rejected_pairs=self.rejected_loop_pairs,
                nms_frames=self.candidate_nms_frames,
                maximum=self.max_candidates_per_target,
            )
            self.candidate_filter_counts.update(counts)
            if not selected:
                self.pending_targets.pop(response.target)
            for candidate in selected:
                pair = (candidate.source, response.target)
                if len(self.loop_candidates) == self.loop_candidates.maxlen:
                    dropped_target, dropped = self.loop_candidates.popleft()
                    dropped_pair = (dropped.source, dropped_target)
                    self.pending_loop_pairs.discard(dropped_pair)
                    if (
                        dropped_target != response.target
                        and not any(
                            queued_target == dropped_target
                            for queued_target, _ in self.loop_candidates
                        )
                    ):
                        self.pending_targets.pop(dropped_target)
                self.pending_loop_pairs.add(pair)
                self.loop_candidates.append((response.target, candidate))

    def _release_pending_target(self, target_id):
        if not any(
            queued_target == target_id
            for queued_target, _ in self.loop_candidates
        ):
            self.pending_targets.pop(target_id)

    def _expire_sparse_hypotheses(self, current_target):
        if getattr(self, "validation_mode", "legacy_dense") != "sparse_se3":
            return
        for hypothesis_id in self.hypothesis_tracker.expire(current_target):
            self.hypothesis_records.append({
                "hypothesis_id": hypothesis_id,
                "target": current_target,
                "state": "expired",
            })

    def _validate_sparse_candidate(self, source, target, candidate):
        if candidate.mutual_matches < self.sparse_geometry_config.min_mutual_matches:
            return SparseLoopResult(
                source=candidate.source,
                target=int(getattr(target, "frame_id", -1)),
                measurement=None,
                inlier_mask=torch.zeros(candidate.mutual_matches, dtype=torch.bool),
                metrics={
                    "raw_knn_matches": candidate.raw_knn_matches,
                    "ratio_matches": candidate.ratio_matches,
                    "mutual_matches": candidate.mutual_matches,
                },
                reason="insufficient_mutual_matches",
            )
        source_depth = self.Frontend.estimate_loop_depth(source.stereo)
        result = self.sparse_validator(
            source=candidate.source,
            target=target.frame_id,
            matches=candidate.matches,
            source_stereo=source.stereo,
            target_stereo=target.stereo,
            source_depth=source_depth,
            target_depth=depth_output_to_device(target.depth, self.device),
            covariance_model=self.ObsCovModel,
            config=self.sparse_geometry_config,
            seed=1000003 + candidate.source * 1009 + target.frame_id,
        )
        result.metrics.update({
            "raw_knn_matches": candidate.raw_knn_matches,
            "ratio_matches": candidate.ratio_matches,
            "mutual_matches": candidate.mutual_matches,
        })
        return result

    def _register_sparse_support(self, candidate, validation):
        tracker_before = copy.deepcopy(self.hypothesis_tracker)
        hypothesis_record_count = len(self.hypothesis_records)
        loop_record_count = len(self.loop_records)
        factors_before = dict(self.pose_factors)
        existing_before = set(self.existing_loop_pairs)
        graph_version_before = self.pose_graph_version
        try:
            return self._register_sparse_support_unchecked(candidate, validation)
        except Exception:
            self.hypothesis_tracker = tracker_before
            del self.hypothesis_records[hypothesis_record_count:]
            del self.loop_records[loop_record_count:]
            self.pose_factors = factors_before
            self.existing_loop_pairs = existing_before
            self.pose_graph_version = graph_version_before
            raise

    def _register_sparse_support_unchecked(self, candidate, validation):
        poses = pp.SE3(self.graph.frames.data["pose"].tensor.double())
        measurement = pp.SE3(validation.measurement)
        correction = (
            poses[validation.source] @ measurement @ poses[validation.target].Inv()
        ).tensor()
        metrics = dict(validation.metrics)
        reprojection_p90 = max(
            float(metrics.get("reprojection_forward_p90_px", float("inf"))),
            float(metrics.get("reprojection_backward_p90_px", float("inf"))),
        )
        support = LoopSupport(
            source=validation.source,
            target=validation.target,
            measurement=validation.measurement,
            correction=correction,
            quality=(
                int(metrics["ransac_inliers"]),
                float(metrics["ransac_ratio"]),
                int(metrics.get("final_grid_cells", 0)),
                -reprojection_p90,
                float(candidate.score),
            ),
            metrics=metrics | {
                "bow_score": float(candidate.score),
                "rank": int(candidate.rank),
            },
        )
        update = self.hypothesis_tracker.add(support)
        self.hypothesis_records.append({
            "hypothesis_id": update.hypothesis_id,
            "source": validation.source,
            "target": validation.target,
            "state": update.state,
            "reason": update.reason,
            "support_count": next((
                len(item.supports)
                for item in self.hypothesis_tracker.hypotheses
                if item.hypothesis_id == update.hypothesis_id
            ), 0),
        })
        if update.emitted is None:
            return False
        emitted = update.emitted
        factor = self.sparse_factor_builder(
            source=emitted.source,
            target=emitted.target,
            measurement=emitted.measurement,
            inliers=int(emitted.metrics["ransac_inliers"]),
            inlier_ratio=float(emitted.metrics["ransac_ratio"]),
            translation_sigma_m=self.sparse_translation_sigma,
            rotation_sigma_deg=self.sparse_rotation_sigma_deg,
            information_scale=getattr(self, "loop_information_scale", 1.0),
        )
        key = (factor.a, factor.b, factor.kind)
        if key in self.pose_factors:
            return False
        self.pose_factors[key] = factor
        self.pose_graph_version += 1
        self.existing_loop_pairs.add((factor.a, factor.b))
        eigenvalues = torch.linalg.eigvalsh(factor.information)
        self.loop_records.append({
            "source": factor.a,
            "target": factor.b,
            "gap": factor.b - factor.a,
            "status": "accepted_sparse",
            "hypothesis_id": update.hypothesis_id,
            "supports": next(
                len(item.supports)
                for item in self.hypothesis_tracker.hypotheses
                if item.hypothesis_id == update.hypothesis_id
            ),
            "observations": factor.observation_count,
            "confidence": factor.confidence,
            "bow_score": float(emitted.metrics["bow_score"]),
            "rank": int(emitted.metrics["rank"]),
            "metrics": emitted.metrics,
            "measurement_se3": emitted.measurement.tolist(),
            "information_eigenvalues": eigenvalues.tolist(),
        })
        self.pose_backend.submit(self._pose_graph_snapshot())
        return True

    def _process_one_loop_candidate(self, current_frame_id):
        if (
            not self.loop_candidates
            or current_frame_id - self.last_loop_match_frame
            < self.min_frames_between_loop_matches
        ):
            return False
        target_id, candidate = self.loop_candidates.popleft()
        pair = (candidate.source, target_id)
        self.pending_loop_pairs.discard(pair)
        target = self.pending_targets.get(target_id)
        if target is None or candidate.source not in self.loop_keyframes.records:
            self.rejected_loop_pairs.add(pair)
            self.loop_records.append({
                "source": candidate.source,
                "target": target_id,
                "status": "expired",
            })
            self._release_pending_target(target_id)
            return False
        try:
            source = self.loop_keyframes.load(candidate.source)
            source.stereo.imageL = source.stereo.imageL.to(target.stereo.imageL)
            source.stereo.imageR = source.stereo.imageR.to(target.stereo.imageR)
        except Exception as error:
            self.last_loop_match_frame = current_frame_id
            self._release_pending_target(target_id)
            self.rejected_loop_pairs.add(pair)
            self.loop_records.append({
                "source": candidate.source,
                "target": target_id,
                "bow_score": candidate.score,
                "status": "validation_failed",
                "reason": str(error),
            })
            return False
        if self.validation_mode == "sparse_se3":
            try:
                validation = self._validate_sparse_candidate(
                    source, target, candidate
                )
            except Exception as error:
                self.last_loop_match_frame = current_frame_id
                self._release_pending_target(target_id)
                self.rejected_loop_pairs.add(pair)
                self.loop_records.append({
                    "source": candidate.source,
                    "target": target_id,
                    "bow_score": candidate.score,
                    "status": "validation_failed",
                    "reason": str(error),
                })
                return False
            self.last_loop_match_frame = current_frame_id
            self._release_pending_target(target_id)
            if validation.reason is not None:
                self.rejected_loop_pairs.add(pair)
                self.loop_records.append({
                    "source": candidate.source,
                    "target": target_id,
                    "bow_score": candidate.score,
                    "status": "rejected_sparse",
                    "reason": validation.reason,
                    "metrics": validation.metrics,
                })
                return False
            try:
                accepted = self._register_sparse_support(candidate, validation)
            except Exception as error:
                self.rejected_loop_pairs.add(pair)
                self.loop_records.append({
                    "source": candidate.source,
                    "target": target_id,
                    "bow_score": candidate.score,
                    "rank": candidate.rank,
                    "status": "validation_failed",
                    "reason": str(error),
                    "metrics": validation.metrics,
                    "measurement_se3": validation.measurement.tolist(),
                })
                return False
            latest = self.hypothesis_records[-1]
            self.loop_records.append({
                "source": candidate.source,
                "target": target_id,
                "bow_score": candidate.score,
                "rank": candidate.rank,
                "status": latest["state"],
                "hypothesis_id": latest["hypothesis_id"],
                "metrics": validation.metrics,
                "measurement_se3": validation.measurement.tolist(),
            })
            return accepted
        source_depth, forward, backward = self.Frontend.estimate_bidirectional(
            source.stereo, target.stereo
        )
        metric = CovisibilityMetrics(
            source=candidate.source,
            target=target_id,
            overlap_forward=1.0,
            overlap_backward=1.0,
            mean_overlap=1.0,
            depth_consistency=1.0,
            median_motion_px=0.0,
            score=max(0.0, 1.0 - candidate.score),
            status="eligible",
            reason=None,
        )
        validation_config = ProximityValidationConfig(
            mahalanobis_threshold=1e9,
            min_mahalanobis_inliers=30,
            min_mahalanobis_inlier_ratio=0.5,
        )
        validation = validate_proximity_match(
            candidate=metric,
            forward=forward,
            backward=backward,
            source_frame=source,
            target_frame=SimpleNamespace(stereo=target.stereo),
            source_depth=source_depth,
            target_depth=depth_output_to_device(target.depth, self.device),
            poses=self.graph.frames.data["pose"].tensor.double(),
            frontend=self.Frontend,
            keypoint_selector=self.KeypointSelector,
            covariance_model=self.ObsCovModel,
            num_point=self.num_point,
            edge_width=self.edge_width,
            match_cov_default=self.match_cov_default,
            device=self.device,
            config=validation_config,
        )
        self.last_loop_match_frame = current_frame_id
        self._release_pending_target(target_id)
        if validation.record is None:
            self.rejected_loop_pairs.add(pair)
            self.loop_records.append({
                "source": candidate.source,
                "target": target_id,
                "bow_score": candidate.score,
                "status": "rejected",
                "reason": validation.reason,
            })
            return False
        accepted = self._compress_and_store(
            validation.record.edge,
            "loop",
            validation.record.confidence,
        )
        if not accepted:
            self.rejected_loop_pairs.add(pair)
            return False
        self.existing_loop_pairs.add(pair)
        self.loop_records.append({
            "source": candidate.source,
            "target": target_id,
            "gap": target_id - candidate.source,
            "bow_score": candidate.score,
            "status": "accepted",
            "observations": len(validation.record.edge.points_a),
        })
        self.pose_backend.submit(self._pose_graph_snapshot())
        return True

    def _pose_graph_snapshot(self):
        poses = self.graph.frames.data["pose"].tensor.double().clone()
        factors = tuple(
            self.pose_factors[key] for key in sorted(self.pose_factors)
        )
        return PoseGraphSnapshot(
            self.pose_graph_version, len(poses), poses, factors
        )

    def _apply_pose_graph_result(self, asynchronous):
        if asynchronous.status != "refined" or asynchronous.result is None:
            self.pose_graph_results.append({
                "status": asynchronous.status,
                "reason": asynchronous.reason,
            })
            return False
        solved = asynchronous.result
        if isinstance(solved, dict):
            return False
        if solved.diagnostics.get("status") != "refined":
            return False
        old = self.graph.frames.data["pose"].tensor.double().clone()
        count = asynchronous.frame_count
        if count > len(old) or len(solved.poses) != count:
            return False
        after = old.clone()
        after[:count] = solved.poses
        if count < len(old):
            correction = pp.SE3(after[count - 1]) @ pp.SE3(
                old[count - 1]
            ).Inv()
            after[count:] = (correction @ pp.SE3(old[count:])).tensor()
        reanchor_points(self.graph, list(range(len(old))), old, after)
        self.graph.frames.data["pose"].tensor[:] = after.float()
        if self.prev_keyframe is not None:
            self.MotionEstimator.update(pp.SE3(after[-1]))
        self.pose_graph_writebacks += 1
        self.pose_graph_results.append(solved.diagnostics)
        return True

    def _poll_pose_backend(self):
        results = self.pose_backend.poll()
        if results:
            self._apply_pose_graph_result(results[-1])

    def _after_window_step(self, frame, depth, adjacent, skip):
        self._compress_and_store(adjacent, "adjacent")
        self._compress_and_store(skip, "skip2")
        frame_id = self.prev_keyframe[1]
        self._expire_sparse_hypotheses(frame_id)
        self._publish_loop_keyframe(frame, depth, frame_id)
        self._poll_orb_candidates()
        self._process_one_loop_candidate(frame_id)
        self._poll_pose_backend()

    def terminate(self):
        if self.terminated:
            return
        if self.loop_provider is not None:
            self.loop_provider.close()
            self._poll_orb_candidates()
            while self.loop_candidates:
                self.last_loop_match_frame = -10**9
                if not self._process_one_loop_candidate(
                    len(self.graph.frames) - 1
                ):
                    continue
        super().terminate()
        if self.pose_factors:
            self.pose_backend.terminate(self._pose_graph_snapshot())
        else:
            self.pose_backend.close()
        results = self.pose_backend.poll()
        if results:
            self._apply_pose_graph_result(results[-1])

    def save_window_diagnostics(self, folder):
        super().save_window_diagnostics(folder)
        folder = Path(folder)
        factors = tuple(
            self.pose_factors[key] for key in sorted(self.pose_factors)
        )
        if factors:
            graph = self.graph.frames.data
            save_pose_graph_archive(
                folder / "pose_graph_factors.npz",
                PoseGraphArchive(
                    graph["pose"].tensor.double(),
                    graph["time_ns"].tensor.cpu().numpy(),
                    factors,
                    {"graph_version": self.pose_graph_version},
                ),
            )
        payload = self._online_loop_diagnostics(factors)
        (folder / "online_loop_diagnostics.json").write_text(
            json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
        )

    def _online_loop_diagnostics(self, factors):
        rejection_counts = Counter(
            item.get("reason") for item in self.loop_records
            if item.get("reason") is not None
        )
        return {
            "loop_enabled": self.loop_enabled,
            "validation_mode": getattr(self, "validation_mode", "legacy_dense"),
            "loop_information_scale": getattr(
                self, "loop_information_scale", 1.0
            ),
            "switch_prior_base": getattr(self, "switch_prior", 1.0),
            "switch_prior_effective": (
                self._effective_switch_prior()
                if hasattr(self, "switch_prior") else 1.0
            ),
            "provider": None if self.loop_provider is None else self.loop_provider.status,
            "protocol_version": (
                None if self.loop_provider is None
                else self.loop_provider.handshake.get("protocol_version")
            ),
            "frontend": self.Frontend.diagnostics,
            "pose_graph_version": self.pose_graph_version,
            "pose_factors": len(factors),
            "factor_kinds": dict(Counter(factor.kind for factor in factors)),
            "loop_records": self.loop_records,
            "hypothesis_records": getattr(self, "hypothesis_records", []),
            "sparse_geometry": asdict(getattr(
                self, "sparse_geometry_config", SparseGeometryConfig()
            )),
            "sparse_factor": {
                "translation_sigma_m": getattr(
                    self, "sparse_translation_sigma", 0.25
                ),
                "rotation_sigma_deg": getattr(
                    self, "sparse_rotation_sigma_deg", 10.0
                ),
            },
            "sparse_rejection_counts": dict(rejection_counts),
            "compression_records": self.compression_records,
            "candidate_filter_counts": dict(self.candidate_filter_counts),
            "expired_targets": self.expired_targets,
            "pose_backend": self.pose_backend.diagnostics,
            "pose_graph_writebacks": self.pose_graph_writebacks,
            "pose_graph_results": self.pose_graph_results,
            "cuda_max_memory_reserved": (
                torch.cuda.max_memory_reserved()
                if torch.cuda.is_available() else 0
            ),
        }
