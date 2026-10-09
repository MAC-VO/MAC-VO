"""Five-frame, pose-only window refinement of the existing Fast ICP frontend."""
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pypose as pp
import torch

from Odometry.MACVO import MACVO
from Module.Optimization.FactorArchive import (
    SCHEMA_VERSION,
    FactorArchive,
    save_factor_archive,
    sensor_to_body_trajectory,
)
from Module.Optimization.GlobalPoseICP import optimize_global_pose_graph
from Module.Optimization.MatchICP import build_match_edge
from Module.Optimization.WindowICP import (
    Edge,
    EdgeWindow,
    edge_tensor_bytes,
    optimize_window,
    normalized_pose,
)
from Utility.Point import pixel2point_NED
from Utility.PrettyPrint import Logger


@torch.no_grad()
def reanchor_points(graph, frame_ids, before, after):
    """Transform each point by the correction of its source/owner frame."""
    for i, frame_id in enumerate(frame_ids):
        if torch.equal(before[i], after[i]):
            continue
        observations = graph.get_frame2match(graph.frames[torch.tensor([frame_id])])
        owner = graph.match2frame1.project(observations.index)
        owned = observations.index[owner == frame_id]
        if owned.numel() == 0:
            continue
        point_ids = graph.match2point.project(owned).unique()
        correction = pp.SE3(normalized_pose(after[i].double())) @ pp.SE3(normalized_pose(before[i].double())).Inv()
        correction = pp.SE3(normalized_pose(correction.tensor()))
        positions = graph.points.data["pos_Tw"][point_ids].double()
        covariances = graph.points.data["cov_Tw"][point_ids].double()
        R = correction.rotation().matrix()
        graph.points.data["pos_Tw"][point_ids] = correction.Act(positions).float()
        graph.points.data["cov_Tw"][point_ids] = R @ covariances @ R.T


class WindowMACVO(MACVO):
    def __init__(self, *, window_size=5, skip_matching=True,
                 window_iterations=10, window_huber_delta=3.0,
                 global_refine=False, global_iterations=5,
                 global_huber_delta=3.0, **kwargs):
        super().__init__(**kwargs)
        self.edge_window = EdgeWindow(window_size)
        self.skip_matching = skip_matching
        self.window_iterations = window_iterations
        self.window_huber_delta = window_huber_delta
        self.global_refine = global_refine
        self.global_iterations = global_iterations
        self.global_huber_delta = global_huber_delta
        self.frame_cache = {}
        self.window_records = []
        self.inactive_edges = {}
        self.global_record = {
            "status": "pending" if global_refine else "disabled"
        }
        self.retained_tensor_bytes = 0
        self.inactive_edge_count = 0
        self._factor_snapshot = None
        self.artifact_record = {
            "status": "pending" if global_refine else "disabled"
        }

    @classmethod
    def from_config(cls, cfg):
        c = cfg.Odometry
        cls.is_valid_config(c)
        if c.args.mapping or c.keyframe.type != "AllKeyframe":
            raise ValueError("Window v1 requires mapping=false and AllKeyframe")
        if c.optimizer.type != "TwoFrame_PGO" or c.optimizer.args.graph_type != "icp":
            raise ValueError("Window initializer must be TwoFrame_PGO with graph_type=icp")
        if c.optimizer.args.parallel:
            raise ValueError("Window v1 requires a synchronous initializer (parallel=false)")
        if c.outlier.type != "CovarianceSanityFilter":
            raise ValueError("Window v1 supports CovarianceSanityFilter")
        if c.args.window_iterations < 1 or c.args.window_huber_delta <= 0:
            raise ValueError("Invalid window solver settings")
        if getattr(c.args, "global_iterations", 5) < 1:
            raise ValueError("Invalid global solver iterations")
        if getattr(c.args, "global_huber_delta", 3.0) <= 0:
            raise ValueError("Invalid global Huber delta")
        return super().from_config(cfg)

    @classmethod
    def is_valid_config(cls, config):
        assert config is not None
        normalized_args = SimpleNamespace(**vars(config.args))
        for key, value in {
            "global_refine": False,
            "global_iterations": 5,
            "global_huber_delta": 3.0,
        }.items():
            if not hasattr(normalized_args, key):
                setattr(normalized_args, key, value)
        base_config = SimpleNamespace(**vars(config))
        base_config.args = SimpleNamespace(**{
            key: getattr(normalized_args, key) for key in (
                "device", "num_point", "edgewidth", "match_cov_default",
                "profile", "mapping",
            )
        })
        super().is_valid_config(base_config)
        cls._enforce_config_spec(normalized_args, {
            "device": lambda value: isinstance(value, str)
                and ("cuda" in value or value == "cpu"),
            "num_point": lambda value: isinstance(value, int) and value > 0,
            "edgewidth": lambda value: isinstance(value, int) and value > 0,
            "match_cov_default": lambda value: isinstance(value, (float, int))
                and value > 0.0,
            "profile": lambda value: isinstance(value, bool),
            "mapping": lambda value: isinstance(value, bool),
            "window_size": lambda value: isinstance(value, int) and 2 <= value <= 5,
            "skip_matching": lambda value: isinstance(value, bool),
            "window_iterations": lambda value: isinstance(value, int) and value >= 1,
            "window_huber_delta": lambda value: isinstance(value, (float, int))
                and value > 0.0,
            "global_refine": lambda value: isinstance(value, bool),
            "global_iterations": lambda value: isinstance(value, int) and value >= 1,
            "global_huber_delta": lambda value: isinstance(value, (float, int))
                and value > 0.0,
        })

    def initialize(self, frame0):
        super().initialize(frame0)
        self.frame_cache[0] = (frame0, self.prev_keyframe[2])

    def _after_window_step(self, frame, depth, adjacent, skip):
        return None

    def _edge(self, a, b, uv0, uv1, d0, d1, cov0, cov1, K0, K1):
        p0 = pixel2point_NED(uv0.cpu(), d0.cpu(), K0.cpu()).double()
        p1 = pixel2point_NED(uv1.cpu(), d1.cpu(), K1.cpu()).double()
        cov0, cov1 = cov0.cpu().double(), cov1.cpu().double()
        valid = (torch.isfinite(p0).all(-1) & torch.isfinite(p1).all(-1)
                 & (d0.cpu() > 0) & (d1.cpu() > 0)
                 & torch.isfinite(cov0).all(dim=(-2, -1)) & torch.isfinite(cov1).all(dim=(-2, -1)))
        ids = torch.nonzero(valid).flatten()
        if len(ids):
            good = ((torch.linalg.cholesky_ex(cov0[ids]).info == 0)
                    & (torch.linalg.cholesky_ex(cov1[ids]).info == 0))
            ids = ids[good]
        if len(ids) < self.min_num_point:
            return None
        return Edge(a, b, p0[ids], p1[ids], cov0[ids], cov1[ids])

    def _estimate_window_inputs(self, frame0, frame1):
        if not self.skip_matching:
            depth1, adjacent, _ = self.Frontend.estimate_window(
                None, frame0.stereo, frame1.stereo
            )
            return depth1, adjacent, None

        prev2 = self.frame_cache.get(self.prev_keyframe[1] - 1)
        frame_t2 = None if prev2 is None else prev2[0].stereo
        return self.Frontend.estimate_window(
            frame_t2, frame0.stereo, frame1.stereo
        )

    def _retain_evicted_edges(self, edges):
        if not self.global_refine:
            return
        for edge in edges:
            self.inactive_edges[(edge.a, edge.b)] = edge

    def _global_edges(self):
        edges = dict(self.inactive_edges)
        edges.update({
            (edge.a, edge.b): edge for edge in self.edge_window.edges
        })
        return [edges[key] for key in sorted(edges)]

    def _capture_factor_snapshot(self, poses):
        if not getattr(self, "global_refine", False):
            return
        graph = self.graph.frames.data
        edges = tuple(self._global_edges())
        edge_kinds = tuple(
            {1: "adjacent", 2: "skip2"}[edge.b - edge.a]
            for edge in edges
        )
        self._factor_snapshot = FactorArchive(
            initial_sensor_poses=poses,
            time_ns=graph["time_ns"].tensor.cpu().numpy(),
            T_BS=graph["T_BS"].tensor[0].double(),
            edges=edges,
            edge_kinds=edge_kinds,
            metadata={
                "source": "WindowMACVO",
                "window_size": self.edge_window.size,
            },
        )

    @torch.inference_mode()
    def _skip_edge_from_match(self, a, b, match):
        frame0, depth0 = self.frame_cache[a]
        frame1, depth1 = self.frame_cache[b]
        return build_match_edge(
            a=a,
            b=b,
            stereo_a=frame0.stereo,
            stereo_b=frame1.stereo,
            depth_a=depth0,
            depth_b=depth1,
            match=match,
            frontend=self.Frontend,
            selector=self.KeypointSelector,
            covariance_model=self.ObsCovModel,
            num_point=self.num_point,
            min_num_point=self.min_num_point,
            edge_width=self.edge_width,
            match_cov_default=self.match_cov_default,
            device=self.device,
        ).edge

    def run_pair(self, frame0, frame1):
        before_match = len(self.graph.match)
        a = self.prev_keyframe[1]
        depth1, adjacent_match, skip_match = self._estimate_window_inputs(
            frame0, frame1
        )
        super().run_pair_from_estimate(frame0, frame1, depth1, adjacent_match)
        b = self.prev_keyframe[1]
        # Flush the two-frame initializer before joint refinement. Clearing this
        # result is essential: a later base write_map must not undo window poses.
        self.Optimizer.write_map(self.graph)
        self.Optimizer.optimize_res = None
        self.Optimizer.has_opt_job = False
        self.frame_cache[b] = (frame1, self.prev_keyframe[2])
        first = max(0, b-self.edge_window.size+1)
        self.frame_cache = {i: value for i, value in self.frame_cache.items() if i >= first}
        evicted = self.edge_window.advance(b)
        self._retain_evicted_edges(evicted)

        obs = self.graph.match[before_match:]
        adjacent = self._edge(
            a, b, obs.data["pixel1_uv"], obs.data["pixel2_uv"],
            obs.data["pixel1_d"].squeeze(-1), obs.data["pixel2_d"].squeeze(-1),
            obs.data["obs1_covTc"], obs.data["obs2_covTc"],
            frame0.stereo.frame_K, frame1.stereo.frame_K)
        if adjacent is not None:
            self.edge_window.add(adjacent)

        skip_start = time.perf_counter()
        skip = None
        if skip_match is not None and b-2 in self.frame_cache:
            skip = self._skip_edge_from_match(b-2, b, skip_match)
            if skip is not None:
                self.edge_window.add(skip)
        skip_seconds = time.perf_counter()-skip_start

        ids = sorted(self.frame_cache)
        old = self.graph.frames.data["pose"][torch.tensor(ids)].double().clone()
        try:
            result = optimize_window(old, ids, self.edge_window.edges,
                                     max_iters=self.window_iterations,
                                     huber_delta=self.window_huber_delta)
        except ValueError as error:
            # Preserve the initializer if observations cannot connect the window.
            Logger.write("warn", f"Window {b} not refined: {error}")
            record = {"frames": ids, "current": b, "status": "not_refined", "reason": str(error)}
        else:
            self._apply_refinement(ids, old, result)
            record = result.diagnostics | {"current": b, "status": "refined"}
        record.update({
            "adjacent_matches": 0 if adjacent is None else len(adjacent.points_a),
            "skip_matches": 0 if skip is None else len(skip.points_a),
            "skip_seconds": skip_seconds, "cached_frames": len(self.frame_cache),
            "cached_edges": len(self.edge_window.edges),
        })
        self.window_records.append(record)
        self._after_window_step(frame1, depth1, adjacent, skip)
        for callback in self.on_optimize_writeback:
            callback(self)

    def _apply_refinement(self, ids, old, result):
        reanchor_points(self.graph, ids, old, result.poses)
        self.graph.frames.data["pose"][torch.tensor(ids)] = result.poses.float()
        # Connected skip constraints can recover a weak-adjacent frame. Do not
        # let termination interpolation overwrite its refined pose later.
        self.graph.frames.data["need_interp"][torch.tensor(ids[1:])] = False

    def _run_global_refinement(self, before):
        if not getattr(self, "global_refine", False):
            self.global_record = {"status": "disabled"}
            self.inactive_edge_count = 0
            self.retained_tensor_bytes = 0
            return

        edges = self._global_edges()
        self.inactive_edge_count = len(self.inactive_edges)
        self.retained_tensor_bytes = sum(
            edge_tensor_bytes(edge) for edge in edges
        )
        result = optimize_global_pose_graph(
            before,
            edges,
            max_iters=self.global_iterations,
            huber_delta=self.global_huber_delta,
        )
        self.global_record = result.diagnostics
        if result.diagnostics["status"] != "refined":
            return

        reanchor_points(
            self.graph, list(range(len(before))), before, result.poses
        )
        self.graph.frames.data["pose"].tensor[:] = result.poses.float()

    def terminate(self):
        if self.terminated:
            return
        before = self.graph.frames.data["pose"].tensor.double().clone()
        self.Optimizer.optimize_res = None
        self.Optimizer.terminate()
        # Base motion interpolation needs at least two relative motions.
        if len(before) >= 3:
            self.MapRefiner.elaborate_map(self.graph.frames)
        after = self.graph.frames.data["pose"].tensor.double().clone()
        reanchor_points(self.graph, list(range(len(before))), before, after)
        self._capture_factor_snapshot(after)
        self._run_global_refinement(after)
        self.frame_cache.clear()
        self.edge_window._edges.clear()
        getattr(self, "inactive_edges", {}).clear()
        self.terminated = True

    def save_window_diagnostics(self, folder):
        folder = Path(folder)
        snapshot = getattr(self, "_factor_snapshot", None)
        if self.global_refine and snapshot is not None:
            try:
                np.save(
                    folder / "poses_before_global.npy",
                    sensor_to_body_trajectory(
                        snapshot.initial_sensor_poses,
                        snapshot.time_ns,
                        snapshot.T_BS,
                    ),
                )
                factor_path = folder / "global_factors.npz"
                save_factor_archive(factor_path, snapshot)
                provenance_path = folder / "run_provenance.json"
                provenance = (
                    json.loads(provenance_path.read_text(encoding="utf-8"))
                    if provenance_path.is_file()
                    else {}
                )
                self.artifact_record = {
                    "status": "saved",
                    "schema_version": SCHEMA_VERSION,
                    "poses": "poses_before_global.npy",
                    "factors": "global_factors.npz",
                    "factor_bytes": factor_path.stat().st_size,
                    "source_id": snapshot.metadata["source_id"],
                    "git_commit": provenance.get("git_commit"),
                    "pose_count": len(snapshot.initial_sensor_poses),
                    "edge_count": len(snapshot.edges),
                    "observation_count": sum(
                        len(edge.points_a) for edge in snapshot.edges
                    ),
                }
                self._factor_snapshot = None
            except Exception as error:
                self.artifact_record = {
                    "status": "failed",
                    "reason": str(error),
                }
        elif self.global_refine and getattr(
            self, "artifact_record", {}
        ).get("status") == "pending":
            self.artifact_record = {
                "status": "unavailable",
                "reason": "factor snapshot was not captured",
            }

        payload = {
            "window_size": self.edge_window.size, "skip_matching": self.skip_matching,
            "marginalization": False, "pose_only": True,
            "initializer": "synchronous TwoFrame_PGO ICP",
            "huber_delta_whitened": self.window_huber_delta,
            "global_refine": self.global_refine,
            "inactive_edges": self.inactive_edge_count,
            "retained_tensor_bytes": self.retained_tensor_bytes,
            "global_refinement": self.global_record,
            "artifacts": getattr(
                self,
                "artifact_record",
                {"status": "disabled" if not self.global_refine else "unavailable"},
            ),
            "windows": self.window_records,
        }
        (folder / "window_diagnostics.json").write_text(
            json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        (folder / "global_refinement.json").write_text(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "global_refinement": self.global_record,
                    "artifacts": payload["artifacts"],
                },
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
