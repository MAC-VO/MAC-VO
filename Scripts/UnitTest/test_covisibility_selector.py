import inspect
import unittest
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pypose as pp
import torch

from Module.Optimization.CovisibilitySelector import (
    CovisibilitySelector,
    CovisibilitySelectorConfig,
    pair_suppressed,
    score_covisibility_pair,
)


def intrinsic():
    K = torch.eye(3, dtype=torch.float64)
    K[0, 0] = 10
    K[1, 1] = 10
    K[0, 2] = 5
    K[1, 2] = 5
    return K


def cache_fixture(frame_ids=(0, 5, 10, 20), depth_value=2.0):
    proxy_uv = np.array(
        [[3, 3], [5, 3], [7, 3], [3, 5], [5, 5], [7, 5],
         [3, 7], [5, 7], [7, 7]],
        dtype=np.float32,
    )
    count = len(frame_ids)
    return SimpleNamespace(
        frame_ids=np.asarray(frame_ids, dtype=np.int64),
        proxy_uv=proxy_uv,
        proxy_depth=np.full((count, len(proxy_uv)), depth_value, dtype=np.float32),
        proxy_depth_cov=np.full((count, len(proxy_uv)), 0.01, dtype=np.float32),
        depth=np.full((count, 10, 10), depth_value, dtype=np.float32),
        depth_cov=np.full((count, 10, 10), 0.01, dtype=np.float32),
        metadata={"height": 10, "width": 10},
    )


def poses(count=21):
    return pp.identity_SE3(count, dtype=torch.float64).tensor()


def intrinsics(frame_ids=(0, 5, 10, 20)):
    return {frame_id: intrinsic() for frame_id in frame_ids}


def permissive_config(**updates):
    config = CovisibilitySelectorConfig(
        min_temporal_gap=0,
        min_directional_overlap=0.0,
        min_mean_overlap=0.0,
        min_depth_consistency=0.0,
        min_median_motion_px=0.0,
        max_median_motion_px=1000.0,
        candidate_nms_frames=5,
        max_candidates_per_target=1,
    )
    return replace(config, **updates)


class CovisibilitySelectorTests(unittest.TestCase):
    def test_identity_projection_has_full_overlap_and_zero_motion(self):
        metrics = score_covisibility_pair(
            0, 20, cache_fixture(), poses(), intrinsics(), permissive_config()
        )

        self.assertAlmostEqual(metrics.overlap_forward, 1.0)
        self.assertAlmostEqual(metrics.overlap_backward, 1.0)
        self.assertAlmostEqual(metrics.mean_overlap, 1.0)
        self.assertAlmostEqual(metrics.depth_consistency, 1.0)
        self.assertAlmostEqual(metrics.median_motion_px, 0.0)
        self.assertEqual(metrics.status, "eligible")

    def test_target_sideways_translation_moves_projected_pixels(self):
        current_poses = poses()
        current_poses[20, 1] = 0.2

        metrics = score_covisibility_pair(
            0, 20, cache_fixture(), current_poses,
            intrinsics(), permissive_config()
        )

        self.assertAlmostEqual(metrics.median_motion_px, 1.0, places=5)

    def test_points_behind_target_camera_have_zero_forward_overlap(self):
        current_poses = poses()
        current_poses[20, 0] = 3.0

        metrics = score_covisibility_pair(
            0, 20, cache_fixture(), current_poses,
            intrinsics(), permissive_config()
        )

        self.assertEqual(metrics.overlap_forward, 0.0)

    def test_depth_mismatch_reduces_consistency(self):
        cache = cache_fixture()
        cache.depth[cache.frame_ids.tolist().index(20)] = 8.0

        metrics = score_covisibility_pair(
            0, 20, cache, poses(), intrinsics(), permissive_config()
        )

        self.assertLess(metrics.depth_consistency, 0.6)

    def test_default_minimum_motion_rejects_identity_pair(self):
        metrics = score_covisibility_pair(
            0, 20, cache_fixture(), poses(), intrinsics(),
            CovisibilitySelectorConfig()
        )

        self.assertEqual(metrics.status, "rejected")
        self.assertEqual(metrics.reason, "motion_too_small")

    def test_pair_space_nms_requires_both_endpoints_near(self):
        accepted = {(0, 20)}

        self.assertTrue(pair_suppressed(5, 25, accepted, radius=5))
        self.assertFalse(pair_suppressed(10, 25, accepted, radius=5))

    def test_selector_excludes_time_near_candidate_and_returns_one(self):
        selector = CovisibilitySelector(
            cache=cache_fixture(),
            poses=poses(),
            intrinsics=intrinsics(),
            config=permissive_config(min_temporal_gap=15),
        )

        selected, evaluated = selector.select(
            target=20,
            history=[0, 5, 10],
            accepted_pairs=set(),
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected.source, 0)
        self.assertEqual(len([item for item in evaluated if item.status == "eligible"]), 2)
        near = next(item for item in evaluated if item.source == 10)
        self.assertEqual(near.reason, "temporal_gap_too_small")

    def test_selector_returns_none_when_nms_suppresses_all_eligible_pairs(self):
        selector = CovisibilitySelector(
            cache=cache_fixture(),
            poses=poses(),
            intrinsics=intrinsics(),
            config=permissive_config(min_temporal_gap=15),
        )

        selected, _ = selector.select(
            target=20,
            history=[0, 5],
            accepted_pairs={(0, 20), (5, 20)},
        )

        self.assertIsNone(selected)

    def test_selector_api_has_no_ground_truth_parameter(self):
        parameters = inspect.signature(CovisibilitySelector.__init__).parameters

        self.assertNotIn("ground_truth", parameters)
        self.assertNotIn("gt", parameters)


if __name__ == "__main__":
    unittest.main()
