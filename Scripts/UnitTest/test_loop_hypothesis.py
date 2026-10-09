import unittest

import pypose as pp
import torch

from Module.LoopClosure.LoopHypothesis import (
    HypothesisConfig,
    LoopHypothesisTracker,
    LoopSupport,
)


def pose(tx=0.0, rz_deg=0.0):
    tangent = torch.tensor(
        [tx, 0.0, 0.0, 0.0, 0.0, torch.deg2rad(torch.tensor(rz_deg))],
        dtype=torch.float64,
    )
    return pp.se3(tangent).Exp().tensor()


def support(source, target, correction=None, quality=None):
    return LoopSupport(
        source=source,
        target=target,
        measurement=pp.identity_SE3(dtype=torch.float64).tensor(),
        correction=pose() if correction is None else correction,
        quality=(40, 0.6, 8, -2.0, 0.1) if quality is None else quality,
        metrics={"ransac_inliers": 40, "ransac_ratio": 0.6},
    )


class LoopHypothesisTrackerTests(unittest.TestCase):
    def tracker(self):
        return LoopHypothesisTracker(HypothesisConfig(
            min_supports=2,
            strong_supports=3,
            source_cluster_frames=10,
            target_support_frames=10,
            max_correction_translation_m=0.25,
            max_correction_rotation_deg=10.0,
        ))

    def test_two_compatible_keyframes_confirm_and_emit_once(self):
        tracker = self.tracker()

        first = tracker.add(support(100, 500))
        second = tracker.add(support(105, 505, correction=pose(0.05, 2.0)))
        third = tracker.add(support(95, 510, correction=pose(0.02, 1.0)))

        self.assertEqual(first.state, "tentative")
        self.assertIsNone(first.emitted)
        self.assertEqual(second.state, "confirmed")
        self.assertIsNotNone(second.emitted)
        self.assertEqual(third.state, "strong")
        self.assertIsNone(third.emitted)

    def test_duplicate_target_is_rejected(self):
        tracker = self.tracker()
        tracker.add(support(100, 500))

        update = tracker.add(support(105, 500))

        self.assertEqual(update.state, "rejected")
        self.assertEqual(update.reason, "duplicate_target")

    def test_incompatible_correction_starts_separate_hypothesis(self):
        tracker = self.tracker()
        first = tracker.add(support(100, 500, correction=pose()))

        second = tracker.add(support(102, 505, correction=pose(1.0, 0.0)))

        self.assertEqual(second.state, "tentative")
        self.assertNotEqual(first.hypothesis_id, second.hypothesis_id)
        self.assertEqual(len(tracker.hypotheses), 2)

    def test_confirmation_emits_highest_quality_support(self):
        tracker = self.tracker()
        weak = support(100, 500, quality=(30, 0.5, 6, -4.0, 0.2))
        strong = support(105, 505, quality=(60, 0.8, 12, -1.0, 0.15))
        tracker.add(weak)

        update = tracker.add(strong)

        self.assertEqual(update.emitted.target, 505)

    def test_expire_marks_stale_hypothesis(self):
        tracker = self.tracker()
        update = tracker.add(support(100, 500))

        expired = tracker.expire(511)

        self.assertEqual(expired, (update.hypothesis_id,))
        hypothesis = next(
            item for item in tracker.hypotheses
            if item.hypothesis_id == update.hypothesis_id
        )
        self.assertEqual(hypothesis.state, "expired")


if __name__ == "__main__":
    unittest.main()
