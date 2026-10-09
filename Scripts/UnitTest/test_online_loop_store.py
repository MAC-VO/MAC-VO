from pathlib import Path
import tempfile
import unittest

import pypose as pp
import torch

from DataLoader import StereoData
from Module.Frontend.StereoDepth import IStereoDepth
from Module.LoopClosure.OnlineLoopStore import (
    LoopKeyframeStore,
    PendingLoopTarget,
    PendingLoopTargetStore,
)


def stereo(frame_id):
    image = torch.arange(3 * 8 * 10, dtype=torch.float32).reshape(1, 3, 8, 10)
    image = (image % 256) / 255
    return StereoData(
        T_BS=pp.identity_SE3(1), K=torch.eye(3)[None],
        baseline=torch.tensor([0.2]), time_ns=[1000 + frame_id],
        height=8, width=10, imageL=image, imageR=image.flip(-1),
    )


def depth():
    values = torch.ones(1, 1, 8, 10)
    return IStereoDepth.Output(depth=values, cov=values * 0.1)


class OnlineLoopStoreTests(unittest.TestCase):
    def test_keyframe_store_round_trips_lossless_uint8_images(self):
        with tempfile.TemporaryDirectory() as directory:
            store = LoopKeyframeStore(Path(directory))
            stored = store.add(5, stereo(5), torch.tensor([1., 2., 3., 0., 0., 0., 1.]))
            loaded = store.load(5)

        self.assertTrue(stored.left_path.name.endswith("left.png"))
        self.assertTrue(torch.equal(
            (loaded.stereo.imageL * 255).round().byte(),
            (stereo(5).imageL * 255).round().byte(),
        ))
        self.assertEqual(loaded.frame_id, 5)

    def test_pending_store_expires_oldest_target_at_capacity(self):
        store = PendingLoopTargetStore(capacity=2)
        first = PendingLoopTarget(5, stereo(5), depth())
        second = PendingLoopTarget(10, stereo(10), depth())
        third = PendingLoopTarget(15, stereo(15), depth())

        self.assertIsNone(store.add(first))
        self.assertIsNone(store.add(second))
        expired = store.add(third)

        self.assertEqual(expired.frame_id, 5)
        self.assertEqual(store.frame_ids, (10, 15))

    def test_pending_depth_is_stored_on_cpu(self):
        packet = PendingLoopTarget(5, stereo(5), depth())
        store = PendingLoopTargetStore(capacity=2)

        store.add(packet)
        loaded = store.get(5)

        self.assertEqual(loaded.depth.depth.device.type, "cpu")
        self.assertEqual(loaded.depth.cov.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
