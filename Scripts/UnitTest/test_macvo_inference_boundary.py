import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from Odometry.MACVO import MACVO


class MACVOInferenceBoundaryTests(unittest.TestCase):
    def test_run_pair_delegates_precomputed_outputs(self):
        system = MACVO.__new__(MACVO)
        depth = object()
        match = object()
        frame0 = SimpleNamespace(stereo=object())
        frame1 = SimpleNamespace(stereo=object())
        self.assertTrue(hasattr(MACVO, "run_pair_from_estimate"))
        system.prev_keyframe = (frame0, 0, object())
        system.KeyframeSelector = SimpleNamespace(isKeyframe=Mock(return_value=True))
        system.Frontend = SimpleNamespace(
            estimate_pair=Mock(return_value=(depth, match))
        )
        system.run_pair_from_estimate = Mock()

        system.run_pair(frame0, frame1)

        system.Frontend.estimate_pair.assert_called_once_with(
            frame0.stereo, frame1.stereo
        )
        system.run_pair_from_estimate.assert_called_once_with(
            frame0, frame1, depth, match
        )


if __name__ == "__main__":
    unittest.main()
