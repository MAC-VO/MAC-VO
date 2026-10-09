import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from Odometry.WindowMACVO import WindowMACVO


class WindowFrontendRoutingTests(unittest.TestCase):
    def make_system(self, skip_matching: bool):
        system = WindowMACVO.__new__(WindowMACVO)
        system.skip_matching = skip_matching
        system.frame_cache = {3: (SimpleNamespace(stereo="t2"), object())}
        system.prev_keyframe = (SimpleNamespace(stereo="t1"), 4, object())
        system.Frontend = SimpleNamespace(
            estimate_window=Mock(return_value=("depth", "adjacent", "skip")),
            estimate_pair=Mock(return_value=("depth", "adjacent")),
        )
        return system

    def test_skip_mode_calls_fused_frontend_once(self):
        system = self.make_system(True)

        result = system._estimate_window_inputs(
            SimpleNamespace(stereo="t1"), SimpleNamespace(stereo="t")
        )

        self.assertEqual(result, ("depth", "adjacent", "skip"))
        system.Frontend.estimate_window.assert_called_once_with("t2", "t1", "t")
        system.Frontend.estimate_pair.assert_not_called()

    def test_first_step_passes_no_two_step_frame(self):
        system = self.make_system(True)
        system.frame_cache.clear()

        result = system._estimate_window_inputs(
            SimpleNamespace(stereo="t1"), SimpleNamespace(stereo="t")
        )

        self.assertEqual(result, ("depth", "adjacent", "skip"))
        system.Frontend.estimate_window.assert_called_once_with(None, "t1", "t")

    def test_adjacent_mode_uses_padded_window_for_fixed_batch3(self):
        system = self.make_system(False)

        result = system._estimate_window_inputs(
            SimpleNamespace(stereo="t1"), SimpleNamespace(stereo="t")
        )

        self.assertEqual(result, ("depth", "adjacent", None))
        system.Frontend.estimate_window.assert_called_once_with(None, "t1", "t")
        system.Frontend.estimate_pair.assert_not_called()


if __name__ == "__main__":
    unittest.main()
