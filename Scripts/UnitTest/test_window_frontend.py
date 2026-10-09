import unittest
from types import SimpleNamespace

import pypose as pp
import torch

from DataLoader import StereoData
from Module.Frontend.Frontend import (
    CUDAGraph_FlowFormerCovFrontend,
    IFrontend,
    build_bidirectional_inputs,
    build_loop_depth_inputs,
    build_window_inputs,
)


def stereo(left: float, right: float) -> StereoData:
    K = torch.eye(3).unsqueeze(0)
    K[0, 0, 0] = 10
    K[0, 1, 1] = 10
    return StereoData(
        T_BS=pp.identity_SE3(1),
        K=K,
        baseline=torch.tensor([0.2]),
        time_ns=[0],
        height=2,
        width=2,
        imageL=torch.full((1, 3, 2, 2), left),
        imageR=torch.full((1, 3, 2, 2), right),
    )


class FusedWindowFrontendTests(unittest.TestCase):
    def test_builds_source_stereo_forward_and_backward_slots(self):
        input_a, input_b = build_bidirectional_inputs(
            stereo(2, 20), stereo(0, 30)
        )

        self.assertEqual(input_a[:, 0, 0, 0].tolist(), [2.0, 2.0, 0.0])
        self.assertEqual(input_b[:, 0, 0, 0].tolist(), [20.0, 0.0, 2.0])

    def test_builds_sparse_loop_depth_slots_from_source_stereo(self):
        input_a, input_b = build_loop_depth_inputs(stereo(2, 20))

        self.assertEqual(input_a[:, 0, 0, 0].tolist(), [2.0, 2.0, 2.0])
        self.assertEqual(input_b[:, 0, 0, 0].tolist(), [20.0, 20.0, 20.0])

    def test_bidirectional_inputs_reject_mismatched_shape(self):
        source = stereo(2, 20)
        target = stereo(0, 30)
        source.imageL = source.imageL[..., :1]

        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            build_bidirectional_inputs(source, target)

    def test_cuda_frontend_routes_bidirectional_slots(self):
        frontend = CUDAGraph_FlowFormerCovFrontend.__new__(
            CUDAGraph_FlowFormerCovFrontend
        )
        frontend.config = SimpleNamespace(
            device="cpu", enforce_positive_disparity=False
        )
        flow = torch.zeros(3, 2, 2, 2)
        covariance = torch.ones(3, 2, 2, 2)
        flow[0, 0] = 2
        flow[1] = 11
        flow[2] = 22
        frontend.cuda_graph_estimate = lambda *_: (flow, covariance)

        depth, forward, backward = frontend.estimate_bidirectional(
            stereo(2, 20), stereo(0, 30)
        )

        self.assertTrue(torch.allclose(depth.depth, torch.ones_like(depth.depth)))
        self.assertTrue(torch.equal(forward.flow, flow[1:2]))
        self.assertTrue(torch.equal(backward.flow, flow[2:3]))

    def test_cuda_frontend_sparse_loop_depth_uses_source_stereo(self):
        frontend = CUDAGraph_FlowFormerCovFrontend.__new__(
            CUDAGraph_FlowFormerCovFrontend
        )
        frontend.config = SimpleNamespace(
            device="cpu", enforce_positive_disparity=False
        )
        captured = {}
        flow = torch.zeros(3, 2, 2, 2)
        covariance = torch.ones(3, 2, 2, 2)
        flow[:, 0] = 2

        def estimate(input_a, input_b):
            captured["a"] = input_a.clone()
            captured["b"] = input_b.clone()
            return flow, covariance

        frontend.cuda_graph_estimate = estimate

        depth = frontend.estimate_loop_depth(stereo(2, 20))

        self.assertEqual(captured["a"][:, 0, 0, 0].tolist(), [2.0, 2.0, 2.0])
        self.assertEqual(captured["b"][:, 0, 0, 0].tolist(), [20.0, 20.0, 20.0])
        self.assertTrue(torch.allclose(depth.depth, torch.ones_like(depth.depth)))

    def test_rejects_non_single_frame_inputs_before_graph_capture(self):
        prev1 = stereo(1, 10)
        current = stereo(0, 30)
        current.imageL = current.imageL.repeat(2, 1, 1, 1)

        with self.assertRaisesRegex(ValueError, "single-frame"):
            build_window_inputs(None, prev1, current)

    def test_rejects_mismatched_image_shapes_before_graph_capture(self):
        prev1 = stereo(1, 10)
        current = stereo(0, 30)
        current.imageR = current.imageR[..., :1]

        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            build_window_inputs(None, prev1, current)

    def test_rejects_mismatched_image_dtype_before_graph_capture(self):
        prev1 = stereo(1, 10)
        current = stereo(0, 30)
        prev1.imageL = prev1.imageL.double()

        with self.assertRaisesRegex(ValueError, "device/dtype mismatch"):
            build_window_inputs(None, prev1, current)

    def test_generic_frontend_falls_back_to_pair_estimates(self):
        depth, adjacent, skip = object(), object(), object()

        class PairFrontend(IFrontend):
            def __init__(self):
                self.calls = []

            @property
            def provide_cov(self):
                return False, False

            def estimate_pair(self, frame0, frame1):
                self.calls.append((frame0, frame1))
                return (depth, adjacent) if frame0 == "t1" else (object(), skip)

            def estimate_depth(self, frame):
                raise NotImplementedError

            @classmethod
            def is_valid_config(cls, config):
                return None

        frontend = PairFrontend()
        result = frontend.estimate_window("t2", "t1", "t")

        self.assertEqual(result, (depth, adjacent, skip))
        self.assertEqual(frontend.calls, [("t1", "t"), ("t2", "t")])

    def test_builds_stereo_adjacent_and_skip_slots(self):
        input_a, input_b, has_skip = build_window_inputs(
            stereo(2, 20), stereo(1, 10), stereo(0, 30)
        )

        self.assertTrue(has_skip)
        self.assertEqual(input_a[:, 0, 0, 0].tolist(), [0.0, 1.0, 2.0])
        self.assertEqual(input_b[:, 0, 0, 0].tolist(), [30.0, 0.0, 0.0])

    def test_first_step_uses_fixed_batch3_without_skip_output(self):
        input_a, input_b, has_skip = build_window_inputs(
            None, stereo(1, 10), stereo(0, 30)
        )

        self.assertFalse(has_skip)
        self.assertEqual(input_a[:, 0, 0, 0].tolist(), [0.0, 1.0, 1.0])
        self.assertEqual(input_b[:, 0, 0, 0].tolist(), [30.0, 0.0, 0.0])

    def test_routes_slots_zero_one_two(self):
        frontend = CUDAGraph_FlowFormerCovFrontend.__new__(
            CUDAGraph_FlowFormerCovFrontend
        )
        frontend.config = SimpleNamespace(
            device="cpu", enforce_positive_disparity=False
        )
        flow = torch.zeros(3, 2, 2, 2)
        covariance = torch.ones(3, 2, 2, 2)
        flow[0, 0] = 2
        flow[1] = 11
        flow[2] = 22
        frontend.cuda_graph_estimate = lambda *_: (flow, covariance)

        depth, adjacent, skip = frontend.estimate_window(
            stereo(2, 20), stereo(1, 10), stereo(0, 30)
        )

        self.assertTrue(torch.allclose(depth.depth, torch.ones_like(depth.depth)))
        self.assertTrue(torch.equal(adjacent.flow, flow[1:2]))
        self.assertIsNotNone(skip)
        assert skip is not None
        self.assertTrue(torch.equal(skip.flow, flow[2:3]))


if __name__ == "__main__":
    unittest.main()
