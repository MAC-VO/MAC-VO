from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pypose as pp
import torch

from Module.Frontend.StereoDepth import IStereoDepth
from Module.Optimization.CovisibilityCache import (
    build_covisibility_cache,
    covisibility_keyframes,
    deterministic_proxy_uv,
    load_covisibility_cache,
)
from Module.Optimization.FactorArchive import FactorArchive


class FakeSequence:
    def __init__(self, count, height=6, width=8):
        self.frames = [
            SimpleNamespace(stereo=SimpleNamespace(
                tag=index,
                frame_ns=1000 + index,
                height=height,
                width=width,
            ))
            for index in range(count)
        ]

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        return self.frames[index]


class FakeFrontend:
    def __init__(self):
        self.calls = []

    def estimate_bidirectional(self, source, target):
        self.calls.append((source.tag, target.tag))
        depth = torch.full(
            (1, 1, target.height, target.width),
            float(target.tag + 1),
        )
        covariance = torch.full_like(depth, float(target.tag + 1) / 100)
        return IStereoDepth.Output(depth=depth, cov=covariance), object(), object()


def source_archive(count):
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


class CovisibilityCacheTests(unittest.TestCase):
    def test_keyframe_schedule_uses_stride_five(self):
        self.assertEqual(covisibility_keyframes(17, 5), [0, 5, 10, 15])

    def test_cache_uses_one_bidirectional_call_per_keyframe(self):
        frontend = FakeFrontend()
        with tempfile.TemporaryDirectory() as directory:
            cache = build_covisibility_cache(
                FakeSequence(16),
                source_archive(16),
                Path(directory, "cache"),
                frontend,
                stride=5,
                proxy_points=8,
            )

        self.assertEqual(
            frontend.calls,
            [(0, 0), (5, 5), (10, 10), (15, 15)],
        )
        self.assertEqual(cache.frame_ids.tolist(), [0, 5, 10, 15])

    def test_proxy_sampling_is_deterministic_and_grid_distributed(self):
        first = deterministic_proxy_uv(8, 6, 8)
        second = deterministic_proxy_uv(8, 6, 8)

        self.assertTrue(np.array_equal(first, second))
        self.assertGreater(len(np.unique(first[:, 0])), 1)
        self.assertGreater(len(np.unique(first[:, 1])), 1)

    def test_cache_round_trip_preserves_float32_maps_and_depth_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory, "cache")
            source = source_archive(6)
            build_covisibility_cache(
                FakeSequence(6), source, root, FakeFrontend(),
                stride=5, proxy_points=8,
            )
            loaded = load_covisibility_cache(root, source)
            output = loaded.depth_output(5)

            self.assertEqual(loaded.depth.dtype, np.float32)
            self.assertEqual(loaded.depth_cov.dtype, np.float32)
            self.assertTrue(torch.equal(
                output.depth, torch.full((1, 1, 6, 8), 6.0)
            ))
            self.assertTrue(torch.equal(
                output.cov, torch.full((1, 1, 6, 8), 0.06)
            ))

    def test_valid_existing_cache_is_reused_without_frontend_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory, "cache")
            source = source_archive(6)
            build_covisibility_cache(
                FakeSequence(6), source, root, FakeFrontend(),
                stride=5, proxy_points=8,
            )
            frontend = FakeFrontend()

            loaded = build_covisibility_cache(
                FakeSequence(6), source, root, frontend,
                stride=5, proxy_points=8,
            )

            self.assertEqual(frontend.calls, [])
            self.assertEqual(loaded.metadata["source_id"], source.metadata["source_id"])

    def test_timestamp_mismatch_is_rejected(self):
        source = source_archive(6)
        source.time_ns[0] += 1
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "timestamp"):
                build_covisibility_cache(
                    FakeSequence(6), source, Path(directory, "cache"),
                    FakeFrontend(), stride=5, proxy_points=8,
                )

    def test_corrupt_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory, "cache")
            root.mkdir()
            (root / "metadata.json").write_text("{}")

            with self.assertRaisesRegex(ValueError, "missing"):
                load_covisibility_cache(root)

    def test_source_identity_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory, "cache")
            source = source_archive(6)
            build_covisibility_cache(
                FakeSequence(6), source, root, FakeFrontend(),
                stride=5, proxy_points=8,
            )
            changed = source_archive(6)
            changed.initial_sensor_poses[1, 0] += 0.01

            with self.assertRaisesRegex(ValueError, "source"):
                load_covisibility_cache(root, changed)

    def test_failed_publication_leaves_no_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory, "cache")
            with patch(
                "Module.Optimization.CovisibilityCache.os.replace",
                side_effect=OSError("disk"),
            ):
                with self.assertRaisesRegex(OSError, "disk"):
                    build_covisibility_cache(
                        FakeSequence(6), source_archive(6), destination,
                        FakeFrontend(), stride=5, proxy_points=8,
                    )

            self.assertFalse(destination.exists())
            self.assertEqual(list(Path(directory).glob("cache.tmp-*")), [])


if __name__ == "__main__":
    unittest.main()
