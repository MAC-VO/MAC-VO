import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pypose as pp
import torch

from Module.Optimization.FactorArchive import (
    FactorArchive,
    filter_factor_archive,
    load_factor_archive,
    merge_factor_archives,
    save_factor_archive,
    sensor_to_body_trajectory,
    trajectory_identity,
)
from Module.Optimization.WindowICP import Edge


def make_edge(a=0, b=1, value=1.0):
    points_a = torch.tensor([[value, 0.1, -0.2]], dtype=torch.float64)
    points_b = torch.tensor([[value + 0.2, 0.1, -0.2]], dtype=torch.float64)
    covariance = torch.eye(3, dtype=torch.float64)[None] * 0.001
    return Edge(a, b, points_a, points_b, covariance, covariance.clone())


def make_archive(edges=(), kinds=(), metadata=None):
    edge_list = tuple(edges)
    pose_count = max([3] + [edge.b + 1 for edge in edge_list])
    tangent = torch.zeros(pose_count, 6, dtype=torch.float64)
    tangent[:, 0] = torch.arange(pose_count, dtype=torch.float64) * 0.1
    poses = pp.se3(tangent).Exp().tensor()
    time_ns = np.arange(pose_count, dtype=np.int64) * 10 + 100
    T_BS = pp.identity_SE3(dtype=torch.float64).tensor()
    return FactorArchive(
        initial_sensor_poses=poses,
        time_ns=time_ns,
        T_BS=T_BS,
        edges=edge_list,
        edge_kinds=tuple(kinds),
        metadata={} if metadata is None else metadata,
    )


def rewrite_npz(path, **updates):
    with np.load(path, allow_pickle=False) as source:
        payload = {name: source[name] for name in source.files}
    payload.update(updates)
    with Path(path).open("wb") as stream:
        np.savez(stream, **payload)


class FactorArchiveTests(unittest.TestCase):
    def test_archive_round_trip_is_exact(self):
        archive = make_archive(
            [make_edge(0, 1), make_edge(0, 5, 2.0)],
            ["adjacent", "gap5"],
            {"run": "fixture"},
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "factors.npz")
            save_factor_archive(path, archive)
            loaded = load_factor_archive(path)

        self.assertTrue(torch.equal(
            loaded.initial_sensor_poses, archive.initial_sensor_poses
        ))
        self.assertTrue(torch.equal(loaded.edges[1].cov_b, archive.edges[1].cov_b))
        self.assertEqual(loaded.edge_kinds, ("adjacent", "gap5"))
        self.assertEqual(loaded.metadata, archive.metadata)

    def test_proximity_edge_kind_round_trips_in_schema_one(self):
        archive = make_archive([make_edge(0, 5)], ["proximity"])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "proximity.npz")
            save_factor_archive(path, archive)
            loaded = load_factor_archive(path)

        self.assertEqual(loaded.edge_kinds, ("proximity",))

    def test_merge_rejects_duplicate_edge_key(self):
        left = make_archive([make_edge(0, 5)], ["gap5"])
        right = make_archive([make_edge(0, 5)], ["gap5"])

        with self.assertRaisesRegex(ValueError, "duplicate"):
            merge_factor_archives(left, right)

    def test_filter_keeps_only_requested_edge_kind(self):
        archive = make_archive(
            [make_edge(0, 5), make_edge(0, 10)],
            ["gap5", "gap10"],
        )

        filtered = filter_factor_archive(archive, {"gap5"})

        self.assertEqual(filtered.edge_kinds, ("gap5",))
        self.assertEqual([(edge.a, edge.b) for edge in filtered.edges], [(0, 5)])

    def test_failed_publication_does_not_replace_destination(self):
        archive = make_archive([make_edge()], ["adjacent"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "factors.npz")
            path.write_bytes(b"original")
            with patch(
                "Module.Optimization.FactorArchive.os.replace",
                side_effect=OSError("disk"),
            ):
                with self.assertRaisesRegex(OSError, "disk"):
                    save_factor_archive(path, archive)
            leftovers = list(Path(directory).glob("*.tmp"))

            self.assertEqual(path.read_bytes(), b"original")
            self.assertEqual(leftovers, [])

    def test_rejects_unsupported_schema_version(self):
        archive = make_archive([make_edge()], ["adjacent"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "factors.npz")
            save_factor_archive(path, archive)
            rewrite_npz(path, schema_version=np.array(2, dtype=np.int64))

            with self.assertRaisesRegex(ValueError, "schema_version"):
                load_factor_archive(path)

    def test_rejects_malformed_offsets(self):
        archive = make_archive([make_edge()], ["adjacent"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "factors.npz")
            save_factor_archive(path, archive)
            rewrite_npz(path, edge_offsets=np.array([0, 2], dtype=np.int64))

            with self.assertRaisesRegex(ValueError, "offset"):
                load_factor_archive(path)

    def test_rejects_nonfinite_pose(self):
        archive = make_archive()
        poses = archive.initial_sensor_poses.clone()
        poses[0, 0] = float("nan")

        with self.assertRaisesRegex(ValueError, "finite Nx7"):
            FactorArchive(
                poses, archive.time_ns, archive.T_BS,
                archive.edges, archive.edge_kinds, archive.metadata,
            )

    def test_rejects_timestamp_count_mismatch(self):
        archive = make_archive()

        with self.assertRaisesRegex(ValueError, "pose count"):
            FactorArchive(
                archive.initial_sensor_poses,
                archive.time_ns[:-1],
                archive.T_BS,
                archive.edges,
                archive.edge_kinds,
                archive.metadata,
            )

    def test_rejects_invalid_extrinsic(self):
        archive = make_archive()
        invalid = archive.T_BS.clone()
        invalid[3:] = 0

        with self.assertRaisesRegex(ValueError, "unit quaternion"):
            FactorArchive(
                archive.initial_sensor_poses,
                archive.time_ns,
                invalid,
                archive.edges,
                archive.edge_kinds,
                archive.metadata,
            )

    def test_rejects_unknown_edge_kind(self):
        with self.assertRaisesRegex(ValueError, "unknown edge kind"):
            make_archive([make_edge()], ["loop"])

    def test_rejects_endpoint_beyond_pose_count(self):
        archive = make_archive()

        with self.assertRaisesRegex(ValueError, "endpoint"):
            FactorArchive(
                archive.initial_sensor_poses,
                archive.time_ns,
                archive.T_BS,
                (make_edge(0, len(archive.initial_sensor_poses)),),
                ("gap5",),
                archive.metadata,
            )

    def test_merge_rejects_source_identity_mismatch(self):
        left = make_archive([make_edge(0, 1)], ["adjacent"])
        poses = left.initial_sensor_poses.clone()
        poses[1, 0] += 0.01
        right = FactorArchive(
            poses,
            left.time_ns,
            left.T_BS,
            (make_edge(0, 2),),
            ("skip2",),
            {},
        )

        with self.assertRaisesRegex(ValueError, "source"):
            merge_factor_archives(left, right)

    def test_trajectory_identity_is_stable_and_sensitive(self):
        archive = make_archive()
        first = trajectory_identity(
            archive.initial_sensor_poses, archive.time_ns, archive.T_BS
        )
        second = trajectory_identity(
            archive.initial_sensor_poses.clone(),
            archive.time_ns.copy(),
            archive.T_BS.clone(),
        )
        changed = archive.initial_sensor_poses.clone()
        changed[1, 0] += 0.01

        self.assertEqual(first, second)
        self.assertNotEqual(
            first, trajectory_identity(changed, archive.time_ns, archive.T_BS)
        )

    def test_sensor_to_body_trajectory_uses_public_format(self):
        archive = make_archive()

        body = sensor_to_body_trajectory(
            archive.initial_sensor_poses, archive.time_ns, archive.T_BS
        )

        self.assertEqual(body.shape, (len(archive.time_ns), 8))
        self.assertTrue(np.array_equal(body[:, 0], archive.time_ns))
        self.assertTrue(np.allclose(
            body[:, 1:], archive.initial_sensor_poses.numpy()
        ))


if __name__ == "__main__":
    unittest.main()
