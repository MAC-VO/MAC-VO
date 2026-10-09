import unittest

import numpy as np
import pypose as pp
import torch

from Module.Optimization.FactorArchive import FactorArchive
from Module.Optimization.ProximityFactorStore import (
    PersistentFactorStore,
    ProximityFactorRecord,
)
from Module.Optimization.WindowICP import Edge


def make_edge(a=0, b=15):
    points = torch.tensor([[3.0, 0.0, 0.0]], dtype=torch.float64)
    covariance = torch.eye(3, dtype=torch.float64)[None] * 0.001
    return Edge(a, b, points, points.clone(), covariance, covariance.clone())


def source_archive(count=20):
    return FactorArchive(
        initial_sensor_poses=pp.identity_SE3(
            count, dtype=torch.float64
        ).tensor(),
        time_ns=np.arange(count, dtype=np.int64) + 1000,
        T_BS=pp.identity_SE3(dtype=torch.float64).tensor(),
        edges=(),
        edge_kinds=(),
        metadata={"fixture": True},
    )


def record(edge=None, **overrides):
    values = {
        "edge": make_edge() if edge is None else edge,
        "score": 0.2,
        "confidence": 0.7,
        "age": 0,
        "state": "accepted",
        "candidate_metrics": {"mean_overlap": 0.6},
        "validation_metrics": {"mahalanobis_inlier_ratio": 0.8},
    }
    values.update(overrides)
    return ProximityFactorRecord(**values)


class ProximityFactorStoreTests(unittest.TestCase):
    def test_store_round_trip_preserves_aligned_records(self):
        store = PersistentFactorStore(source_archive())
        store.add(record())

        archive = store.to_archive()
        loaded = PersistentFactorStore.from_archive(archive)

        self.assertEqual(loaded.records[0].edge.b, 15)
        self.assertEqual(loaded.records[0].confidence, 0.7)
        self.assertEqual(loaded.records[0].candidate_metrics["mean_overlap"], 0.6)

    def test_store_rejects_record_count_mismatch(self):
        store = PersistentFactorStore(source_archive())
        store.add(record())
        archive = store.to_archive()
        archive.metadata["edge_records"] = []

        with self.assertRaisesRegex(ValueError, "record count"):
            PersistentFactorStore.from_archive(archive)

    def test_store_rejects_invalid_confidence(self):
        store = PersistentFactorStore(source_archive())

        with self.assertRaisesRegex(ValueError, "confidence"):
            store.add(record(confidence=1.1))

    def test_store_rejects_invalid_age_or_state(self):
        store = PersistentFactorStore(source_archive())
        with self.assertRaisesRegex(ValueError, "age"):
            store.add(record(age=-1))
        with self.assertRaisesRegex(ValueError, "state"):
            store.add(record(state="inactive"))

    def test_store_rejects_duplicate_endpoints(self):
        store = PersistentFactorStore(source_archive())
        store.add(record())

        with self.assertRaisesRegex(ValueError, "duplicate"):
            store.add(record())

    def test_store_rejects_non_proximity_archive(self):
        source = source_archive()
        archive = FactorArchive(
            source.initial_sensor_poses,
            source.time_ns,
            source.T_BS,
            (make_edge(0, 5),),
            ("gap5",),
            {"edge_records": [{}]},
        )

        with self.assertRaisesRegex(ValueError, "proximity"):
            PersistentFactorStore.from_archive(archive)

    def test_store_exposes_immutable_edge_keys(self):
        store = PersistentFactorStore(source_archive())
        store.add(record())

        self.assertEqual(store.keys, frozenset({(0, 15)}))


if __name__ == "__main__":
    unittest.main()
