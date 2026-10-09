"""Persistent records for validated dynamic-covisibility ICP factors."""

from dataclasses import dataclass, replace
import json
import math

from Module.Optimization.FactorArchive import FactorArchive
from Module.Optimization.WindowICP import Edge


@dataclass(frozen=True)
class ProximityFactorRecord:
    edge: Edge
    score: float
    confidence: float
    age: int
    state: str
    candidate_metrics: dict
    validation_metrics: dict


def _validate_record(record):
    if not isinstance(record, ProximityFactorRecord):
        raise TypeError("record must be a ProximityFactorRecord")
    if not math.isfinite(record.score) or record.score < 0:
        raise ValueError("proximity score must be finite and non-negative")
    if not math.isfinite(record.confidence) or not 0 <= record.confidence <= 1:
        raise ValueError("proximity confidence must be within [0, 1]")
    if not isinstance(record.age, int) or record.age < 0:
        raise ValueError("proximity age must be a non-negative integer")
    if record.state != "accepted":
        raise ValueError("proximity state must be accepted")
    if not isinstance(record.candidate_metrics, dict) or not isinstance(
        record.validation_metrics, dict
    ):
        raise ValueError("proximity metrics must be dictionaries")
    try:
        json.dumps(
            {
                "candidate": record.candidate_metrics,
                "validation": record.validation_metrics,
            },
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("proximity metrics must be finite JSON values") from error


def _record_to_json(record):
    return {
        "a": record.edge.a,
        "b": record.edge.b,
        "score": record.score,
        "confidence": record.confidence,
        "age": record.age,
        "state": record.state,
        "candidate_metrics": dict(record.candidate_metrics),
        "validation_metrics": dict(record.validation_metrics),
    }


class PersistentFactorStore:
    def __init__(self, source_archive):
        if not isinstance(source_archive, FactorArchive):
            raise TypeError("source_archive must be a FactorArchive")
        self.source_archive = source_archive
        self.records = []
        self._keys = set()

    @property
    def keys(self):
        return frozenset(self._keys)

    def add(self, record):
        _validate_record(record)
        key = (record.edge.a, record.edge.b)
        if key in self._keys:
            raise ValueError(f"duplicate proximity edge {key}")
        if record.edge.b >= len(self.source_archive.initial_sensor_poses):
            raise ValueError("proximity edge endpoint exceeds source poses")
        self.records.append(record)
        self._keys.add(key)

    def to_archive(self):
        metadata = dict(self.source_archive.metadata)
        metadata["generator"] = "dynamic_covisibility_proximity"
        metadata["edge_records"] = [
            _record_to_json(record) for record in self.records
        ]
        return FactorArchive(
            initial_sensor_poses=self.source_archive.initial_sensor_poses,
            time_ns=self.source_archive.time_ns,
            T_BS=self.source_archive.T_BS,
            edges=tuple(record.edge for record in self.records),
            edge_kinds=("proximity",) * len(self.records),
            metadata=metadata,
        )

    @classmethod
    def from_archive(cls, archive):
        if not isinstance(archive, FactorArchive):
            raise TypeError("archive must be a FactorArchive")
        if any(kind != "proximity" for kind in archive.edge_kinds):
            raise ValueError("factor archive must contain only proximity edges")
        payloads = archive.metadata.get("edge_records")
        if not isinstance(payloads, list) or len(payloads) != len(archive.edges):
            raise ValueError("proximity edge record count does not match edges")
        source_metadata = dict(archive.metadata)
        source_metadata.pop("edge_records", None)
        source = replace(
            archive,
            edges=(),
            edge_kinds=(),
            metadata=source_metadata,
        )
        store = cls(source)
        for edge, payload in zip(archive.edges, payloads):
            if not isinstance(payload, dict):
                raise ValueError("proximity edge record must be an object")
            if payload.get("a") != edge.a or payload.get("b") != edge.b:
                raise ValueError("proximity edge record endpoints do not align")
            store.add(ProximityFactorRecord(
                edge=edge,
                score=payload.get("score"),
                confidence=payload.get("confidence"),
                age=payload.get("age"),
                state=payload.get("state"),
                candidate_metrics=payload.get("candidate_metrics"),
                validation_metrics=payload.get("validation_metrics"),
            ))
        return store
