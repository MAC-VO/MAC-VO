"""Temporal and covisibility consistency for sparse loop hypotheses."""

from dataclasses import dataclass, field
import math

import pypose as pp
import torch


@dataclass(frozen=True)
class HypothesisConfig:
    min_supports: int = 2
    strong_supports: int = 3
    source_cluster_frames: int = 10
    target_support_frames: int = 10
    max_correction_translation_m: float = 0.25
    max_correction_rotation_deg: float = 10.0

    def __post_init__(self):
        if self.min_supports < 2:
            raise ValueError("loop confirmation requires at least two supports")
        if self.strong_supports < self.min_supports:
            raise ValueError("strong support count must not precede confirmation")
        if self.source_cluster_frames < 0 or self.target_support_frames < 1:
            raise ValueError("loop support frame windows are invalid")
        if (
            self.max_correction_translation_m <= 0
            or self.max_correction_rotation_deg <= 0
        ):
            raise ValueError("loop correction thresholds must be positive")


@dataclass(frozen=True)
class LoopSupport:
    source: int
    target: int
    measurement: torch.Tensor
    correction: torch.Tensor
    quality: tuple
    metrics: dict

    def __post_init__(self):
        measurement = torch.as_tensor(self.measurement).detach().cpu().double().clone()
        correction = torch.as_tensor(self.correction).detach().cpu().double().clone()
        if measurement.shape != (7,) or correction.shape != (7,):
            raise ValueError("loop support transforms must be SE3 tensors")
        if not torch.isfinite(measurement).all() or not torch.isfinite(correction).all():
            raise ValueError("loop support transforms must be finite")
        object.__setattr__(self, "measurement", measurement)
        object.__setattr__(self, "correction", correction)
        object.__setattr__(self, "quality", tuple(self.quality))
        object.__setattr__(self, "metrics", dict(self.metrics))


@dataclass
class LoopHypothesis:
    hypothesis_id: int
    supports: list[LoopSupport] = field(default_factory=list)
    state: str = "tentative"
    emitted: bool = False

    @property
    def last_target(self):
        return self.supports[-1].target


@dataclass(frozen=True)
class HypothesisUpdate:
    hypothesis_id: int | None
    state: str
    emitted: LoopSupport | None
    reason: str | None = None


class LoopHypothesisTracker:
    def __init__(self, config):
        self.config = config
        self._hypotheses = []
        self._next_id = 0

    @property
    def hypotheses(self):
        return tuple(self._hypotheses)

    def _correction_compatible(self, left, right):
        delta = (pp.SE3(left).Inv() @ pp.SE3(right)).Log().tensor()
        translation = float(delta[:3].norm())
        rotation = math.degrees(float(delta[3:].norm()))
        return (
            translation <= self.config.max_correction_translation_m
            and rotation <= self.config.max_correction_rotation_deg
        )

    def _source_compatible(self, hypothesis, support):
        sources = torch.tensor(
            [item.source for item in hypothesis.supports], dtype=torch.float64
        )
        center = float(sources.median())
        return abs(support.source - center) <= self.config.source_cluster_frames

    def _compatible(self, hypothesis, support):
        if hypothesis.state == "expired":
            return False
        gap = support.target - hypothesis.last_target
        return (
            self._source_compatible(hypothesis, support)
            and 0 < gap <= self.config.target_support_frames
            and self._correction_compatible(
                hypothesis.supports[0].correction, support.correction
            )
        )

    def add(self, support):
        for hypothesis in self._hypotheses:
            if (
                hypothesis.state != "expired"
                and any(item.target == support.target for item in hypothesis.supports)
                and self._source_compatible(hypothesis, support)
            ):
                return HypothesisUpdate(
                    hypothesis.hypothesis_id,
                    "rejected",
                    None,
                    "duplicate_target",
                )

        selected = next(
            (
                hypothesis for hypothesis in self._hypotheses
                if self._compatible(hypothesis, support)
            ),
            None,
        )
        if selected is None:
            selected = LoopHypothesis(self._next_id, [support])
            self._next_id += 1
            self._hypotheses.append(selected)
            return HypothesisUpdate(selected.hypothesis_id, "tentative", None)

        selected.supports.append(support)
        count = len(selected.supports)
        if count >= self.config.strong_supports:
            selected.state = "strong"
        elif count >= self.config.min_supports:
            selected.state = "confirmed"
        emitted = None
        if count >= self.config.min_supports and not selected.emitted:
            emitted = max(selected.supports, key=lambda item: item.quality)
            selected.emitted = True
        return HypothesisUpdate(selected.hypothesis_id, selected.state, emitted)

    def expire(self, current_target):
        expired = []
        for hypothesis in self._hypotheses:
            if (
                hypothesis.state != "expired"
                and current_target - hypothesis.last_target
                > self.config.target_support_frames
            ):
                hypothesis.state = "expired"
                expired.append(hypothesis.hypothesis_id)
        return tuple(expired)
