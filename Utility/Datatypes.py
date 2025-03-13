import math
import numpy as np

from dataclasses import dataclass
from typing import Iterable


def median(data: Iterable[float]) -> float:
    sorted_data = sorted(data)
    n = len(sorted_data)
    if n == 0:
        raise ValueError("median() arg is an empty sequence")  # Handle empty input
    mid = n // 2
    if n % 2 == 1:
        return float(sorted_data[mid])
    else:
        return (sorted_data[mid - 1] + sorted_data[mid]) / 2.0  # Even case: return mean of two middle values

def mean(data: Iterable[float]) -> float:
    data = list(data)  # Convert iterable to list to allow multiple passes
    if not data:
        raise ValueError("mean() arg is an empty sequence")  # Handle empty input
    
    return sum(data) / len(data)  # Compute mean


@dataclass
class FlowPerformance:
    masked_epe: float
    epe : float
    px1 : float
    px3 : float
    px5 : float
    
    @classmethod
    def mean(cls, values: list["FlowPerformance"]) -> "FlowPerformance":
        return FlowPerformance(
            masked_epe  =sum([v.masked_epe for v in values]) / len(values),
            epe         =sum([v.epe        for v in values]) / len(values),
            px1         =sum([v.px1        for v in values]) / len(values),
            px3         =sum([v.px3        for v in values]) / len(values),
            px5         =sum([v.px5        for v in values]) / len(values),
        )


@dataclass
class FlowCovPerformance:
    masked_nll: float
    q25_nll: float
    q50_nll: float
    q75_nll: float
    
    @classmethod
    def mean(cls, values: list["FlowCovPerformance"]) -> "FlowCovPerformance":
        return FlowCovPerformance(
            masked_nll = mean([v.masked_nll for v in values]),
            q25_nll    = mean([v.q25_nll    for v in values]),
            q50_nll    = mean([v.q50_nll    for v in values]),
            q75_nll    = mean([v.q75_nll    for v in values]),
        )


@dataclass
class DepthPerformance:
    masked_err: float
    err_25 : float
    err_50 : float
    err_75 : float

    @classmethod
    def median(cls, values: list["DepthPerformance"]) -> "DepthPerformance":
        return DepthPerformance(
            masked_err= median([v.masked_err for v in values]),
            err_25    = median([v.err_25 for v in values]),
            err_50    = median([v.err_50 for v in values]),
            err_75    = median([v.err_75 for v in values]),
        )


@dataclass
class DepthCovPerformance:
    masked_nll: float
    q25_nll: float
    q50_nll: float
    q75_nll: float
    
    @classmethod
    def mean(cls, values: list["DepthCovPerformance"]) -> "DepthCovPerformance":
        return DepthCovPerformance(
            masked_nll = mean([v.masked_nll for v in values]),
            q25_nll    = mean([v.q25_nll    for v in values]),
            q50_nll    = mean([v.q50_nll    for v in values]),
            q75_nll    = mean([v.q75_nll    for v in values])
        )
