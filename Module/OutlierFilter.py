from abc import ABC, abstractmethod

import torch
from types import SimpleNamespace

from Module.Map import BatchObservation
from DataLoader import MetaInfo
from Utility.Extensions import ConfigTestableSubclass


class IObservationFilter(ABC, ConfigTestableSubclass):
    """
    This class filters observation made. It helps for more robust tracking and removes outlierp at
    early stage of the system.
    """
    def __init__(self, config: SimpleNamespace):
        self.config = config
    
    def set_meta(self, meta: MetaInfo):
        """
        This method is used to receive meta info (e.g. camera intrinsic, image shape, etc.) on the first frame received by MAC-VO.
        The filter can then initialize some behavior dyanmically based on these information.
        """
        pass

    @abstractmethod
    def filter(self, observations: BatchObservation) -> torch.Tensor:
        """
        Given a batch of N observation (`BatchObservation`), the filter returns a boolean tensor of shape (N,) that 
            * sets True for "good" observation 
            * sets False for observations to filter away.
        """
        ...


class FilterCompose(IObservationFilter):
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.filters = [
            IObservationFilter.instantiate(filter_arg.type, filter_arg.args)
            for filter_arg in self.config.filter_args
        ]

    def set_meta(self, meta: MetaInfo):
        for f in self.filters:
            f.set_meta(meta)

    def filter(self, observations: BatchObservation) -> torch.Tensor:
        mask = torch.ones((len(observations),), dtype=torch.bool, device=observations.device)
        for f in self.filters:
            mask = torch.logical_and(mask, f.filter(observations))
        
        return mask
    
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        assert config is not None
        assert isinstance(config.filter_args, list)
        for filter_arg in config.filter_args:
            IObservationFilter.is_valid_config(filter_arg)


class IdentityFilter(IObservationFilter):
    def filter(self, observations: BatchObservation) -> torch.Tensor:
        return torch.ones((len(observations),), dtype=torch.bool, device=observations.device)
    
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None: return


class CovarianceSanityFilter(IObservationFilter):
    def filter(self, observations: BatchObservation) -> torch.Tensor:
        cov_has_nan = observations.cov_Tc.isnan().any(dim=[-1, -2])
        cov_has_inf = observations.cov_Tc.isinf().any(dim=[-1, -2])
        return ~torch.logical_or(cov_has_nan, cov_has_inf)
    
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None: return


class SimpleDepthFilter(IObservationFilter):
    def set_meta(self, meta: MetaInfo):
        if self.config.max_depth == "auto":
            self.config.max_depth = meta.fx * meta.baseline

    def filter(self, observations: BatchObservation) -> torch.Tensor:
        return ~torch.logical_or(observations.pixel_d < self.config.min_depth, 
                                 observations.pixel_d  > self.config.max_depth)
    
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        assert config is not None
        if isinstance(config.max_depth, (float, int)): assert config.max_depth > config.min_depth
        cls._enforce_config_spec(config, {
            "min_depth": lambda dist: isinstance(dist, (int, float)) and dist > 0.,
            "max_depth": lambda dist: (dist == "auto") or (isinstance(dist, (int, float)) and dist > 0.)
        })


class DepthFilter(IObservationFilter):
    def set_meta(self, meta: MetaInfo):
        if self.config.max_depth == "auto": self.config.max_depth = meta.baseline * meta.fx
    
    def filter(self, observations: BatchObservation) -> torch.Tensor:
        current_max: float = self.config.max_depth
        current_mask = torch.ones((len(observations),), dtype=torch.bool, device=observations.device)
        
        while current_max < self.config.final_depth:
            current_mask = torch.logical_and(
                observations.pixel_d < current_max,
                observations.pixel_d > self.config.min_depth
            )
            if (current_mask).sum().item() >= self.config.expect_num:
                break
            else:
                current_max *= 1.5
        
        return current_mask

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        assert config is not None
        if isinstance(config.max_depth, (float, int)): assert config.max_depth > config.min_depth
        cls._enforce_config_spec(config, {
            "min_depth": lambda dist: isinstance(dist, (int, float)) and dist > 0.,
            "max_depth": lambda dist: (dist == "auto") or (isinstance(dist, (int, float)) and dist > 0.)
        })


class LikelyFrontOfCamFilter(IObservationFilter):
    def filter(self, observations: BatchObservation) -> torch.Tensor:
        if (observations.cov_pixel_d == -1).any():
            return torch.tensor([True] * len(observations), dtype=torch.bool)
        return (observations.pixel_d.square() - (observations.cov_pixel_d * 4)) > 0.
    
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None: return


