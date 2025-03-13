from abc import ABC, abstractmethod

import torch
from types import SimpleNamespace
from typing_extensions import LiteralString

from Module.Map import TensorBundle
from DataLoader import StereoData
from Utility.Extensions import ConfigTestableSubclass
from Utility.PrettyPrint import Logger


class IObservationFilter(ABC, ConfigTestableSubclass):
    """
    This class filters observation made. It helps for more robust tracking and removes outlierp at
    early stage of the system.
    """
    def __init__(self, config: SimpleNamespace):
        self.config = config
    
    @property
    @abstractmethod
    def required_keys(self) -> set[LiteralString]: ...
    
    def verify_shape(self, value: TensorBundle): return all([k in value.data.keys() for k in self.required_keys])
    
    def set_meta(self, meta: StereoData):
        """
        This method is used to receive meta info (e.g. camera intrinsic, image shape, etc.) on the first frame received by MAC-VO.
        The filter can then initialize some behavior dyanmically based on these information.
        """
        pass

    @abstractmethod
    def filter(self, values: TensorBundle, device: torch.device) -> torch.Tensor:
        """
        Given a batch of N observation (`TensorBundle`), the filter returns a boolean tensor of shape (N,) that 
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
        if not hasattr(self.config, "verbose"): self.config.verbose = False

    @property
    def required_keys(self) -> set[LiteralString]: return {
        k for f in self.filters 
          for k in f.required_keys
    }

    def set_meta(self, meta: StereoData):
        for f in self.filters:
            f.set_meta(meta)

    def filter(self, values: TensorBundle, device: torch.device) -> torch.Tensor:
        mask = torch.ones((len(values),), dtype=torch.bool, device=device)
        for f in self.filters:
            mask = torch.logical_and(mask, f.filter(values, device))
            if self.config.verbose:
                Logger.write("info", f"\t{f.__class__.__name__} | => {mask.sum().item()}")
        
        return mask
    
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        assert config is not None
        assert isinstance(config.filter_args, list)
        for filter_arg in config.filter_args:
            IObservationFilter.is_valid_config(filter_arg)


class IdentityFilter(IObservationFilter):
    @property
    def required_keys(self) -> set[LiteralString]: return set()
    
    def filter(self, values: TensorBundle, device: torch.device) -> torch.Tensor:
        return torch.ones((len(values),), dtype=torch.bool, device=device)
    
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None: return


class CovarianceSanityFilter(IObservationFilter):
    @property
    def required_keys(self) -> set[LiteralString]: return {"obs1_covTc", "obs2_covTc"}
    
    def filter(self, values: TensorBundle, device: torch.device) -> torch.Tensor:
        cov1_has_nan = values.data["obs1_covTc"].isnan().any(dim=[-1, -2])
        cov1_has_inf = values.data["obs1_covTc"].isinf().any(dim=[-1, -2])
        cov2_has_nan = values.data["obs2_covTc"].isnan().any(dim=[-1, -2])
        cov2_has_inf = values.data["obs2_covTc"].isinf().any(dim=[-1, -2])
        return ~(cov1_has_nan | cov1_has_inf | cov2_has_nan | cov2_has_inf)
    
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None: return


class SimpleDepthFilter(IObservationFilter):
    def set_meta(self, meta: StereoData):
        if self.config.max_depth == "auto":
            self.config.max_depth = meta.fx * meta.frame_baseline
    
    @property
    def required_keys(self) -> set[LiteralString]: return {"pixel1_d", "pixel2_d"}

    def filter(self, values: TensorBundle, device: torch.device) -> torch.Tensor:
        return ~(  (values.data["pixel1_d"] < self.config.min_depth) 
                 | (values.data["pixel1_d"] > self.config.max_depth)
                 | (values.data["pixel2_d"] < self.config.min_depth) 
                 | (values.data["pixel2_d"] > self.config.max_depth)).squeeze(-1)
    
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        assert config is not None
        if isinstance(config.max_depth, (float, int)): assert config.max_depth > config.min_depth
        cls._enforce_config_spec(config, {
            "min_depth": lambda dist: isinstance(dist, (int, float)) and dist > 0.,
            "max_depth": lambda dist: (dist == "auto") or (isinstance(dist, (int, float)) and dist > 0.)
        })


class LikelyFrontOfCamFilter(IObservationFilter):
    @property
    def required_keys(self) -> set[LiteralString]: return {"pixel1_d", "pixel1_d_cov", "pixel2_d", "pixel2_d_cov"}
    
    def filter(self, values: TensorBundle, device: torch.device) -> torch.Tensor:
        if (values.data["pixel1_d_cov"] == -1).any(): 
            # This means we don't have covariance estimation, 
            # it's just a placeholder.
            return torch.ones((len(values),), dtype=torch.bool)
        
        return (  ((values.data["pixel1_d"] - (values.data["pixel1_d_cov"].sqrt() * 2)) > 0.)
                & ((values.data["pixel2_d"] - (values.data["pixel2_d_cov"].sqrt() * 2)) > 0.)).squeeze(-1)
    
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None: return
