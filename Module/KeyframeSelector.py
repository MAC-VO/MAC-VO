from abc import ABC, abstractmethod
from types import SimpleNamespace

from Utility.Extensions import ConfigTestableSubclass
from DataLoader import SourceDataFrame


class IKeyframeSelector(ABC, ConfigTestableSubclass):
    """
    Keyframe selector - decide whether a frame is considered as "keyframe" for backend optimization.
    
    In current implementation of MAC-VO, non-keyframes will be linearly interpolated on termination of system.
    """
    def __init__(self, config: SimpleNamespace):
        self.config = config
    
    @abstractmethod
    def isKeyframe(self, frame: SourceDataFrame) -> bool: ...


class AllKeyframe(IKeyframeSelector):
    def isKeyframe(self, frame: SourceDataFrame) -> bool:
        return True

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {})


class UniformKeyframe(IKeyframeSelector):    
    def isKeyframe(self, frame: SourceDataFrame) -> bool:
        return (frame.meta.idx % self.config.keyframe_freq) == 0

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "keyframe_freq": lambda freq: isinstance(freq, int) and (freq >= 1)
        })
