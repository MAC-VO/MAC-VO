from types import SimpleNamespace
from .AutoScalingTensor import AutoScalingTensor
from .Testable import ConfigTestable
from .SubclassRegistry import SubclassRegistry


# A mixin class between two traits - SubclassRegistry (dynamic reflection) and ConfigTestable
class ConfigTestableSubclass(SubclassRegistry, ConfigTestable):
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        assert config is not None
        cls.get_class(config.type).is_valid_config(config.args)
