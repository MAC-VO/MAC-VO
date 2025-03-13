from types import SimpleNamespace
from .TensorExtension import AutoScalingTensor, TensorQueue
from .Testable import ConfigTestable
from .SubclassRegistry import SubclassRegistry
from .Chain import Chain
from .OnCallCompiler import OnCallCompiler
from .GridRecorder import GridRecorder

# A mixin class between two traits - SubclassRegistry (dynamic reflection) and ConfigTestable
class ConfigTestableSubclass(SubclassRegistry, ConfigTestable):
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        assert config is not None
        if (not hasattr(config, 'type')) or (not hasattr(config, 'args')):
            raise ValueError(f"Unable to dynamically delegate a subclass to test provided config. Incorrect config shape! {config=}")
        cls.get_class(config.type).is_valid_config(config.args)
