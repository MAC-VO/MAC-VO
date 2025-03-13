from typing import Callable
from types import SimpleNamespace
from ..PrettyPrint import Logger

T_ConfigValue = int | float | str | bool | None | SimpleNamespace
T_LiteralSpec = Callable[[T_ConfigValue], bool]
T_ConfigSpec  = dict[str, T_LiteralSpec | "T_ConfigSpec"] | T_LiteralSpec


class ConfigTestable:
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        """
        `is_valid_config`
        
        This method is for minimum sanity check on config passed to the instance when instantiate.
        It *should not* run heavy workload (e.g. loading a pytorch module to check validity of weight / data shape)
        """
        Logger.write("warn", f"Class {cls} did not implement is_valid_config classmethod."
                              " Could not test if the config is valid or not (assume valid)")

    @staticmethod
    def _enforce_config_spec(config: SimpleNamespace | T_ConfigValue, spec: T_ConfigSpec, allow_excessive_cfg: bool=False):
        if not isinstance(spec, dict):
            is_valid = spec(config)
            if not is_valid:
                Logger.write("error", f"Config does not match specification! ({config} does not pass test)")
                raise ValueError(f"Config does not match specification! ({config} does not pass test)")
            return
        
        assert isinstance(config, SimpleNamespace), f"Config does not have same shape as the spec! Expect to receive a dict (simpleNamespace) but get literal value of {config}"
        for key, test_fn in spec.items():
            if key not in config.__dict__:
                raise KeyError(f"Config does not match specification! (expect to have key {key} but did not found)")
            ConfigTestable._enforce_config_spec(config.__dict__[key], test_fn)
        
        if not allow_excessive_cfg:
            spec_keys = spec.keys()
            cfg_keys  = dict(**vars(config)).keys()
            
            if len(spec_keys) != len(cfg_keys): raise KeyError(f"Excessive Keys: {set(cfg_keys) - set(spec_keys)} from {list(spec_keys)}")
