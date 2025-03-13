import torch
from functools import wraps
from typing import Callable, ParamSpec, TypeVar

from Utility.PrettyPrint import Logger


I = ParamSpec("I")
O = TypeVar("O")

class OnCallCompiler:
    """
    A decorator class that attempts to compile a given function using `torch.compile` for optimization.
    If the compilation fails, it falls back to the original unoptimized function.
    Attributes:
        compile_enabled (bool): A flag indicating whether the function can be compiled.
        optimized (Callable | None): The compiled version of the function, if available.
    Methods:
        __call__(func: Callable[I, O]) -> Callable[I, O]:
            Wraps the given function and attempts to compile it. If compilation fails,
            it logs a warning and uses the unoptimized function instead.
    """
    
    def __init__(self):
        self.compile_enabled: bool = True
        self.optimized  : Callable | None = None
    
    def __call__(self, func: Callable[I, O]) -> Callable[I, O]:
        @wraps(func)
        def implement(*args: I.args, **kwargs: I.kwargs) -> O:
            if not self.compile_enabled:
                return func(*args, **kwargs)
            
            if self.optimized is None:
                try:
                    self.optimized = torch.compile(func)
                    return self.optimized(*args, **kwargs)
                except Exception as e:
                    Logger.write("warn", f"Failed to compile function {func} - reason {e}. Will use unoptimized function instead.")
                    self.compile_enabled = False
                    return func(*args, **kwargs)
            
            assert self.optimized is not None
            return self.optimized(*args, **kwargs)
        return implement
