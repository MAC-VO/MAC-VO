from __future__ import annotations
import typing as T

I = T.TypeVar("I")
O = T.TypeVar("O")
O2 = T.TypeVar("O2")


class Chain(T.Generic[I, O]):
    """
    To have SML-like function chaining operator 
    (f1 >> f2)(x) = f2(f1(x))
    """
    def __init__(self, func: T.Callable[[I,], O]) -> None:
        super().__init__()
        self.func = func
    
    def __call__(self, arg: I) -> O:
        return self.func(arg)
    
    def __rshift__(self, other: T.Callable[[O], O2]) -> Chain[I, O2]:
        return Chain(lambda arg: other(self(arg)))

    @classmethod
    def side_effect(cls, other: T.Callable[[I], T.Any]) -> Chain[I, I]:
        def impl(arg: I) -> I:
            other(arg)
            return arg
        return cls[I, I](impl)  #type: ignore   # PyRight cannot infer this type correctly.
