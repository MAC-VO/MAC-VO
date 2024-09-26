"""
Provides `AutoScalingTensor` class, an efficient tensor wrapper for data accumulation along certain dimension
with little abstraction cost.

Github Repo: https://github.com/MarkChenYutian/AutoScalingTensor
"""

# MIT License
# 
# Copyright (c) 2024 Yutian Chen
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import math
from typing import Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    # Since extending torch.Tensor class using __torch_function__ is not supported by 
    # static type checker like MyPy and Pyright, we use this dummy class to fool the 
    # static analysis tool that AutoScalingTensor behaves like a torch.Tensor.
    # https://github.com/pytorch/pytorch/issues/75568
    # https://github.com/pytorch/pytorch/pull/75484
    # 
    # Due to the auto attribute delegation to torch.Tensor in the AutoScalingTensor.__getattribute__(...)
    # this version visible to type hinting actually matches all valid usages of the AutoScalingTensor
    # so there is no significant discrepency between static analysis bahavior and actual runtime result.
    class AutoScalingTensor(torch.Tensor):
        def __init__(self, 
                     shape: torch.Size | Sequence[int] | None, 
                     grow_on: int, 
                     init_tensor: torch.Tensor | None = None,
                     init_val: int | float | None = None,
                     **kwargs) -> None: ...
        def __new__(cls, *args, **kwargs) -> "AutoScalingTensor": ...
        def push(self, x: torch.Tensor) -> None: ...
        @property
        def current_size(self) -> int: ...
        @property
        def _curr_max_size(self) -> int: ...
        @property
        def tensor(self) -> torch.Tensor: ...
else:
    class AutoScalingTensor:
        def __init__(self, 
                    shape: torch.Size | Sequence[int] | None, 
                    grow_on: int, 
                    init_tensor: torch.Tensor | None = None,
                    init_val: int | float | None = None,
                    **kwargs
                    ) -> None:
            self.device = "cpu"
            self.grow_on = grow_on
            self.init_val = init_val
            self.current_size = 0
            if shape is not None:
                self._tensor = self._alloc_new_tensor(shape, **kwargs)
                self._curr_max_size = shape[grow_on]
            else:
                assert init_tensor is not None
                self._tensor = init_tensor
                self._curr_max_size = self._tensor.size(grow_on)
        
        def _alloc_new_tensor(self, shape, **kwargs):
            if self.init_val is None:
                return torch.empty(shape, device=self.device, **kwargs)
            else:
                return torch.full(shape, fill_value=self.init_val, device=self.device, **kwargs)
        
        def _scale_up_to(self, size: int):
            grow_to = int(2 ** math.ceil(math.log2(size + 1)))
            orig_shape = list(self._tensor.shape)
            orig_shape[self.grow_on] = grow_to
            
            new_storage = self._alloc_new_tensor(orig_shape, dtype=self._tensor.dtype)
            new_storage.narrow(dim=self.grow_on, start=0, length=self.current_size).copy_(
                self._tensor.narrow(dim=self.grow_on, start=0, length=self.current_size)
            )
            
            self._tensor = new_storage
            self._curr_max_size = grow_to
        
        @property
        def tensor(self) -> torch.Tensor:
            return self._tensor.narrow(dim=self.grow_on, start=0, length=self.current_size)
        
        def __repr__(self) -> str:
            return f"AutoScalingTensor(alloc={self._curr_max_size}, actual={self.current_size}, \n\tdata={self.tensor}\n)"

        def push(self, x: torch.Tensor) -> None:
            data_size = x.size(self.grow_on)
            
            if self.current_size + data_size >= self._curr_max_size:
                self._scale_up_to(self.current_size + data_size)
            assert self.current_size < self._curr_max_size
            
            self._tensor.narrow(dim=0, start=self.current_size, length=data_size).copy_(x, non_blocking=True)
            self.current_size += data_size
        
        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            if kwargs is None:
                kwargs = {}
            args = [data.tensor if isinstance(data, AutoScalingTensor) else data for data in args]
            return func(*args, **kwargs)

        # A further enhancement - we want AutoScale to behave exactly like the tensor it contains
        def __getattribute__(self, name: str):
            if name in {'_tensor', 'tensor', 'push', '__class__'}:
                return object.__getattribute__(self, name)
            try:
                return object.__getattribute__(self, name)
            except AttributeError:
                # If it's not a class attribute, delegate to the tensor
                tensor = object.__getattribute__(self, 'tensor')
                return getattr(tensor, name)

        # Magic methods cannot be forwarded automatically, so has to do this
        def __getitem__(self, slice): return self.tensor.__getitem__(slice)
        def __setitem__(self, slice, val): return self.tensor.__setitem__(slice, val)
        def __rsub__(self, other): return self.tensor.__rsub__(other)
        def __rdiv__(self, other): return self.tensor.__rdiv__(other)
        __rtruediv__ = __rdiv__
        def __itruediv__(self, *args, **kwargs): return self.tensor.__itruediv__(*args, **kwargs)
        def __pow__(self, exponent): return self.tensor.__pow__(exponent)
        def __ipow__(self, exponent): return self.tensor.__ipow__(exponent)
        def __rmod__(self, other): return self.tensor.__rmod__(other)
        def __format__(self, spec): return self.tensor.__format__(spec)
        def __rpow__(self, other): return self.tensor.__rpow__(other)
        def __add__(self, other): return self.tensor.__add__(other)
        def __floordiv__(self, other): return self.tensor.__floordiv__(other)
        def __rfloordiv__(self, other): return self.tensor.__rfloordiv__(other)
        def __rlshift__(self, other): return self.tensor.__rlshift__(other)
        def __rrshift__(self, other): return self.tensor.__rrshift__(other)
        def __rmatmul__(self, other): return self.tensor.__rmatmul__(other)
        def __pos__(self): return self.tensor.__pos__()
        def __neg__(self): return self.tensor.__neg__()
        def __abs__(self): return self.tensor.__abs__()
        def __len__(self): return self.tensor.__len__()
        def __iter__(self): return self.tensor.__iter__()
        def __hash__(self): return self.tensor.__hash__()
        def __dir__(self): return self.tensor.__dir__()
        def __reversed__(self): return self.tensor.__reversed__()
