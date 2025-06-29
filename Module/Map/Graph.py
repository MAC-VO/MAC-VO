"""
Provide graph infrastructure (node with arbitrary features and edges)
"""
from __future__ import annotations

import torch
import numpy as np
import typing as T
from typing_extensions import LiteralString, Self

from abc import ABC, abstractmethod
from Utility.Extensions import AutoScalingTensor


T_Fields = T.TypeVar("T_Fields", bound=LiteralString, covariant=True)


@T.no_type_check    # Jaxtyping and typeguard does not support StringLiteral well.
class TensorBundle(T.Generic[T_Fields]):
    """
    Support indexing and slicing through a dictionary with torch.tensor.
    Used as a template for the SoA (structure of array) design pattern.
    Ref: https://en.wikipedia.org/wiki/AoS_and_SoA
    
    index: the index for each torch.tensor on the dimention of N.
    data: dict[T_Fields: Tensor(N, ...)]
    """
    def __init__(self, index: torch.Tensor, data: dict[T_Fields, torch.Tensor]):
        self.index = index
        self.data  = data
    
    @classmethod
    def init(cls: type[Self], data: dict[T_Fields, torch.Tensor]) -> Self:
        sizes = [v.size(0) for v in data.values()]
        assert all([s == sizes[0] for s in sizes]), f"TensorBundle requires all features have (Nx...) shape with same 'N', get {sizes}"
        
        index   = torch.arange(0, sizes[0], dtype=torch.long)
        return cls(index, data)

    def __getitem__(self, index) -> TensorBundle[T_Fields]:
        selected_dict: dict[T_Fields, torch.Tensor] = {
            k : v.__getitem__(index) for k, v in self.data.items()
        }
        return self.__class__(self.index.__getitem__(index), selected_dict)

    def __setitem__(self, index, value: TensorBundle[T_Fields]) -> None:
        for k in self.data:
            self.data[k].__setitem__(index, value.data[k])

    def apply(self, func: T.Callable[[torch.Tensor,], torch.Tensor]) -> None:
        for k in self.data:
            self.data[k] = func(self.data[k])
  
    def __len__(self) -> int:
        return self.index.size(0)
    
    def __repr__(self) -> str:
        return f"TensorBundle(size={len(self)}, keys=[{', '.join(self.data.keys())}])"
    
    def serialize(self, prefix: str) -> dict[str, np.ndarray]:
        return {
            f"{prefix}/{k}": v.cpu().numpy()
            for k, v in self.data.items()
        }
    
    @classmethod
    def deserialize(cls, prefix: str, value: dict[str, np.ndarray]) -> Self:
        allowed_fields: T_Fields | None = T.get_args(cls.__orig_bases__[0].__args__[0])  # type: ignore
        if not allowed_fields:
            raise Exception("T_Fields must be specified with concrete string literal types.")
        
        data: dict[T_Fields, torch.Tensor] = T.cast(dict[T_Fields, torch.Tensor], {
            k : torch.tensor(value[f"{prefix}/k"])
            for k in allowed_fields
        })
        return cls.init(data)
  

@T.no_type_check    # Jaxtyping and typeguard does not support StringLiteral well.
class AutoScalingBundle(TensorBundle[T_Fields]):
    def __init__(self, index: AutoScalingTensor, data: dict[T_Fields, AutoScalingTensor]):
        self.index = index
        self.data  = data
        self.edges_from: list[Scaling_DenseEdge_Multi | Scaling_SparseEdge_Multi | Scaling_SingleEdge] = []
    
    def __getitem__(self, index) -> TensorBundle[T_Fields]:
        selected_dict: dict[T_Fields, torch.Tensor] = {
            k : v.__getitem__(index) for k, v in self.data.items()
        }
        return TensorBundle[T_Fields](self.index.__getitem__(index), selected_dict)
    
    def push(self, value: TensorBundle[T_Fields]) -> torch.Tensor:
        start_idx = self.index.size(0)
        new_index = torch.arange(start_idx, start_idx + len(value))
        self.index.push(new_index)
        for k in self.data:
            self.data[k].push(value.data[k])
        
        for edge in self.edges_from:
            if isinstance(edge, Scaling_SparseEdge_Multi):
                edge.push(SparseEdge_Multi(len(value), edge.max_deg))
            elif isinstance(edge, Scaling_DenseEdge_Multi):
                edge.push(DenseEdge_Multi(len(value), edge.max_deg))
            elif isinstance(edge, Scaling_SingleEdge):
                edge.push(SingleEdge(len(value)))
            else: raise Exception("Impossible")
        
        return new_index

    def register_edge(self, edge: Scaling_SparseEdge_Multi | Scaling_DenseEdge_Multi | Scaling_SingleEdge):
        self.edges_from.append(edge)
    
    def __repr__(self) -> str:
        return f"ScalingBundle(size={len(self)}, keys=[{', '.join(self.data.keys())}])"


T_From = T.TypeVar("T_From", bound=TensorBundle)
T_To   = T.TypeVar("T_To"  , bound=TensorBundle)


class EdgeLike(ABC):
    @abstractmethod
    def project(self, from_index: torch.Tensor) -> torch.Tensor: ...
    
    @abstractmethod
    def serialize(self, prefix: str) -> dict[str, np.ndarray]: ...
    
    @classmethod
    @abstractmethod
    def deserialize(cls, prefix: str, value: dict[str, np.ndarray]) -> Self: ...


class SparseEdge_Multi(EdgeLike):
    """
    An arbitrary one-to-multi mapping relationship.
    """
    def __init__(self, num_from: int, max_deg: int):
        self.out_deg = torch.zeros((num_from,), dtype=torch.long)
        self.edges   = torch.ones((num_from, max_deg), dtype=torch.long) * -1
        self.max_deg = max_deg

    def add(self, from_idx: torch.Tensor, to_idx: torch.Tensor):
        assert from_idx.shape == to_idx.shape
        self.edges[from_idx, self.out_deg[from_idx]] = to_idx
        self.out_deg[from_idx] += 1

    def project(self, from_index: torch.Tensor) -> torch.Tensor:
        to_idx = self.edges[from_index].flatten()
        return to_idx[to_idx >= 0]
    
    def serialize(self, prefix: str) -> dict[str, np.ndarray]:
        return {
            f"{prefix}/edges": self.edges.cpu().numpy(),
            f"{prefix}/deg"  : self.out_deg.cpu().numpy()
        }
    
    @classmethod
    def deserialize(cls, prefix: str, value: dict[str, np.ndarray]) -> Self:
        edges = torch.Tensor(value[f"{prefix}/edges"])
        deg   = torch.Tensor(value[f"{prefix}/deg"])
        num_from, max_deg = edges.shape[0], edges.shape[1]
        
        edge_instance = cls(num_from, max_deg)
        edge_instance.edges = edges
        edge_instance.out_deg = deg
        return edge_instance


class DenseEdge_Multi(EdgeLike):
    """
    In case of one-to-multi mapping with continuous index. Can significantly reduce the 
    memory footprint.
    """
    def __init__(self, num_from: int, max_deg: int) -> None:
        self.ranges     = torch.ones((num_from, max_deg, 2), dtype=torch.long) * -1
        self.num_ranges = torch.zeros((num_from,), dtype=torch.long)
        self.max_deg    = max_deg

    def add(self, from_idx: torch.Tensor, to_idx_start: torch.Tensor, to_idx_length: torch.Tensor):
        self.ranges[from_idx, self.num_ranges[from_idx], 0] = to_idx_start
        self.ranges[from_idx, self.num_ranges[from_idx], 1] = to_idx_length
        self.num_ranges[from_idx] += 1
    
    def project(self, from_index: torch.Tensor) -> torch.Tensor:
        offsets = self.ranges[from_index, :, 0].flatten()
        offsets = offsets[offsets >= 0]
        
        lengths = self.ranges[from_index, :, 1].flatten()
        lengths = lengths[lengths >= 0]
        
        if torch.any(lengths > 0):
            expanded_offsets = torch.repeat_interleave(offsets, lengths)
            range_indices    = torch.arange(lengths.sum().item())
            range_starts     = torch.repeat_interleave(lengths.cumsum(0) - lengths, lengths)
            return expanded_offsets + (range_indices - range_starts)
        else:
            return torch.tensor([], dtype=torch.long)
    
    def serialize(self, prefix: str) -> dict[str, np.ndarray]:
        return {
            f"{prefix}/ranges": self.ranges.numpy(),
            f"{prefix}/deg"   : self.num_ranges.numpy()
        }
    
    @classmethod
    def deserialize(cls, prefix: str, value: dict[str, np.ndarray]) -> Self:
        ranges     = torch.tensor(value[f"{prefix}/ranges"])
        num_ranges = torch.tensor(value[f"{prefix}/deg"])
        num_from, max_deg = ranges.shape[0], ranges.shape[1]
        
        edge_instance = cls(num_from, max_deg)
        edge_instance.ranges     = ranges
        edge_instance.num_ranges = num_ranges
        return edge_instance


class SingleEdge(EdgeLike):
    """
    A one-to-one mapping relationship between
    """
    def __init__(self, num_elem: int) -> None:
        self.mapping = torch.empty((num_elem,), dtype=torch.long).fill_(-1)
    
    def project(self, from_index: torch.Tensor) -> torch.Tensor:
        mapped = self.mapping[from_index]
        return mapped[mapped >= 0]

    def set(self, elem_idx: torch.Tensor, map_idx: torch.Tensor) -> None:
        self.mapping[elem_idx] = map_idx
    
    def serialize(self, prefix: str) -> dict[str, np.ndarray]:
        return {f"{prefix}/mapping": self.mapping.cpu().numpy()}

    @classmethod
    def deserialize(cls, prefix: str, value: dict[str, np.ndarray]) -> Self:
        mapping = torch.tensor(value[f"{prefix}/mapping"])
        num_elem = mapping.shape[0]
        
        edge_instance = cls(num_elem)
        edge_instance.mapping = mapping
        return edge_instance


class Scaling_SparseEdge_Multi(SparseEdge_Multi):
    def __init__(self, num_from: int, max_deg: int):
        self.out_deg = AutoScalingTensor((num_from,), grow_on=0, dtype=torch.long, init_val=0)
        self.edges   = AutoScalingTensor((num_from, max_deg), grow_on=0, dtype=torch.long, init_val=-1)
        self.max_deg    = max_deg
        
    
    def push(self, edge_multi: SparseEdge_Multi):
        self.out_deg.push(edge_multi.out_deg)
        self.edges  .push(edge_multi.edges)
    
    @classmethod
    def deserialize(cls, prefix: str, value: dict[str, np.ndarray]) -> Self:
        tensor_edge = super().deserialize(prefix, value)
        
        # Convert from torch.Tensor to AutoScalingTensor
        tensor_edge.out_deg = AutoScalingTensor(None, grow_on=0, init_tensor=tensor_edge.out_deg)
        tensor_edge.edges   = AutoScalingTensor(None, grow_on=0, init_tensor=tensor_edge.edges)
        return tensor_edge


class Scaling_DenseEdge_Multi(DenseEdge_Multi):
    def __init__(self, num_from: int, max_deg: int) -> None:
        self.ranges     = AutoScalingTensor((num_from, max_deg, 2), grow_on=0, dtype=torch.long, init_val=-1)
        self.num_ranges = AutoScalingTensor((num_from,), grow_on=0, dtype=torch.long, init_val=0)
        self.max_deg    = max_deg

    def push(self, edges_multi: DenseEdge_Multi):
        self.ranges.push(edges_multi.ranges)
        self.num_ranges.push(edges_multi.num_ranges)
    
    @classmethod
    def deserialize(cls, prefix: str, value: dict[str, np.ndarray]) -> Self:
        tensor_edge = super().deserialize(prefix, value)
        
        # Convert from torch.Tensor to AutoScalingTensor
        tensor_edge.ranges     = AutoScalingTensor(None, grow_on=0, init_tensor=tensor_edge.ranges)
        tensor_edge.num_ranges = AutoScalingTensor(None, grow_on=0, init_tensor=tensor_edge.num_ranges)
        return tensor_edge


class Scaling_SingleEdge(SingleEdge):
    def __init__(self, num_elem: int) -> None:
        self.mapping = AutoScalingTensor((num_elem,), grow_on=0, dtype=torch.long, init_val=-1)
    
    def push(self, edges_single: SingleEdge):
        self.mapping.push(edges_single.mapping)

    @classmethod
    def deserialize(cls, prefix: str, value: dict[str, np.ndarray]) -> Self:
        tensor_edge = super().deserialize(prefix, value)

        # Convert from torch.Tensor to AutoScalingTensor
        tensor_edge.mapping = AutoScalingTensor(None, grow_on=0, init_tensor=tensor_edge.mapping)
        return tensor_edge
