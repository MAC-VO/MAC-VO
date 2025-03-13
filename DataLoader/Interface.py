import torch
import pypose as pp
import numpy  as np

import typing as T
from typing_extensions import Self
from dataclasses import dataclass
from itertools import chain


Tp = T.TypeVar("Tp")
CollateFn = T.Callable[[T.Sequence[Tp],], Tp]

@dataclass(kw_only=True)
class Collatable:
    collate_handlers: T.ClassVar[dict[str, CollateFn]] = dict()
    
    @classmethod
    def collate(cls, batch: T.Sequence[Self]) -> Self:
        """
        A default collate function that will handle torch.Tensor, pp.LieTensor and
        np.array automatically. You can perform more customized collate by one of the following methods:
        
        1. Overriding the collate method
        
        2. Setting the class attribute `collate_handlers` to a dictionary that maps the attribute name to the collate function corresponding to that field.
        
        """
        data_dict = dict()
        for key, value in batch[0].__dict__.items():
            if key in cls.collate_handlers:
                collate_fn = cls.collate_handlers[key]
            elif isinstance(value, torch.Tensor):
                collate_fn = lambda seq: torch.cat(seq, dim=0)
            elif isinstance(value, pp.LieTensor):
                collate_fn = lambda seq: torch.stack(seq, dim=0)
            elif isinstance(value, np.ndarray):
                collate_fn = lambda seq: np.concatenate(seq, axis=0)
            elif isinstance(value, list):
                collate_fn = lambda seq: list(chain.from_iterable(seq))
            elif isinstance(value, Collatable):
                collate_fn = value.collate
            elif value is None:
                collate_fn = lambda seq: None
            else:
                raise ValueError(f"Unsupported data type {type(value)}, you need to overrider the collate method.")
            data_dict[key] = cls._collate([getattr(x, key) for x in batch], collate_fn)
        return cls(**data_dict)
    
    @staticmethod
    def _collate(batch: T.Sequence[Tp | None], collate_fn: CollateFn) -> Tp | None:
        if any([x is None for x in batch]): return None
        return collate_fn([x for x in batch if x is not None])


@dataclass(kw_only=True)
class StereoData(Collatable):
    # Transformation from body frame to sensor frame
    T_BS: pp.LieTensor      # torch.float32, pp.SE3 of shape Bx7
    K   : torch.Tensor      # torch.float32 of shape Bx3x3
    baseline: torch.Tensor   # Baseline (m) between left and right camera, len(list) = B
    time_ns : list[int]     # Time (ns) of data received, len(list) = B
    height: int             # H
    width : int             # W
    
    @property
    def frame_ns(self) -> int:
        assert len(self.time_ns) == 1, "Can only use frame_ns on unbatched data."
        return self.time_ns[0]
    @property
    def frame_ms(self) -> float: return self.frame_ns / 1000.
    @property
    def frame_baseline(self) -> float:
        assert self.baseline.size(0) == 1, "Can only use frame_baseline on unbatched data"
        return self.baseline.item()
    @property
    def frame_K(self) -> torch.Tensor:
        assert self.K.size(0) == 1, "Can only use frame_K on unbatched data"
        return self.K[0]
    
    @property
    def time_ms(self) -> list[float]: return [t / 1000. for t in self.time_ns]
    @property
    def fx(self) -> float:
        assert self.K.size(0) == 1, "Can only use property shortcut on unbatched data"
        return self.K[0, 0, 0].item()
    @property
    def fy(self) -> float:
        assert self.K.size(0) == 1, "Can only use property shortcut on unbatched data"
        return self.K[0, 1, 1].item()
    @property
    def cx(self) -> float:
        assert self.K.size(0) == 1, "Can only use property shortcut on unbatched data"
        return self.K[0, 0, 2].item()
    @property
    def cy(self) -> float:
        assert self.K.size(0) == 1, "Can only use property shortcut on unbatched data"
        return self.K[0, 1, 2].item()
    
    # Sensor Data
    imageL: torch.Tensor    # torch.float32 of shape Bx3xHxW
    imageR: torch.Tensor    # torch.float32 of shape Bx3xHxW
    
    # Label & Ground Truth
    gt_flow  : torch.Tensor | None = None    # torch.float32 of shape Bx2xHxW 
    flow_mask: torch.Tensor | None = None    # torch.bool    of shape Bx1xHxW
    gt_depth : torch.Tensor | None = None    # torch.float32 of shape Bx1xHxW 
    
    collate_handlers = {
        "height": lambda batch: batch[0],
        "width" : lambda batch: batch[0],
    }


@dataclass(kw_only=True)
class IMUData(Collatable):
    """
    (N) IMU measurements from a certain period of time
    """
    # Transformation from body frame to sensor frame
    T_BS: pp.LieTensor          # torch.float32, pp.SE3 of shape Bx7
    time_ns: torch.Tensor       # torch.int64 of shape BxNx1
    gravity: list[float]        # gravity constant
    
    @property
    def time_delta(self) -> torch.Tensor: return self.time_ns[:, 1:] - self.time_ns[:, :-1]
    @property
    def time_ms(self) -> torch.Tensor   : return self.time_ns.double() / 1000.
    @property
    def frame_gravity(self) -> float:
        assert len(self.gravity) == 1, "frame_gravity can only be used on unbatched data"
        return self.gravity[0]
    
    # acc: Raw acceleration of IMU body frame with gravity added
    acc   : torch.Tensor                # torch.float32 of shape BxNx3
    # gyro: Angular rate of the IMU body frame
    gyro  : torch.Tensor                # torch.float32 of shape BxNx3


@dataclass(kw_only=True)
class AttitudeData(Collatable):
    # Transformation from body frame to sensor frame
    T_BS: pp.LieTensor          # torch.float32, pp.SE3 of shape Bx7
    time_ns: torch.Tensor       # torch.int64 of shape BxNx1
    gravity: list[float]        # gravity constant
    
    @property
    def time_delta(self) -> torch.Tensor: return self.time_ns[:, 1:] - self.time_ns[:, :-1]
    @property
    def time_ms(self) -> torch.Tensor   : return self.time_ns.double() / 1000.
    @property
    def frame_gravity(self) -> float:
        assert len(self.gravity) == 1, "frame_gravity can only be used on unbatched data"
        return self.gravity[0]
    
    # Ground truth velocity, position and rotation
    gt_vel: torch.Tensor      # torch.float32 of shape BxNx3
    gt_pos: torch.Tensor      # torch.float32 of shape BxNx3
    gt_rot: pp.LieTensor      # torch.float32 of shape BxNx4, pp.SO3 rotation.
    
    # Initial condition for IMU preintegration
    init_vel: torch.Tensor      # torch.float32 of shape Bx1x3
    init_pos: torch.Tensor      # torch.float32 of shape Bx1x3
    init_rot: pp.LieTensor      # torch.float32 of shape Bx1x4, pp.SO3 rotation.


@dataclass(kw_only=True)
class DataFrame(Collatable):
    idx: list[int]
    # Ground truth pose under body frame
    gt_pose  : pp.LieTensor | None = None   # pp.SE3 of shape Bx7
    
    # Time in nanoseconds
    time_ns  : list[int]
    
    @property
    def frame_idx(self) -> int:
        assert len(self.idx) == 1, "frame_idx property is only valid on unbatched data"
        return self.idx[0]

    @property
    def frame_time_ns(self) -> int:
        assert len(self.time_ns) == 1, "frame_time_ns property is only valid on unbatched data"
        return self.time_ns[0]

T_Data = T.TypeVar("T_Data", bound=DataFrame)

@dataclass(kw_only=True)
class DataFramePair(DataFrame, T.Generic[T_Data]):
    cur : T_Data
    nxt : T_Data

@dataclass(kw_only=True)
class StereoFrame(DataFrame):
    stereo   : StereoData

@dataclass(kw_only=True)
class StereoInertialFrame(StereoFrame):
    imu        : IMUData
    gt_attitude: AttitudeData | None = None
