from typing import Optional, Sequence, Callable, TypeVar
from typing_extensions import Self

import numpy as np
import pypose as pp
import torch
from torchvision.transforms.functional import resize

T = TypeVar("T")
CollateFn = Callable[[Sequence[T],], T]

class DataFrame:
    collate_handlers: dict[str, CollateFn] = dict()
    
    @classmethod
    def collate(cls, batch: Sequence[Self]) -> Self:
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
            
            elif value is None:
                collate_fn = lambda seq: None
            
            else:
                raise ValueError(f"Unsupported data type {type(value)}, you need to overrider the collate method.")
            
            data_dict[key] = cls._collate([getattr(x, key) for x in batch], collate_fn)
        return cls(**data_dict)
    
    @staticmethod
    def _collate(batch: Sequence[T | None], collate_fn: CollateFn) -> T | None:
        if any([x is None for x in batch]): return None
        return collate_fn([x for x in batch if x is not None])


class MetaInfo:
    def __init__(self, idx: int, K: torch.Tensor, baseline: float, width: int, height: int):
        self.idx = idx
        self.K = K
        self.baseline = baseline
        self.width = width
        self.height = height

    @property
    def fx(self) -> float: return self.K[0, 0].item()
    @property
    def fy(self) -> float: return self.K[1, 1].item()
    @property
    def cx(self) -> float: return self.K[0, 2].item()
    @property
    def cy(self) -> float: return self.K[1, 2].item()


class IMUData:
    def __init__(self,
        dt: torch.Tensor,
        time: torch.Tensor, acc: torch.Tensor, gyro: torch.Tensor,
        gtVel: torch.Tensor, gtPos: torch.Tensor, gtRot: torch.Tensor,
        initPos: torch.Tensor, initRot: torch.Tensor, initVel: torch.Tensor
    ):
        self.dt = dt
        self.time = time
        
        self.acc = acc
        self.gyro = gyro
        
        self.gtVel = gtVel
        self.gtPos = gtPos
        self.gtRot = gtRot
        
        self.initPos = initPos
        self.initRot = initRot
        self.initVel = initVel


class SourceDataFrame(DataFrame):
    collate_handlers: dict[str, CollateFn] = {
        "meta": lambda batch: batch[0],
        "imu" : lambda imus : None    ,
    }
    
    def __init__(self, meta: MetaInfo, imageL: torch.Tensor, imageR: torch.Tensor, imu: Optional[IMUData], 
                 gtFlow: Optional[torch.Tensor], flowMask: Optional[torch.Tensor], gtDepth: Optional[torch.Tensor], gtPose: Optional[pp.LieTensor]) -> None:
        self.meta = meta
        self.imageL = imageL
        self.imageR = imageR
        self.imu = imu
        self.gtFlow = gtFlow
        self.flowMask = flowMask
        self.gtDepth = gtDepth
        self.gtPose = gtPose
    
    def resize_image(self, scale_u: float, scale_v: float) -> "SourceDataFrame":
        raw_height = self.meta.height
        raw_width  = self.meta.width
        
        target_h   = int(raw_height / scale_v)
        target_w   = int(raw_width  / scale_u)
        
        round_scale_v = raw_height / target_h
        round_scale_u = raw_width  / target_w
        
        self.meta.height = target_h
        self.meta.width  = target_w
        self.meta.K[0] /= round_scale_u
        self.meta.K[1] /= round_scale_v
        
        self.imageL = resize(self.imageL, [target_h, target_w])
        self.imageR = resize(self.imageR, [target_h, target_w])
        
        return self

    @staticmethod
    def _collate_tensor(batch: Sequence[Optional[torch.Tensor]]) -> Optional[torch.Tensor]:
        if any([x is None for x in batch]): return None
        return torch.cat(batch, dim=0)    #type: ignore
    
    @staticmethod
    def _collate_lie(batch: Sequence[Optional[pp.LieTensor]]) -> Optional[pp.LieTensor]:
        if any([x is None for x in batch]): return None
        return torch.cat(batch, dim=0)  #type: ignore


class FramePair(DataFrame):
    collate_handlers = {
        "cur": lambda batch: SourceDataFrame.collate(batch),
        "nxt": lambda batch: SourceDataFrame.collate(batch),
    }
    
    def __init__(self, cur: SourceDataFrame, nxt: SourceDataFrame) -> None:
        self.cur = cur
        self.nxt = nxt 
