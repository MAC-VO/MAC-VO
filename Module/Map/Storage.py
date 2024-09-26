import torch
import pypose as pp
from enum import IntFlag, auto
from typing_extensions import Callable, overload, Literal

from Utility.Point import pixel2point_NED
from Utility.Extensions import AutoScalingTensor

# 
# NOTE: Difference between Batch<...>Storage and Batch<...>
#       * Batch...Storage - is a set of autoscaling tensor. It provides efficient way of accumulating data in it.
#         However, it's initialization has overheads and requires user to import AutoScalingTensor in all places.
#
#       * Batch... - therefore, we use Batch... as an interface between the computation and storage. Everytime user
#         retrieve stuff from the underlying AutoScalingTensor, we return a Batch... object. Everytime user want to 
#         add new data into underlying storage, they need to create a Batch... object.
#
#       In a word - Batch...Storage is for Storage, Batch... is for computation and used by user.
#

BoolType = Literal[True, False]
MAP_DEVICE = torch.device("cpu")


def fast_tensor_intersect(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    When a, b are (N, 1) and (M, 1) tensor, return a tensor of shape (K, 1) such that
    
    @ensures
    all([(ret in a) and (ret in b) for ret in return]
    """
    both = torch.cat([a, b], dim=0)
    both_val, both_count = both.unique(return_counts=True)
    return both_val[both_count > 1]


class BatchPoints:
    def __init__(self, position: torch.Tensor, cov_Tw: torch.Tensor, color: torch.Tensor,
                 point_idx: torch.Tensor | None = None,
                 observed_count: torch.Tensor | None = None, observed_by: torch.Tensor | None = None,
                 device: torch.device = MAP_DEVICE):
        self.position       = position       # Nx3, torch.float
        self.cov_Tw         = cov_Tw         # Nx3x3, torch.float
        self.color          = color          # Nx3, torch.uint8
        self.point_idx      = point_idx      # N, torch.long
        self.observed_count = observed_count # N, torch.uint8
        self.observed_by    = observed_by    # N, Storage.OBSERVE_MAX_CNT, torch.long
        self.apply(lambda x: x.to(device), inplace=True)
    
    def __repr__(self) -> str:
        return f"BatchPoints(#pts={self.position.size(0)})"
    
    def __len__(self) -> int:
        return self.position.size(0)
    
    @property
    def is_batched(self) -> bool: return self.position.dim() == 2 
    
    @property
    def device(self) -> torch.device: return self.position.device
    
    def to(self, device: torch.device) -> "BatchPoints":
        return self.apply(lambda x: x.to(device), inplace=False)
    
    def inflate(self) -> "BatchPoints":
        if self.is_batched: return self
        return self.apply(lambda x: x.unsqueeze(0), inplace=False)
    
    def squeeze(self) -> "BatchPoints":
        if not self.is_batched: return self
        return self.apply(lambda x: x.squeeze(0), inplace=False)
    
    def __getitem__(self, slice) -> "BatchPoints":
        return self.apply(lambda x: x.__getitem__(slice), inplace=False)
    
    def __and__(self, other: "BatchPoints") -> "BatchPoints":
        """
        Calculate the intersection between two BatchPoints.
        The intersection is based on the point index. If two points have the same index, they are considered
        the same point.
        
        Can only intersect registered points (points with index).
        """
        assert self.point_idx is not None and other.point_idx is not None, "Can only intersect registered points"
        
        intersect_idx = fast_tensor_intersect(self.point_idx, other.point_idx)
        self_selector  = torch.isin(self.point_idx, intersect_idx)  #TODO: This lines maybe slow, need to fix them
        
        return self[self_selector]
    
    @overload
    def apply(self, func: Callable[[torch.Tensor,], torch.Tensor], inplace: Literal[True]) -> None: ...
    @overload
    def apply(self, func: Callable[[torch.Tensor,], torch.Tensor], inplace: Literal[False]) -> "BatchPoints": ...
    
    def apply(self, func: Callable[[torch.Tensor,], torch.Tensor], inplace: BoolType) -> "BatchPoints | None":
        if inplace:
            self.position = func(self.position)
            self.cov_Tw = func(self.cov_Tw)
            self.color = func(self.color)
            self.point_idx = func(self.point_idx) if self.point_idx is not None else None
            self.observed_count = func(self.observed_count) if self.observed_count is not None else None
            self.observed_by = func(self.observed_by) if self.observed_by is not None else None
        else:
            return BatchPoints(
                func(self.position),
                func(self.cov_Tw),
                func(self.color),
                point_idx=func(self.point_idx) if self.point_idx is not None else None,
                observed_count=func(self.observed_count) if self.observed_count is not None else None,
                observed_by=func(self.observed_by) if self.observed_by is not None else None
            )


class BatchObservation:
    def __init__(self, point_idx: torch.Tensor, frame_idx: torch.Tensor, pixel_uv: torch.Tensor, pixel_d: torch.Tensor, cov_Tc: torch.Tensor,
                 cov_pixel_uv: torch.Tensor, cov_pixel_d: torch.Tensor, device: torch.device=MAP_DEVICE) -> None:
        self.point_idx      : torch.Tensor = point_idx
        self.frame_idx      : torch.Tensor = frame_idx
        self.pixel_uv       : torch.Tensor = pixel_uv
        self.pixel_d        : torch.Tensor = pixel_d
        self.cov_Tc         : torch.Tensor = cov_Tc
        self.cov_pixel_uv   : torch.Tensor = cov_pixel_uv
        self.cov_pixel_d    : torch.Tensor = cov_pixel_d
        self.apply(lambda x: x.to(device), inplace=True)
    
    def __repr__(self) -> str:
        return f"BatchObservation(#obs={self.point_idx.size(0)})"
    
    def __len__(self) -> int:
        return self.point_idx.size(0)

    def __getitem__(self, slice) -> "BatchObservation":
        return self.apply(lambda x: x.__getitem__(slice), inplace=False).inflate()
    
    @property
    def is_batched(self) -> bool: return self.point_idx.dim() == 1
    
    @property
    def device(self) -> torch.device: return self.point_idx.device
    
    def to(self, device: torch.device) -> "BatchObservation":
        return self.apply(lambda x: x.to(device), inplace=False)
    
    def inflate(self) -> "BatchObservation":
        if self.is_batched: return self
        return self.apply(lambda x: x.unsqueeze(0), inplace=False)
    
    def squeeze(self) -> "BatchObservation":
        if not self.is_batched: return self
        return self.apply(lambda x: x.squeeze(0), inplace=False)
    
    def project_Tc(self, K: torch.Tensor) -> torch.Tensor:
        return pixel2point_NED(self.pixel_uv, self.pixel_d, K)    

    @overload
    def apply(self, func: Callable[[torch.Tensor,], torch.Tensor], inplace: Literal[True]) -> None: ...
    @overload
    def apply(self, func: Callable[[torch.Tensor,], torch.Tensor], inplace: Literal[False]) -> "BatchObservation": ...
    def apply(self, func: Callable[[torch.Tensor,], torch.Tensor], inplace: BoolType) -> "BatchObservation | None":
        if inplace:
            self.point_idx = func(self.point_idx)
            self.frame_idx = func(self.frame_idx)
            self.pixel_uv = func(self.pixel_uv)
            self.pixel_d = func(self.pixel_d)
            self.cov_Tc = func(self.cov_Tc)
            self.cov_pixel_uv = func(self.cov_pixel_uv)
            self.cov_pixel_d = func(self.cov_pixel_d)
        else:
            return BatchObservation(
                func(self.point_idx), func(self.frame_idx), func(self.pixel_uv),
                func(self.pixel_d), func(self.cov_Tc), func(self.cov_pixel_uv), func(self.cov_pixel_d),
            )


class BatchFrame:
    FLAG_EMPTY = 0
    FLAG_VO_LOSTTRACK = 1
    FLAG_NEED_INTERP = 2
    
    def __init__(self, K: torch.Tensor, pose: pp.LieTensor, obs_range: torch.Tensor,
                 flag: torch.Tensor, quality: torch.Tensor, frame_idx: torch.Tensor | None = None,
                 device: torch.device = MAP_DEVICE) -> None:
        self.K                  = K
        self.pose: pp.LieTensor = pose   #type: ignore
        self.obs_range          = obs_range
        self.quality            = quality
        self.flag               = flag
        self.frame_idx          = frame_idx
        self.apply(lambda x: x.to(device), inplace=True)
    
    def __repr__(self) -> str:
        return f"BatchFrame(#frame={self.K.size(0)})"
    
    def __len__(self) -> int:
        return self.K.size(0)
    
    def __getitem__(self, slice) -> "BatchFrame":
        return BatchFrame(
            self.K.__getitem__(slice),
            pp.SE3(self.pose.__getitem__(slice)),
            self.obs_range.__getitem__(slice),
            self.flag.__getitem__(slice),
            self.quality.__getitem__(slice),
            frame_idx=None if self.frame_idx is None else self.frame_idx.__getitem__(slice)
        ).inflate()
    
    @property
    def is_batched(self) -> bool: return self.K.dim() == 3
    
    @property
    def device(self) -> torch.device: return self.K.device
    
    def to(self, device: torch.device) -> "BatchFrame":
        return self.apply(lambda x: x.to(device), inplace=False)
    
    def inflate(self) -> "BatchFrame":
        if self.is_batched: return self
        return self.apply(lambda x: x.unsqueeze(0), inplace=False)
    
    def squeeze(self) -> "BatchFrame":
        if not self.is_batched: return self
        return self.apply(lambda x: x.squeeze(0), inplace=False)
    
    def unique_contraction(self) -> "BatchFrame":
        """Dedupe a batch of frames.
        """
        assert self.is_batched
        assert self.frame_idx is not None
        if len(self) == 1: return self
        
        _, unique_idx = self.frame_idx.unique(return_inverse=True)
        unique_selector = unique_idx.unique()
        return self.apply(lambda x: x[unique_selector], inplace=False)
    
    @overload
    def apply(self, func: Callable[[torch.Tensor,], torch.Tensor], inplace: Literal[True]) -> None: ...
    @overload
    def apply(self, func: Callable[[torch.Tensor,], torch.Tensor], inplace: Literal[False]) -> "BatchFrame": ...
    
    def apply(self, func: Callable[[torch.Tensor,], torch.Tensor], inplace: BoolType) -> "BatchFrame | None":
        if inplace:
            self.K = func(self.K)
            self.pose = func(self.pose) #type: ignore
            self.obs_range = func(self.obs_range)
            self.flag = func(self.flag)
            self.quality = func(self.quality)
            self.frame_idx = func(self.frame_idx) if self.frame_idx is not None else None
        else:
            return BatchFrame(
                func(self.K),
                func(self.pose), #type: ignore
                func(self.obs_range),
                func(self.flag),
                func(self.quality),
                frame_idx=func(self.frame_idx) if self.frame_idx is not None else None
            )


class BatchPointsStorage:
    class Scatter(IntFlag):
        POSITION=auto()
        COLOR=auto()
        COV_TW=auto()
        OBSERVED_COUNT=auto()
        OBSERVED_BY=auto()
        ALL= POSITION | COLOR | COV_TW | OBSERVED_BY | OBSERVED_COUNT
    
    def __init__(self) -> None:
        self.INIT_SIZE = 1024
        self.OBSERVE_MAX_CNT = 5
        assert self.OBSERVE_MAX_CNT < 255   # We use uint8 to store #observe for a point.
        
        # torch.long ID for points
        self.point_idx = AutoScalingTensor((self.INIT_SIZE,), grow_on=0, dtype=torch.long)
        
        # Stores coordinate of each point in R3 under world coordinate
        self.position = AutoScalingTensor((self.INIT_SIZE, 3), grow_on=0, dtype=torch.float)
        
        # Stores color of each point in R3
        self.color = AutoScalingTensor((self.INIT_SIZE, 3), grow_on=0, dtype=torch.uint8)
        
        # Stores covariance under world coordinate
        self.cov_Tw = AutoScalingTensor((self.INIT_SIZE, 3, 3), grow_on=0, dtype=torch.double)
        
        # Stores heterogeneous point->observation lists
        self.observed_count = AutoScalingTensor((self.INIT_SIZE, ), grow_on=0, dtype=torch.int)
        self.observed_by = AutoScalingTensor((self.INIT_SIZE, self.OBSERVE_MAX_CNT), grow_on=0, dtype=torch.long, init_val=-1)
    
    def __len__(self) -> int:
        return self.point_idx.current_size
    
    def __getitem__(self, slice) -> BatchPoints:
        return BatchPoints(
            position = self.position.__getitem__(slice),
            cov_Tw = self.cov_Tw.__getitem__(slice),
            color = self.color.__getitem__(slice),
            point_idx = self.point_idx.__getitem__(slice),
            observed_count = self.observed_count.__getitem__(slice),
            observed_by = self.observed_by.__getitem__(slice)
        ).inflate()
    
    def __setitem__(self, slice, val: BatchPoints) -> None:
        val = val.to(MAP_DEVICE)
        if val.point_idx is not None:
            self.point_idx.__setitem__(slice, val.point_idx)
        if val.observed_by is not None:
            self.observed_by.__setitem__(slice, val.observed_by)
        if val.observed_count is not None:
            self.observed_count.__setitem__(slice, val.observed_count)
        self.position.__setitem__(slice, val.position)
        self.cov_Tw.__setitem__(slice, val.cov_Tw)
        self.color.__setitem__(slice, val.color)
    
    def push(self, data: BatchPoints):
        r"""Given a batch of points, add it into internal storage and return the tensor representing 
            index of points.
        """
        data = data.to(MAP_DEVICE)
        begin_size = self.position.current_size
        
        self.position.push(data.position)
        self.cov_Tw.push(data.cov_Tw)
        self.color.push(data.color)
        self.observed_count.push(torch.zeros((len(data),), dtype=torch.long))
        self.observed_by.push(torch.full((len(data), self.OBSERVE_MAX_CNT), -1, dtype=torch.long))
        
        end_size = self.position.current_size
        
        indices = torch.arange(begin_size, end_size, 1, dtype=torch.long)
        self.point_idx.push(indices)
        return indices

    def update(self, data: BatchPoints, flag: Scatter):
        data = data.to(MAP_DEVICE)
        assert data.point_idx is not None, "point_idx is required to scatter_ data into storage. Consider use __setitem__ instead."
        
        if self.point_idx.untyped_storage().data_ptr() == data.point_idx.untyped_storage().data_ptr():
            return  # No need to do other operation, data is just a view of underlying storage.
        
        if self.Scatter.POSITION in flag:
            self.position[data.point_idx] = data.position
        if self.Scatter.COLOR in flag:
            self.color[data.point_idx] = data.color
        if self.Scatter.COV_TW in flag:
            self.cov_Tw[data.point_idx] = data.cov_Tw
        if self.Scatter.OBSERVED_BY in flag and data.observed_by is not None:
            self.observed_by[data.point_idx] = data.observed_by
        if self.Scatter.OBSERVED_COUNT in flag and data.observed_count is not None:
            self.observed_count[data.point_idx] = data.observed_count


class BatchObservationStorage:
    def __init__(self) -> None:
        self.INIT_SIZE = 1024
        
        # Stores index of 3D keypoint observed by the observation
        self.point_idx = AutoScalingTensor((self.INIT_SIZE,), grow_on=0, dtype=torch.long)
        
        # Stores index of Frame observed by the observation
        self.frame_idx = AutoScalingTensor((self.INIT_SIZE,), grow_on=0, dtype=torch.long)
        
        # Stores uv of observation
        self.pixel_uv = AutoScalingTensor((self.INIT_SIZE, 2), grow_on=0, dtype=torch.float)
        
        # Stores depth of observation
        self.pixel_d = AutoScalingTensor((self.INIT_SIZE,), grow_on=0, dtype=torch.float)
        
        # Stores covariance of observation (under camera coordinate)
        self.cov_Tc = AutoScalingTensor((self.INIT_SIZE, 3, 3), grow_on=0, dtype=torch.double)
        
        # Stores covariance of raw inputs
        self.cov_pixel_uv = AutoScalingTensor((self.INIT_SIZE, 2), grow_on=0, dtype=torch.float)
        self.cov_pixel_d = AutoScalingTensor((self.INIT_SIZE, ), grow_on=0, dtype=torch.float)

    def __len__(self) -> int:
        return self.frame_idx.current_size

    def __getitem__(self, slice) -> BatchObservation:
        return BatchObservation(
            self.point_idx.__getitem__(slice),
            self.frame_idx.__getitem__(slice),
            self.pixel_uv.__getitem__(slice),
            self.pixel_d.__getitem__(slice),
            self.cov_Tc.__getitem__(slice),
            self.cov_pixel_uv.__getitem__(slice),
            self.cov_pixel_d.__getitem__(slice),
        ).inflate()
    
    def __setitem__(self, slice, val: BatchObservation):
        val = val.to(MAP_DEVICE)
        
        self.point_idx.__setitem__(slice, val.point_idx)
        self.frame_idx.__setitem__(slice, val.frame_idx)
        self.pixel_uv.__setitem__(slice, val.pixel_uv)
        self.pixel_d.__setitem__(slice, val.pixel_d)
        self.cov_Tc.__setitem__(slice, val.cov_Tc)
        self.cov_pixel_uv.__setitem__(slice, val.cov_pixel_uv)
        self.cov_pixel_d.__setitem__(slice, val.cov_pixel_d)
    
    def push(self, data: BatchObservation):
        r"""Given a batch of observations, add it to internal storage and return tensor representing
            index of observations.
        """
        data = data.to(MAP_DEVICE)
        begin_size = self.point_idx.current_size
        
        self.point_idx.push(data.point_idx)
        self.frame_idx.push(data.frame_idx)
        self.pixel_uv.push(data.pixel_uv)
        self.pixel_d.push(data.pixel_d)
        self.cov_Tc.push(data.cov_Tc)
        self.cov_pixel_uv.push(data.cov_pixel_uv)
        self.cov_pixel_d.push(data.cov_pixel_d)
        
        end_size = self.point_idx.current_size
        return torch.arange(begin_size, end_size, 1, dtype=torch.long)


class BatchFrameStorage:
    class Scatter(IntFlag):
        K=auto()
        POSE=auto()
        OBS_RANGE=auto()
        FLAG=auto()
        QUALITY=auto()
        ALL=K | POSE | OBS_RANGE | FLAG | QUALITY
    
    def __init__(self) -> None:
        self.INIT_SIZE = 1024
        
        # Stores frame index for each frame
        self.frame_idx = AutoScalingTensor((self.INIT_SIZE,), grow_on=0, dtype=torch.long)
        
        # Stores intrinsic matrix
        self.K = AutoScalingTensor((self.INIT_SIZE, 3, 3), grow_on=0, dtype=torch.float)
        
        # Stores pose
        self.pose = AutoScalingTensor((self.INIT_SIZE, 7), grow_on=0, dtype=torch.float)
        
        # Stores range of observation made in this frame
        self.obs_range = AutoScalingTensor((self.INIT_SIZE, 2), grow_on=0, dtype=torch.long)
        
        # Stores status flag as int8
        self.flag = AutoScalingTensor((self.INIT_SIZE, ), grow_on=0, dtype=torch.uint8)
        
        # Stores a quality measurement for each frame, -1 if not available.
        self.quality = AutoScalingTensor((self.INIT_SIZE, ), grow_on=0, dtype=torch.float)

    def __len__(self) -> int:
        return self.frame_idx.current_size

    def __getitem__(self, slice) -> BatchFrame:
        return BatchFrame(
            self.K.__getitem__(slice),
            pp.SE3(self.pose.__getitem__(slice)),
            self.obs_range.__getitem__(slice),
            self.flag.__getitem__(slice),
            self.quality.__getitem__(slice),
            frame_idx=self.frame_idx.__getitem__(slice)
        ).inflate()
    
    def __setitem__(self, slice, val: BatchFrame):
        val = val.to(MAP_DEVICE)
        if val.frame_idx is not None:
            self.frame_idx.__setitem__(slice, val.frame_idx)
        if val.obs_range is not None:
            self.obs_range.__setitem__(slice, val.obs_range)
        self.K.__setitem__(slice, val.K)
        self.pose.__setitem__(slice, val.pose)
        self.flag.__setitem__(slice, val.flag)
        self.quality.__setitem__(slice, val.quality)

    def push(self, data: BatchFrame) -> torch.Tensor:
        data = data.to(MAP_DEVICE)
        begin_size = self.K.current_size
        
        self.K.push(data.K)
        self.pose.push(data.pose)
        self.obs_range.push(data.obs_range)
        self.flag.push(data.flag)
        self.quality.push(data.quality)
        
        end_size = self.K.current_size
        indices = torch.arange(begin_size, end_size, 1, dtype=torch.long)
        self.frame_idx.push(indices)
        return indices

    def update(self, data: BatchFrame, flag: Scatter):
        data = data.to(MAP_DEVICE)
        assert data.frame_idx is not None, "Need frame_idx to scatter data back. Consider use __setitem__ instead."
        if self.frame_idx.untyped_storage().data_ptr() == data.frame_idx.untyped_storage().data_ptr():
            return  # No need to do other operation, data is just a view of underlying storage.
        
        if self.Scatter.K in flag:
            self.K[data.frame_idx] = data.K
        if self.Scatter.POSE in flag:
            self.pose[data.frame_idx] = data.pose
        if self.Scatter.OBS_RANGE in flag:
            self.obs_range[data.frame_idx] = data.obs_range
        if self.Scatter.FLAG in flag:
            self.flag[data.frame_idx] = data.flag
        if self.Scatter.QUALITY in flag:
            self.quality[data.frame_idx] = data.quality
