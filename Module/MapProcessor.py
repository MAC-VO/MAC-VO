import torch
import pypose as pp

from abc import ABC, abstractmethod
from types import SimpleNamespace

from Module.Map import TensorMap, BatchFrame
from Utility.Extensions import ConfigTestableSubclass
from Utility.Math import interpolate_pose, NormalizeQuat


class IMapProcessor(ABC, ConfigTestableSubclass):
    """
    A class to post-process the TensorMap to improve mapping and estimated trajectory.
    """
    def __init__(self, config: SimpleNamespace | None):
        self.config = config
        
    @abstractmethod
    def elaborate_map(self, gmap: TensorMap) -> tuple[TensorMap, torch.Tensor]: ...


class Naive(IMapProcessor):
    """
    For all frames that marked as `FLAG_NEED_INTERP`, linearly interpolate the poses (under se3 space)
    using two adjacent frames (that are not marked as NEED_INTERP).
    """
    @torch.inference_mode()
    def elaborate_map(self, gmap: TensorMap) -> tuple[TensorMap, torch.Tensor]:
        # Interpolate for VOLostTrack poses.
        poses = pp.SE3(gmap.frames.pose.tensor)
        flags = gmap.frames.flag.tensor
        
        bad_mask = (flags & BatchFrame.FLAG_NEED_INTERP).bool()
        bad_mask[-5:] = False
        bad_idx = torch.nonzero(bad_mask).flatten()
        
        interp_poses = interpolate_pose(poses[~bad_mask], torch.nonzero(~bad_mask).flatten(), bad_idx)   #type: ignore
        
        gmap.frames.pose[bad_mask] = interp_poses
        return gmap, bad_idx

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {})


class DisplacementInterpolate(IMapProcessor):
    """
    For all frames that marked as `FLAG_NEED_INTERP`, linearly interpolate the poses (under se3 space)
    using two adjacent frames (that are not marked as NEED_INTERP).
    
    Then for all poses that show excessive translation (> mean + 4std), interpolate them as they might be 
    outlier / lost track cases.
    """
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.fill_interpolate_poses = Naive(None)
    
    @torch.inference_mode()
    def elaborate_map(self, gmap: TensorMap) -> tuple[TensorMap, torch.Tensor]:
        gmap, _ = self.fill_interpolate_poses.elaborate_map(gmap)
        
        poses = pp.SE3(gmap.frames.pose.tensor).double()
        motions = poses[:-1].Inv() @ poses[1:]  #type: ignore
        
        translation_norm = motions.translation().norm(dim=1)
        bad_mask = translation_norm > (translation_norm.mean() + translation_norm.std() * 4)
        bad_mask[:5] = False
        bad_mask[-5:] = False
        
        interp_idx = torch.nonzero(bad_mask).flatten()
        interp_motions = interpolate_pose(motions[~bad_mask], torch.nonzero(~bad_mask).flatten(), interp_idx)
        motions[bad_mask] = interp_motions
        
        # NOTE: pp.cumprod is not numerically stable, so we use this cumops with NormalizeQuat for
        # more stable performance.
        # https://github.com/pypose/pypose/issues/346
        # NOTE: the behavior of Pypose v0.6.7 and v0.6.8 are different for pp.cumops
        interp_poses = pp.cumops(motions, dim=0, ops=lambda a, b: NormalizeQuat(a) @ NormalizeQuat(b))
        
        gmap.frames.pose[1:] = interp_poses.float()
        return gmap, interp_idx

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {})
