import torch
import pypose as pp

from abc import ABC, abstractmethod
from types import SimpleNamespace

from Module.Map import FrameStore
from Utility.Extensions import ConfigTestableSubclass
from Utility.Math import interpolate_pose, NormalizeQuat


class IMapProcessor(ABC, ConfigTestableSubclass):
    """
    A class to post-process the TensorMap to improve mapping and estimated trajectory.
    """
    def __init__(self, config: SimpleNamespace | None):
        self.config = config
    
    @abstractmethod
    def elaborate_map(self, frames: FrameStore) -> tuple[FrameStore, torch.Tensor]:
        """
        Given a sequence of frames, elaborate the trajectory (frame poses) and handle the 
        'need_interp' (i.e. lost track / skipped) frames.
        """
        ...


class PoseInterpolate(IMapProcessor):
    """
    For all frames that marked as `NEED_INTERP`, linearly interpolate the poses (under se3 space)
    using two adjacent frames (that are not marked as NEED_INTERP).
    """
    @torch.inference_mode()
    def elaborate_map(self, frames: FrameStore) -> tuple[FrameStore, torch.Tensor]:
        # Interpolate for VOLostTrack poses.
        poses    = pp.SE3(frames.data["pose"].tensor)
        bad_mask = frames.data["need_interp"].tensor
        bad_mask[:5]  = False
        bad_mask[-5:] = False
        bad_idx = torch.nonzero(bad_mask).flatten()
        
        interp_poses, _ = interpolate_pose(poses[~bad_mask], torch.nonzero(~bad_mask).flatten(), bad_idx)   #type: ignore
        
        frames.data["pose"][bad_mask] = interp_poses.tensor()
        return frames, bad_idx

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {})


class MotionInterpolate(IMapProcessor):
    """
    For all frames that marked as `NEED_INTERP`, linearly interpolate the poses (under se3 space)
    using two adjacent frames (that are not marked as NEED_INTERP).
    """
    @torch.inference_mode()
    def elaborate_map(self, frames: FrameStore) -> tuple[FrameStore, torch.Tensor]:
        # Interpolate for VOLostTrack poses.
        poses = pp.SE3(frames.data["pose"].tensor).double()
        bad_mask = frames.data["need_interp"][1:].bool()
        motions = poses[:-1].Inv() @ poses[1:]  #type: ignore
        bad_mask[:2] = False
        bad_mask[-2:] = False
        
        interp_idx = torch.nonzero(bad_mask).flatten()
        interp_motions, _ = interpolate_pose(motions[~bad_mask], torch.nonzero(~bad_mask).flatten(), interp_idx)
        motions[bad_mask] = interp_motions
        
        # NOTE: pp.cumprod is not numerically stable, so we use this cumops with NormalizeQuat for
        # more stable performance.
        # https://github.com/pypose/pypose/issues/346
        # NOTE: the behavior of Pypose v0.6.7 and v0.6.8 are different for pp.cumops
        interp_poses = pp.cumops(motions, dim=0, ops=lambda a, b: NormalizeQuat(a) @ NormalizeQuat(b))
        frames.data["pose"][1:] = (pp.SE3(frames.data["pose"][0:1]).double() @ interp_poses).float()
        return frames, interp_idx

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {})