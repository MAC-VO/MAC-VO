import torch
import pypose as pp
import numpy as np

from pathlib import Path

from .Storage import (
    BatchFrameStorage, BatchObservationStorage, BatchPointsStorage,
    BatchFrame, BatchObservation, BatchPoints
)


class TensorMap:
    def __init__(self) -> None:
        self.frames = BatchFrameStorage()
        self.observations = BatchObservationStorage()
        self.points = BatchPointsStorage()
    
    def __repr__(self) -> str:
        return f"TensorMap(#frame={len(self.frames)}, #obs={len(self.observations)}, #pts={len(self.points)})"

    def add_frame(self, K: torch.Tensor, pose: pp.LieTensor, num_obs: int, 
                  quality: torch.Tensor | float | None, flag: int = BatchFrame.FLAG_EMPTY) -> torch.Tensor:
        r"""Given the intrinsic matrix and pose of a frame, add it to internal storage and return tensor
            representing index of frame.
        """
        obs_idx_start = len(self.observations)
        
        if quality is None:
            quality = torch.tensor([-1.], dtype=torch.float)
        elif isinstance(quality, torch.Tensor):
            if quality.ndim == 0:
                quality = quality.unsqueeze(0)
        else:
            quality = torch.tensor([quality])
        
        return self.frames.push(BatchFrame(
            K.unsqueeze(0),
            pose.unsqueeze(0) if pose.ndim == 1 else pose,  #type: ignore
            torch.tensor([[obs_idx_start, obs_idx_start + num_obs]]),
            torch.tensor([flag]),
            quality
        ))
    
    def add_observation(self, observation: BatchObservation) -> torch.Tensor:
        # Sanity check - TensorMap is currently restricted to maximize map overhead
        _, point_idx_count = observation.point_idx.unique(sorted=False, return_counts=True)
        frame_idx = observation.frame_idx.unique(sorted=False)
        assert (point_idx_count == 1).all(), "add_observation only handles one-observation per point."
        assert torch.numel(frame_idx) <= 1, "add_observation only handles one-frame as source of all observations."
        if torch.numel(frame_idx) == 0: return torch.tensor([], dtype=torch.long) # Do nothing
        
        # FIXME: this conflict with current design of KeyframeSelector
        # Currently need to comment out this line to make non-trival KeyframeSelector work.
        # assert frame_idx.item() == len(self.frames) - 1, f"Random add of observation is not supported yet. (try to add obs to #{frame_idx} but currently you can only add to {len(self.frames) - 1})"
        # end
        
        result = self.observations.push(observation)
        self.points.observed_by[observation.point_idx, self.points.observed_count[observation.point_idx]] = result
        self.points.observed_count[observation.point_idx] += 1
        self.frames.obs_range[frame_idx, 1] += len(observation)
        return result

    def get_frame_observes(self, frames: BatchFrame) -> BatchObservation:
        assert frames.frame_idx is not None
        lengths = frames.obs_range[:, 1] - frames.obs_range[:, 0]
        offsets = frames.obs_range[:, 0]

        # Use torch.repeat_interleave to expand indices according to lengths
        if torch.any(lengths > 0):
            expanded_offsets = torch.repeat_interleave(offsets, lengths)
            range_indices = torch.arange(lengths.sum().item())
            range_starts = torch.repeat_interleave(lengths.cumsum(0) - lengths, lengths)
            selector = expanded_offsets + (range_indices - range_starts)
        else:
            selector = torch.tensor([], dtype=torch.long)
        return self.observations[selector]

    def get_frame_points(self, frames: BatchFrame) -> BatchPoints:
        observations = self.get_frame_observes(frames)
        points = self.points[observations.point_idx]
        return points

    def get_obs_points(self, obs: BatchObservation) -> BatchPoints:
        return self.points[obs.point_idx]
    
    def get_obs_frames(self, obs: BatchObservation) -> BatchFrame:
        return self.frames[obs.frame_idx]

    def get_point_observes(self, points: BatchPoints) -> BatchObservation:
        assert points.observed_by is not None
        
        observation_idx = points.observed_by.flatten()
        return self.observations[observation_idx[observation_idx >= 0]]

    def save_poses(self, file_path: Path | str):
        poses = self.frames.pose.tensor.numpy()
        np.save(file_path, poses)

    def save_flags(self, file_path: Path | str):
        torch.save(self.frames.flag.tensor, file_path)

    def save_map(self, file_path: Path | str):
        torch.save(self, file_path)
    
    @classmethod
    def load_map(cls, file_path: Path | str) -> "TensorMap":
        tmap = torch.load(file_path)
        return tmap
