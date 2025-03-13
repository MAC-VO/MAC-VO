import torch
import pypose as pp
import pypose.module as pm

from abc import ABC, abstractmethod
from pathlib import Path

from typing import Generic, cast
from types import SimpleNamespace
from Utility.Extensions import ConfigTestableSubclass,TensorQueue
from Utility.PrettyPrint import Logger
from DataLoader import StereoFrame, StereoInertialFrame, T_Data
from Utility.Timer import Timer


class IMotionModel(ABC, Generic[T_Data], ConfigTestableSubclass):
    """
    A motion model class receives informations (e.g. frames, estimated flow and depth) and produce an
    initial guess to the pose of incoming frame **under global coordinate**.
    """
    def __init__(self, config: SimpleNamespace):
        self.config : SimpleNamespace = config
    
    @abstractmethod
    def predict(self, frame: T_Data, flow: torch.Tensor | None, depth: torch.Tensor | None) -> pp.LieTensor:
        """
        Estimate the pose of next frame given current frame, estimated depth and flow.
        
        NOTE: returned pose should be under global coordinate!
        
        Returns
        *   pose  - 7, shaped pypose.LieTensor (SE3 ltype) under world coordinate
                  predicted pose of next frame.
        """
        ...
    
    @abstractmethod
    def update(self, pose: pp.LieTensor) -> None:
        """
        Receive a feedback (optimized pose) and may (or may not) use this method to refine next prediction.
        """
        ...


class GTMotionwithNoise(IMotionModel[StereoFrame]):
    """
    Apply GT motion with noise (can be disabled by setting `noise_std` to 0.0 in config) on previous optimized pose to predict next pose.
    """
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.prev_pose: pp.LieTensor | None = None
        self.prev_gt_pose: pp.LieTensor | None = None
    
    def init_context(self) -> None: return None
    
    def _stableNoiseModel(self) -> pp.LieTensor:
        if self.config.noise_std == 0.0:
            return pp.identity_SE3()
        noise: pp.LieTensor = pp.randn_SE3(sigma=self.noise_std)    #type:ignore
        return noise

    def predict(self, frame: StereoFrame, flow: torch.Tensor | None, depth: torch.Tensor | None) -> pp.LieTensor:
        assert frame.gt_pose is not None
        frame_gtpose = cast(pp.LieTensor, frame.gt_pose.squeeze(0))

        if self.prev_pose is None or self.prev_gt_pose is None:
            self.prev_pose    = pp.identity_SE3()
            self.prev_gt_pose = frame_gtpose
            return pp.identity_SE3()

        gtMotion = self.prev_gt_pose.Inv() @ frame_gtpose
        gtMotion_w_noise = gtMotion @ self._stableNoiseModel()
        predict = self.prev_pose @ gtMotion_w_noise
        
        self.prev_pose = predict
        self.prev_gt_pose = frame_gtpose
        
        return predict

    def update(self, pose: pp.LieTensor) -> None:
        self.prev_pose = pose

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "noise_std": lambda noise: isinstance(noise, (int, float)) and noise >= 0.0
        })


class TartanMotionNet(IMotionModel[StereoFrame]):
    """
    Apply motion estimated by MotionNet adapted from TartanVO on previously optimized pose to predict next pose.
    """
    def __init__(self, config: SimpleNamespace):
        from .Network.TartanVOStereo import TartanStereoVOMotion
        
        super().__init__(config)
        self.model = TartanStereoVOMotion(self.config.weight, True, self.config.device)
        self.prev_pose = None

    @Timer.cpu_timeit("MotionModel")
    @Timer.gpu_timeit("MotionModel")
    @torch.inference_mode()
    def predict(self, frame: StereoFrame, flow: torch.Tensor | None, depth: torch.Tensor | None) -> pp.LieTensor:
        if self.prev_pose is None:
            self.prev_pose = pp.identity_SE3(device=self.config.device)
            return pp.identity_SE3(device=self.config.device)

        assert flow is not None and depth is not None, "Motion model requires flow and depth to predict motion"
        motion_se3: torch.Tensor = self.model.inference(frame, flow, depth)
        new_pose = self.prev_pose @ pp.se3(motion_se3).Exp()
        self.prev_pose = new_pose
        return new_pose
    
    def update(self, pose: pp.LieTensor) -> None:
        self.prev_pose = pose.to(self.prev_pose.device)    #type: ignore

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "weight": lambda f: isinstance(f, str),
            "device": lambda dev: isinstance(dev, str) and (("cuda" in dev) or (dev == "cpu"))
        })


class StaticMotionModel(IMotionModel[StereoFrame]):
    """
    Assumes the camera is static and simply record and returns the pose of previous frame.
    """
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.prev_pose: pp.LieTensor | None = None
    
    def predict(self, frame: StereoFrame, flow: torch.Tensor | None, depth: torch.Tensor | None) -> pp.LieTensor:
        if self.prev_pose is None:
            self.prev_pose = pp.identity_SE3()
            return pp.identity_SE3()
        return self.prev_pose

    def update(self, pose: pp.LieTensor) -> None:
        assert self.prev_pose is not None
        self.prev_pose = pose.to(self.prev_pose.device) # type: ignore
    
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None: return


class ReadPoseFile(IMotionModel[StereoFrame]):
    """
    Use an external file of Nx7 SE3 poses as motion model output poses.
    
    NOTE: Specifically, the module will *not* output these poses directly but calculate the motion
    and apply motion on modified poses (potentially by optimizer) iteratively.
    """
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.prev_pose: None | pp.LieTensor = None
        self.prev_gt_pose: None | pp.LieTensor = None
        self.poses: pp.LieTensor = self.load_poses()
    
    def load_poses(self) -> pp.LieTensor:
        pose_file = Path(self.config.pose_file)
        if not pose_file.exists():
            Logger.write("error", f"Cannot read pose file at {pose_file} - File Not Exist!")
            raise FileNotFoundError(f"Cannot read pose file at {pose_file} - File Not Exist!")
        
        poses: pp.LieTensor
        match pose_file.suffix:
            case ".npy":
                import numpy as np
                poses_data: torch.Tensor = torch.from_numpy(np.load(str(pose_file)))
            case ".pt" | ".pth":
                poses_data: torch.Tensor = torch.load(str(pose_file), weights_only=False)
            case ".txt":
                import numpy as np
                poses_data: torch.Tensor = torch.from_numpy(np.loadtxt(str(pose_file)))
            case suffix:
                raise NameError(f"Cannot handle a file with suffix '{suffix}'. Consider change it to .npy/.pt/.pth/.txt or write a custom loader.")
        assert poses_data.ndim == 2 and poses_data.shape[1] == 7
        poses = pp.SE3(poses_data)
        return poses
        
    def predict(self, frame: StereoFrame, flow: torch.Tensor | None, depth: torch.Tensor | None) -> pp.LieTensor:
        if self.prev_pose is None or self.prev_gt_pose is None:
            self.prev_pose = pp.identity_SE3()
            self.prev_gt_pose = pp.SE3(self.poses[frame.frame_idx])
            return pp.identity_SE3()

        motion = self.prev_gt_pose.Inv() @ pp.SE3(self.poses[frame.frame_idx])
        predict = self.prev_pose @ motion
        
        self.prev_pose = predict
        self.prev_gt_pose = pp.SE3(self.poses[frame.frame_idx])
        return predict

    def update(self, pose: pp.LieTensor) -> None:
        self.prev_pose = pose

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "pose_file": lambda s: isinstance(s, str)
        })
