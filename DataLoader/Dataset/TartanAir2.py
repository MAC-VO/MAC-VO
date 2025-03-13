import torch
import pypose as pp
import numpy  as np
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from ..SequenceBase import SequenceBase
from ..Interface import StereoData, StereoFrame, StereoInertialFrame


from .TartanAir import TartanAirMonocularDataset, TartanAirIMUSimulator, loadTartanAirGT, IMUNoiseGenerator



class TartanAirV2_Sequence(SequenceBase[StereoInertialFrame]):
    # The default_imu_simulate_spec is from
    # Epson m365 parameters
    # This is for simplicity of loading. You can override this by changing the config.
    default_imu_simulate_spec = SimpleNamespace(**{
        # Accelerometer Noise
        "acc_bias": (0.02, -0.01, 0.05),
        "acc_init_bias_noise": (0.01, 0.01, 0.01),
        "acc_bias_instability": (1.47e-4, 1.47e-4, 1.47e-4),
        "acc_random_walk": (1.96e-7, 1.96e-7, 1.96e-7),

        # Gyroscope Noise
        "gyro_bias": (5.e-3, -2.e-3, 5.e-3),
        "gyro_init_bias_noise": (0.01, 0.01, 0.01),
        "gyro_bias_instability": (5.8e-6, 5.8e-6, 5.8e-6),
        "gyro_random_walk": (3.8e-7, 3.8e-7, 3.8e-7),
    })
    
    @classmethod
    def name(cls) -> str: return "TartanAirv2"
    
    def __init__(self, config: SimpleNamespace | dict[str, Any]):
        cfg = self.config_dict2ns(config)
        
        self.stereo_sequence = TartanAirV2_StereoSequence(cfg)
        self.imu_sequence    = TartanAirIMUSimulator(cfg.imu_sim, Path(cfg.root, "pose_lcam_front.txt"), fps=cfg.imu_freq)
        super().__init__(len(self.stereo_sequence))

    def __getitem__(self, local_index: int) -> StereoInertialFrame:
        index   = self.get_index(local_index)
        stereo_frame = self.stereo_sequence[index]
        if index == 0:
            imu_data, attitude_data     = self.imu_sequence[0]
        else:
            imu_data, attitude_data     = self.imu_sequence.frameRangeQuery(index - 1, index)
        return StereoInertialFrame(
            idx=[local_index],
            stereo=stereo_frame.stereo,
            imu=imu_data, gt_attitude=attitude_data,
            gt_pose=stereo_frame.gt_pose,
            time_ns=stereo_frame.time_ns
        )
    
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        assert config is not None
        cls._enforce_config_spec(config, {
            "root"      : lambda s: isinstance(s, str),
            "compressed": lambda b: isinstance(b, bool),
            "imu_freq"  : lambda f: isinstance(f, int),
            "gtFlow"    : lambda b: isinstance(b, bool),
            "gtDepth"   : lambda b: isinstance(b, bool),
            "gtPose"    : lambda b: isinstance(b, bool),
        }, allow_excessive_cfg=True)
        IMUNoiseGenerator.is_valid_config(config.imu_sim)


class TartanAirV2_StereoSequence(SequenceBase[StereoFrame]):
    @classmethod
    def name(cls) -> str: return "TartanAirv2_NoIMU"
    
    def __init__(self, config: SimpleNamespace | dict[str, Any]):
        cfg = self.config_dict2ns(config)
        
        # Metadata
        self.lcam_T_BS = pp.identity_SE3(1)
        self.lcam_K    = torch.tensor([[320.0, 0.0, 320.0], [0.0, 320.0, 320.0], [0.0, 0.0, 1.0]]).unsqueeze(0)
        self.baseline  = 0.25
        self.width     = 640
        self.height    = 640
        # End
        
        # Stereo Loader
        self.lcam_loader = TartanAirMonocularDataset(Path(cfg.root, "image_lcam_front"))
        self.rcam_loader = TartanAirMonocularDataset(Path(cfg.root, "image_rcam_front"))
        
        cam_time_file_path = Path(cfg.root, "imu", "cam_time.txt")
        if cam_time_file_path.exists():
            self.lcam_time   = (np.loadtxt(str(cam_time_file_path), dtype=np.float64) * 1_000_000_000).astype(np.int64)
        else:
            # Fake data, assume 10Hz image
            self.lcam_time = (np.arange(len(self.lcam_loader), dtype=np.float64) * 0.1 * 1_000_000_000).astype(np.int64)

        if cfg.gtDepth:
            from .TartanAir import TartanAirGTDepthDataset
            self.depth_loader = TartanAirGTDepthDataset(Path(cfg.root, "depth_lcam_front"), cfg.compressed)
        else: self.depth_loader = None

        if cfg.gtFlow:
            from .TartanAir import TartanAirGTFlowDataset
            self.flow_loader = TartanAirGTFlowDataset(Path(cfg.root, "flow_lcam_front"), cfg.compressed)
            length = len(self.flow_loader)
        else:
            self.flow_loader = None
            length = len(self.lcam_loader)

        if cfg.gtPose:
            self.gt_poses = loadTartanAirGT(Path(cfg.root, "pose_lcam_front.txt"))
        else: self.gt_poses = None
        
        super().__init__(length)
    
    def __getitem__(self, local_index: int) -> StereoFrame:
        index   = self.get_index(local_index)
        gt_flow = self.flow_loader[index] if self.flow_loader else None
        return StereoFrame(
            idx=[local_index],
            stereo=StereoData(
                T_BS      = self.lcam_T_BS,
                K         = self.lcam_K,
                baseline  = torch.tensor([self.baseline]),
                time_ns   = [self.lcam_time[index].item()],  # Fake data, assume 10Hz image
                height    = 640,
                width     = 640,
                imageL    = self.lcam_loader[index],
                imageR    = self.rcam_loader[index],
                
                # Ground truth and labels
                gt_depth  = self.depth_loader[index] if self.depth_loader else None,
                gt_flow   = gt_flow[0] if gt_flow else None,
                flow_mask = gt_flow[1] if gt_flow else None,
            ),
            time_ns   = [self.lcam_time[index].item()],  # Fake data, assume 10Hz image
            gt_pose   = cast(pp.LieTensor, self.gt_poses[index].unsqueeze(0)) if (self.gt_poses is not None) else None,
        )
    
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        assert config is not None
        cls._enforce_config_spec(config, {
            "root"      : lambda s: isinstance(s, str),
            "compressed": lambda b: isinstance(b, bool),
            "gtFlow"    : lambda b: isinstance(b, bool),
            "gtDepth"   : lambda b: isinstance(b, bool),
            "gtPose"    : lambda b: isinstance(b, bool),
        }, allow_excessive_cfg=True)
