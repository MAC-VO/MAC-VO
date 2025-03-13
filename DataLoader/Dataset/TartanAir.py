import cv2
import torch
import numpy as np
import pypose as pp
from types import SimpleNamespace
from typing import Any, cast
from pathlib import Path
from scipy import interpolate
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation, RotationSpline

from Utility.Extensions import ConfigTestable
from Utility.PrettyPrint import Logger

from ..SequenceBase import SequenceBase
from ..Interface    import StereoInertialFrame, StereoFrame, StereoData, IMUData, AttitudeData


class TartanAir_Sequence(SequenceBase[StereoInertialFrame]):
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
    def name(cls) -> str: return "TartanAir"
    
    def __init__(self, config: SimpleNamespace | dict[str, Any]):
        cfg = self.config_dict2ns(config)
        self.stereo_sequence = TartanAir_StereoSequence(cfg)
        
        if cfg.imu_freq is None:
            self.imu_sequence    = TartanAirIMULoader(Path(cfg.root, "imu"))
        else:
            self.imu_sequence    = TartanAirIMUSimulator(cfg.imu_sim, Path(cfg.root, "pose_left.txt"), fps=cfg.imu_freq)
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
            time_ns=stereo_frame.time_ns,
            stereo=stereo_frame.stereo,
            imu=imu_data, gt_attitude=attitude_data,
            gt_pose=stereo_frame.gt_pose
        )

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        assert config is not None
        cls._enforce_config_spec(config, {
            "root"      : lambda s: isinstance(s, str),
            "compressed": lambda b: isinstance(b, bool),
            "imu_freq"  : lambda f: isinstance(f, (int, type(None))),
            "gtFlow"    : lambda b: isinstance(b, bool),
            "gtDepth"   : lambda b: isinstance(b, bool),
            "gtPose"    : lambda b: isinstance(b, bool),
        }, allow_excessive_cfg=True)
        IMUNoiseGenerator.is_valid_config(config.imu_sim)


class TartanAir_StereoSequence(SequenceBase[StereoFrame]):
    @classmethod
    def name(cls) -> str: return "TartanAir_NoIMU"
    
    def __init__(self, config: SimpleNamespace | dict[str, Any]):
        cfg = self.config_dict2ns(config)
        
        # Metadata
        self.lcam_T_BS = pp.identity_SE3(1)
        self.lcam_K    = torch.tensor([[320.0, 0.0, 320.0], [0.0, 320.0, 240.0], [0.0, 0.0, 1.0]]).unsqueeze(0)
        self.baseline  = 0.25
        self.width     = 640
        self.height    = 480
        # End
        
        # Stereo Loader
        self.lcam_loader = TartanAirMonocularDataset(Path(cfg.root, "image_left"))
        self.rcam_loader = TartanAirMonocularDataset(Path(cfg.root, "image_right"))

        cam_time_file_path = Path(cfg.root, "imu", "cam_time.npy")
        if cam_time_file_path.exists():
            self.lcam_time = (np.load(str(cam_time_file_path)) * 1_000_000_000).astype(np.int64)
        else:
            # Fake data, assume 10Hz image
            self.lcam_time = (np.arange(len(self.lcam_loader)) * 0.1 * 1_000_000_000).astype(np.int64)

        # Depth Loader
        if cfg.gtDepth:
            self.depth_loader = TartanAirGTDepthDataset(Path(cfg.root, "depth_left"), cfg.compressed)
        else:
            self.depth_loader = None
        
        # Flow loader
        if cfg.gtFlow :
            self.flow_loader  = TartanAirGTFlowDataset (Path(cfg.root, "flow"), cfg.compressed)
            length = len(self.flow_loader)
        else:
            self.flow_loader  = None
            length = len(self.lcam_loader)
        
        # Pose Loader
        if cfg.gtPose:
            # gt_poses is originally on left camera sensor frame, need to convert to body frame
            self.gt_poses = self.lcam_T_BS @ loadTartanAirGT(Path(cfg.root, "pose_left.txt")) @ self.lcam_T_BS.Inv()
        else:
            self.gt_poses = None

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
                height    = 480,
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
        cls._enforce_config_spec(config, {
            "root": lambda s: isinstance(s, str),
            "compressed": lambda b: isinstance(b, bool),
            "gtFlow"  : lambda b: isinstance(b, bool),
            "gtDepth" : lambda b: isinstance(b, bool),
            "gtPose"  : lambda b: isinstance(b, bool),
        })


# Specific dataset for a single sensor

class TartanAirMonocularDataset(Dataset):
    """
    Return images in the given directory ends with .png
    Return the image in shape (1, 3, H, W) with dtype=float32 
    and normalized (image in [0, 1])
    """
    def __init__(self, directory: Path) -> None:
        super().__init__()
        self.directory = directory
        assert self.directory.exists(), f"Monocular image directory {self.directory} does not exist"

        self.file_names = [f for f in self.directory.iterdir() if f.suffix == ".png"]
        self.file_names.sort()
        self.length = len(self.file_names)
        assert self.length > 0, f"No flow with '.png' suffix is found under {self.directory}"

    @staticmethod
    def load_png_format(path: Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def __len__(self):
        return self.length

    def __getitem__(self, index: int) -> torch.Tensor:
        # Output image tensor in shape of (1, C, H, W)
        result = self.load_png_format(self.file_names[index])
        result = torch.tensor(result, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        result /= 255.

        return result


class TartanAirGTDepthDataset(Dataset):
    """
    Returns pixel depth in shape of (1, H, W)
    """
    def __init__(self, directory: Path, compressed: bool) -> None:
        super().__init__()
        self.directory = directory
        self.compressed = compressed
        assert self.directory.exists(), f"Depth image directory {self.directory} does not exist"

        if self.compressed:
            self.file_names = [f for f in self.directory.iterdir() if f.suffix == ".png"]
        else:
            self.file_names = [f for f in self.directory.iterdir() if f.suffix == ".npy"]
        self.file_names.sort()
        self.length = len(self.file_names)
        assert len(self.file_names) > 0, f"No depth with '.npy' suffix is found in {self.directory}"
        
    def __len__(self): return self.length
    
    @staticmethod
    def load_npy_format(path: Path):
        return np.load(str(path))

    @staticmethod
    def load_png_format(path: Path):
        depth_rgba = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        assert depth_rgba is not None, f"Unable to load depth image at {path}"
        depth = TartanAirGTDepthDataset.depth_rgba_float32(depth_rgba)
        return depth
    
    @staticmethod
    def depth_rgba_float32(depth_rgba):
        """
        Referenced from TartanVO codebase
        """
        depth = depth_rgba.view("<f4")
        return np.squeeze(depth, axis=-1)

    def __getitem__(self, index) -> torch.Tensor:
        # Output (1, 1, H, W) tensor
        if self.compressed:
            depth_np = TartanAirGTDepthDataset.load_png_format(self.file_names[index])
        else:
            depth_np = TartanAirGTDepthDataset.load_npy_format(self.file_names[index])
        depth = torch.tensor(depth_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return depth


class TartanAirGTFlowDataset(Dataset):
    """
    Returns raw optical flow in shape of (1, 2, H, W)
    """
    def __init__(self, directory: Path, compressed: bool) -> None:
        super().__init__()
        self.directory = directory
        self.compressed = compressed

        assert self.directory.exists(), f"Optical flow directory {self.directory} does not exist"

        if self.compressed:
            self.file_names = [f for f in self.directory.iterdir() if f.name.endswith("_flow.png")]
        else:
            self.file_names = [f for f in self.directory.iterdir() if f.name.endswith("_flow.npy")]
        
        self.file_names.sort()
        self.length = len(self.file_names)
        assert self.length > 0, f"No flow with *.png is found under {self.directory}"
    
    @staticmethod
    def load_npy_format(path: Path) -> np.ndarray:
        return np.load(str(path))
    
    @staticmethod
    def load_png_format(path: Path) -> tuple[np.ndarray, np.ndarray]:
        flow16 = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        assert flow16 is not None, f"Error reading flow at {path}"
        flow32, mask = TartanAirGTFlowDataset.flow16to32(flow16)
        return flow32, mask
    
    @staticmethod
    def flow16to32(flow16):
        """
        Referenced from TartanVO project.
        flow_32b (float32) [-512.0, 511.984375]
        flow_16b (uint16) [0 - 65535]
        flow_32b = (flow16 -32768) / 64
        """
        flow32 = flow16[:,:,:2].astype(np.float32)
        flow32 = (flow32 - 32768) / 64.0

        mask8 = flow16[:,:,2].astype(np.uint8)
        return flow32, mask8
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.compressed:
            flow_np, mask = self.load_png_format(self.file_names[index])
        else:
            flow_npy = self.load_npy_format(self.file_names[index])
            flow_np = flow_npy[:,:,:2]
            mask = flow_npy[:,:,2:]
        flow = torch.tensor(flow_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return flow, mask


class TartanAirIMULoader(Dataset[tuple[IMUData, AttitudeData]]):
    def __init__(self, imuPath: Path) -> None:
        super().__init__()
        assert imuPath.exists(), f"IMU Data path ({imuPath}) does not exist"
        self.limu_T_BS = pp.identity_SE3(1)     # Body -> Sensor Transformation
        self.limu_g    = 9.81                   # Gravity

        self.imuPath = imuPath
        # IMU Data
        self.lin_acc = torch.tensor(
            np.load(str(Path(imuPath, "accel_left.npy"))), dtype=torch.float
        ).unsqueeze(0)  # (1, N, 3)
        self.rot_vel = torch.tensor(
            np.load(str(Path(imuPath, "gyro_left.npy"))), dtype=torch.float
        ).unsqueeze(0)  # (1, N, 3)
        self.timestamp = torch.tensor(
            np.load(str(Path(imuPath, "imu_time.npy"))) * 1_000_000_000, dtype=torch.long
        ).unsqueeze(0)  # (1, N,)

        # Ground Truth labels
        self.gt_vel = torch.tensor(
            np.load(str(Path(imuPath, "vel_left.npy"))), dtype=torch.float
        ).unsqueeze(0)  # (1, N, 3)
        self.gt_pos = torch.tensor(
            np.load(str(Path(imuPath, "xyz_left.npy"))), dtype=torch.float
        ).unsqueeze(0)  # (1, N, 3)
        angle_left = Rotation.from_euler(
            "XYZ", np.load(str(Path(imuPath, "angles_left.npy"))), degrees=False
        )
        self.gt_rot = pp.euler2SO3(torch.tensor(
            angle_left.as_euler("xyz", degrees=False), dtype=torch.float
        )).unsqueeze(0)  # (1, N, 4)

        # Camera alignment
        self.camtime = torch.tensor(np.load(str(Path(imuPath, "cam_time.npy"))) * 1_000_000_000, dtype=torch.long)  # (M,)
        self.cam2imuIdx = torch.ones_like(self.camtime, dtype=torch.long) * -1  # (M,)

        self.length = self.lin_acc.size(1) - 1

        self.alignWithCameraTime()

    def __len__(self) -> int:
        return self.length

    def alignWithCameraTime(self) -> None:
        """
        Align IMU Sequence with camera time

        The alignment relationship is stored in self.cam2imuIdx array as
        camidx -> imuidx
        """
        imu_idx, cam_idx, failFlag = 0, 0, False
        while imu_idx < self.length:
            frame_cam_time = self.camtime[cam_idx]
            imu_time = self.timestamp[0, imu_idx]
            next_imu_time = self.timestamp[0, imu_idx + 1]
            if imu_time <= frame_cam_time < next_imu_time:
                self.cam2imuIdx[cam_idx] = imu_idx
                cam_idx += 1
                imu_idx += 1
            elif imu_time > frame_cam_time:
                cam_idx += 1
                failFlag = True
            else:
                imu_idx += 1

        if failFlag or (cam_idx < self.camtime.size(0) - 1):
            Logger.write("warn", "Not all camera times are aligned with IMU Trajectory")
        if imu_idx < self.timestamp.size(0):
            Logger.write(
                "warn",
                f"{self.timestamp.size(0) - imu_idx} IMU samples remain unmatched",
            )

    def frameRangeQuery(self, start_frame, end_frame) -> tuple[IMUData, AttitudeData]:
        """Retrieve IMU data in range of [start_frame, end_frame)

        Args:
            start_frame (int): start image frame to get IMU sequence
            end_frame (int): end image frame to get (exclusive)

        Returns:
            IMUResult: A range of IMU Results
        """
        start_imu_idx = self.cam2imuIdx[start_frame]
        end_imu_idx = self.cam2imuIdx[end_frame]
        assert (
            start_imu_idx != -1 and end_imu_idx != -1
        ), "Requested frame is not aligned with IMU Sequence"
        return IMUData(
            T_BS=self.limu_T_BS,
            gravity=[self.limu_g],
            time_ns=self.timestamp[:, start_imu_idx:end_imu_idx],
            
            acc =self.lin_acc [:, start_imu_idx:end_imu_idx],
            gyro=self.rot_vel [:, start_imu_idx:end_imu_idx],
        ), AttitudeData(
            T_BS=self.limu_T_BS,
            gravity=[self.limu_g],
            time_ns=self.timestamp[:, start_imu_idx:end_imu_idx],
            
            gt_pos=self.gt_pos[:, start_imu_idx:end_imu_idx],
            gt_vel=self.gt_vel[:, start_imu_idx:end_imu_idx],
            gt_rot=cast(pp.LieTensor, self.gt_rot[:, start_imu_idx:end_imu_idx]),
            
            init_pos=self.gt_pos[:, start_imu_idx : start_imu_idx + 1],
            init_vel=self.gt_vel[:, start_imu_idx : start_imu_idx + 1],
            init_rot=cast(pp.LieTensor, self.gt_rot[:, start_imu_idx : start_imu_idx + 1]), 
        )

    def __getitem__(self, index) -> tuple[IMUData, AttitudeData]:
        """
        Args:
            index (int): the index of IMU record to retrieve

        Returns: {
            "acc": (1, 3) tensor for linear acceleration
            "ang_vel": (1, 3) tensor for angular velocity
            "time": (1,) tnesor for timestamp
        }
        """
        return IMUData(
            T_BS=self.limu_T_BS,
            gravity=[self.limu_g],
            time_ns=self.timestamp[:, index : index + 1],
            
            acc =self.lin_acc[:, index : index + 1],
            gyro=self.rot_vel[:, index : index + 1],
        ), AttitudeData(
            T_BS=self.limu_T_BS,
            gravity=[self.limu_g],
            time_ns=self.timestamp[:, index : index + 1],
            
            gt_pos=self.gt_pos[:, index : index + 1],
            gt_vel=self.gt_vel[:, index : index + 1],
            gt_rot=cast(pp.LieTensor, self.gt_rot[:, index : index + 1]),
            
            init_pos=self.gt_pos[:, index : index + 1],
            init_vel=self.gt_vel[:, index : index + 1],
            init_rot=cast(pp.LieTensor, self.gt_rot[:, index : index + 1]),
        )


def loadTartanAirGT(path: Path) -> pp.LieTensor:
    se3_data = np.loadtxt(str(path))
    return pp.SE3(se3_data)


class TartanAirIMUSimulator(Dataset[tuple[IMUData, AttitudeData]]):
    def __init__(self, config: SimpleNamespace, gtPath: Path, fps=100) -> None:
        super().__init__()
        self.fps = fps
        self.g_factor = 9.81
        self.imu_T_BS = pp.identity_SE3(1)
        
        self.g      = np.array([0, 0, self.g_factor])
        self.gtPath = gtPath
        self.camFPS = 10.0

        raw_poses = np.loadtxt(self.gtPath)
        img_time  = np.arange(raw_poses.shape[0]) / self.camFPS
        imu_time, accel_body, vel, pose, gyro, angles, _, _ = (
            self._interpolate_trajectory(img_time, raw_poses)
        )
        # time_interpolate, accel_body, vel, pose, gyro, angles, vel_body, accel
    
        # IMU Data
        self.lin_acc = torch.tensor(accel_body, dtype=torch.float).unsqueeze(0)  # (1, N, 3)
        self.rot_vel = torch.tensor(gyro, dtype=torch.float)      .unsqueeze(0)  # (1, N, 3)
        self.timestamp = torch.tensor(imu_time * 1_000_000, dtype=torch.long)    # (1, N,)

        noiseSim = IMUNoiseGenerator(**vars(config))
        self.lin_acc, self.rot_vel = noiseSim.propogate(self.lin_acc, self.rot_vel)

        # Ground Truth labels
        self.gt_vel = torch.tensor(vel, dtype=torch.float) .unsqueeze(0)  # (1, N, 3)
        self.gt_pos = torch.tensor(pose, dtype=torch.float).unsqueeze(0)  # (1, N, 3)
        rotations   = Rotation.from_euler("xyz", angles, degrees=False).as_quat(canonical=False)
        self.gt_rot = pp.SO3(torch.tensor(rotations, dtype=torch.float)).unsqueeze(0)  # (1, N, 4)

        # Camera alignment
        self.camtime    = torch.tensor(img_time * 1_000_000, dtype=torch.long)  # (M,)
        self.cam2imuIdx = torch.ones_like(self.camtime, dtype=torch.long) * -1  # (M,)
        self.length = self.lin_acc.size(1) - 1

        self.alignWithCameraTime()
        self.timestamp = self.timestamp.unsqueeze(0)

    def _interpolate_translation(self, time, data):
        time_interpolate = np.arange(round(time.max() * self.fps)) / self.fps
        pose, vel, accel = [], [], []

        for i in range(3):
            x = data[:, i]
            tck = interpolate.splrep(time, x, s=0, k=4)
            x_new = interpolate.splev(time_interpolate, tck, der=0)
            vel_new = interpolate.splev(time_interpolate, tck, der=1)
            accel_new = interpolate.splev(time_interpolate, tck, der=2)
            pose.append(x_new)
            vel.append(vel_new)
            accel.append(accel_new)
        accel = np.array(accel).T
        vel = np.array(vel).T
        pose = np.array(pose).T
        return time_interpolate, accel, vel, pose

    def _interpolate_rotation(self, time, data):
        rotations = Rotation.from_quat(data)
        spline = RotationSpline(time, rotations)

        time_interpolate = np.arange(round(time.max() * self.fps)) / self.fps
        angles               = spline(time_interpolate).as_euler("xyz", degrees=False)  # XYZ
        gyro                 = spline(time_interpolate, 1)
        angular_acceleration = spline(time_interpolate, 2)

        return time_interpolate, angular_acceleration, gyro, angles

    def _interpolate_trajectory(self, time, data):
        time_interpolate, accel, vel, pose = self._interpolate_translation(
            time, data[:, :3]
        )
        _, _, gyro, angles = self._interpolate_rotation(time, data[:, 3:])

        rotations = Rotation.from_euler("xyz", angles, degrees=False).as_matrix()
        accel_body = np.matmul(np.expand_dims(accel + self.g, 1), rotations).squeeze(1)
        vel_body = np.matmul(np.expand_dims(vel, 1), rotations).squeeze(1)

        return time_interpolate, accel_body, vel, pose, gyro, angles, vel_body, accel

    def alignWithCameraTime(self) -> None:
        """
        Align IMU Sequence with camera time

        The alignment relationship is stored in self.cam2imuIdx array as
        camidx -> imuidx
        """
        imu_idx, cam_idx, failFlag = 0, 0, False
        while imu_idx < self.length:
            frame_cam_time = self.camtime[cam_idx]
            imu_time = self.timestamp[imu_idx]
            next_imu_time = self.timestamp[imu_idx + 1]
            if imu_time <= frame_cam_time < next_imu_time:
                self.cam2imuIdx[cam_idx] = imu_idx
                cam_idx += 1
                imu_idx += 1
            elif imu_time > frame_cam_time:
                cam_idx += 1
                failFlag = True
            else:
                imu_idx += 1

        if failFlag or (cam_idx < self.camtime.size(0) - 1):
            Logger.write("warn", "Not all camera times are aligned with IMU Trajectory")
        if imu_idx < self.timestamp.shape[0]:
            Logger.write(
                "warn",
                f"{self.timestamp.shape[0] - imu_idx} IMU samples remain unmatched",
            )

    def frameRangeQuery(self, start_frame, end_frame) -> tuple[IMUData, AttitudeData]:
        """Retrieve IMU data in range of [start_frame, end_frame)

        Args:
            start_frame (int): start image frame to get IMU sequence
            end_frame (int): end image frame to get (exclusive)

        Returns:
            IMUResult: A range of IMU Results
        """
        start_imu_idx = self.cam2imuIdx[start_frame]
        end_imu_idx = self.cam2imuIdx[end_frame]
        assert (
            start_imu_idx != -1 and end_imu_idx != -1
        ), "Requested frame is not aligned with IMU Sequence"
        
        return IMUData(
            T_BS=self.imu_T_BS,
            gravity=[self.g_factor],
            time_ns=self.timestamp[:, start_imu_idx:end_imu_idx],
            
            acc    = self.lin_acc[:, start_imu_idx:end_imu_idx],
            gyro   = self.rot_vel[:, start_imu_idx:end_imu_idx],
        ), AttitudeData(
            T_BS=self.imu_T_BS,
            gravity=[self.g_factor],
            time_ns=self.timestamp[:, start_imu_idx:end_imu_idx],
            
            gt_pos = self.gt_pos [:, start_imu_idx:end_imu_idx],
            gt_vel = self.gt_vel [:, start_imu_idx:end_imu_idx],
            gt_rot = cast(pp.LieTensor, self.gt_rot [:, start_imu_idx:end_imu_idx]),
            
            init_pos=self.gt_pos [:, start_imu_idx:start_imu_idx+1],
            init_vel=self.gt_vel [:, start_imu_idx:start_imu_idx+1],
            init_rot=cast(pp.LieTensor, self.gt_rot [:, start_imu_idx:start_imu_idx+1]),
        )

    def __getitem__(self, index) -> tuple[IMUData, AttitudeData]:
        """
        Args:
            index (int): the index of IMU record to retrieve

        Returns: {
            "acc": (1, 3) tensor for linear acceleration
            "ang_vel": (1, 3) tensor for angular velocity
            "time": (1,) tnesor for timestamp
        }
        """
        start_imu_idx = index
        end_imu_idx   = index + 1
        return IMUData(
            T_BS=self.imu_T_BS,
            gravity=[self.g_factor],
            time_ns=self.timestamp[:, start_imu_idx:end_imu_idx],
            
            acc    = self.lin_acc[:, start_imu_idx:end_imu_idx],
            gyro   = self.rot_vel[:, start_imu_idx:end_imu_idx],
        ), AttitudeData(
            T_BS=self.imu_T_BS,
            gravity=[self.g_factor],
            time_ns=self.timestamp[:, start_imu_idx:end_imu_idx],
            
            gt_pos = self.gt_pos [:, start_imu_idx:end_imu_idx],
            gt_vel = self.gt_vel [:, start_imu_idx:end_imu_idx],
            gt_rot = cast(pp.LieTensor, self.gt_rot [:, start_imu_idx:end_imu_idx]),
            
            init_pos=self.gt_pos [:, start_imu_idx:start_imu_idx+1],
            init_vel=self.gt_vel [:, start_imu_idx:start_imu_idx+1],
            init_rot=cast(pp.LieTensor, self.gt_rot [:, start_imu_idx:start_imu_idx+1]),
        )


class IMUNoiseGenerator(ConfigTestable):
    T_Vec3 = tuple[float, float, float]
    
    def __init__(self, acc_bias: T_Vec3, gyro_bias: T_Vec3, 
                 acc_init_bias_noise: T_Vec3 , acc_bias_instability: T_Vec3 , acc_random_walk: T_Vec3,
                 gyro_init_bias_noise: T_Vec3, gyro_bias_instability: T_Vec3, gyro_random_walk: T_Vec3) -> None:
        """
        Noise Parameters:
            - Initial bias Gaussian: Shape (, 3) the noise on xyz
            - Bias statability noise
            - White Gaussian noise
            - scale noise

        Adding noise:
            1. apply init bias noise 
            2. during each iteration, get the imu input and apply the random walk with bias statability noise
        """
        self.acc_bias = torch.tensor(acc_bias)
        self.acc_init_bias_noise = torch.tensor(acc_init_bias_noise)
        self.acc_bias_instability = torch.tensor(acc_bias_instability)
        self.acc_random_walk = torch.tensor(acc_random_walk)
        
        self.gyro_bias = torch.tensor(gyro_bias)
        self.gyro_init_bias_noise = torch.tensor(gyro_init_bias_noise)
        self.gyro_bias_instatability = torch.tensor(gyro_bias_instability)
        self.gyro_random_walk = torch.tensor(gyro_random_walk)

        self.scale = torch.ones(1, 3) # TODO: consider a scale axis matrix
        self.acc_bias_list = []
        self.gyro_bias_list = []

    @staticmethod
    def _gaussian_xyz(x: torch.Tensor, noise: torch.Tensor):
        std = torch.ones_like(x)
        std[..., 0] *= noise[0]
        std[..., 1] *= noise[1]
        std[..., 2] *= noise[2]
        return torch.normal(mean=0, std=std)

    def get_acc_bias(self):
        return self.acc_bias
    
    def get_gyro_bias(self):
        return self.gyro_bias
        
    def propogate(self, acc, gyro):
        acc += self.acc_bias
        acc += self._gaussian_xyz(acc, noise = self.acc_random_walk)
        gyro += self.gyro_bias
        gyro += self._gaussian_xyz(gyro, noise = self.gyro_random_walk)
        
        ## propogate bias
        self.acc_bias_list.append(self.acc_bias.clone())
        self.gyro_bias_list.append(self.gyro_bias.clone())

        self.acc_bias += self._gaussian_xyz(self.acc_bias, noise = self.acc_bias_instability)
        self.gyro_bias += self._gaussian_xyz(self.gyro_bias, noise = self.gyro_bias_instatability)
    
        return acc, gyro

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        def is_triplet_of(value, t: type):
            return len(value) == 3 and all([isinstance(value[idx], t) for idx in range(len(value))])

        cls._enforce_config_spec(config, {
            "acc_bias": lambda v: is_triplet_of(v, float),
            "acc_init_bias_noise": lambda v: is_triplet_of(v, float),
            "acc_bias_instability": lambda v: is_triplet_of(v, float),
            "acc_random_walk": lambda v: is_triplet_of(v, float),
            
            "gyro_bias": lambda v: is_triplet_of(v, float),
            "gyro_init_bias_noise": lambda v: is_triplet_of(v, float),
            "gyro_bias_instability": lambda v: is_triplet_of(v, float),
            "gyro_random_walk": lambda v: is_triplet_of(v, float)
        })
