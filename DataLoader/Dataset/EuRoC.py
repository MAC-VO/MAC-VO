from __future__ import annotations
import cv2
import torch
import numpy as np
import pypose as pp

from types import SimpleNamespace
from typing import cast, Any
from pathlib import Path
from torch.utils.data import Dataset

from Utility.PrettyPrint import Logger
from Utility.Config import load_config
from Utility.Math import qinterp, interpolate_pose

from ..Interface import StereoData, IMUData, StereoFrame, StereoInertialFrame, AttitudeData
from ..SequenceBase import SequenceBase


EDN2NED = pp.from_matrix(torch.tensor([
    [0., 0., 1., 0.],
    [1., 0., 0., 0.],
    [0., 1., 0., 0.],
    [0., 0., 0., 1.],
]), pp.SE3_type)
NED2EDN = EDN2NED.Inv()


class EuRoC_Sequence(SequenceBase[StereoInertialFrame]):
    @classmethod
    def name(cls) -> str: return "EuRoC"

    def __init__(self, config: SimpleNamespace | dict[str, Any]) -> None:
        cfg = self.config_dict2ns(config)
        self.stereo_seq = EuRoC_StereoSequence(cfg)
        self.imu_seq    = EurocIMULoader(Path(cfg.root, "imu0"))
        self.stereo_seq = self.stereo_seq.clip(end_idx=self.imu_seq.idx_end_cam)
        self.offset     = self.imu_seq.idx_start_cam
        super().__init__(len(self.stereo_seq)-self.offset)

    def __getitem__(self, local_index: int) -> StereoInertialFrame:
        index = self.get_index(local_index)
        stereo_frame = self.stereo_seq[self.offset + index]
        
        if index == 0:
            imu, attitude = self.imu_seq[self.imu_seq.idx_imu_align_start]
        else:
            imu, attitude = self.imu_seq.frameRangeQuery(self.offset + index - 1, self.offset + index)
        
        return StereoInertialFrame(
            idx=[local_index],
            time_ns=stereo_frame.time_ns,
            gt_pose=stereo_frame.gt_pose,
            stereo=stereo_frame.stereo,
            imu=imu, gt_attitude=attitude
        )
    
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "root"   : lambda v: isinstance(v, str),
            "gt_pose": lambda b: isinstance(b, bool)
        })


class EuRoC_StereoSequence(SequenceBase[StereoFrame]):
    @classmethod
    def name(cls) -> str: return "EuRoC_NoIMU"

    def __init__(self, config: SimpleNamespace | dict[str, Any]) -> None:
        cfg = self.config_dict2ns(config)
        self.seqRoot  = Path(cfg.root)
        
        # ref: https://github.com/raulmur/ORB_SLAM2/blob/master/Examples/Stereo/EuRoC.yaml
        # in this file only bl * fx is provided , the baseline here is derived by bf/fx
        self.baseline = 0.1100778422
        self.width = 752
        self.height = 480
        
        # Left Camera
        l_sensor_config, _ = load_config(Path(self.seqRoot, "cam0", "sensor.yaml"))
        T_BS_lcam = np.array(l_sensor_config.T_BS.data).reshape(4, 4)
        self.ImageL = EurocMonocularDataset(
            Path(self.seqRoot, "cam0", "data"), 
            K=self.build_intrinsic(l_sensor_config.intrinsics),
            T_BS=T_BS_lcam,
            undistort=np.array([-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05, 0.0])
        )
        
        # Right Camera
        r_sensor_config, _ = load_config(Path(self.seqRoot, "cam1", "sensor.yaml"))
        T_BS_rcam = np.array(r_sensor_config.T_BS.data).reshape(4, 4)
        self.ImageR = EurocMonocularDataset(
            Path(self.seqRoot, "cam1", "data"),
            K=self.build_intrinsic(r_sensor_config.intrinsics),
            T_BS=T_BS_rcam,
            undistort=np.array([-0.28368365, 0.07451284, -0.00010473, -3.555907e-05, 0.0])
        )
        
        # Sync left-right camera
        rectified_K = self.sync_LR(self.ImageL, self.ImageR)
        self.K = torch.tensor(rectified_K[:3, :3], dtype=torch.float).unsqueeze(0)
        self.cam_timestamps = self.ImageL.cam_timestamps
        assert len(self.ImageL) == len(self.ImageR), f"ImageL={len(self.ImageL)} and ImageR={len(self.ImageR)} is not sync'd as expected."
        
        # Setup metadata
        self.T_BS_lcam = pp.from_matrix(
            torch.tensor(T_BS_lcam, dtype=torch.float32).unsqueeze(0), pp.SE3_type
        ) @ NED2EDN.unsqueeze(0)
        
        # Load ground truth pose
        if cfg.gt_pose:
            self.gt_pose_data, self.cam_time_mask = load_EurocGTPose(
                Path(self.seqRoot, "state_groundtruth_estimate0/data.csv"),
                self.ImageL.cam_timestamps
            )
            self.ImageL.apply_mask(self.cam_time_mask)
            self.ImageR.apply_mask(self.cam_time_mask)
        else:
            self.gt_pose_data = None
        
        super().__init__(len(self.ImageL))

    def __getitem__(self, local_index: int) -> StereoFrame:
        index = self.get_index(local_index)

        return StereoFrame(
            idx=[local_index],
            time_ns=[int(self.cam_timestamps[index].item())], 
            stereo=StereoData(
                T_BS=self.T_BS_lcam,
                K   =self.K,
                baseline=torch.tensor([self.baseline]),
                width=self.width,
                height=self.height,
                time_ns=[int(self.cam_timestamps[index].item())],
                imageL=self.ImageL[index],
                imageR=self.ImageR[index],
            ),
            gt_pose= None if self.gt_pose_data is None else cast(pp.LieTensor, self.gt_pose_data[index].unsqueeze(0))
        )
    
    @staticmethod
    def sync_LR(left: EurocMonocularDataset, right: EurocMonocularDataset) -> np.ndarray:
        # Constant - Transformation from cam 1 to cam 2 (L -> R)
        T_LR = np.linalg.inv(right.T_BS) @ left.T_BS
        
        # Align timestamps, discard time stamp with only Left/Right image.
        left_time = {t_l.item() for t_l in left.cam_timestamps}
        right_time = {t_r.item() for t_r in right.cam_timestamps}
        common_time = left_time.intersection(right_time)
        common_time = np.array(sorted(list(common_time)))
        
        left_sync_mask  = np.isin(left.cam_timestamps, common_time, assume_unique=True)
        right_sync_mask = np.isin(right.cam_timestamps, common_time, assume_unique=True)
        
        left.cam_timestamps = left.cam_timestamps[left_sync_mask]
        right.cam_timestamps = right.cam_timestamps[right_sync_mask]
        left.file_names = [f for idx, f in enumerate(left.file_names) if left_sync_mask[idx].item()]
        right.file_names = [f for idx, f in enumerate(right.file_names) if right_sync_mask[idx].item()]
        left.length = len(left.file_names)
        right.length = len(right.file_names)
        
        # Rectify stereo and undistort based on Left and Right camera.
        R1, R2, P1, P2, Q, validRoi1, validRoi2 = cv2.stereoRectify(left.K, left.distort_factor, 
                          right.K, right.distort_factor, (752, 480),
                          T_LR[:3, :3], T_LR[:3, 3], flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1)

        left.undistort_map = cv2.initUndistortRectifyMap(left.K, left.distort_factor, R1, P1, (752, 480), cv2.CV_32FC1)
        right.undistort_map = cv2.initUndistortRectifyMap(right.K, right.distort_factor, R2, P2, (752, 480), cv2.CV_32FC1)
        left.K = P1[:3, :3]
        right.K = P2[:3, :3]
        
        return P1

    @staticmethod
    def build_intrinsic(intrinsic: list[float]) -> np.ndarray:
        fx, fy, cx, cy = intrinsic
        return np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])
    
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "root"   : lambda v: isinstance(v, str),
            "gt_pose": lambda b: isinstance(b, bool)
        })


class EurocMonocularDataset(Dataset):
    """
    Return images in the given directory ends with .png
    """
    def __init__(self, directory: Path, K: np.ndarray, undistort: np.ndarray, T_BS: np.ndarray) -> None:
        super().__init__()
        self.directory = directory
        self.K = K
        self.T_BS = T_BS
        self.distort_factor = undistort
        self.undistort_map: None | tuple[np.ndarray, np.ndarray] = None
        
        assert self.directory.exists(), f"Monocular image directory {self.directory} does not exist"

        self.file_names = [f for f in self.directory.iterdir() if f.suffix == ".png"]
        self.file_names.sort()
        self.length = len(self.file_names)
        assert self.length > 0, f"No flow with '.png' suffix is found under {self.directory}"
        
        self.cam_timestamps = np.loadtxt(Path(directory, "..", "data.csv"), delimiter=",", skiprows=1, usecols=0, dtype=np.int64)

    def apply_mask(self, cam_mask: np.ndarray):
        self.file_names = [f for idx, f in enumerate(self.file_names) if (idx < cam_mask.shape[0]) and cam_mask[idx].item()]
        self.file_names.sort()
        self.length = len(self.file_names)
        self.cam_timestamps = self.cam_timestamps[cam_mask]

    def load_png(self, path: Path) -> np.ndarray:
        return self.correct_distortion(cv2.imread(str(path)))

    def __len__(self):
        return self.length
    
    def __getitem__(self, index: int) -> torch.Tensor:
        # Output image tensor in shape of (1, C, H, W)
        result = self.load_png(self.file_names[index])
        result = torch.tensor(result, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        result /= 255.        
        return result

    def correct_distortion(self, image: np.ndarray) -> np.ndarray:
        if self.undistort_map is None:
            raise Exception("Monocular sequence is not rectified.")
        else:
            undistorted_image = cv2.remap(image, self.undistort_map[0], self.undistort_map[1], cv2.INTER_LINEAR)
        return undistorted_image


def load_EurocGTPose(csv_file_path: Path, cam_time: np.ndarray) -> tuple[pp.LieTensor, np.ndarray]:
    csv_path = csv_file_path
    
    raw_data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    pose_time = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.int64, usecols=0)
    position_xyz, rotation_wxyz = raw_data[..., 1:4], raw_data[..., 4:8]
    rotation_xyzw = np.roll(rotation_wxyz, axis=1, shift=-1)
    pose_SE3 = pp.SE3(np.concatenate([position_xyz, rotation_xyzw], axis=1))
    
    # Need to mask only the valid part of cam_time (cannot be smaller than or greater to gt_pose)
    cam_time_mask = (cam_time > pose_time[0]) & (cam_time < pose_time[-1])
    # end
    
    bodyPose_SE3, _ = interpolate_pose(pose_SE3, torch.tensor(pose_time), torch.tensor(cam_time[cam_time_mask]))
    return bodyPose_SE3, cam_time_mask


class EurocIMULoader(Dataset[tuple[IMUData, AttitudeData]]):
    def __init__(self, imu_path: Path) -> None:
        super().__init__()
        imu_data          = Path(imu_path, "data.csv")
        gt_data           = Path(imu_path, "..", "state_groundtruth_estimate0", "data.csv")
        cam_data          = Path(imu_path, "..", "cam0", "data.csv")
        assert imu_path.exists(), f"IMU Data path ({imu_path}) does not exist"
        assert gt_data.exists(), f"GT Data path ({gt_data}) does not exist"

        rawIMU = np.genfromtxt(str(imu_data), dtype=np.float64, delimiter=",")
        rawGT  = torch.tensor(np.genfromtxt(str(gt_data), dtype=np.float64, delimiter=","))

        gt_time = rawGT[:, 0:1]
        imu_time = rawIMU[..., 0:1]
        self.timestamp = torch.tensor(imu_time, dtype=torch.long)
        
        t_start = np.max([self.timestamp.numpy()[0] , gt_time.numpy()[0]])
        t_end   = np.min([self.timestamp.numpy()[-1], gt_time.numpy()[-1]])

        idx_start_imu = np.searchsorted(self.timestamp.squeeze().numpy(), t_start)
        idx_start_gt  = np.searchsorted(gt_time.squeeze().numpy(), t_start)
        idx_end_imu   = np.searchsorted(self.timestamp.squeeze().numpy(), t_end, 'right')
        idx_end_gt    = np.searchsorted(gt_time.squeeze().numpy(), t_end, 'right')

        rawGT   = rawGT[idx_start_gt:idx_end_gt]
        gt_time = gt_time[idx_start_gt:idx_end_gt]

        rawIMU = rawIMU[idx_start_imu:idx_end_imu]
        self.timestamp = self.timestamp[idx_start_imu:idx_end_imu]

        self.gyro = torch.tensor(rawIMU[..., 1:4]).unsqueeze(0)  # With batch-dim (dim0), Coord. S, (w, x, y, z) format
        self.acc  = torch.tensor(rawIMU[..., 4:7]).unsqueeze(0)  # With batch-dim (dim0), Coord. S

        self.gt_vel = self.interpolate_vecN  (self.timestamp, gt_time, rawGT[:, -9:-6]).unsqueeze(0)
        self.gt_pos = self.interpolate_vecN  (self.timestamp, gt_time, rawGT[:, 1:4]).unsqueeze(0)
        self.gt_rot = self.interpolate_rotate(self.timestamp, gt_time, rawGT[:, 4:8]).unsqueeze(0)

        self.camtime = self.read_camera_time(cam_data)
        self.idx_start_cam = np.searchsorted(self.camtime.numpy(), t_start)
        self.idx_end_cam   = np.searchsorted(self.camtime.numpy(), t_end, 'right')
    
        self.cam2imuIdx = torch.ones_like(self.camtime, dtype=torch.long) * -1
        
        self.length    = self.timestamp.size(0) - 1
        self.alignWithCameraTime()
        self.idx_imu_align_start = int(self.cam2imuIdx[self.idx_start_cam].item())
        
        self.timestamp = self.timestamp.unsqueeze(0)    # Add batch dimension
        
    def __len__(self):
        return self.length

    def __getitem__(self, index) -> tuple[IMUData, AttitudeData]:
        return IMUData(
            T_BS    =pp.identity_SE3(0),
            time_ns = self.timestamp[:, index:index + 1,:],
            gravity = [9.81007],
            acc     = self.acc[:, index : index + 1],
            gyro    = self.gyro[:, index : index + 1],
        ), AttitudeData(
            T_BS    =pp.identity_SE3(0),
            time_ns = self.timestamp[:, index:index + 1],
            gravity = [9.81007],
            
            gt_pos=self.gt_pos[:, index : index + 1],
            gt_vel=self.gt_vel[:, index : index + 1],
            gt_rot=cast(pp.LieTensor, self.gt_rot[:, index : index + 1]),
            
            init_pos=self.gt_pos[:, index : index + 1],
            init_vel=self.gt_vel[:, index : index + 1],
            init_rot=cast(pp.LieTensor, self.gt_rot[:, index : index + 1]),
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
        end_imu_idx   = self.cam2imuIdx[end_frame]
        assert (start_imu_idx != -1 and end_imu_idx != -1), "Requested frame is not aligned with IMU Sequence"
        
        return IMUData(
            T_BS    =pp.identity_SE3(0),
            time_ns = self.timestamp[:, start_imu_idx:end_imu_idx],
            gravity = [9.81007],
            acc     = self.acc [:, start_imu_idx:end_imu_idx],
            gyro    = self.gyro[:, start_imu_idx:end_imu_idx],
        ), AttitudeData(
            T_BS    =pp.identity_SE3(0),
            time_ns = self.timestamp[:, start_imu_idx:end_imu_idx],
            gravity = [9.81007],
            
            gt_pos=self.gt_pos[:, start_imu_idx:end_imu_idx],
            gt_vel=self.gt_vel[:, start_imu_idx:end_imu_idx],
            gt_rot=cast(pp.LieTensor, self.gt_rot[:, start_imu_idx:end_imu_idx]),
            
            init_pos=self.gt_pos[:, start_imu_idx:start_imu_idx + 1],
            init_vel=self.gt_vel[:, start_imu_idx:start_imu_idx + 1],
            init_rot=cast(pp.LieTensor, self.gt_rot[:, start_imu_idx:start_imu_idx + 1]),
        )

    @staticmethod
    def interpolate_vecN(ev_time: torch.Tensor, time: torch.Tensor, value: torch.Tensor, N=3):
        assert time.size(0) == value.size(0)
        ev_time, time = ev_time.squeeze(), time.squeeze()
        interp_res = np.zeros((ev_time.size(0), 3))
        for i in range(N):
            interp_res[:, i] = np.interp(
                ev_time.numpy(), xp=time.numpy(), fp=value[:, i].numpy()
            )
        return torch.tensor(interp_res)

    @staticmethod
    def interpolate_rotate(ev_time: torch.Tensor, time: torch.Tensor, quat: torch.Tensor):
        assert time.size(0) == quat.size(0)
        ev_time, time = ev_time.squeeze(), time.squeeze()

        # interpolation in the log space
        quat = qinterp(quat, time, ev_time.double()).double()
        rot = torch.zeros_like(quat)
        rot[:, 3] = quat[:, 0]
        rot[:, :3] = quat[:, 1:]

        return pp.SO3(rot)

    def read_camera_time(self, cam_data_path: Path):
        # cam_data_path = Path(self.imuPath, "..", "cam0", "data.csv")
        assert cam_data_path.exists()
        cam_time = np.genfromtxt(str(cam_data_path), delimiter=",", dtype=np.float64)
        cam_time = cam_time[:, 0]
        return torch.tensor(cam_time)

    def alignWithCameraTime(self) -> None:
        """
        Align IMU Sequence with camera time

        The alignment relationship is stored in self.cam2imuIdx array as
        camidx -> imuidx
        """
        imu_idx, cam_idx, failFlag = 0, 0, False
        while imu_idx < self.length and cam_idx < self.camtime.size(0):
            frame_cam_time = self.camtime[cam_idx]
            imu_time       = self.timestamp[imu_idx]
            next_imu_time  = self.timestamp[imu_idx + 1]
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
            Logger.write("warn", f"{self.timestamp.size(0) - imu_idx} IMU samples remain unmatched")
