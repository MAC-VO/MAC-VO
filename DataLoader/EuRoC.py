import cv2
import torch
import numpy as np
import pypose as pp
from pathlib import Path
from torch.utils.data import Dataset

from .SequenceBase import GenericSequence
from .Interface import IMUData, MetaInfo, SourceDataFrame
from Utility.PrettyPrint import Logger
from Utility.Config import load_config
from Utility.Math import qinterp, interpolate_pose


EDN2NED = pp.from_matrix(torch.tensor([
    [0., 0., 1., 0.],
    [1., 0., 0., 0.],
    [0., 1., 0., 0.],
    [0., 0., 0., 1.],
]), pp.SE3_type)
NED2EDN = EDN2NED.Inv()


class EuRoCSequence(GenericSequence[SourceDataFrame]):
    @classmethod
    def name(cls) -> str: return "EuRoC"

    def __init__(self, root, gtFlow, gtDepth, gtPose, useIMU, **_) -> None:
        # Metadata
        self.seqRoot  = Path(root)
        
        # ref: https://github.com/raulmur/ORB_SLAM2/blob/master/Examples/Stereo/EuRoC.yaml
        # in this file only bl * fx is provided , the baseline here is derived by bf/fx
        self.baseline = 0.1100778422
        self.K = torch.tensor([[458.654, 0, 367.215], [0, 457.296, 248.375], [0, 0, 1.0]])
        self.g = 9.81007
        self.width = 752
        self.height = 480
        
        self.gtFlow = gtFlow
        self.gtDepth = gtDepth
        self.gtPose = gtPose
        self.useIMU = useIMU
        self.ImageL = EurocMonocularDataset(
            Path(self.seqRoot, "cam0", "data"), 
            K=np.array([458.654, 0.0, 367.215, 0.0, 457.296, 248.375, 0.0, 0.0, 1.0]).reshape(3,3),
            T_BS=np.array([
                [0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,],
                [0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768,],
                [-0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949,],
                [0.0, 0.0, 0.0, 1.0]
            ]),
            undistort=np.array([-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05, 0.0])
        )
        self.ImageR = EurocMonocularDataset(
            Path(self.seqRoot, "cam1", "data"),
            K=np.array([457.587, 0.0, 379.999, 0.0, 456.134, 255.238, 0.0, 0.0, 1]).reshape(3,3),
            T_BS=np.array([
                [0.0125552670891, -0.999755099723, 0.0182237714554, -0.0198435579556],
                [0.999598781151, 0.0130119051815, 0.0251588363115, 0.0453689425024],
                [-0.0253898008918, 0.0179005838253, 0.999517347078, 0.00786212447038],
                [0.0, 0.0, 0.0, 1.0]
            ]),
            undistort=np.array([-0.28368365, 0.07451284, -0.00010473, -3.555907e-05, 0.0])
        )
        rectified_K = self.sync_LR(self.ImageL, self.ImageR)
        self.K = torch.tensor(rectified_K[:3, :3], dtype=torch.float)
        
        assert len(self.ImageL) == len(self.ImageR), \
               f"ImageL={len(self.ImageL)} and ImageR={len(self.ImageR)} is not sync'd as expected."
        
        if self.useIMU:
            self.IMUL = EurocIMULoader(Path(self.seqRoot, "imu0"))
        else:
            self.IMUL = None

        assert not self.gtDepth, "EuRoC does not provide ground truth depth data"
        assert not self.gtFlow , "EuRoC does not provide ground truth flow data"
        
        if self.gtPose:
            self.gtPose_data, self.cam_time_mask = load_EurocGTPose(
                Path(self.seqRoot, "state_groundtruth_estimate0/data.csv"),
                Path(self.seqRoot, "cam0"),
                self.ImageL.cam_timestamps
            )
            self.ImageL.apply_mask(self.cam_time_mask)
            self.ImageR.apply_mask(self.cam_time_mask)
        else:
            self.gtPose_data = None
        
        super().__init__(len(self.ImageL))

    def __getitem__(self, local_index: int) -> SourceDataFrame:
        index = self.get_index(local_index)
        if self.IMUL is None:
            imuData = None
        else:
            if index != 0:
                imuData = self.IMUL.frameRangeQuery(index - 1, index)
            else:
                imuData = self.IMUL[0]

        return SourceDataFrame(
            meta=MetaInfo(
                idx=local_index,
                K=self.K,
                baseline=self.baseline,
                width=self.width,
                height=self.height
            ),
            imageL=self.ImageL[index],
            imageR=self.ImageR[index],
            gtDepth=None,
            gtFlow=None,
            flowMask=None,
            gtPose=None if self.gtPose_data is None else self.gtPose_data[index], #type: ignore
            imu=imuData,
        )
    
    @staticmethod
    def sync_LR(left: "EurocMonocularDataset", right: "EurocMonocularDataset") -> np.ndarray:
        # Constant - Transformation from cam 1 to cam 2 (L -> R)
        T_LR = np.linalg.inv(right.T_BS) @ left.T_BS
        
        # Align timestamps, discard time stamp with only Left/Right image.
        left_time = {t_l.item() for t_l in left.cam_timestamps}
        right_time = {t_r.item() for t_r in right.cam_timestamps}
        common_time = left_time.intersection(right_time)
        common_time = torch.tensor(sorted(list(common_time)))
        
        left_sync_mask = torch.isin(left.cam_timestamps, common_time, assume_unique=True)
        right_sync_mask = torch.isin(right.cam_timestamps, common_time, assume_unique=True)
        
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


class EurocMonocularDataset(Dataset):
    """
    Return images in the given directory ends with .png
    """
    def __init__(self, directory: Path, K, undistort, T_BS) -> None:
        super().__init__()
        self.directory = directory
        self.K = K
        self.T_BS = T_BS
        self.distort_factor = undistort
        self.undistort_map = None
        
        assert self.directory.exists(), f"Monocular image directory {self.directory} does not exist"

        self.file_names = [f for f in self.directory.iterdir() if f.suffix == ".png"]
        self.file_names.sort()
        self.length = len(self.file_names)
        assert self.length > 0, f"No flow with '.png' suffix is found under {self.directory}"
        
        self.cam_timestamps = torch.tensor(
            np.loadtxt(Path(directory, "..", "data.csv"), delimiter=",", skiprows=1, usecols=0, dtype=np.int64)
        )

    def apply_mask(self, cam_mask: torch.Tensor):
        self.file_names = [f for idx, f in enumerate(self.file_names) if (idx < cam_mask.size(0)) and cam_mask[idx].item()]
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
            # undistorted_image = cv2.undistort(image, self.K, self.distort_factor)
        else:
            undistorted_image = cv2.remap(image, self.undistort_map[0], self.undistort_map[1], cv2.INTER_LINEAR)
        return undistorted_image


class EurocIMULoader(Dataset[IMUData]):
    def __init__(self, imuPath: Path) -> None:
        super().__init__()
        self.imuPath = imuPath
        imuData = Path(self.imuPath, "data.csv")
        gtData = Path(self.imuPath, "..", "state_groundtruth_estimate0", "data.csv")
        assert imuPath.exists(), f"IMU Data path ({imuPath}) does not exist"
        assert gtData.exists(), f"GT Data path ({gtData}) does not exist"

        rawIMU = np.genfromtxt(str(imuData), dtype=np.float64, delimiter=",")
        rawGT = torch.tensor(
            np.genfromtxt(str(gtData), dtype=np.float64, delimiter=",")
        )

        gt_time = rawGT[:, 0:1] / 1e9

        self.timestamp = torch.tensor(rawIMU[..., 0:1]) / 1e9
        self.dt = self.timestamp[1:] - self.timestamp[:-1]

        self.gyro = torch.tensor(rawIMU[..., 1:4])  # Coord. S, (w, x, y, z) format
        self.acc = torch.tensor(rawIMU[..., 4:7])  # Coord. S

        self.gt_vel = self.interpolate_vecN(self.timestamp, gt_time, rawGT[:, -9:-6])
        self.gt_pos = self.interpolate_vecN(self.timestamp, gt_time, rawGT[:, 1:4])
        self.gt_rot = self.interpolate_rotate(self.timestamp, gt_time, rawGT[:, 4:8])

        self.camtime = self.read_camera_time()
        self.cam2imuIdx = torch.ones_like(self.camtime, dtype=torch.long) * -1

        self.length = self.dt.size(0)
        self.alignWithCameraTime()

        ## Move to global coordinate system
        self.gyro = self.gt_rot @ self.gyro
        self.acc = self.gt_rot @ self.acc

    def __len__(self):
        return self.length

    def __getitem__(self, index) -> IMUData:
        return IMUData(
            dt=self.dt[index : index + 1],
            acc=self.acc[index : index + 1],
            gyro=self.gyro[index : index + 1],
            time=self.timestamp[index : index + 1],
            gtPos=self.gt_pos[index : index + 1],
            gtVel=self.gt_vel[index : index + 1],
            gtRot=self.gt_rot[index : index + 1],
            initPos=self.gt_pos[index : index + 1],
            initVel=self.gt_vel[index : index + 1],
            initRot=self.gt_rot[index : index + 1],
        )

    def frameRangeQuery(self, start_frame, end_frame) -> IMUData:
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
            dt=self.dt[start_imu_idx:end_imu_idx],
            time=self.timestamp[start_imu_idx:end_imu_idx],
            acc=self.acc[start_imu_idx:end_imu_idx],
            gyro=self.gyro[start_imu_idx:end_imu_idx],
            gtPos=self.gt_pos[start_imu_idx:end_imu_idx],
            gtVel=self.gt_vel[start_imu_idx:end_imu_idx],
            gtRot=self.gt_rot[start_imu_idx:end_imu_idx],
            initPos=self.gt_pos[start_imu_idx : start_imu_idx + 1],
            initVel=self.gt_vel[start_imu_idx : start_imu_idx + 1],
            initRot=self.gt_rot[start_imu_idx : start_imu_idx + 1],
        )

    @staticmethod
    def interpolate_vecN(
        ev_time: torch.Tensor, time: torch.Tensor, value: torch.Tensor, N=3
    ):
        assert time.size(0) == value.size(0)
        ev_time, time = ev_time.squeeze(), time.squeeze()
        interp_res = np.zeros((ev_time.size(0), 3))
        for i in range(N):
            interp_res[:, i] = np.interp(
                ev_time.numpy(), xp=time.numpy(), fp=value[:, i].numpy()
            )
        return torch.tensor(interp_res)

    @staticmethod
    def interpolate_rotate(
        ev_time: torch.Tensor, time: torch.Tensor, quat: torch.Tensor
    ):
        assert time.size(0) == quat.size(0)
        ev_time, time = ev_time.squeeze(), time.squeeze()

        # interpolation in the log space
        quat = qinterp(quat, time, ev_time).double()
        rot = torch.zeros_like(quat)
        rot[:, 3] = quat[:, 0]
        rot[:, :3] = quat[:, 1:]

        return pp.SO3(rot)

    def read_camera_time(self):
        cam_data_path = Path(self.imuPath, "..", "cam0", "data.csv")
        assert cam_data_path.exists()
        cam_time = np.genfromtxt(str(cam_data_path), delimiter=",", dtype=np.float64)
        cam_time = cam_time[:, 0] / 1e9
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
            Logger.write("warn", "[Warning]: Not all camera times are aligned with IMU Trajectory")
        if imu_idx < self.timestamp.size(0):
            Logger.write("warn", f"[Warning]: {self.timestamp.size(0) - imu_idx} IMU samples remain unmatched")


def load_EurocGTPose(csv_file_path: Path, cam0_data_path: Path, cam_time: torch.Tensor) -> tuple[pp.LieTensor, torch.Tensor]:
    csv_path = csv_file_path
    cam_0_path = cam0_data_path
    
    raw_data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    pose_time = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.int64, usecols=0)
    position_xyz, rotation_wxyz = raw_data[..., 1:4], raw_data[..., 4:8]
    rotation_xyzw = np.roll(rotation_wxyz, axis=1, shift=-1)
    pose_SE3 = pp.SE3(np.concatenate([position_xyz, rotation_xyzw], axis=1))
    
    gtPose_SE3 = pose_SE3
    gtPose_time = torch.tensor(pose_time)
    
    # Need to mask only the valid part of cam_time (cannot be smaller than or greater to gtPose)
    cam_time_mask = torch.logical_and(cam_time > gtPose_time[0], cam_time < gtPose_time[-1])
    # end
    
    camPose_SE3 = interpolate_pose(gtPose_SE3, gtPose_time, cam_time[cam_time_mask])
    
    # Transform from Body frame -> Camera (Sensor) frame
    sensor_config, _ = load_config(Path(cam_0_path, "sensor.yaml"))
    T_BS = pp.from_matrix(torch.tensor(sensor_config.T_BS.data).reshape(4, 4), pp.SE3_type)
    
    camPose_SE3 = T_BS.Inv() @ camPose_SE3 @ T_BS
    camPose_SE3 = EDN2NED @ camPose_SE3 @ NED2EDN
    return camPose_SE3, cam_time_mask
