import cv2
import torch
import numpy as np
import pypose as pp
from pathlib import Path

from scipy import interpolate
from scipy.spatial.transform import Rotation, RotationSpline
from torch.utils.data import Dataset

from .SequenceBase import GenericSequence
from .Interface import IMUData, MetaInfo, SourceDataFrame
from Utility.PrettyPrint import Logger


class TartanAirSequence(GenericSequence[SourceDataFrame]):
    @classmethod
    def name(cls) -> str: return "TartanAir"

    def __init__(self, root, ips, compressed, gtFlow, gtDepth, gtPose, useIMU, **_):
        # metadata
        self.seqRoot = Path(root)
        self.K = torch.tensor([[320.0, 0.0, 320.0], [0.0, 320.0, 240.0], [0.0, 0.0, 1.0]])
        self.g = 9.81
        self.baseline = 0.25
        self.width = 640
        self.height = 480
        # end
        
        self.gtFlow = gtFlow
        self.gtDepth = gtDepth
        self.gtPose = gtPose
        self.useIMU = useIMU
        self.ImageL = TartanAirMonocularDataset(Path(self.seqRoot, "image_left"))
        self.ImageR = TartanAirMonocularDataset(Path(self.seqRoot, "image_right"))
        
        match (useIMU, ips):
            case (True, None):
                self.IMUL = TartanAirIMULoader(Path(self.seqRoot, "imu"))
            case (True, ips_val):
                Logger.write("info", f"Use simulated IMU data with ips={ips_val}")
                self.IMUL = TartanAirIMUSimulator(
                    Path(self.seqRoot, "pose_left.txt"), fps=ips
                )
            case _:
                Logger.write("info", f"IMU DataLoader disabled.")
                self.IMUL = None
        
        if self.IMUL is None:
            self.length = len(self.ImageL)
        else:
            self.length = 0
            while self.length < len(self.ImageL):
                if self.IMUL.cam2imuIdx[self.length].item() == -1:
                    break
                self.length += 1
            else:
                Logger.write(
                    "warn",
                    f"Not all image matched to IMU data, {len(self.ImageL) - self.length} images unmatched",
                )

        if self.gtDepth:
            self.DepthL = TartanAirGTDepthDataset(
                Path(self.seqRoot, "depth_left"), compressed
            )

        if self.gtFlow:
            self.Flow = TartanAirGTFlowDataset(Path(self.seqRoot, "flow"), compressed)
            self.length = len(self.Flow)

        if self.gtPose:
            self.Poses = loadTartanAirGT(Path(self.seqRoot, "pose_left.txt"))
        
        super().__init__(self.length)

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
                local_index,
                self.K,
                self.baseline,
                self.width,
                self.height
            ),
            imageL=self.ImageL[index],
            imageR=self.ImageR[index],
            gtDepth=self.DepthL[index] if self.gtDepth else None,
            gtFlow=self.Flow[index][0] if self.gtFlow and index < len(self.Flow) else None,
            flowMask=self.Flow[index][1] if self.gtFlow else None,
            gtPose=self.Poses[index] if self.gtPose else None,  #type: ignore
            imu=imuData,
        )


class TartanAirIMULoader(Dataset[IMUData]):
    def __init__(self, imuPath: Path) -> None:
        super().__init__()
        assert imuPath.exists(), f"IMU Data path ({imuPath}) does not exist"
        self.imuPath = imuPath

        # IMU Data
        self.lin_acc = torch.tensor(
            np.load(str(Path(imuPath, "accel_left.npy"))), dtype=torch.float
        )  # (N, 3)
        self.rot_vel = torch.tensor(
            np.load(str(Path(imuPath, "gyro_left.npy"))), dtype=torch.float
        )  # (N, 3)
        self.timestamp = torch.tensor(
            np.load(str(Path(imuPath, "imu_time.npy"))), dtype=torch.float
        )  # (N,)
        self.dt = (self.timestamp[1:] - self.timestamp[:-1]).unsqueeze(
            dim=-1
        )  # (N-1, 1)

        # Ground Truth labels
        self.gt_vel = torch.tensor(
            np.load(str(Path(imuPath, "vel_left.npy"))), dtype=torch.float
        )  # (N, 3)
        self.gt_pos = torch.tensor(
            np.load(str(Path(imuPath, "xyz_left.npy"))), dtype=torch.float
        )  # (N, 3)
        angle_left = Rotation.from_euler(
            "XYZ", np.load(str(Path(imuPath, "angles_left.npy"))), degrees=False
        )
        self.gt_rot = torch.tensor(
            angle_left.as_euler("xyz", degrees=False), dtype=torch.float
        )  # (N, 3)

        # Camera alignment
        self.camtime = torch.tensor(np.load(str(Path(imuPath, "cam_time.npy"))))  # (M,)
        self.cam2imuIdx = torch.ones_like(self.camtime, dtype=torch.long) * -1  # (M,)

        self.length = self.lin_acc.size(0) - 1

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
        if imu_idx < self.timestamp.size(0):
            Logger.write(
                "warn",
                f"{self.timestamp.size(0) - imu_idx} IMU samples remain unmatched",
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
            acc=self.lin_acc[start_imu_idx:end_imu_idx],
            gyro=self.rot_vel[start_imu_idx:end_imu_idx],
            gtPos=self.gt_pos[start_imu_idx:end_imu_idx],
            gtVel=self.gt_vel[start_imu_idx:end_imu_idx],
            gtRot=self.gt_rot[start_imu_idx:end_imu_idx],
            initPos=self.gt_pos[start_imu_idx : start_imu_idx + 1],
            initVel=self.gt_vel[start_imu_idx : start_imu_idx + 1],
            initRot=self.gt_rot[start_imu_idx : start_imu_idx + 1],
        )

    def __getitem__(self, index) -> IMUData:
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
            dt=self.dt[index : index + 1],
            acc=self.lin_acc[index : index + 1],
            gyro=self.rot_vel[index : index + 1],
            time=self.timestamp[index : index + 1],
            gtPos=self.gt_pos[index : index + 1],
            gtVel=self.gt_vel[index : index + 1],
            gtRot=self.gt_rot[index : index + 1],
            initPos=self.gt_pos[index : index + 1],
            initVel=self.gt_vel[index : index + 1],
            initRot=self.gt_rot[index : index + 1],
        )


class TartanAirIMUSimulator(Dataset[IMUData]):
    def __init__(self, gtPath: Path, fps=100) -> None:
        super().__init__()
        self.fps = fps
        self.g = np.array([0, 0, 9.81])
        self.gtPath = gtPath
        self.camFPS = 10.0

        raw_poses = np.loadtxt(self.gtPath)
        img_time = np.arange(raw_poses.shape[0]) / self.camFPS
        imu_time, accel_body, vel, pose, gyro, angles, _, _ = (
            self._interpolate_trajectory(img_time, raw_poses)
        )
        # time_interpolate, accel_body, vel, pose, gyro, angles, vel_body, accel

        # IMU Data
        self.lin_acc = torch.tensor(accel_body, dtype=torch.float)  # (N, 3)
        self.rot_vel = torch.tensor(gyro, dtype=torch.float)  # (N, 3)
        self.timestamp = torch.tensor(imu_time, dtype=torch.float)  # (N,)
        self.dt = (self.timestamp[1:] - self.timestamp[:-1]).unsqueeze(
            dim=-1
        )  # (N-1, 1)

        # Ground Truth labels
        self.gt_vel = torch.tensor(vel, dtype=torch.float)  # (N, 3)
        self.gt_pos = torch.tensor(pose, dtype=torch.float)  # (N, 3)
        rotations = Rotation.from_euler("xyz", angles, degrees=False).as_quat(
            canonical=False
        )
        self.gt_rot = torch.tensor(rotations, dtype=torch.float)  # (N, 4)

        # Camera alignment
        self.camtime = torch.tensor(img_time, dtype=torch.float)  # (M,)
        self.cam2imuIdx = torch.ones_like(self.camtime, dtype=torch.long) * -1  # (M,)

        self.length = self.lin_acc.size(0) - 1

        self.alignWithCameraTime()

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
        angles = spline(time_interpolate).as_euler("xyz", degrees=False)  # XYZ
        gyro = spline(time_interpolate, 1)
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
        if imu_idx < self.timestamp.size(0):
            Logger.write(
                "warn",
                f"{self.timestamp.size(0) - imu_idx} IMU samples remain unmatched",
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
            acc=self.lin_acc[start_imu_idx:end_imu_idx],
            gyro=self.rot_vel[start_imu_idx:end_imu_idx],
            gtPos=self.gt_pos[start_imu_idx:end_imu_idx],
            gtVel=self.gt_vel[start_imu_idx:end_imu_idx],
            gtRot=self.gt_rot[start_imu_idx:end_imu_idx],
            initPos=self.gt_pos[start_imu_idx : start_imu_idx + 1],
            initVel=self.gt_vel[start_imu_idx : start_imu_idx + 1],
            initRot=self.gt_rot[start_imu_idx : start_imu_idx + 1],
        )

    def __getitem__(self, index) -> IMUData:
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
            dt=self.dt[index : index + 1],
            acc=self.lin_acc[index : index + 1],
            gyro=self.rot_vel[index : index + 1],
            time=self.timestamp[index : index + 1],
            gtPos=self.gt_pos[index : index + 1],
            gtVel=self.gt_vel[index : index + 1],
            gtRot=self.gt_rot[index : index + 1],
            initPos=self.gt_pos[index : index + 1],
            initVel=self.gt_vel[index : index + 1],
            initRot=self.gt_rot[index : index + 1],
        )


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
        # Black and white mode.
        # return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[..., np.newaxis].repeat(3, axis=-1)
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
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        return flow, mask


def loadTartanAirGT(path: Path) -> pp.LieTensor:
    se3_data = np.loadtxt(str(path))
    return pp.SE3(se3_data)

if __name__ == "__main__":
    import argparse
    from Utility.Config import load_config
    from torch.utils.data import DataLoader

    args = argparse.ArgumentParser()
    args.add_argument("--data", default="Config/Sequence/TartanAir_seaside_000.yaml", type=str)
    args = args.parse_args()
    datacfg, datacfg_dict = load_config(Path(args.data))
    dataset = GenericSequence[SourceDataFrame].instantiate(**vars(datacfg))
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False ,num_workers=1, collate_fn=dataset.collate_fn)
    
    batch: SourceDataFrame
    for batch in dataloader:
        print(batch.imageL.shape, batch.imageR.shape, batch.meta.idx)
