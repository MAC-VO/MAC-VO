import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

from scipy.spatial.transform import Rotation
from .SequenceBase import GenericSequence
from .Interface import IMUData, MetaInfo, SourceDataFrame
from .TartanAir import TartanAirMonocularDataset, TartanAirIMUSimulator, loadTartanAirGT

from Utility.PrettyPrint import Logger


class TartanAirV2Sequence(GenericSequence[SourceDataFrame]):
    @classmethod
    def name(cls) -> str: return "TartanAirv2"

    def __init__(self, root: str, ips: int | None, 
                 compressed: bool, gtFlow: bool, gtDepth: bool, gtPose: bool, useIMU: bool, **_) -> None:
        # metadata
        self.seqRoot = Path(root)
        self.K = torch.tensor([[320.0, 0.0, 320.0], [0.0, 320.0, 320.0], [0.0, 0.0, 1.0]])
        self.g = 9.81
        self.baseline = 0.25
        self.width = 640
        self.height = 640
        # end

        self.gtFlow=gtFlow
        self.gtDepth=gtDepth
        self.gtPose=gtPose
        self.useIMU = useIMU
        self.ImageL = TartanAirMonocularDataset(Path(self.seqRoot, "image_lcam_front"))
        self.ImageR = TartanAirMonocularDataset(Path(self.seqRoot, "image_rcam_front"))
        
        match (self.useIMU, ips):
            case (True, None):
                self.IMUL = TartanAirv2IMULoader(Path(self.seqRoot, "imu"))
            case (True, ips_val):
                Logger.write("info", f"Using simulated IMU signal with ips={ips_val}")
                self.IMUL = TartanAirIMUSimulator(Path(self.seqRoot, "pose_lcam_front.txt"), fps=ips_val)
            case (False, _):
                Logger.write("info", f"IMU DataLoader disabled.")
                self.IMUL = None

        if self.useIMU and self.IMUL:
            self.length = 0
            while self.length < len(self.ImageL):
                if self.IMUL.cam2imuIdx[self.length].item() == -1:
                    break
                self.length += 1
            else:
                print(
                    f"[Warning]: Not all image matched to IMU data, {len(self.ImageL) - self.length} images unmatched"
                )
        else:
            self.length = len(self.ImageL)
        

        if self.gtDepth:
            from DataLoader.TartanAir import TartanAirGTDepthDataset

            self.DepthL = TartanAirGTDepthDataset(
                Path(self.seqRoot, "depth_lcam_front"), compressed
            )

        if self.gtFlow:
            from DataLoader.TartanAir import TartanAirGTFlowDataset

            self.Flow = TartanAirGTFlowDataset(
                Path(self.seqRoot, "flow_lcam_front"), compressed
            )
            self.length = len(self.Flow)

        if self.gtPose:
            self.Poses = loadTartanAirGT(Path(self.seqRoot, "pose_lcam_front.txt"))
        
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
                K=self.K,
                baseline=self.baseline,
                width=self.width,
                height=self.height,
            ),
            imageL=self.ImageL[index],
            imageR=self.ImageR[index],
            gtDepth=self.DepthL[index] if self.gtDepth else None,
            gtFlow=self.Flow[index][0] if self.gtFlow and index < len(self.Flow) else None,
            flowMask=self.Flow[index][1] if self.gtFlow else None,
            gtPose=self.Poses[index] if self.gtPose else None,  #type: ignore
            imu=imuData,
        )


class TartanAirv2IMULoader(Dataset[IMUData]):
    def __init__(self, imuPath: Path) -> None:
        super().__init__()
        assert imuPath.exists(), f"IMU Data path ({imuPath}) does not exist"
        self.imuPath = imuPath

        # IMU Data
        self.lin_acc = torch.tensor(np.load(str(Path(imuPath, "acc.npy"))), dtype=torch.float)  # (N, 3)
        self.rot_vel = torch.tensor(np.load(str(Path(imuPath, "gyro.npy"))), dtype=torch.float)  # (N, 3)
        self.timestamp = torch.tensor(np.load(str(Path(imuPath, "imu_time.npy"))), dtype=torch.float)  # (N,)
        self.dt = (self.timestamp[1:] - self.timestamp[:-1]).unsqueeze(dim=-1)  # (N-1, 1)

        # Ground Truth labels
        self.gt_vel = torch.tensor(np.load(str(Path(imuPath, "vel_body.npy"))), dtype=torch.float)  # (N, 3)
        self.gt_pos = torch.tensor(np.load(str(Path(imuPath, "pos_global.npy"))), dtype=torch.float)  # (N, 3)
        angle_left = Rotation.from_euler("XYZ", np.load(str(Path(imuPath, "ori_global.npy"))), degrees=False)   #type: ignore
        self.gt_rot = torch.tensor(angle_left.as_euler('xyz', degrees=False), dtype=torch.float) # (N, 3)

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
            Logger.write("warn", f"{self.timestamp.size(0) - imu_idx} IMU samples remain unmatched")

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
        assert start_imu_idx != -1 and end_imu_idx != -1, "Requested frame is not aligned with IMU Sequence"
        return IMUData(
            dt=self.dt[start_imu_idx:end_imu_idx],
            time=self.timestamp[start_imu_idx:end_imu_idx],
            acc=self.lin_acc[start_imu_idx:end_imu_idx],
            gyro=self.rot_vel[start_imu_idx:end_imu_idx],

            gtPos=self.gt_pos[start_imu_idx:end_imu_idx],
            gtVel=self.gt_vel[start_imu_idx:end_imu_idx],
            gtRot=self.gt_rot[start_imu_idx:end_imu_idx],

            initPos=self.gt_pos[start_imu_idx:start_imu_idx + 1],
            initVel=self.gt_vel[start_imu_idx:start_imu_idx + 1],
            initRot=self.gt_rot[start_imu_idx:start_imu_idx + 1]
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
            dt=self.dt[index:index + 1],
            acc=self.lin_acc[index:index + 1],
            gyro=self.rot_vel[index:index + 1],
            time=self.timestamp[index:index + 1],

            gtPos=self.gt_pos[index:index + 1],
            gtVel=self.gt_vel[index:index + 1],
            gtRot=self.gt_rot[index:index + 1],

            initPos=self.gt_pos[index:index + 1],
            initVel=self.gt_vel[index:index + 1],
            initRot=self.gt_rot[index:index + 1]
        )
