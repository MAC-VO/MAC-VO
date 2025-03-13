import cv2
import torch
import numpy as np
import pypose as pp

from pathlib import Path
from torch.utils.data import Dataset

from types import SimpleNamespace
from typing import Any, cast
import yaml

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

class VBRMonocularDataset(Dataset):
    def __init__(self, image_path: Path, K: np.ndarray, undistort: np.ndarray, T_BS: np.ndarray) -> None:
        super().__init__()
        self.K = K
        self.T_BS = T_BS
        self.distort_factor = undistort
        self.undistort_map: None | tuple[np.ndarray, np.ndarray] = None
        
        self.file_names = sorted([file for file in image_path.glob("*.png")])
        self.length     = len(self.file_names)
        self.cam_stamps = (np.loadtxt(
            Path(image_path,"..", "timestamps.txt"), delimiter=" ", dtype="str"
        )).astype("datetime64[ns]").astype(np.int64)
    
    def apply_mask(self, cam_mask: np.ndarray):
        self.file_names = [f for idx, f in enumerate(self.file_names) if (idx < cam_mask.shape[0]) and cam_mask[idx].item()]
        self.file_names.sort()
        self.length = len(self.file_names)
        self.cam_stamps = self.cam_stamps[cam_mask]
    
    def __len__(self) -> int: return self.length
    
    def __getitem__(self, index) -> torch.Tensor:
        image = cv2.imread(str(self.file_names[index]), cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.correct_distortion(image)
        image_tensor = torch.tensor(image, dtype=torch.float).permute(2, 0, 1).unsqueeze(0)
        image_tensor /= 255.
        return image_tensor
    
    def correct_distortion(self, image: np.ndarray) -> np.ndarray:
        if self.undistort_map is None:
            raise Exception("Monocular sequence is not rectified.")
        else:
            undistorted_image = cv2.remap(image, self.undistort_map[0], self.undistort_map[1], cv2.INTER_LINEAR)
        return undistorted_image

class VBR_StereoSequence(SequenceBase[StereoFrame]):
    @classmethod
    def name(cls) -> str: return "VBR_Stereo"
    
    def __init__(self, config: SimpleNamespace | dict[str, Any], **_):
        cfg = self.config_dict2ns(config)
        
        self.root = Path(cfg.root)
        self.sequence_name = self.root.name
        
        self.width = 1388
        self.high = 700
        with open(Path(self.root, "vbr_calib.yaml"), "r") as f:
                calib_data = yaml.safe_load(f)
                # left camera
                cam_l =calib_data["cam_l"]
                distortion_l = np.array(cam_l["distortion_coeffs"])
                intrinsics_l = cam_l["intrinsics"] ## intrinsics: [Fx, Fy, Cx, Cy]
                self.cam_l_K_np = np.array([[intrinsics_l[0],   0.0,               intrinsics_l[2]],
                                            [0.0,               intrinsics_l[1], intrinsics_l[3]],
                                            [0.0,               0.0,               1]])
                cam_l_T = np.array(cam_l["T_b"])
                self.cam_l_t = cam_l_T[:3, 3]
                self.cam_l_R = cam_l_T[:3, :3]
                self.cam_l_K = torch.tensor(self.cam_l_K_np).float().unsqueeze(0)
                
                # right camera
                cam_r = calib_data["cam_r"]
                distortion_r = np.array(cam_r["distortion_coeffs"])
                intrinsics_l = cam_r["intrinsics"] ## intrinsics: [Fx, Fy, Cx, Cy]
                self.cam_r_K_np = np.array([[intrinsics_l[0], 0.0,               intrinsics_l[2]],
                                            [0.0,               intrinsics_l[1], intrinsics_l[3]],
                                            [0.0,               0.0,               1]])
                cam_r_T = np.array(cam_r["T_b"])
                self.cam_r_t = cam_r_T[:3, 3]
                self.cam_r_R = cam_r_T[:3, :3]
                self.cam_r_K = torch.tensor(self.cam_r_K_np).float().unsqueeze(0)
            
        # TODO: check the undistort
        self.imageL = VBRMonocularDataset(Path(self.root, "camera_left","data"), K=self.cam_l_K_np, T_BS=cam_l_T, undistort=distortion_l)
        self.imageR = VBRMonocularDataset(Path(self.root, "camera_right","data"), K=self.cam_r_K_np, T_BS=cam_r_T, undistort=distortion_r)
        
        rectified_K = self.sync_LR(self.imageL, self.imageR)
        self.K = torch.tensor(rectified_K[:3, :3], dtype=torch.float).unsqueeze(0)
        self.cam_stamps = self.imageL.cam_stamps
        assert self.imageL.length == self.imageR.length

        if cfg.gt_pose:
            self.gtPose_data, self.cam_time_mask = loadVBRGTPoses(Path(self.root, self.sequence_name + "_gt.txt"), self.imageL.cam_stamps)
            self.imageL.apply_mask(self.cam_time_mask)
            self.imageR.apply_mask(self.cam_time_mask)            
        else:
            self.gtPose_data = None
        
  
        self.baseline = np.linalg.norm(self.cam_l_t - self.cam_r_t).item()
        T_BS_ext = cam_l_T[np.newaxis, ...]
        self.T_BS = pp.from_matrix(T_BS_ext, pp.SE3_type).float() @ NED2EDN.unsqueeze(0)
        
        super().__init__(self.imageL.length)

    def __getitem__(self, local_index: int) -> StereoFrame:
        index = self.get_index(local_index)
        imageL = self.imageL[index]
        return StereoFrame(
            stereo=StereoData(
                T_BS    = cast(pp.LieTensor, self.T_BS),
                K       = self.cam_l_K,
                baseline= torch.tensor([self.baseline]),
                time_ns = [self.imageL.cam_stamps[index]],
                height  = imageL.size(2),
                width   = imageL.size(3),
                imageL  = imageL,
                imageR  = self.imageR[index]
            ),
            time_ns = [self.imageL.cam_stamps[index]],
            idx=[local_index],
            gt_pose= None if self.gtPose_data is None else cast(pp.LieTensor, self.gtPose_data[index].unsqueeze(0))
        )

    @staticmethod
    def unique_sync_mask(stamps: np.ndarray, common_time: np.ndarray) -> np.ndarray:
        _, first_indices = np.unique(stamps, return_index=True)
        mask = np.zeros_like(stamps, dtype=bool)
        mask[first_indices] = np.isin(stamps[first_indices], common_time)
        return mask
    
    @staticmethod
    def sync_LR(left: VBRMonocularDataset, right: VBRMonocularDataset) -> np.ndarray:
        # Constant - Transformation from cam 1 to cam 2 (L -> R)
        T_LR = np.linalg.inv(right.T_BS) @ left.T_BS
        
        # Align timestamps, discard time stamp with only Left/Right image.
        left_time = {t_l.item() for t_l in left.cam_stamps}
        right_time = {t_r.item() for t_r in right.cam_stamps}
        common_time = np.intersect1d(left.cam_stamps, right.cam_stamps)

        left_sync_mask = VBR_StereoSequence.unique_sync_mask(left.cam_stamps, common_time)
        right_sync_mask = VBR_StereoSequence.unique_sync_mask(right.cam_stamps, common_time)
        
        left.cam_stamps = left.cam_stamps[left_sync_mask]
        right.cam_stamps = right.cam_stamps[right_sync_mask]
        left.file_names = [f for idx, f in enumerate(left.file_names) if left_sync_mask[idx].item()]
        right.file_names = [f for idx, f in enumerate(right.file_names) if right_sync_mask[idx].item()]
        left.length = len(left.file_names)
        right.length = len(right.file_names)
        
        # Rectify stereo and undistort based on Left and Right camera.
        R1, R2, P1, P2, Q, validRoi1, validRoi2 = cv2.stereoRectify(left.K, left.distort_factor, 
                          right.K, right.distort_factor, (1388, 700),
                          T_LR[:3, :3], T_LR[:3, 3], flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1)

        left.undistort_map = cv2.initUndistortRectifyMap(left.K, left.distort_factor, R1, P1,  (1388, 700), cv2.CV_32FC1)
        right.undistort_map = cv2.initUndistortRectifyMap(right.K, right.distort_factor, R2, P2,  (1388, 700), cv2.CV_32FC1)
        left.K = P1[:3, :3]
        right.K = P2[:3, :3]
        
        return P1

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "root"   : lambda v: isinstance(v, str),
            "gt_pose": lambda b: isinstance(b, bool)
        })



def loadVBRGTPoses(pose_dir: Path, cam_time: np.ndarray)  -> tuple[pp.LieTensor, np.ndarray]:
    ##timestamp tx ty tz qx qy qz qw
    gt_data = np.loadtxt(pose_dir)
    pose_time = gt_data[:,0] *  1_000_000_000 
    
    pose_SE3 = pp.SE3(torch.tensor(gt_data[:,1:]))
    cam_time_mask = (cam_time > pose_time[0]) & (cam_time < pose_time[-1])
    bodyPose_SE3, _ = interpolate_pose(pose_SE3, torch.tensor(pose_time), torch.tensor(cam_time[cam_time_mask]))
   
    return bodyPose_SE3, cam_time_mask
