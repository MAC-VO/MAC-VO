import cv2
import torch
import numpy as np
import pypose as pp
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from torch.utils.data import Dataset

from ..Interface import StereoFrame, StereoData
from ..SequenceBase import SequenceBase

EDN2NED = pp.from_matrix(torch.tensor([
    [0., 0., 1., 0.],
    [1., 0., 0., 0.],
    [0., 1., 0., 0.],
    [0., 0., 0., 1.],
]), pp.SE3_type)
NED2EDN = EDN2NED.Inv()


class KITTI_StereoSequence(SequenceBase[StereoFrame]):
    @classmethod
    def name(cls) -> str: return "KITTI"
    
    def __init__(self, config: SimpleNamespace | dict[str, Any], **_):
        cfg = self.config_dict2ns(config)
        
        self.root = Path(cfg.root)
        self.sequence_name = self.root.name
        
        self.imageL = KITTIMonocularDataset(Path(self.root, "image_2"))
        self.imageR = KITTIMonocularDataset(Path(self.root, "image_3"))
        assert len(self.imageL) == len(self.imageR)

        if cfg.gt_pose:
            self.gtPose_data = loadKITTIGTPoses(Path(self.root.parent.parent, "poses", self.sequence_name + ".txt"))
        else:
            self.gtPose_data = None
        
        with open(Path(self.root, "calib.txt"), "r") as f:
            lines = f.read().strip().splitlines()
            P2    = np.array(list(map(float, lines[2][4:].split(" "))))
            P2.resize((3, 4))
            self.cam2_K_np, self.cam2_R, self.cam2_t, _, _, _, _ = cv2.decomposeProjectionMatrix(P2)
            self.cam2_t = self.cam2_t[:3] / self.cam2_t[3]
            self.cam2_K = torch.tensor(self.cam2_K_np).float().unsqueeze(0)
            
            P3    = np.array(list(map(float, lines[3][4:].split(" "))))
            P3.resize((3, 4))
            self.cam3_K_np, self.cam3_R, self.cam3_t, _, _, _, _ = cv2.decomposeProjectionMatrix(P3)
            self.cam3_t = self.cam3_t[:3] / self.cam3_t[3]
        
        self.baseline = np.linalg.norm(self.cam2_t - self.cam3_t).item()
        T_BS_ext      = np.eye(4)[np.newaxis, ...]
        T_BS_ext[0, :3, :3] = self.cam2_R
        T_BS_ext[0, :3,  3] = self.cam2_t[..., 0]
        self.T_BS = pp.from_matrix(T_BS_ext, pp.SE3_type).float() @ NED2EDN.unsqueeze(0)
        
        super().__init__(len(self.imageL))

    def __getitem__(self, local_index: int) -> StereoFrame:
        index = self.get_index(local_index)
        imageL = self.imageL[index]
        return StereoFrame(
            stereo=StereoData(
                T_BS    = cast(pp.LieTensor, self.T_BS),
                K       = self.cam2_K,
                baseline= torch.tensor([self.baseline]),
                time_ns = [self.imageL.cam_timestamps[index]],
                height  = imageL.size(2),
                width   = imageL.size(3),
                imageL  = imageL,
                imageR  = self.imageR[index]
            ),
            idx=[local_index],
            time_ns=[self.imageL.cam_timestamps[index]],
            gt_pose= None if self.gtPose_data is None else cast(pp.LieTensor, self.gtPose_data[index].unsqueeze(0))
        )
    
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "root"   : lambda v: isinstance(v, str),
            "gt_pose": lambda b: isinstance(b, bool)
        })


class KITTIMonocularDataset(Dataset):
    def __init__(self, image_path: Path) -> None:
        super().__init__()
        self.file_names = sorted([file for file in image_path.glob("*.png")])
        self.length = len(self.file_names)
        self.cam_timestamps = (np.loadtxt(
            Path(image_path, "..", "times.txt"), delimiter=" ", dtype=np.float64
        ) * 1_000_000_000).astype(np.int64)
    
    def __len__(self) -> int: return self.length
    
    def __getitem__(self, index) -> torch.Tensor:
        image = cv2.imread(str(self.file_names[index]), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = torch.tensor(image, dtype=torch.float).permute(2, 0, 1).unsqueeze(0)
        image_tensor /= 255.
        return image_tensor


def loadKITTIGTPoses(pose_dir: Path) -> pp.LieTensor:
    poses = torch.tensor(np.loadtxt(pose_dir)).reshape((-1, 3, 4))
    
    padding = torch.tensor([[[0., 0., 0., 1.]]]).repeat(poses.size(0), 1, 1)
    poses = torch.cat([poses, padding], dim=1)
    poses = pp.from_matrix(poses.float(), pp.SE3_type)
    return poses
