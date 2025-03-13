from types import SimpleNamespace
import torch
import pypose as pp

from typing import Any
# CI/CD PyRight pass cannot cover the DPVO as it requires to compile CUDA kernels
from dpvo.dpvo import DPVO      #type: ignore
from dpvo.config import cfg     #type: ignore
DPVO: Any
cfg : Any

from DataLoader import SequenceBase, StereoFrame
from Module.Map import VisualMap, FrameNode

from .Interface import IOdometry


class DeepPatchVO(IOdometry[StereoFrame]):
    def __init__(self, config_file: str, weight_file: str, height: int, width: int, **kwargs) -> None:
        super().__init__()
        self.config_file = config_file
        self.weight_file = weight_file
        self.height = height
        self.width  = width
        
        self.cfg = cfg
        self.cfg.merge_from_file(self.config_file)
        self.cfg.BUFFER_SIZE = 8192
        
        self.map  = VisualMap()
        self.dpvo = DPVO(self.cfg, self.weight_file, ht=self.height, wd=self.width, viz=False)
        
        self.Ks, self.poses, self.timestep = [], None, None
        self.Ts    = []
        self.T_BSs = []
    
    @classmethod
    def from_config(cls: type["DeepPatchVO"], cfg: SimpleNamespace, seq: SequenceBase[StereoFrame]) -> "DeepPatchVO":
        sample_frame = seq[0]
        return cls(**vars(cfg.Odometry.args), height=sample_frame.stereo.height, width=sample_frame.stereo.width)
    
    @torch.no_grad()
    @torch.inference_mode()
    def run(self, frame: StereoFrame) -> None:
        self.Ks.append(frame.stereo.K)
        self.T_BSs.append(frame.stereo.T_BS)
        self.Ts.append(frame.stereo.frame_ns)
        # NOTE: DPVO will perform /255 operation internally.
        image_cu = frame.stereo.imageL.cuda()[0] * 255
        intrinsic_cu = torch.tensor([frame.stereo.fx, frame.stereo.fy, frame.stereo.cx, frame.stereo.cy], device="cuda")
        self.dpvo(frame.frame_idx, image_cu, intrinsic_cu)
        torch.cuda.empty_cache()
        
    def get_map(self) -> VisualMap:
        return self.map
    
    @torch.no_grad()
    @torch.inference_mode()
    def terminate(self) -> None:
        super().terminate()
        # As per the official DPVO repository on evaluate_tartan.py
        # We use 12 iteration here.
        for _ in range(12):
            self.dpvo.update()
        self.poses, self.timestep = self.dpvo.terminate()
        self.poses = self.poses[..., [2, 0, 1, 5, 3, 4, 6]]
        self.poses = pp.SE3(self.poses)
        
        n_frame = self.poses.size(0)
        self.map.frames.push(FrameNode.init({
            "pose": self.poses,
            "T_BS": torch.stack(self.T_BSs),
            "K"   : torch.cat(self.Ks, dim=0),
            "need_interp": torch.zeros((n_frame,), dtype=torch.bool),
            "time_ns"    : torch.tensor(self.Ts, dtype=torch.long),
        }))
