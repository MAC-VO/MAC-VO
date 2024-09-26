from types import SimpleNamespace
import torch
import pypose as pp

from dpvo.dpvo import DPVO
from dpvo.config import cfg
from DataLoader import GenericSequence, SourceDataFrame
from Module.Map import TensorMap

from .Interface import IVisualOdometry


class DeepPatchVO(IVisualOdometry[SourceDataFrame]):
    def __init__(self, config_file: str, weight_file: str, height: int, width: int, **kwargs) -> None:
        super().__init__()
        self.config_file = config_file
        self.weight_file = weight_file
        self.height = height
        self.width  = width
        
        self.cfg = cfg
        self.cfg.merge_from_file(self.config_file)
        self.cfg.BUFFER_SIZE = 8192
        
        self.map = TensorMap()
        self.dpvo = DPVO(self.cfg, self.weight_file, ht=self.height, wd=self.width, viz=False)
        
        self.Ks, self.poses, self.timestep = [], None, None
    
    @classmethod
    def from_config(cls: type["DeepPatchVO"], cfg: SimpleNamespace, seq: GenericSequence[SourceDataFrame]) -> "DeepPatchVO":
        sample_frame = seq[0]
        return cls(**vars(cfg.Odometry.args), height=sample_frame.meta.height, width=sample_frame.meta.width)
    
    @torch.no_grad()
    @torch.inference_mode()
    def run(self, frame: SourceDataFrame) -> None:
        self.Ks.append(frame.meta.K)
        # NOTE: DPVO will perform /255 operation internally.
        image_cu = frame.imageL.cuda()[0] * 255
        intrinsic_cu = torch.tensor([frame.meta.fx, frame.meta.fy, frame.meta.cx, frame.meta.cy], device="cuda")
        self.dpvo(frame.meta.idx, image_cu, intrinsic_cu)
        torch.cuda.empty_cache()
        
    def get_map(self) -> TensorMap:
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
        
        for idx, pose in enumerate(self.poses):
            self.map.add_frame(self.Ks[idx], pose, 0, None)   #type: ignore
