import torch
from DataLoader import SourceDataFrame, GenericSequence
from Module.Map import TensorMap, BatchFrame
from types import SimpleNamespace

from .Interface import IVisualOdometry
from Module import IMatcher, IStereoDepth, IKeyframeSelector, IMapProcessor
from Module.MotionModel import TartanMotionNet
from Utility.Extensions import ConfigTestableSubclass


class TartanVO(IVisualOdometry[SourceDataFrame], ConfigTestableSubclass):
    def __init__(self, match_estimator: IMatcher, depth_estimator: IStereoDepth, kf_selector: IKeyframeSelector, tvo_cfg):
        super().__init__()
        self.gmap = TensorMap()
        
        self.tartanvo = TartanMotionNet(tvo_cfg)
        
        self.match_estimator = match_estimator
        self.depth_estimator = depth_estimator
        self.keyframe_select = kf_selector
        self.map_refiner = IMapProcessor.instantiate("Naive", None)
        self.prev_frame = None
    
    @classmethod
    def from_config(cls: type["TartanVO"], cfg: SimpleNamespace, seq: GenericSequence[SourceDataFrame]) -> "TartanVO":
        match_estimator   = IMatcher.instantiate(cfg.match.type, cfg.match.args)
        depth_estimator   = IStereoDepth.instantiate(cfg.depth.type, cfg.depth.args)
        keyframe_selector = IKeyframeSelector.instantiate(cfg.keyframe.type, cfg.keyframe.args)
        
        return cls(match_estimator=match_estimator, depth_estimator=depth_estimator, kf_selector=keyframe_selector,
                   tvo_cfg=cfg.tartanvo)
    
    @torch.no_grad()
    @torch.inference_mode()
    def run(self, frame: SourceDataFrame) -> None:
        if not self.keyframe_select.isKeyframe(frame):
            self.gmap.add_frame(frame.meta.K, 
                                self.gmap.frames.pose[-1],  #type: ignore
                                0, None, 
                                flag=BatchFrame.FLAG_NEED_INTERP)
            return
        
        if self.prev_frame is not None:
            flow_map, _ = self.match_estimator(self.prev_frame, frame)
        else:
            flow_map = None
        
        depth_map, _ = self.depth_estimator(frame)
        
        est_pose = self.tartanvo.predict(frame, flow_map, depth_map)
        self.gmap.add_frame(frame.meta.K, est_pose, 0, None)
        self.tartanvo.update(est_pose)
        self.prev_frame = frame
    
    def get_map(self) -> TensorMap:
        return self.gmap

    def terminate(self) -> None:
        super().terminate()
        self.gmap, _ = self.map_refiner.elaborate_map(self.gmap)

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        assert config is not None
        IMatcher.is_valid_config(config.match)
        IStereoDepth.is_valid_config(config.depth)
        IKeyframeSelector.is_valid_config(config.keyframe)
        TartanMotionNet.is_valid_config(config.tartanvo)
