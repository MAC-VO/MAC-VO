from types import SimpleNamespace
import torch

from rich.columns import Columns
from rich.panel import Panel

import Module
from DataLoader.SequenceBase import GenericSequence
from DataLoader import SourceDataFrame
from Module.Map import BatchFrame, BatchObservation, BatchPoints, TensorMap
from Utility.Point import filterPointsInRange, pixel2point_NED
from Utility.PrettyPrint import Logger, GlobalConsole
from Utility.Visualizer import PLTVisualizer
from Utility.Extensions import ConfigTestable

from .Interface import IVisualOdometry

class MACVO(IVisualOdometry[SourceDataFrame], ConfigTestable):
    def __init__(
        self,
        device, num_point, edgewidth, match_cov_default, profile,
        frontend        : Module.IFrontend, 
        motion_model    : Module.IMotionModel,
        kp_selector     : Module.IKeypointSelector,
        obs_filter      : Module.IObservationFilter,
        obs_covmodel    : Module.IObservationCov,
        post_process    : Module.IMapProcessor,
        kf_selector     : Module.IKeyframeSelector,
        optimizer       : Module.IOptimizer,
        **_excessive_args,
    ) -> None:
        super().__init__(profile=profile)
        if len(_excessive_args) > 0:
            Logger.write("warn", f"Receive excessive arguments for __init__ {_excessive_args}, update/clean up your config!")
        
        self.gmap = TensorMap()
        self.device = device
        self.match_cov_default = match_cov_default

        # Modules
        self.Frontend = frontend
        self.MotionEstimator = motion_model
        self.KeypointSelector = kp_selector
        self.OutlierFilter = obs_filter
        self.ObsCovModel = obs_covmodel
        self.MapRefiner = post_process
        self.KeyframeSelector = kf_selector
        self.Optimizer = optimizer
        # end

        self.min_num_point = 10
        self.num_point = num_point
        self.edge_width = edgewidth
        
        # Context
        self.prev_frame: SourceDataFrame | None = None
        self.prev_handle: int | None = None
        self.prev_depth_map: torch.Tensor | None = None
        self.prev_depth_map_cov: torch.Tensor | None = None

        self.report_config()
    
    @classmethod
    def from_config(cls: type["MACVO"], cfg: SimpleNamespace, seq: GenericSequence[SourceDataFrame]) -> "MACVO":
        odomcfg = cfg.Odometry
        # Initialize modules for VO
        Frontend            = Module.IFrontend.instantiate(odomcfg.frontend.type, odomcfg.frontend.args)
        MotionEstimator     = Module.IMotionModel.instantiate(odomcfg.motion.type, odomcfg.motion.args)
        KeypointSelector    = Module.IKeypointSelector.instantiate(odomcfg.keypoint.type, odomcfg.keypoint.args)
        ObservationFilter   = Module.IObservationFilter.instantiate(odomcfg.outlier.type, odomcfg.outlier.args)
        ObserveCovModel     = Module.IObservationCov.instantiate(odomcfg.cov.obs.type, odomcfg.cov.obs.args)
        MapRefiner          = Module.IMapProcessor.instantiate(odomcfg.postprocess.type, odomcfg.postprocess.args)
        KeyframeSelector    = Module.IKeyframeSelector.instantiate(odomcfg.keyframe.type, odomcfg.keyframe.args)
        Optimizer = Module.IOptimizer.instantiate(odomcfg.optimizer.type, odomcfg.optimizer.args)
        
        return cls(
            frontend=Frontend,
            motion_model=MotionEstimator,
            kp_selector=KeypointSelector,
            obs_filter=ObservationFilter,
            obs_covmodel=ObserveCovModel,
            post_process=MapRefiner,
            kf_selector=KeyframeSelector,
            optimizer=Optimizer,
            **vars(odomcfg.args),
        )
    
    def report_config(self):
        # Cute fine-print boxes
        box1 = Panel.fit(
            "\n".join(
                [
                    f"DepthEstimator cov: {self.Frontend.provide_cov[0]}",
                    f"MatchEstimator cov: {self.Frontend.provide_cov[1]}",
                    f"Observation cov:    {self.ObsCovModel.__class__.__name__}",
                ]
            ),
            title="Odometry Covariance",
            title_align="left",
        )
        box2 = Panel.fit(
            "\n".join(
                [
                    f"Frontend        -'{self.Frontend        .__class__.__name__}'",
                    f"MotionEstimator -'{self.MotionEstimator .__class__.__name__}'",
                    f"KeypointSelector-'{self.KeypointSelector.__class__.__name__}'",
                    f"OutlierFilter   -'{self.OutlierFilter   .__class__.__name__}'",
                    f"MapRefiner      -'{self.MapRefiner      .__class__.__name__}'",
                ]
            ),
            title="Odometry Modules",
            title_align="left",
        )
        GlobalConsole.print(Columns([box1, box2]))

    def new_keypoint(self, 
                     data_frame: SourceDataFrame, 
                     map_frame: BatchFrame,
                     depth_map: torch.Tensor,
                     depth_map_cov: torch.Tensor | None,
                     flow_map_cov : torch.Tensor | None) -> BatchObservation:
        """
        Generate new keypoint and register to TensorMap.

        Note
        ---
        Requires `self.DepthEstimator` to be initialized correctly. Will not call
        `self.DepthEstimator.receive_stereo(curr_frame)` internally.
        """
        assert len(map_frame) == 1
        assert map_frame.frame_idx is not None

        kp_uv = self.KeypointSelector.select_point(
            data_frame, self.num_point, depth_map, depth_map_cov, flow_map_cov 
        )
        num_kp = kp_uv.size(0)
        
        kp_d     = self.Frontend.retrieve_pixels(kp_uv, depth_map).squeeze(0)
        kp_d_cov = self.Frontend.retrieve_pixels(kp_uv, depth_map_cov)
        kp_d_cov = kp_d_cov.squeeze(0) if kp_d_cov is not None else None
        
        if kp_uv.size(0) == 0:
            # NOTE: Refer to https://github.com/pypose/pypose/issues/342
            kp_3d = torch.empty((0, 3))
        else:
            kp_3d = map_frame.pose[0].Act(pixel2point_NED(kp_uv, kp_d, data_frame.meta.K).cpu())   #type: ignore
        
        kp_covTc = self.ObsCovModel.estimate(
            data_frame, kp_uv, depth_map=depth_map, depth_cov_map=depth_map_cov,
            depth_cov=kp_d_cov, flow_cov=None,
        )

        # Record color of keypoints for visualization
        kp_uv_cpu = kp_uv.cpu()
        kp_color = data_frame.imageL[..., kp_uv_cpu[..., 1], kp_uv_cpu[..., 0]].squeeze(0).permute(1, 0)
        kp_color = (kp_color * 255).to(torch.uint8)
        
        # Convert covariance to world coordinate
        est_R = map_frame.pose.rotation().matrix().repeat((num_kp, 1, 1)).double()
        kp_covTw = torch.bmm(torch.bmm(est_R, kp_covTc), est_R.transpose(1, 2))
        
        # Register points and observations
        kp_idx = self.gmap.points.push(BatchPoints(kp_3d, kp_covTw, kp_color))
        obs = BatchObservation(
            point_idx=kp_idx,
            frame_idx=map_frame.frame_idx[0].repeat((num_kp,)),
            pixel_uv=kp_uv_cpu, pixel_d=kp_d, cov_Tc=kp_covTc,
            cov_pixel_uv=torch.ones((num_kp, 2)) * self.match_cov_default,
            cov_pixel_d=(torch.ones((num_kp,)) * -1) if (kp_d_cov is None) else kp_d_cov
        )
        obs = obs[self.OutlierFilter.filter(obs)]
        self.gmap.add_observation(obs)
        
        return obs

    def match_keypoint(self,
                       to_frame_data: SourceDataFrame,
                       orig_obs: BatchObservation,
                       to_frame: BatchFrame,
                       depth_map    : torch.Tensor,
                       depth_map_cov: torch.Tensor | None,
                       flow_map     : torch.Tensor,
                       flow_map_cov : torch.Tensor | None):
        """
        Given a set of observations, match these observations to corresponding kps
        in other frame.

        Note
        ---
        Requires `self.DepthEstimator` and `self.OutlierFilter` 
        to be initialized properly.
        `self.DepthEstimator` receives the SourceDataFrame of latest frame
        """
        assert len(to_frame) == 1
        assert to_frame.frame_idx is not None
        meta = to_frame_data.meta

        kp2_uv_old  = orig_obs.pixel_uv.to(flow_map.device)
        kp2_uv      = kp2_uv_old + self.Frontend.retrieve_pixels(kp2_uv_old, flow_map).T
        kp2_uv_cov  = self.Frontend.retrieve_pixels(kp2_uv_old, flow_map_cov)
        kp2_uv_cov  = kp2_uv_cov.T if kp2_uv_cov is not None else None
        
        inbound_mask = filterPointsInRange(kp2_uv, 
                                       (self.edge_width, meta.width - self.edge_width),
                                       (self.edge_width, meta.height - self.edge_width))

        orig_obs    = orig_obs[inbound_mask.cpu()]
        kp2_uv      = kp2_uv[inbound_mask]
        kp2_uv_cov  = None if (kp2_uv_cov is None) else kp2_uv_cov[inbound_mask]
        
        num_points = len(orig_obs)
        kp2_d      = self.Frontend.retrieve_pixels(kp2_uv, depth_map).squeeze(0)
        kp2_d_cov  = self.Frontend.retrieve_pixels(kp2_uv, depth_map_cov)
        kp2_d_cov  = kp2_d_cov.squeeze(0) if kp2_d_cov is not None else None

        obs_covTc = self.ObsCovModel.estimate(
            to_frame_data, kp2_uv,
            depth_map=depth_map, depth_cov_map=depth_map_cov,
            depth_cov=kp2_d_cov, flow_cov=kp2_uv_cov,
        )
        
        obs = BatchObservation(
            point_idx=orig_obs.point_idx,
            frame_idx=to_frame.frame_idx[0].repeat((kp2_uv.size(0),)),
            pixel_uv=kp2_uv,
            pixel_d=kp2_d,
            cov_Tc=obs_covTc,
            cov_pixel_uv=(torch.ones((num_points, 2)) * self.match_cov_default) if (kp2_uv_cov is None) else kp2_uv_cov,
            cov_pixel_d=(torch.ones((num_points,)) * -1) if (kp2_d_cov is None) else kp2_d_cov
        )
        obs = obs[self.OutlierFilter.filter(obs)]
        self.gmap.add_observation(obs)
        
        return obs
    
    def epilog(self, curr_frame: SourceDataFrame, depth_map: torch.Tensor, depth_map_cov: torch.Tensor | None) -> None:
        self.prev_frame = curr_frame
        self.prev_handle = len(self.gmap.frames) - 1
        self.prev_depth_map = depth_map
        self.prev_depth_map_cov = depth_map_cov
        
    def initialize(self, curr_frame: SourceDataFrame) -> None:
        # Estimate Depth
        depth_map, depth_map_cov, _, _ = self.Frontend(None, curr_frame)
        est_pose = self.MotionEstimator.predict(curr_frame, None, depth_map)
        
        # num_obs is set to zero since this will be updated in new_keypoint later
        self.OutlierFilter.set_meta(curr_frame.meta)
        self.gmap.add_frame(curr_frame.meta.K, est_pose, 0, None)
        self.MotionEstimator.update(self.gmap.frames[-1].squeeze().pose)
        
        # Store context
        self.epilog(curr_frame, depth_map, depth_map_cov)

    def run(self, frame: SourceDataFrame) -> None:
        if self.prev_frame is None:
            return self.initialize(frame)
        assert self.prev_handle is not None
        assert self.prev_depth_map is not None
        
        if not self.KeyframeSelector.isKeyframe(frame):
            self.gmap.add_frame(frame.meta.K, self.gmap.frames[self.prev_handle].pose, 0, None, 
                                flag=BatchFrame.FLAG_NEED_INTERP)
            return
    
        # Frontend inference
        depth_map, depth_map_cov, flow_map, flow_map_cov = self.Frontend(self.prev_frame, frame)
        
        # NOTE: should always writeback optimized pose to global map before selecting new keypoints (register
        # new 3D point) on that frame.
        self.Optimizer.write_map(self.gmap)
        
        # Update motion model (this must be after write_back to get latest result)
        self.MotionEstimator.update(self.gmap.frames[self.prev_handle].squeeze().pose)
        
        # Make new prediction
        est_pose = self.MotionEstimator.predict(frame, flow_map, depth_map)
        
        # We will use the estimated matching quality between frame (t-1) and frame t
        # to select new keypoints on frame (t-1).
        prev_obs = self.new_keypoint(self.prev_frame, self.gmap.frames[self.prev_handle], 
                                     self.prev_depth_map, self.prev_depth_map_cov, flow_map_cov)
        
        self.gmap.add_frame(frame.meta.K, est_pose, 0, None)
        curr_frame_map = self.gmap.frames[-1]
        
        # Update keypoints from previous frame to current frame
        new_obs = self.match_keypoint(frame, prev_obs, curr_frame_map, depth_map, depth_map_cov, flow_map, flow_map_cov)
        
        # Visualizer start
        PLTVisualizer.visualize_Obs("observation", self.prev_frame.imageL, frame.imageL,
                                    new_obs,
                                    depth_map_cov, flow_map_cov, None)
        # Visualizer end
        
        if len(new_obs) > 0:
            curr_frame_map.quality[0] = new_obs.cov_Tc.det().mean()
        else:
            curr_frame_map.quality[0] = -1
        
        if len(new_obs) < self.min_num_point:
            Logger.write("error", f"VOLostTrack @ {frame.meta.idx} - only get {len(new_obs)} observations")

            curr_frame_map.flag |= curr_frame_map.FLAG_VO_LOSTTRACK
            self.gmap.frames.update(curr_frame_map, self.gmap.frames.Scatter.FLAG | self.gmap.frames.Scatter.QUALITY)
            
            self.epilog(frame, depth_map, depth_map_cov)
            return
        
        # Construct graph optimization problem and execute optimization
        self.Optimizer.optimize(self.gmap, [-1])
        
        self.epilog(frame, depth_map, depth_map_cov)
    
    def terminate(self) -> None:
        super().terminate()
        self.Optimizer.write_map(self.gmap)
        self.Optimizer.terminate()
        
        self.gmap, _ = self.MapRefiner.elaborate_map(self.gmap)

    def get_map(self) -> TensorMap:
        return self.gmap
    
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        assert config is not None
        Module.IKeyframeSelector.is_valid_config(config.keyframe)
        Module.IMapProcessor.is_valid_config(config.postprocess)
        Module.IObservationFilter.is_valid_config(config.outlier)
        Module.IMotionModel.is_valid_config(config.motion)
        Module.IKeypointSelector.is_valid_config(config.keypoint)
        Module.IObservationCov.is_valid_config(config.cov.obs)
        Module.IFrontend.is_valid_config(config.frontend)
        
        cls._enforce_config_spec(config.args, {
            "device"            : lambda s: isinstance(s, str) and (("cuda" in s) or (s == "cpu")),
            "num_point"         : lambda b: isinstance(b, int) and b > 0, 
            "edgewidth"         : lambda b: isinstance(b, int) and b > 0, 
            "match_cov_default" : lambda b: isinstance(b, (float, int)) and b > 0.0, 
            "profile"           : lambda b: isinstance(b, bool),
        })

