# pyright: reportAttributeAccessIssue=none, reportPrivateImportUsage=none
from __future__ import annotations
import traceback
from io import BytesIO
from pathlib import Path
from typing import Callable, Optional, TypeVar, ParamSpec, Concatenate, Generic, Any, Sequence
from matplotlib.figure import Figure, FigureBase, Axes
from matplotlib.colors import Normalize
from functools import wraps
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import pypose as pp
import rerun as rr
import torch

import Utility.Plot as Plot
from Utility.PrettyPrint import Logger
from Module.Map import TensorMap, BatchObservation


I = ParamSpec("I")
O = TypeVar("O")

class RerunVisualizer:
    class State(Enum):
        INACTIVE = 0
        ONLINE = 1
        OFFLINE = 2
    status: State = State.INACTIVE
    context: dict[str, Any] = {
        "LABEL_TO_CLASSID": dict()
    }

    @staticmethod
    def setup(project_name: str, interact_mode: bool, save_rrd_path: Optional[Path]=None, useRR=True, show_plots=True) -> None:
        match (interact_mode, save_rrd_path, useRR):
            case (_, _, False):
                Logger.write("warn", "Rerun visualizer not active.")
            case (True, _, True):
                Logger.write("info", "Rerun visualizer run in interactive mode")
                RerunVisualizer.status = RerunVisualizer.State.ONLINE
                
                rr.init(project_name, spawn=True)
                rr.connect()
                rr.log(
                    "/",
                    rr.ViewCoordinates(xyz=rr.ViewCoordinates.FRD),
                    timeless=True,
                )
            case (False, save_path, True) if save_path is not None:
                Logger.write("info", f"Rerun visualizer will write to {str(save_rrd_path)}")
                RerunVisualizer.status = RerunVisualizer.State.OFFLINE
                
                rr.init(project_name)
                rr.save(str(save_rrd_path))
                rr.log(
                    "/",
                    rr.ViewCoordinates(xyz=rr.ViewCoordinates.FRD),
                    timeless=True,
                )
                
                if show_plots:
                    rr.send_blueprint(rr.blueprint.Blueprint(
                        rr.blueprint.Horizontal(
                            rr.blueprint.Spatial3DView(origin="/map"),
                            rr.blueprint.Tabs(
                                rr.blueprint.Spatial2DView(origin="/plots"),
                                rr.blueprint.Spatial2DView(origin="/map/Frame/cam")
                            )
                            
                        )
                    ))
                else:
                    rr.send_blueprint(rr.blueprint.Blueprint(
                        rr.blueprint.Spatial3DView(origin="/map"),
                    ))
            case _:
                raise ValueError("Invalid configuration for rerun visualizer.")

    @staticmethod
    def close():
        rr.rerun_shutdown()

    @staticmethod
    def checkIsActive(func):
        """
        Decorator, the decorated method will be executed if and only if the
        visualizer is active (--useRR is provided).
        """
        def wrapped(*args, **kwargs):
            if RerunVisualizer.status == RerunVisualizer.State.INACTIVE:
                return None
            return func(*args, **kwargs)
        return wrapped

    @checkIsActive
    @staticmethod
    def visualizeFrameAt(gmap: TensorMap, frame_idx: int):
        frame = gmap.frames[frame_idx]
        K = frame.K.squeeze(0)
        cx = K[0][2].item()
        cy = K[1][2].item()
        
        frame_position = frame.pose.translation()
        frame_rotation = frame.pose.rotation().numpy()

        rr.log(
            "/map/Frame",
            rr.Transform3D(
                translation=frame_position[0].numpy(),
                rotation=rr.datatypes.Quaternion(xyzw=frame_rotation[0]),
            ),
        )
        rr.log(
            "/map/Frame/cam",
            rr.Pinhole(
                resolution=[cx * 2, cy * 2],
                image_from_camera=K.numpy(),
                camera_xyz=rr.ViewCoordinates.FRD,
            ),
        )
        points = gmap.get_frame_points(frame)
        rr.log(
            "/map/Matches",
            rr.LineStrips3D(
                torch.stack(
                    [frame_position.repeat(len(points), 1), points.position], dim=1
                ).numpy(),
                radii=[0.001],
                colors=[[0, 0, 255]],
            )
        )
        rr.log(
            "/map/Frame/cam/keypoints",
            rr.Points2D(
                gmap.get_frame_observes(frame).pixel_uv.numpy(),
                colors=[34, 138, 167],
                radii=2.,
            ),
        )

    @checkIsActive
    @staticmethod
    def visualizeAllPoints(gmap: TensorMap, show_cov: bool = False):
        point_3d = gmap.points.position.tensor
        point_color = gmap.points.color.tensor
        
        rr.log("/map/Points", rr.Points3D(point_3d.numpy(), colors=point_color))

        if show_cov:
            covs = gmap.points.cov_Tw.tensor
            eigen_val, eigen_vec = torch.linalg.eig(covs)
            eigen_val, eigen_vec = torch.clamp(eigen_val.real, min=0.03), eigen_vec.real

            delta = point_3d.repeat(1, 3, 1).reshape(-1, 3).numpy()
            eigen_vec_Tw = eigen_vec.transpose(-1, -2).reshape(-1, 3).numpy()
            eigen_val = eigen_val.unsqueeze(-1).repeat(1, 1, 3).reshape(-1, 3).numpy()
            eigen_vec_Tw = eigen_vec_Tw * eigen_val.sqrt()
            eigen_vec_Tw_a = delta + eigen_vec_Tw
            eigen_vec_Tw_b = delta - eigen_vec_Tw

            rr.log(
                "/map/PointsVar",
                rr.LineStrips3D(
                    torch.stack([eigen_vec_Tw_a, eigen_vec_Tw_b], dim=1).numpy(),
                    radii=[0.003],
                    colors=[[200, 200, 10]],
                ),
            )

    @checkIsActive
    @staticmethod
    def visualizeRecentPoints(gmap: TensorMap, index=-1):
        points = gmap.get_frame_points(gmap.frames[index])
        rr.log("/map/Points", rr.Points3D(points.position.numpy(), colors=points.color.numpy()))

    @checkIsActive
    @staticmethod
    def visualizePointCov(gmap: TensorMap):
        cov_det = gmap.points.cov_Tw.det()
        cov_det_normalized = Normalize(vmin=0, vmax=cov_det.quantile(0.99).item())(cov_det)
        colormap = plt.cm.plasma
        color = colormap(cov_det_normalized)[..., :3]        
        rr.log("/map/PointsCov", rr.Points3D(gmap.points.position.numpy(), colors=color))

    @checkIsActive
    @staticmethod
    def visualizeImageOnCamera(image: torch.Tensor | np.ndarray, subpath: str | None = None, apply_cmap: bool = False, vmax: float | None = None):
        subpath = "img" if subpath is None else subpath
        if isinstance(image, torch.Tensor):
            np_image = image.cpu().numpy()
        else:
            np_image = image
        np_image: np.ndarray
        
        if apply_cmap:    
            data_normalized = Normalize(vmax=vmax)(np_image)
            colormap = plt.cm.plasma
            np_image = colormap(data_normalized)[..., :3]
        
        if np_image.dtype != np.uint8:
            np_image = (np_image * 255).astype(np.uint8)
        
        rr.log(f"/map/Frame/cam/{subpath}", rr.Image(np_image).compress())

    @checkIsActive
    @staticmethod
    def visualizeTrajectory(gmap: TensorMap, to_frame: int | None = None, from_frame: int | None = None, color=(244, 97, 221), radii=0.02, **rr_log_kwargs):
        frame_pos = pp.SE3(gmap.frames.pose.tensor[from_frame:to_frame]).translation().cpu().numpy()
        from_pos_np = frame_pos[:-1]
        to_pos_np = frame_pos[1:]
        rr.log(
            "/map/Trajectory",
            rr.LineStrips3D(
                np.stack([from_pos_np, to_pos_np], axis=1),
                radii=[radii],
                colors=[[color]],
            ),
            **rr_log_kwargs
        )

    @checkIsActive
    @staticmethod
    def visualizeEncodedImage(buf: BytesIO, path: str):
        """
        Interface to interact with PLTVisualizer.
        """
        rr.log(path, rr.ImageEncoded(contents=buf, format=rr.ImageFormat.JPEG))

    @checkIsActive
    @staticmethod
    def visualizePoints(path: str, positions: np.ndarray, colors: np.ndarray | None | tuple[float, float, float] = None, radii: float | np.ndarray=0.05):
        rr.log(f"/map/{path}", rr.Points3D(positions, colors=colors, radii=radii))

    @checkIsActive
    @staticmethod
    def visualizePath(path: str, trajectory: pp.LieTensor, colors: np.ndarray | None | Sequence[float], radii: float | np.ndarray=0.2):
        position = trajectory.translation().cpu().numpy()
        from_pos = position[:-1]
        to_pos = position[1:]
        rr.log(
            f"/map/{path}",
            rr.LineStrips3D(
                np.stack([from_pos, to_pos], axis=1),
                radii=radii, colors=colors,
            ),
        )

    @checkIsActive
    @staticmethod
    def visualizeAnnotatedCamera(path: str, K: torch.Tensor, pose: pp.LieTensor, label: str | None = None):
        cx, cy = K[0][2].item(), K[1][2].item()
        translation = pose.translation().squeeze().numpy()
        rotation = pose.rotation().squeeze().numpy()
        
        if label is not None:
            class_id = RerunVisualizer.context["LABEL_TO_CLASSID"].get(label, len(RerunVisualizer.context["LABEL_TO_CLASSID"]))
            RerunVisualizer.context["LABEL_TO_CLASSID"][label] = class_id
            rr.log(f"{path}/label", rr.Points3D(positions=[(0, 0, 0)], labels=[label], class_ids=[class_id]))
        
        rr.log(path, rr.Transform3D(translation=translation, rotation=rr.datatypes.Quaternion(xyzw=rotation)))
        rr.log(f"{path}/cam", rr.Pinhole(
            resolution=[cx * 2, cy * 2], image_from_camera=K.numpy(), camera_xyz=rr.ViewCoordinates.FRD,
        ))


class PLTVisualizer:
    DPI = 300
    save_path = Path(".")
    
    class State(Enum):
        INACTIVE  = 0
        SAVE_FILE = 1
        LOG_RERUN = 2
    
    class PlotArg(Generic[I, O]):
        def __init__(self, fn: Callable[Concatenate[Axes, I], O], args: dict[str, Any], title: str | None = None) -> None:
            self.fn = fn
            self.args = args
            self.title = title
    
    default_state = State.INACTIVE
    method_states: dict[str, State] = dict()
    method_call_cnt: dict[str, int] = dict()
    

    @classmethod
    def setup(cls, state: State | dict[Callable, State], save_path: Path, dpi: int = 300):
        """
        active can be either a boolean as a master switch to all methods
        or
        a list of method names (function name) that you want to enable during runtime.
        """
        assert save_path.exists()
        PLTVisualizer.save_path = save_path
        PLTVisualizer.DPI = dpi
        
        if isinstance(state, cls.State):
            cls.default_state = state
        else:
            cls.method_states = {func.__name__: state[func] for func in state}
            Logger.write("info", f"PLTVisualizer default to {cls.default_state} but following methods override the default mode")
            for method_key in state:
                Logger.write("info", f"\t{method_key.__name__}={state[method_key]}")

    @staticmethod
    def isActive(func: Callable[Concatenate[str, I], None]) -> bool:
        method_state = PLTVisualizer.method_states.get(func.__name__, PLTVisualizer.default_state)
        return method_state != PLTVisualizer.State.INACTIVE

    @staticmethod
    def AddVisualizerContext(func: Callable[I, FigureBase]) -> Callable[Concatenate[str, I], None]:
        """
        Decorator, the decorated method will be executed if and only if the
        visualizer is active (--useRR is provided).
        """

        @wraps(func)
        def wrapped(name: str, *args: I.args, **kwargs: I.kwargs):
            method_state = PLTVisualizer.method_states.get(func.__name__, PLTVisualizer.default_state)
            method_count = PLTVisualizer.method_call_cnt.get(func.__name__, 0)
            PLTVisualizer.method_call_cnt[func.__name__] = method_count + 1
            
            match method_state:
                case PLTVisualizer.State.INACTIVE:
                    return None
                case PLTVisualizer.State.SAVE_FILE:
                    fig_handle = func(*args, **kwargs)
                    
                    curr_name = name + f"_{method_count}.jpg"
                    try:
                        fig_handle.savefig(str(Path(PLTVisualizer.save_path, curr_name)))
                        Logger.write("info", f"PLTVisualizer: {func.__name__} ==> {curr_name}")
                    except KeyboardInterrupt as e:
                        Logger.write("fatal", f"Failed to save file from {func.__name__} to {curr_name}. Interrupted.")
                        raise e
                    except:
                        Logger.write("error", f"Failed to save file from {func.__name__} to {curr_name}")
                        Logger.write("error", traceback.format_exc())
                    finally:
                        if isinstance(fig_handle, Figure):
                            plt.close(fig_handle)
                case PLTVisualizer.State.LOG_RERUN:
                    fig_handle = func(*args, **kwargs)
                    try:
                        buf = BytesIO()
                        # A lower dpi since Rerun Viewer takes a lot of RAM with high-res images.
                        fig_handle.savefig(buf, format='jpg', dpi=200)
                        buf.seek(0)
                        RerunVisualizer.visualizeEncodedImage(buf, f"/plots/{name}")
                    finally:
                        if isinstance(fig_handle, Figure):
                            plt.close(fig_handle)

        return wrapped

    @staticmethod
    @AddVisualizerContext
    def visualize_Obs(
        image0: torch.Tensor,
        image1: torch.Tensor,
        obs: BatchObservation,
        depth_cov_map: torch.Tensor | None,
        flow_cov_map: torch.Tensor  | None,
        include_mask: torch.Tensor | None,
    ):
        """
        Pixel coordinate: (height, width) (row, col in image plane, left-top as origin)
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, dpi=PLTVisualizer.DPI)
        Plot.plot_image(ax1, image0[0].permute(1, 2, 0))
        
        Plot.plot_whiten_image(ax2, image1[0].permute(1, 2, 0))
        Plot.plot_observations(ax2, obs, include_mask=include_mask)
        Plot.plot_obs_flowcov(ax2, obs)
        
        if depth_cov_map is not None:
            cax3 = Plot.plot_scalarmap(ax3, depth_cov_map[0, 0].log10())
            fig.colorbar(cax3, ax=ax3, orientation="vertical", fraction=0.05)
        else:
            ax3.axis("off")
            
        if flow_cov_map is not None:
            cax4 = Plot.plot_scalarmap(ax4, flow_cov_map.norm(dim=1)[0].log10())
            fig.colorbar(cax4, ax=ax4, orientation="vertical", fraction=0.05)
        else:
            ax4.axis("off") 
        fig.tight_layout(pad=0.5)
        return fig

    @staticmethod
    @AddVisualizerContext
    def visualize_dpatch(patches: torch.Tensor, local_filters: torch.Tensor, depth_var: torch.Tensor, flow_var: torch.Tensor):
        assert depth_var.size(0) == patches.size(0)
        assert depth_var.size(0) == local_filters.size(0)
        assert depth_var.size(0) == flow_var.size(0)
        
        fig, axes = plt.subplots(5, 10, dpi=PLTVisualizer.DPI)
        cmin, cmax = 0.0, patches[:25].max().item()
        
        for idx, ax in enumerate(axes[:, :5].flatten()):
            ax.axis("off")
            
            if idx >= patches.size(0): continue
            Plot.plot_scalarmap(ax, patches[idx], vmin=cmin, vmax=cmax)
            ax.set_title(f"σ={round(depth_var[idx].item(), 3)}", fontsize=8)
        
        for idx, ax in enumerate(axes[:, 5:].flatten()):
            ax.axis("off")
            
            if idx >= local_filters.size(0): continue
            Plot.plot_scalarmap(ax, local_filters[idx])
            ax.set_title(f"⟨{round(flow_var[idx, 0].item())}, {round(flow_var[idx, 1].item())}⟩", fontsize=8)
        
        fig.tight_layout(pad=1.0)
        return fig


    @staticmethod
    @AddVisualizerContext
    def visualize_flow(flow_uv: torch.Tensor):
        ax: Axes
        fig, ax = plt.subplots(1, 1, dpi=PLTVisualizer.DPI)   #type: ignore
        Plot.plot_flow(ax, flow_uv)
        fig.tight_layout()
        return fig

    @staticmethod
    @AddVisualizerContext
    def visualize_depth(depth_map: torch.Tensor):
        ax: Axes
        fig, ax = plt.subplots(1, 1, dpi=PLTVisualizer.DPI)   #type: ignore
        cax = Plot.plot_scalarmap(ax, depth_map[0, 0], vmax=200.)
        fig.colorbar(cax, ax=ax)
        fig.tight_layout()
        return fig

    @staticmethod
    @AddVisualizerContext
    def visualize_depthcov(depth_cov: torch.Tensor):
        ax: Axes
        fig, ax = plt.subplots(1, 1, dpi=PLTVisualizer.DPI)   #type: ignore
        cax = Plot.plot_scalarmap(ax, depth_cov[0, 0].log10(), vmax=3.)
        fig.colorbar(cax, ax=ax)
        fig.tight_layout()
        return fig

    @staticmethod
    @AddVisualizerContext
    def visualize_covTc(obs: BatchObservation):
        valid_obs = obs[:25].cov_Tc
        vmax = max([c.max().item() for c in valid_obs if c is not None])
        vmin = 0.0

        fig, axes = plt.subplots(5, 5, dpi=PLTVisualizer.DPI)
        for idx, ax in enumerate(axes.flatten()):
            ax.axis("off")
            if idx >= len(valid_obs): continue
            
            cov = valid_obs[idx]
            if cov is None: continue
            Plot.plot_scalarmap(ax, cov, vmin, vmax)
        fig.tight_layout()
        return fig

    @staticmethod
    def __visualize_grid(n_rows: int, n_cols: int, plot_args: Sequence[None | PLTVisualizer.PlotArg], h_factor: float = 2, w_factor: float = 2):
        fig = plt.figure(1, dpi=PLTVisualizer.DPI, figsize=(n_rows * 2, n_cols * 2))
        
        assert len(plot_args) <= n_rows * n_cols
        axes = fig.subplots(n_rows, n_cols).flatten()   #type: ignore
        for ax, plot_arg in zip(axes, plot_args):
            ax: Axes
            if plot_arg is None:
                ax.axis("off")
                continue
            plot_arg.fn(ax, **plot_arg.args)
            if plot_arg.title is not None:
                ax.set_title(plot_arg.title)
        
        for idx in range(len(plot_args), axes.size):
            axes[idx].axis("off")

        
        fig.set_figheight(n_rows * h_factor)
        fig.set_figwidth(n_cols * w_factor)
        fig.tight_layout()
        return fig

    @AddVisualizerContext
    @staticmethod
    def visualize_flow_grid(
            frame0L: torch.Tensor | None, frame1L: torch.Tensor | None,
            flow_est: torch.Tensor | None, flow_gt: torch.Tensor | None,
            cov_est: torch.Tensor | None, cov_gt: torch.Tensor | None,
            est_curve: np.ndarray, oracle_curve: np.ndarray
        ) -> FigureBase:
        plot_args = [
            PLTVisualizer.PlotArg(Plot.plot_image, dict(image=frame0L), "Frame A")
                if frame0L is not None else None,
            PLTVisualizer.PlotArg(Plot.plot_image, dict(image=frame1L), "Frame B")
                if frame0L is not None else None,
            
            PLTVisualizer.PlotArg(Plot.plot_flow, dict(flow=flow_est), "Est Flow")
                if flow_est is not None else None,
            PLTVisualizer.PlotArg(Plot.plot_flow, dict(flow=flow_gt), "GT Flow")
                if flow_gt is not None else None,
            
            PLTVisualizer.PlotArg(Plot.plot_scalarmap, dict(data=cov_est, vmin=0, vmax=150), "Est FlowCov")
                if cov_est is not None else None,
            PLTVisualizer.PlotArg(Plot.plot_scalarmap, dict(data=cov_gt, vmin=0, vmax=150), "GT FlowCov")
                if cov_gt is not None else None,
            
            PLTVisualizer.PlotArg(Plot.plot_sparsification, dict(est_curve=est_curve, oracle_curve=oracle_curve), "Sparsify Curve")
                if (est_curve is not None) and (oracle_curve is not None) else None
        ]
        return PLTVisualizer.__visualize_grid(4, 2, plot_args, h_factor=2, w_factor=3)
    
    
    @AddVisualizerContext
    @staticmethod
    def visualize_depth_grid(
        frame0L: torch.Tensor | None, frame0R: torch.Tensor | None,
        gt_depth_map: torch.Tensor | None, est_depth_map: torch.Tensor | None,
        gt_cov_map: torch.Tensor | None, est_cov_map: torch.Tensor | None
    ) -> FigureBase:
        plot_args = [
            PLTVisualizer.PlotArg(Plot.plot_image, dict(image=frame0L), "Frame 0 Left") 
                if frame0L is not None else None,
            PLTVisualizer.PlotArg(Plot.plot_image, dict(image=frame0R), "Frame 0 Right")
                if frame0R is not None else None,
            PLTVisualizer.PlotArg(Plot.plot_scalarmap, dict(data=gt_depth_map, vmin=0., vmax=100.), "Depth GT")
                if gt_depth_map is not None else None,
            PLTVisualizer.PlotArg(Plot.plot_scalarmap, dict(data=est_depth_map, vmin=0., vmax=100.), "Depth Est")
                if est_depth_map is not None else None,
            PLTVisualizer.PlotArg(Plot.plot_scalarmap, dict(data=gt_cov_map, vmax=10), "Depth GT Cov")
                if gt_cov_map is not None else None,
            PLTVisualizer.PlotArg(Plot.plot_scalarmap, dict(data=est_cov_map, vmax=10), "Depth Est Cov")
                if est_cov_map is not None else None
        ]
        return PLTVisualizer.__visualize_grid(3, 2, plot_args)

    @AddVisualizerContext
    @staticmethod
    def visualize_kp_masks(kp_masks: list[torch.Tensor | None], titles: list[str]) -> FigureBase:
        plot_args = [
            (
                PLTVisualizer.PlotArg(Plot.plot_scalarmap, args=dict(data=mask.float()[0, 0]), title=title)
                if mask is not None else
                None
            )
            for mask, title in zip(kp_masks, titles)
        ]
        return PLTVisualizer.__visualize_grid(1, len(kp_masks), plot_args)

    @AddVisualizerContext
    @staticmethod
    def visualize_stereo(imageL: torch.Tensor, imageR: torch.Tensor) -> FigureBase:
        plot_args = [
            PLTVisualizer.PlotArg(Plot.plot_image, dict(image=imageL[0].permute(1, 2, 0)), title="Left Cam"),
            PLTVisualizer.PlotArg(Plot.plot_image, dict(image=imageR[0].permute(1, 2, 0)), title="Right Cam")
        ]
        return PLTVisualizer.__visualize_grid(1, 2, plot_args)

    @AddVisualizerContext
    @staticmethod
    def visualize_covaware_selector(imageL: torch.Tensor, quality_map: torch.Tensor, original_points: torch.Tensor, 
                                    final_points: torch.Tensor) -> FigureBase:
        ax: Axes
        fig, axs = plt.subplots(1, 3, dpi=PLTVisualizer.DPI)   #type: ignore
        ax = axs[0]
        Plot.plot_whiten_image(ax, imageL)
        Plot.plot_scalarmap(ax, quality_map, alpha=0.5)
        Plot.plot_keypoints(ax, original_points, color=(1., 0., 0.))
        Plot.plot_keypoints(ax, final_points)
        
        ax = axs[1]
        Plot.plot_scalarmap(ax, quality_map, vmax=10)
        
        ax = axs[2]
        Plot.plot_whiten_image(ax, imageL)
        Plot.plot_keypoints(ax, final_points, color=(1., 0., 0.), s=0.1)
        
        fig.tight_layout()
        return fig

    @AddVisualizerContext
    @staticmethod
    def visualize_keypoint(imageL: torch.Tensor | np.ndarray, keypoint: torch.Tensor, color=(1., 0., 0.)) -> FigureBase:
        ax: Axes
        fig, ax = plt.subplots(1, 1, dpi=PLTVisualizer.DPI)   #type: ignore
        Plot.plot_whiten_image(ax, imageL)
        Plot.plot_keypoints(ax, keypoint, s=2., color=color)
        fig.tight_layout()
        return fig

    @AddVisualizerContext
    @staticmethod
    def visualize_random_selector(imageL: torch.Tensor | np.ndarray, points: torch.Tensor) -> FigureBase:
        ax: Axes
        fig, ax = plt.subplots(1, 1, dpi=PLTVisualizer.DPI)   #type: ignore
        Plot.plot_whiten_image(ax, imageL)
        Plot.plot_keypoints(ax, points)
        fig.tight_layout()
        return fig

    @AddVisualizerContext
    @staticmethod
    def visualize_image_patches(patches: torch.Tensor | np.ndarray) -> FigureBase:
        fig, axes = plt.subplots(5, 5, dpi=PLTVisualizer.DPI)
        for idx, ax in enumerate(axes[:, :5].flatten()):
            ax.axis("off")
            
            if idx >= patches.shape[0]: continue
            Plot.plot_image(ax, patches[idx])        
        return fig
