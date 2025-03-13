from __future__ import annotations
from functools import wraps
from pathlib import Path
from io import BytesIO

from matplotlib.figure import Figure
from matplotlib.axes   import Axes
import matplotlib.pyplot as plt
import torch
import numpy as np

import typing as T
try:
    import rerun as rr
except ImportError:
    rr = None

import Utility.Plot as Plot
from Utility.Extensions import Chain
from Utility.PrettyPrint import Logger

if T.TYPE_CHECKING:
    from DataLoader import StereoFrame
    from Module.Map import MatchObs
    from Module.Frontend.Matching import IMatcher
    from Module.Frontend.StereoDepth import IStereoDepth



# A global dictionary stores the state of each function registered.
T_Mode   = T.Literal["none", "rerun", "image"]

T_Input  = T.ParamSpec("T_Input")
T_Output = T.TypeVar("T_Output")


class Matplotlib_Visualizer:
    func_mode: T.ClassVar[dict[str, T_Mode | T.Literal["default"]]] = dict()
    default_mode: T.ClassVar[T_Mode] = "none"
    default_dpi  : T.ClassVar[int]    = 250
    default_save : T.ClassVar[Path]   = Path(".")

    @classmethod
    def set_fn_mode(cls, func: T.Callable[T.Concatenate[T_Mode, T_Input], T_Output], mode: T_Mode | T.Literal["default"]):
        cls.func_mode[func.__name__] = mode
    
    @classmethod
    def get_fn_mode(cls, func: T.Callable[T.Concatenate[T_Mode, T_Input], T_Output]) -> T_Mode:
        assert func.__name__ in cls.func_mode
        func_mode = cls.func_mode[func.__name__]
        if func_mode == 'default': return cls.default_mode
        return func_mode
    
    @staticmethod
    def register(func: T.Callable[T_Input, Figure]) -> T.Callable[T.Concatenate[str, T_Input], None]:
        """
        Register a classmethod of Matplotlib Visualizer
        """
        @wraps(func)
        def implement(file_prefix: str, *args: T_Input.args, **kwargs: T_Input.kwargs) -> None:
            if func.__name__ not in Matplotlib_Visualizer.func_mode:
                Matplotlib_Visualizer.func_mode[func.__name__] = "default"
            
            func_calls: int = implement.calls   #type: ignore   # PyRight does not support this.
            implement.calls += 1                #type: ignore
            func_mode  = Matplotlib_Visualizer.func_mode[func.__name__]
            if func_mode == 'default': func_mode = Matplotlib_Visualizer.default_mode
            
            match func_mode:
                case "none":
                    return None
                case "image":
                    fig = func(*args, **kwargs)
                    fig.savefig(Path(Matplotlib_Visualizer.default_save, (save_path := f"{file_prefix}_{func_calls:05}.png")), dpi=Matplotlib_Visualizer.default_dpi)
                    Logger.write("info", f"Save figure at {save_path}")
                    plt.close(fig)
                case "rerun":
                    assert rr is not None
                    fig = func(*args, **kwargs)
                    buf = BytesIO()
                    fig.savefig(buf, format='jpg', dpi=Matplotlib_Visualizer.default_dpi)
                    buf.seek(0)
                    rr.log(file_prefix, rr.ImageEncoded(contents=buf, format=rr.ImageFormat.JPEG))
        implement.calls = 0         #type: ignore   # PyRight does not support this.
        return implement
    
    @staticmethod
    def __plot_grid(dpi: int, n_rows: int, n_cols: int, plot_args: T.Sequence[T.Callable[[Axes], T.Any] | None], h_factor: float = 4, w_factor: float = 4) -> Figure:
        fig = plt.figure(1, dpi=dpi, figsize=(n_rows * h_factor, n_cols * w_factor))
        
        assert len(plot_args) <= n_rows * n_cols
        axes = fig.subplots(n_rows, n_cols)
        if isinstance(axes, np.ndarray): axes = axes.flatten()
        else:
            assert isinstance(axes, Axes)
            axes = np.array([axes], dtype=object)
        for ax, plot_arg in zip(axes, plot_args):
            ax: Axes
            if plot_arg is None:
                ax.axis("off")
                continue
            plot_arg(ax)
        
        for idx in range(len(plot_args), axes.size):
            axes[idx].axis("off")
        
        fig.set_figheight(n_rows * h_factor)
        fig.set_figwidth(n_cols * w_factor)
        fig.tight_layout()
        return fig

    @register
    @staticmethod
    def plot_imatcher(output:IMatcher.Output, frame0: StereoFrame, frame1: StereoFrame) -> Figure:
        """ Plot the IMatcher module output in various subplots. Show mask, covariance and flow as much as possible.
        """    
        if frame0.stereo.gt_flow is not None:
            plot_epe = (output.flow.detach().cpu() - frame0.stereo.gt_flow).abs()
        else:
            plot_epe = None
        
        plot_args = [
            # Row 1
            # Plot image 0
            Plot.plot_image(frame0.stereo.imageL.detach().cpu().permute(0, 2, 3, 1)[0])
                >> Plot.plot_no_border()
                >> Chain.side_effect(lambda ax: ax.set_title(f"Frame {frame0.frame_idx}", loc="left")),
            
            # Plot predicted flow
            Plot.plot_flow(output.flow[0].detach().cpu())
                >> Plot.plot_no_border()
                >> Chain.side_effect(lambda ax: ax.set_title(f"Predict flow", loc="left")),
            
            # Plot predicted flow masked by gt mask
            Plot.plot_flow(output.flow[0].detach().cpu())
                >> Plot.plot_mask(None if frame0.stereo.flow_mask is None else frame0.stereo.flow_mask.detach().cpu()[0, 0])
                >> Plot.plot_no_border()
                >> Chain.side_effect(lambda ax: ax.set_title(f"Predict flow (GTMask)", loc="left")),
            
            # Plot predicted flow masked by predicted mask
            Plot.plot_flow(output.flow[0].detach().cpu())
                >> Plot.plot_mask(None if output.mask is None else output.mask.detach().cpu()[0, 0])
                >> Plot.plot_no_border()
                >> Chain.side_effect(lambda ax: ax.set_title(f"Predict flow (Pred Mask)", loc="left")),
            
            # Row 2
            # Plot image 1
            Plot.plot_image(frame1.stereo.imageL.detach().cpu().permute(0, 2, 3, 1)[0])
                >> Plot.plot_no_border()
                >> Chain.side_effect(lambda ax: ax.set_title(f"Frame {frame1.frame_idx}", loc="left")),
            
            # Plot gtFlow
            Plot.plot_flow(None if frame0.stereo.gt_flow is None else frame0.stereo.gt_flow[0].detach().cpu())
                >> Plot.plot_no_border()
                >> Chain.side_effect(lambda ax: ax.set_title(f"Ground truth flow", loc="left")),
            
            # Plot gtFlow masked by gt mask
            Plot.plot_flow(None if frame0.stereo.gt_flow is None else frame0.stereo.gt_flow[0].detach().cpu())
                >> Plot.plot_mask(None if frame0.stereo.flow_mask is None else frame0.stereo.flow_mask.detach().cpu()[0, 0])
                >> Plot.plot_no_border()
                >> Chain.side_effect(lambda ax: ax.set_title(f"Predict flow (GTMask)", loc="left")),
            
            # Plot gtFlow masked by pred mask
            Plot.plot_flow(None if frame0.stereo.gt_flow is None else frame0.stereo.gt_flow[0].detach().cpu())
                >> Plot.plot_mask(None if output.mask is None else output.mask.detach().cpu()[0, 0])
                >> Plot.plot_no_border()
                >> Chain.side_effect(lambda ax: ax.set_title(f"Predict flow (Pred Mask)", loc="left")),

            # Row 3
            # Plot flow covariance on u direction
            Plot.plot_scalarmap(None if output.cov is None else output.cov[0, 0].log10().detach().cpu(), colorbar=True)
                >> Plot.plot_no_border()
                >> Chain.side_effect(lambda ax: ax.set_title("Pred cov_u (log)", loc="left")),
            
            # Plot flow covariance on u direction, masked by predMask
            Plot.plot_scalarmap(None if output.cov is None else output.cov[0, 0].log10().detach().cpu(), colorbar=True)
                >> Plot.plot_mask(None if output.mask is None else output.mask[0, 0].detach().cpu())
                >> Plot.plot_no_border()
                >> Chain.side_effect(lambda ax: ax.set_title("Pred cov_u (log, Pred Mask)", loc="left")),
            
            # Plot epe (u direction)
            Plot.plot_scalarmap(None if plot_epe is None else plot_epe[0, 0], colorbar=True)
                >> Plot.plot_no_border()
                >> Chain.side_effect(lambda ax: ax.set_title("EPE u", loc="left")),
            
            # Plot epe (u direction), masked by predMask
            Plot.plot_scalarmap(None if plot_epe is None else plot_epe[0, 0], colorbar=True)
                >> Plot.plot_mask(None if output.mask is None else output.mask[0, 0].detach().cpu())
                >> Plot.plot_no_border()
                >> Chain.side_effect(lambda ax: ax.set_title("EPE u (Pred Mask)", loc="left")),
            
            # Row 4
            # Plot flow covariance on v direction
            Plot.plot_scalarmap(None if output.cov is None else output.cov[0, 1].log10().detach().cpu(), colorbar=True)
                >> Plot.plot_no_border()
                >> Chain.side_effect(lambda ax: ax.set_title("Pred cov_v (log)", loc="left")),
            
            # Plot flow covariance on v direction, masked by predMask
            Plot.plot_scalarmap(None if output.cov is None else output.cov[0, 1].log10().detach().cpu(), colorbar=True)
                >> Plot.plot_mask(None if output.mask is None else output.mask[0, 0].detach().cpu())
                >> Plot.plot_no_border()
                >> Chain.side_effect(lambda ax: ax.set_title("Pred cov_v (log, Pred Mask)", loc="left")),
            
            # Plot epe (v direction)
            Plot.plot_scalarmap(None if plot_epe is None else plot_epe[0, 1], colorbar=True)
                >> Plot.plot_no_border()
                >> Chain.side_effect(lambda ax: ax.set_title("EPE v", loc="left")),
            
            
            # Plot epe (v direction), masked by predMask
            Plot.plot_scalarmap(None if plot_epe is None else plot_epe[0, 1], colorbar=True)
                >> Plot.plot_mask(None if output.mask is None else output.mask[0, 0].detach().cpu())
                >> Plot.plot_no_border()
                >> Chain.side_effect(lambda ax: ax.set_title("EPE v (Pred Mask)", loc="left")),   
        ]
        fig = Matplotlib_Visualizer.__plot_grid(dpi=Matplotlib_Visualizer.default_dpi ,n_rows=4, n_cols=4, plot_args=plot_args, h_factor=2, w_factor=2)
        return fig

    @register
    @staticmethod
    def plot_istereo(output: IStereoDepth.Output, frame: StereoFrame) -> Figure:
        """Plot the IStereoDepth output in various subplots. Plot depth, gtdepth, cov and mask as much as possible.
        """
        if frame.stereo.gt_depth is not None:
            depth_err = (output.depth.detach().cpu() - frame.stereo.gt_depth).abs()
        else:
            depth_err = None
        
        plot_args = [
            # Row 1
            # Plot left camera
            Plot.plot_image(frame.stereo.imageL.detach().cpu().permute(0, 2, 3, 1)[0])
                >> Plot.plot_no_border()
                >> Chain.side_effect(lambda ax: ax.set_title(f"Frame {frame.frame_idx} L", loc="left")),
            
            # Plot right camera
            Plot.plot_image(frame.stereo.imageR.detach().cpu().permute(0, 2, 3, 1)[0])
                >> Plot.plot_no_border()
                >> Chain.side_effect(lambda ax: ax.set_title(f"Frame {frame.frame_idx} R", loc="left")),
            
            # Empty
            None,
            
            # Row 2
            # Plot Predicted Depth
            Plot.plot_scalarmap(output.depth.detach().cpu()[0, 0], colorbar=True)
                >> Plot.plot_no_border()
                >> Chain.side_effect(lambda ax: ax.set_title(f"Pred Depth", loc="left")),
            
            # Plot predicted cov
            Plot.plot_scalarmap(None if output.cov is None else output.cov.sqrt().detach().cpu()[0, 0], colorbar=True)
                >> Plot.plot_no_border()
                >> Chain.side_effect(lambda ax: ax.set_title(f"Depth Cov (sqrt)", loc="left")),
                
            # Plot depth error
            Plot.plot_scalarmap(None if depth_err is None else depth_err[0, 0].log10(), colorbar=True) 
                >> Plot.plot_no_border()
                >> Chain.side_effect(lambda ax: ax.set_title("Depth Err (log)", loc="left")),
            
            # Row 3
            # Plot gt Depth
            Plot.plot_scalarmap(None if frame.stereo.gt_depth is None else frame.stereo.gt_depth.detach().cpu()[0, 0], colorbar=True)
                >> Plot.plot_no_border()
                >> Chain.side_effect(lambda ax: ax.set_title(f"GT Depth", loc="left")),
                
            # Plot predicted cov (pred mask)
            Plot.plot_scalarmap(None if output.cov is None else output.cov.sqrt().detach().cpu()[0, 0], colorbar=True)
                >> Plot.plot_mask(None if output.mask is None else output.mask[0, 0].detach().cpu())
                >> Plot.plot_no_border()
                >> Chain.side_effect(lambda ax: ax.set_title(f"Depth Cov (sqrt, Pred Mask)", loc="left")),
            
            # Plot depth error (pred mask)
            Plot.plot_scalarmap(None if depth_err is None else depth_err[0, 0].log10(), colorbar=True)
                >> Plot.plot_mask(None if output.mask is None else output.mask[0, 0].detach().cpu())
                >> Plot.plot_no_border()
                >> Chain.side_effect(lambda ax: ax.set_title("Depth Err (log, Pred Mask)", loc="left")),
        ]
        
        fig = Matplotlib_Visualizer.__plot_grid(dpi=Matplotlib_Visualizer.default_dpi, n_rows=3, n_cols=3, plot_args=plot_args)
        return fig

    @register
    @staticmethod
    def plot_macvo(obs: MatchObs, depth: IStereoDepth.Output, match: IMatcher.Output, frame0: StereoFrame, frame1: StereoFrame) -> Figure:
        if match.cov is not None:
            flow_det = (match.cov[:, 0] * match.cov[:, 1] - match.cov[:, 2].square())[0]
        else:
            flow_det = None
        
        plot_args = [
            # Plot left camera at frame 0
            Plot.plot_image(frame0.stereo.imageL[0].permute(1, 2, 0))
                >> Plot.plot_no_border()
                >> Chain.side_effect(lambda ax: ax.set_title(f"Frame {frame0.frame_idx} Left", loc="left")),
            
            # Plot left camera at frame 1, with keypoints overlayed on it
            Plot.plot_whiten_image(frame1.stereo.imageL[0].permute(1, 2, 0), whiten=0.75)
                >> Plot.plot_no_border()
                >> Plot.plot_flow_cov(obs.data["pixel2_uv"], obs.data["pixel2_uv_cov"])
                >> Plot.plot_keypoints(obs.data["pixel2_uv"], obs.data["pixel2_d_cov"], s=2, marker='.')
                >> Chain.side_effect(lambda ax: ax.set_title(f"Frame {frame1.frame_idx} Left", loc="left")),
            
            # Plot depth cov estimation
            Plot.plot_scalarmap(None if depth.cov is None else depth.cov.sqrt().detach().cpu()[0, 0], colorbar=True)
                >> Plot.plot_mask(None if depth.mask is None else depth.mask[0, 0].detach().cpu())
                >> Plot.plot_no_border()
                >> Chain.side_effect(lambda ax: ax.set_title(f"Depth Cov (sqrt, Pred Mask)", loc="left")),
            
            # Plot flow cov estimation
            Plot.plot_scalarmap(None if flow_det is None else flow_det.log10().detach().cpu(), colorbar=True)
                >> Plot.plot_no_border()
                >> Chain.side_effect(lambda ax: ax.set_title("Pred cov_det (log)", loc="left")),
        ]
        return Matplotlib_Visualizer.__plot_grid(dpi=Matplotlib_Visualizer.default_dpi, n_rows=2, n_cols=2, plot_args=plot_args)

    @register
    @staticmethod
    def plot_reprojerr(proj_kp1: torch.Tensor, kp2: torch.Tensor, cov2x2: torch.Tensor, frame1: StereoFrame) -> Figure:
        plot_args = [
            Plot.plot_whiten_image(frame1.stereo.imageL[0].permute(1, 2, 0))
                >> Plot.plot_no_border()
                >> Plot.plot_flow_cov(kp2, cov2x2, scale=1.)
                >> Plot.plot_keypoints(proj_kp1, None, s=0.5)
                >> Plot.plot_keypoints(kp2, None, s=0.5)
                >> Plot.plot_kp_correspondence(proj_kp1, kp2, color='green', linewidth=1.)
                >> Chain.side_effect(lambda ax: ax.set_title("Reprojection Error"))
        ]
        return Matplotlib_Visualizer.__plot_grid(dpi=Matplotlib_Visualizer.default_dpi, n_rows=1, n_cols=1, plot_args=plot_args)
