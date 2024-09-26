import torch
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.collections import LineCollection

from Utility.Trajectory import PlotableMotions, PlotableTrajectory
from Utility.Utils import getConsecutiveRange

from Module.Map import BatchFrame


def plot_losttrack_frames(ax: Axes, frame_status: torch.Tensor):
    lost_track_ranges = getConsecutiveRange(frame_status.int().tolist(), lambda flag: bool(BatchFrame.FLAG_VO_LOSTTRACK & flag))
    for start, end in lost_track_ranges:
        ax.axvspan(start, end-0.99, color="red", alpha=0.3)

def plot_motionRTE(ax: Axes, ref_traj: PlotableMotions, est_traj: PlotableMotions) -> Line2D:
    ref_motions = ref_traj.data.motions
    est_motions = est_traj.data.motions
    err_motions = ref_motions @ est_motions.Inv()
    err_trans   = err_motions.translation().norm(dim=1).numpy()
    handle, = ax.plot(err_trans, scalex=True, scaley=True, label=est_traj.name, **est_traj.plot_kwargs)
    
    plot_losttrack_frames(ax, est_traj.data.frame_status)
    return handle

def plot_MotionRTE_axes(ax: Axes, ref_traj: PlotableMotions, est_traj: PlotableMotions, axis: int) -> Line2D:
    ref_motions = ref_traj.data.motions
    est_motions = est_traj.data.motions
    err_motions = ref_motions @ est_motions.Inv()
    err_trans   = err_motions.translation().numpy()
    handle, = ax.plot(np.abs(err_trans[..., axis]), scalex=True, scaley=True, label=est_traj.name, **est_traj.plot_kwargs)

    plot_losttrack_frames(ax, est_traj.data.frame_status)    
    return handle

def plot_Translation_axes(ax: Axes, traj: PlotableMotions | PlotableTrajectory, axis: int) -> Line2D:
    motions = traj.data.translation
    handle, = ax.plot(motions[..., axis], scalex=True, scaley=True, label=traj.name, **traj.plot_kwargs)
    
    plot_losttrack_frames(ax, traj.data.frame_status)
    return handle

def plot_MotionROE(ax: Axes, ref_traj: PlotableMotions, est_traj: PlotableMotions) -> Line2D:
    ref_motions = ref_traj.data.motions
    est_motions = est_traj.data.motions
    err_motions = ref_motions @ est_motions.Inv()
    err_rots    = err_motions.rotation().Log().norm(dim=1).numpy()
    handle, = ax.plot(err_rots, scalex=True, scaley=True, label=est_traj.name, **est_traj.plot_kwargs)
    
    plot_losttrack_frames(ax, est_traj.data.frame_status)
    return handle

def plot_MotionROE_axes(ax: Axes, ref_traj: PlotableMotions, est_traj: PlotableMotions, axis: int) -> Line2D:
    ref_motions = ref_traj.data.motions
    est_motions = est_traj.data.motions
    err_motions = ref_motions @ est_motions.Inv()
    err_rots    = err_motions.rotation().euler()[..., axis].abs().numpy()
    handle, = ax.plot(err_rots, scalex=True, scaley=True, label=est_traj.name, **est_traj.plot_kwargs)
    
    plot_losttrack_frames(ax, est_traj.data.frame_status)
    return handle

def plot_Rotation_axes(ax: Axes, traj: PlotableMotions | PlotableTrajectory, axis: int) -> Line2D:
    motions = traj.data.rotation.euler()
    handle, = ax.plot(motions[..., axis], scalex=True, scaley=True, label=traj.name, **traj.plot_kwargs)
    return handle

def plot_Trajectory(ax: Axes, traj: PlotableTrajectory) -> Line2D:
    traj_transitions = traj.data.translation
    x, y = traj_transitions[:, 0], traj_transitions[:, 1]
    handle, = ax.plot(x, y, scalex=True, scaley=True, label=traj.name, **traj.plot_kwargs)
    return handle

def plot_LostTrackAt(ax: Axes, traj: PlotableTrajectory):
    traj_transitions = traj.data.translation
    vo_lost_mask = np.array(
        [(BatchFrame.FLAG_VO_LOSTTRACK & flag) for flag in traj.data.frame_status.int().tolist()],
        dtype=bool
    )
    lost_x, lost_y = traj_transitions[vo_lost_mask, 0], traj_transitions[vo_lost_mask, 1]
    scatter_handle = ax.scatter(lost_x, lost_y, color="red", marker=MarkerStyle("x"), zorder=100)
    _, labels = ax.get_legend_handles_labels()
    
    # We only want this to occur once
    if "VO Lost Track" not in labels:
        scatter_handle.set_label("VO Lost Track")

def plot_LinewithAlpha(ax: Axes, x: np.ndarray, y: np.ndarray, color, alpha: np.ndarray, linewidth: float=1., linestyle: str="-", label: str | None = None):
    assert len(x.shape) == 1
    assert len(alpha.shape) == 1
    assert x.shape == y.shape
    assert x.shape[0] == alpha.shape[0] + 1

    positions = np.concatenate([x[:, np.newaxis], y[:, np.newaxis]], axis=1)
    start_pos = positions[:-1]
    end_pos = positions[1:]
    line_collect = LineCollection(segments=[[
        (start_pos[idx, 0], start_pos[idx, 1]), (end_pos[idx, 0], end_pos[idx, 1]),
    ] for idx in range(start_pos.shape[0])], label=label)

    line_collect.set_linewidth(linewidth)
    line_collect.set_linestyle(linestyle)
    line_collect.set_color(color)
    line_collect.set_alpha(alpha.tolist())
    ax.add_collection(line_collect) #type: ignore
    ax.set_xlim(min(x.min(), ax.get_xlim()[0]), max(x.max(), ax.get_xlim()[1]))
    ax.set_ylim(min(y.min(), ax.get_ylim()[0]), max(y.max(), ax.get_ylim()[1]))
