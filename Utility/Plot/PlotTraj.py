import torch
import numpy as np
from typing import Literal
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle

from Utility.Trajectory import PlotableMotions, PlotableTrajectory
from Utility.Utils import getConsecutiveRange, IgnoreException

# from Module.Map import BatchFrame


@IgnoreException
def plot_losttrack_frames(ax: Axes, frame_status: torch.Tensor):
    lost_track_ranges = getConsecutiveRange(frame_status.int().tolist(), lambda flag: flag)
    for start, end in lost_track_ranges:
        ax.axvspan(start, end-0.99, color="red", alpha=0.3)

@IgnoreException
def plot_motionRTE(ax: Axes, ref_traj: PlotableMotions, est_traj: PlotableMotions) -> Line2D:
    ref_motions = ref_traj.data.motions
    est_motions = est_traj.data.motions
    err_motions = ref_motions @ est_motions.Inv()
    err_trans   = err_motions.translation().norm(dim=1).numpy()
    handle, = ax.plot(err_trans, scalex=True, scaley=True, label=est_traj.name, **est_traj.plot_kwargs)
    
    plot_losttrack_frames(ax, est_traj.data.frame_status)
    return handle

@IgnoreException
def plot_MotionRTE_axes(ax: Axes, ref_traj: PlotableMotions, est_traj: PlotableMotions, axis: int) -> Line2D:
    ref_motions = ref_traj.data.motions
    est_motions = est_traj.data.motions
    err_motions = ref_motions @ est_motions.Inv()
    err_trans   = err_motions.translation().numpy()
    handle, = ax.plot(np.abs(err_trans[..., axis]), scalex=True, scaley=True, label=est_traj.name, **est_traj.plot_kwargs)

    plot_losttrack_frames(ax, est_traj.data.frame_status)    
    return handle

@IgnoreException
def plot_Translation_axes(ax: Axes, traj: PlotableMotions | PlotableTrajectory, axis: int) -> Line2D:
    motions = traj.data.translation
    handle, = ax.plot(motions[..., axis], scalex=True, scaley=True, label=traj.name, **traj.plot_kwargs)
    
    plot_losttrack_frames(ax, traj.data.frame_status)
    return handle

@IgnoreException
def plot_MotionROE(ax: Axes, ref_traj: PlotableMotions, est_traj: PlotableMotions) -> Line2D:
    ref_motions = ref_traj.data.motions
    est_motions = est_traj.data.motions
    err_motions = ref_motions @ est_motions.Inv()
    err_rots    = err_motions.rotation().Log().norm(dim=1).numpy()
    handle, = ax.plot(err_rots, scalex=True, scaley=True, label=est_traj.name, **est_traj.plot_kwargs)
    
    plot_losttrack_frames(ax, est_traj.data.frame_status)
    return handle

@IgnoreException
def plot_MotionROE_axes(ax: Axes, ref_traj: PlotableMotions, est_traj: PlotableMotions, axis: int) -> Line2D:
    ref_motions = ref_traj.data.motions
    est_motions = est_traj.data.motions
    err_motions = ref_motions @ est_motions.Inv()
    err_rots    = err_motions.rotation().euler()[..., axis].abs().numpy()
    handle, = ax.plot(err_rots, scalex=True, scaley=True, label=est_traj.name, **est_traj.plot_kwargs)
    
    plot_losttrack_frames(ax, est_traj.data.frame_status)
    return handle

@IgnoreException
def plot_Rotation_axes(ax: Axes, traj: PlotableMotions | PlotableTrajectory, axis: int) -> Line2D:
    motions = traj.data.rotation.euler()
    handle, = ax.plot(motions[..., axis], scalex=True, scaley=True, label=traj.name, **traj.plot_kwargs)
    return handle

@IgnoreException
def plot_Trajectory(ax: Axes, axis_0: int, axis_1: int, traj: PlotableTrajectory) -> Line2D:
    traj_transitions = traj.data.translation
    x, y = traj_transitions[:, axis_0], traj_transitions[:, axis_1]
    handle, = ax.plot(x, y, scalex=True, scaley=True, label=traj.name, **traj.plot_kwargs)
    return handle

@IgnoreException
def plot_LostTrackAt(ax: Axes, axis_0: int, axis_1: int, traj: PlotableTrajectory):
    traj_transitions = traj.data.translation
    vo_lost_mask = traj.data.frame_status.bool()
    lost_x, lost_y = traj_transitions[vo_lost_mask, axis_0], traj_transitions[vo_lost_mask, axis_1]
    scatter_handle = ax.scatter(lost_x, lost_y, color="red", marker=MarkerStyle("x"), zorder=100)
    _, labels = ax.get_legend_handles_labels()
    
    # We only want this to occur once
    if "VO Lost Track" not in labels:
        scatter_handle.set_label("VO Lost Track")

@IgnoreException
def plot_CumulativeROECurve(ax: Axes, ref_traj: PlotableMotions, est_traj: PlotableMotions, axis: Literal[0, 1, 2] | None) -> Line2D:
    ref_motions = ref_traj.data.motions
    est_motions = est_traj.data.motions
    err_motions = ref_motions @ est_motions.Inv()
    
    if axis is None:
        err_rots    = err_motions.rotation().Log().norm(dim=1).numpy()
    else:
        err_rots    = err_motions.rotation().Log().norm[..., axis].numpy()
    
    return ax.ecdf(err_rots, label=est_traj.name, **est_traj.plot_kwargs)

@IgnoreException
def plot_CumulativeRTECurve(ax: Axes, ref_traj: PlotableMotions, est_traj: PlotableMotions, axis: Literal[0, 1, 2] | None) -> Line2D:
    ref_motions = ref_traj.data.motions
    est_motions = est_traj.data.motions
    err_motions = ref_motions @ est_motions.Inv()
    
    if axis is None:
        err_trans   = err_motions.translation().norm(dim=1).numpy()
    else:
        err_trans   = err_motions.translation()[..., axis].numpy()
    
    return ax.ecdf(err_trans, label=est_traj.name, **est_traj.plot_kwargs)
