import torch
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from .PlotTraj import (
    plot_MotionROE,
    plot_MotionROE_axes,
    plot_motionRTE,
    plot_MotionRTE_axes,
    plot_Rotation_axes,
    plot_Trajectory,
    plot_Translation_axes,
    plot_LostTrackAt, 
)
from Utility.Trajectory import PlotableMotions, PlotableTrajectory


def AnalyzeTranslation(
    ref_traj: PlotableMotions,
    est_trajs: list[PlotableMotions], file_name: Path
):
    GRID_SHAPE = (3, 6)
    _ = plt.figure(figsize=(16, 6), dpi=300)
    ax = plt.subplot2grid(GRID_SHAPE, (0, 0), rowspan=3, colspan=2)
    for est_traj in est_trajs:
        plot_motionRTE(ax, ref_traj, est_traj)
    ax.legend(frameon=False)
    ax.grid(visible=True, linestyle="--")
    ax.set_title("Translation Error (m)", loc="left")

    ax = plt.subplot2grid(GRID_SHAPE, (0, 2), rowspan=1, colspan=2)
    for est_traj in est_trajs:
        plot_MotionRTE_axes(ax, ref_traj, est_traj, axis=0)
    ax.grid(visible=True, linestyle="--")
    ax.set_title("Translation Error on x-axis (m)", loc="left")

    ax = plt.subplot2grid(GRID_SHAPE, (1, 2), rowspan=1, colspan=2)
    for est_traj in est_trajs:
        plot_MotionRTE_axes(ax, ref_traj, est_traj, axis=1)
    ax.grid(visible=True, linestyle="--")
    ax.set_title("Translation Error on y-axis (m)", loc="left")

    ax = plt.subplot2grid(GRID_SHAPE, (2, 2), rowspan=1, colspan=2)
    for est_traj in est_trajs:
        plot_MotionRTE_axes(ax, ref_traj, est_traj, axis=2)
    ax.grid(visible=True, linestyle="--")
    ax.set_title("Translation Error on z-axis (m)", loc="left")

    ax = plt.subplot2grid(GRID_SHAPE, (0, 4), rowspan=1, colspan=2)
    plot_Translation_axes(ax, ref_traj, axis=0)
    for est_traj in est_trajs:
        plot_Translation_axes(ax, est_traj, axis=0)
    ax.grid(visible=True, linestyle="--")
    # ax.set_ylim(bottom=-1., top=1.)
    ax.set_title("Translation on x-axis (m)", loc="left")

    ax = plt.subplot2grid(GRID_SHAPE, (1, 4), rowspan=1, colspan=2)
    plot_Translation_axes(ax, ref_traj, axis=1)
    for est_traj in est_trajs:
        plot_Translation_axes(ax, est_traj, axis=1)
    ax.grid(visible=True, linestyle="--")
    # ax.set_ylim(bottom=-1., top=1.)
    ax.set_title("Translation on y-axis (m)", loc="left")

    ax = plt.subplot2grid(GRID_SHAPE, (2, 4), rowspan=1, colspan=2)
    plot_Translation_axes(ax, ref_traj, axis=2)
    for est_traj in est_trajs:
        plot_Translation_axes(ax, est_traj, axis=2)
    ax.grid(visible=True, linestyle="--")
    # ax.set_ylim(bottom=-1., top=1.)
    ax.set_title("Translation on z-axis (m)", loc="left")

    plt.tight_layout()
    plt.savefig(str(file_name))
    plt.close()


def AnalyzeRotation(ref_traj: PlotableMotions, est_trajs: list[PlotableMotions], file_name: Path):
    GRID_SHAPE = (3, 6)
    _ = plt.figure(figsize=(16, 6), dpi=300)
    ax = plt.subplot2grid(GRID_SHAPE, (0, 0), rowspan=3, colspan=2)
    for est_traj in est_trajs:
        plot_MotionROE(ax, ref_traj, est_traj)
    ax.legend(frameon=False)
    ax.grid(visible=True, linestyle="--")
    ax.set_title("Rotation Error (rad)", loc="left")

    ax = plt.subplot2grid(GRID_SHAPE, (0, 2), rowspan=1, colspan=2)
    for est_traj in est_trajs:
        plot_MotionROE_axes(ax, ref_traj, est_traj, axis=0)
    ax.grid(visible=True, linestyle="--")
    ax.set_title("Rotation Error on x-axis (rad)", loc="left")

    ax = plt.subplot2grid(GRID_SHAPE, (1, 2), rowspan=1, colspan=2)
    for est_traj in est_trajs:
        plot_MotionROE_axes(ax, ref_traj, est_traj, axis=1)
    ax.grid(visible=True, linestyle="--")
    ax.set_title("Rotation Error on y-axis (rad)", loc="left")

    ax = plt.subplot2grid(GRID_SHAPE, (2, 2), rowspan=1, colspan=2)
    for est_traj in est_trajs:
        plot_MotionROE_axes(ax, ref_traj, est_traj, axis=2)
    ax.grid(visible=True, linestyle="--")
    ax.set_title("Rotation Error on z-axis (rad)", loc="left")

    ax = plt.subplot2grid(GRID_SHAPE, (0, 4), rowspan=1, colspan=2)
    plot_Rotation_axes(ax, ref_traj, axis=0)
    for est_traj in est_trajs:
        plot_Rotation_axes(ax, est_traj, axis=0)
    ax.grid(visible=True, linestyle="--")
    ax.set_title("Rotation on x-axis (rad)", loc="left")

    ax = plt.subplot2grid(GRID_SHAPE, (1, 4), rowspan=1, colspan=2)
    plot_Rotation_axes(ax, ref_traj, axis=1)
    for est_traj in est_trajs:
        plot_Rotation_axes(ax, est_traj, axis=1)
    ax.grid(visible=True, linestyle="--")
    ax.set_title("Rotation on y-axis (rad)", loc="left")

    ax = plt.subplot2grid(GRID_SHAPE, (2, 4), rowspan=1, colspan=2)
    plot_Rotation_axes(ax, ref_traj, axis=2)
    for est_traj in est_trajs:
        plot_Rotation_axes(ax, est_traj, axis=2)
    ax.grid(visible=True, linestyle="--")
    ax.set_title("Rotation on z-axis (rad)", loc="left")

    plt.tight_layout()
    plt.savefig(str(file_name))
    plt.close()


def PlotTrajectory(trajs: list[PlotableTrajectory], file_name: Path):
    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    for traj in trajs:
        plot_Trajectory(ax, traj)
        plot_LostTrackAt(ax, traj)
    ax.legend(frameon=False)
    ax.set_title("Trajectory on x-y plane")
    ax.set_aspect("equal", adjustable="datalim")

    plt.tight_layout()
    plt.savefig(str(file_name))
    plt.close()

def AnalyzeRTE_with_Covariance(gt_traj: PlotableMotions, trajs: list[PlotableMotions], quality: torch.Tensor, file_name: Path):
    # Start plotting
    ax1: Axes
    fig, (ax1, ax2) = plt.subplots(2, 1, dpi=300)
    for est_traj in trajs:
        plot_MotionROE(ax1, gt_traj, est_traj)
    
    ax2: Axes
    ax2.set_title("Estimated quality over time")
    ax2.plot(quality.numpy(), label="flow_cov")
    ax2.set_ylim(bottom=quality[1:].min().item(), top=quality[1:].max().item())

    ax1.grid(visible=True, linestyle="--")
    ax2.grid(visible=True, linestyle="--")
    
    plt.tight_layout()
    plt.savefig(str(file_name))
    plt.close()
