import torch
from pathlib import Path
from typing import Literal
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
    plot_CumulativeRTECurve,
    plot_CumulativeROECurve,
)
from .PlotAxes import plot_histogram
from .Color import getColor
from ..Trajectory import PlotableMotions, PlotableTrajectory
from ..Datatypes  import DepthCovPerformance, DepthPerformance, FlowCovPerformance, FlowPerformance


def AnalyzeTranslation(
    runs: list[tuple[PlotableMotions, PlotableMotions]], file_name: Path
):
    GRID_SHAPE = (3, 6)
    _ = plt.figure(figsize=(16, 6), dpi=300)
    ax = plt.subplot2grid(GRID_SHAPE, (0, 0), rowspan=3, colspan=2)
    for ref_traj, est_traj in runs:
        plot_motionRTE(ax, ref_traj, est_traj)
    ax.legend(frameon=False)
    ax.grid(visible=True, linestyle="--")
    ax.set_title("Translation Error (m)", loc="left")

    ax = plt.subplot2grid(GRID_SHAPE, (0, 2), rowspan=1, colspan=2)
    for ref_traj, est_traj in runs:
        plot_MotionRTE_axes(ax, ref_traj, est_traj, axis=0)
    ax.grid(visible=True, linestyle="--")
    ax.set_title("Translation Error on x-axis (m)", loc="left")

    ax = plt.subplot2grid(GRID_SHAPE, (1, 2), rowspan=1, colspan=2)
    for ref_traj, est_traj in runs:
        plot_MotionRTE_axes(ax, ref_traj, est_traj, axis=1)
    ax.grid(visible=True, linestyle="--")
    ax.set_title("Translation Error on y-axis (m)", loc="left")

    ax = plt.subplot2grid(GRID_SHAPE, (2, 2), rowspan=1, colspan=2)
    for ref_traj, est_traj in runs:
        plot_MotionRTE_axes(ax, ref_traj, est_traj, axis=2)
    ax.grid(visible=True, linestyle="--")
    ax.set_title("Translation Error on z-axis (m)", loc="left")

    ax = plt.subplot2grid(GRID_SHAPE, (0, 4), rowspan=1, colspan=2)
    plot_Translation_axes(ax, runs[0][0], axis=0)
    for _, est_traj in runs:
        plot_Translation_axes(ax, est_traj, axis=0)
    ax.grid(visible=True, linestyle="--")
    # ax.set_ylim(bottom=-1., top=1.)
    ax.set_title("Translation on x-axis (m)", loc="left")

    ax = plt.subplot2grid(GRID_SHAPE, (1, 4), rowspan=1, colspan=2)
    plot_Translation_axes(ax, runs[0][0], axis=1)
    for _, est_traj in runs:
        plot_Translation_axes(ax, est_traj, axis=1)
    ax.grid(visible=True, linestyle="--")
    # ax.set_ylim(bottom=-1., top=1.)
    ax.set_title("Translation on y-axis (m)", loc="left")

    ax = plt.subplot2grid(GRID_SHAPE, (2, 4), rowspan=1, colspan=2)
    plot_Translation_axes(ax, runs[0][0], axis=2)
    for _, est_traj in runs:
        plot_Translation_axes(ax, est_traj, axis=2)
    ax.grid(visible=True, linestyle="--")
    # ax.set_ylim(bottom=-1., top=1.)
    ax.set_title("Translation on z-axis (m)", loc="left")

    plt.tight_layout()
    plt.savefig(str(file_name))
    plt.close()


def AnalyzeRotation(runs: list[tuple[PlotableMotions, PlotableMotions]], file_name: Path):
    GRID_SHAPE = (3, 6)
    _ = plt.figure(figsize=(16, 6), dpi=300)
    ax = plt.subplot2grid(GRID_SHAPE, (0, 0), rowspan=3, colspan=2)
    for ref_traj, est_traj in runs:
        plot_MotionROE(ax, ref_traj, est_traj)
    ax.legend(frameon=False)
    ax.grid(visible=True, linestyle="--")
    ax.set_title("Rotation Error (rad)", loc="left")

    ax = plt.subplot2grid(GRID_SHAPE, (0, 2), rowspan=1, colspan=2)
    for ref_traj, est_traj in runs:
        plot_MotionROE_axes(ax, ref_traj, est_traj, axis=0)
    ax.grid(visible=True, linestyle="--")
    ax.set_title("Rotation Error on x-axis (rad)", loc="left")

    ax = plt.subplot2grid(GRID_SHAPE, (1, 2), rowspan=1, colspan=2)
    for ref_traj, est_traj in runs:
        plot_MotionROE_axes(ax, ref_traj, est_traj, axis=1)
    ax.grid(visible=True, linestyle="--")
    ax.set_title("Rotation Error on y-axis (rad)", loc="left")

    ax = plt.subplot2grid(GRID_SHAPE, (2, 2), rowspan=1, colspan=2)
    for ref_traj, est_traj in runs:
        plot_MotionROE_axes(ax, ref_traj, est_traj, axis=2)
    ax.grid(visible=True, linestyle="--")
    ax.set_title("Rotation Error on z-axis (rad)", loc="left")

    ax = plt.subplot2grid(GRID_SHAPE, (0, 4), rowspan=1, colspan=2)
    plot_Rotation_axes(ax, runs[0][0], axis=0)
    for ref_traj, est_traj in runs:
        plot_Rotation_axes(ax, est_traj, axis=0)
    ax.grid(visible=True, linestyle="--")
    ax.set_title("Rotation on x-axis (rad)", loc="left")

    ax = plt.subplot2grid(GRID_SHAPE, (1, 4), rowspan=1, colspan=2)
    plot_Rotation_axes(ax, runs[0][0], axis=1)
    for ref_traj, est_traj in runs:
        plot_Rotation_axes(ax, est_traj, axis=1)
    ax.grid(visible=True, linestyle="--")
    ax.set_title("Rotation on y-axis (rad)", loc="left")

    ax = plt.subplot2grid(GRID_SHAPE, (2, 4), rowspan=1, colspan=2)
    plot_Rotation_axes(ax, runs[0][0], axis=2)
    for ref_traj, est_traj in runs:
        plot_Rotation_axes(ax, est_traj, axis=2)
    ax.grid(visible=True, linestyle="--")
    ax.set_title("Rotation on z-axis (rad)", loc="left")

    plt.tight_layout()
    plt.savefig(str(file_name))
    plt.close()


def PlotTrajectory(trajs: list[PlotableTrajectory], file_name: Path):
    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = fig.add_subplot(2, 2, 1)
    for traj in trajs:
        plot_Trajectory(ax, 0, 1, traj)
        plot_LostTrackAt(ax, 0, 1, traj)
    ax.legend(frameon=False)
    ax.set_title("Trajectory on x-y plane")
    ax.set_aspect("equal", adjustable="datalim")
    
    ax = fig.add_subplot(2, 2, 2)
    for traj in trajs:
        plot_Trajectory(ax, 0, 2, traj)
        plot_LostTrackAt(ax, 0, 2, traj)
    ax.legend(frameon=False)
    ax.set_title("Trajectory on x-z plane")
    ax.set_aspect("equal", adjustable="datalim")
    
    ax = fig.add_subplot(2, 2, 3)
    for traj in trajs:
        plot_Trajectory(ax, 1, 2, traj)
        plot_LostTrackAt(ax, 1, 2, traj)
    ax.legend(frameon=False)
    ax.set_title("Trajectory on y-z plane")
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


def plot_flow_performance(flow_results: list[FlowPerformance], file_name: str):
    GRID_SHAPE = (1, 2)
    _ = plt.figure(figsize=(16, 6), dpi=300)
    
    fields = ["masked_epe", "epe"]
    for i, field in enumerate(fields):
        row: list[float] = [getattr(res, field) for res in flow_results]
        ax = plt.subplot2grid(GRID_SHAPE, (i // 2, i % 2), rowspan=1, colspan=1)
        ax.set_title(f"Distribution {field}", loc="left")
        plot_histogram(row, bins=('width', 2.))(ax)
    
    plt.tight_layout()
    plt.savefig(str(file_name))
    plt.close()


def AnalyzeRTE_cdf(runs: list[tuple[PlotableMotions, PlotableMotions]], axis: Literal[0, 1, 2] | None, file_name: Path):
    ax: Axes
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

    for ref_traj, est_traj in runs:
        plot_CumulativeRTECurve(ax, ref_traj, est_traj, axis)
    
    ax.legend(frameon=False)
    ax.set_xlim(left=1e-5)
    ax.set_ylim(bottom=0., top=1.)
    ax.set_xlabel("RTE (unit: m, log-scaled)")
    ax.set_ylabel("Proportion of frames")
    ax.set_xscale("log")
    ax.grid(True, linestyle="--")
    plt.tight_layout()
    fig.savefig(str(file_name))
    plt.close()

def AnalyzeROE_cdf(runs: list[tuple[PlotableMotions, PlotableMotions]], axis: Literal[0, 1, 2, None], file_name: Path):
    ax: Axes
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

    for ref_traj, est_traj in runs:
        plot_CumulativeROECurve(ax, ref_traj, est_traj, axis)
    
    ax.legend(frameon=False)
    ax.set_xlim(left=1e-5)
    ax.set_ylim(bottom=0., top=1.)
    ax.set_xlabel("ROE (unit: rad, log-scaled)")
    ax.set_ylabel("Proportion of frames")
    ax.set_xscale("log")
    ax.grid(True, linestyle="--")
    plt.tight_layout()
    fig.savefig(str(file_name))
    plt.close()
