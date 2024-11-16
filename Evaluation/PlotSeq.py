import argparse
from pathlib import Path

from Utility.Plot import getColor, AnalyzeRotation, AnalyzeTranslation, PlotTrajectory
from Utility.PrettyPrint import ColoredTqdm, Logger
from Utility.Sandbox import Sandbox
from Utility.Trajectory import Trajectory

NEED_ALIGN_SCALE = {"dpvo", "droid", "tartanvo_mono"}


def plot_separately(spaces: list[str]):
    for spaceid in ColoredTqdm(spaces, desc="Plotting"):
        exp_space = Sandbox.load(spaceid)
        config = exp_space.config
        if not exp_space.is_finished:
            Logger.write(
                "error",
                f"Experiment space {spaceid} (project: {config.Project if hasattr(config, 'Project') else 'N/A'}) did not finish. Is this run crashed?",
            )
            continue

        gt_traj, est_traj = Trajectory.from_sandbox(exp_space)
        est_traj.plot_kwargs |= {"color": getColor("-", 4, 0)}
        gt_traj.plot_kwargs  |= {"linewidth": 4, "linestyle": "--"}
        
        if gt_traj is None:
            Logger.write("warn", f"Failed to read ground truth trajectory, alignments may not work properly.")
        else:
            if any([kw in est_traj.name.lower() for kw in NEED_ALIGN_SCALE]):
                Logger.write("info", f"{est_traj} --[align]-> {gt_traj}")
                est_traj.data = est_traj.data.align_scale(gt_traj.data)

        name = config.Project if hasattr(config, "Project") else est_traj.name
        AnalyzeTranslation(
            gt_traj.apply(lambda traj: traj.as_motion), 
            [est_traj.apply(lambda traj: traj.as_motion)],
            Path("Results", f"{name}_TranslationErr.png")
        )
        AnalyzeRotation(
            gt_traj.apply(lambda traj: traj.as_motion),
            [est_traj.apply(lambda traj: traj.as_motion)],
            Path("Results", f"{name}_RotationErr.png")
        )
        PlotTrajectory([gt_traj, est_traj], Path("Results", f"{name}_Trajectory.png"))


def plot_jointly(spaces: list[str]):
    trajs = [Trajectory.from_sandbox(Sandbox.load(space)) for space in spaces]

    for gt_traj, est_traj in trajs:
        est_traj.plot_kwargs |= {"color": getColor("-", 4, 0)}
        gt_traj.plot_kwargs  |= {"linewidth": 4, "linestyle": "--"}
        
        if any([kw in est_traj.name.lower() for kw in NEED_ALIGN_SCALE]):
            Logger.write("info", f"{est_traj} --[align]-> {gt_traj}")
            est_traj.data = est_traj.data.align_scale(gt_traj.data)

    gt_traj, est_trajs = trajs[0][0], [est for gt, est in trajs]
    PlotTrajectory([gt_traj] + est_trajs, Path("Results", f"{gt_traj.name}_Compare.png"))


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--spaces", type=str, nargs="+", default=[])
    args.add_argument("--recursive", action="store_true", help="Find and evaluate on leaf sandboxes only.")
    args = args.parse_args()
    
    if args.recursive:
        spaces = []
        for space in args.spaces:
            spaces.extend([str(child.folder.absolute()) for child in Sandbox.load(space).get_leaves()])
        Logger.write("info", f"Found {len(spaces)} spaces to plot.")
    else:
        spaces = args.spaces
    plot_separately(spaces)
    plot_jointly(spaces)
