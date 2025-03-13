import argparse
from pathlib import Path
from typing import Literal

from Utility.Plot import getColor, AnalyzeRotation, AnalyzeTranslation, PlotTrajectory, AnalyzeRTE_cdf, AnalyzeROE_cdf
from Utility.PrettyPrint import ColoredTqdm, Logger
from Utility.Sandbox import Sandbox
from Utility.Trajectory import Trajectory

NEED_ALIGN_SCALE: dict[str, Literal["Dynamic"] | float] = {
    "dpvo"         : "Dynamic",
    "droid"        : "Dynamic",
    "tartanvo_mono": "Dynamic",
    "mast3r"       : "Dynamic",
}


def plot_separately(spaces: list[str]):
    for spaceid in ColoredTqdm(spaces, desc="Plotting"):
        exp_space = Sandbox.load(spaceid)
        config = exp_space.config

        try:
            gt_traj, est_traj = Trajectory.from_sandbox(exp_space, align_time="est->gt")
            est_traj.plot_kwargs |= {"color": getColor("-", 4, 0)}
            if gt_traj is None:
                Logger.write("warn", f"Unable to plot error analysis for {est_traj} since could not load GT trajectory.")
                return

            gt_traj.plot_kwargs  |= {"linewidth": 3, "linestyle": ":"}
            
            for key, scale in NEED_ALIGN_SCALE.items():
                if key not in est_traj.name.lower(): continue
                
                Logger.write("info", f"{est_traj} --[align_scale={scale}]-> {gt_traj}")
                if scale == "Dynamic": est_traj.data = est_traj.data.align_scale(gt_traj.data)
                else: est_traj.data = est_traj.data.scale(scale)
                break

            est_traj.data = est_traj.data.align_origin(gt_traj.data)
            name = config.Project if hasattr(config, "Project") else est_traj.name
            AnalyzeTranslation(
                [(gt_traj.apply(lambda traj: traj.as_motion), est_traj.apply(lambda traj: traj.as_motion))],
                Path("Results", f"{name}_TranslationErr.png")
            )
            AnalyzeRotation(
                [(gt_traj.apply(lambda traj: traj.as_motion), est_traj.apply(lambda traj: traj.as_motion))],
                Path("Results", f"{name}_RotationErr.png")
            )
            PlotTrajectory([gt_traj, est_traj], Path("Results", f"{name}_Trajectory.png"))
        except Exception as e:
            Logger.show_exception()


def plot_jointly(spaces: list[str]):
    trajs = [Trajectory.from_sandbox(Sandbox.load(space)) for space in spaces]

    for idx, (gt_traj, est_traj) in enumerate(trajs):
        est_traj.plot_kwargs |= {"color": getColor("-", (idx * 2) % 7, 0)}
        gt_traj.plot_kwargs  |= {"linewidth": 4, "linestyle": "--"}
        
        for key, scale in NEED_ALIGN_SCALE.items():
            if key not in est_traj.name.lower(): continue
            
            Logger.write("info", f"{est_traj} --[align_scale={scale}]-> {gt_traj}")
            if scale == "Dynamic": est_traj.data = est_traj.data.align_scale(gt_traj.data)
            else: est_traj.data = est_traj.data.scale(scale)
            break
        
        est_traj.data = est_traj.data.align_origin(gt_traj.data)

    gt_traj, est_trajs = trajs[0][0], [est for _, est in trajs]
    PlotTrajectory([gt_traj] + est_trajs, Path("Results", f"{gt_traj.name}_Compare.png"))
    
    trajs_motion = [
        (gt_traj.apply(lambda x: x.as_motion), est_traj.apply(lambda x: x.as_motion))
        for (gt_traj, _), est_traj in zip(trajs, est_trajs)
    ]
    
    AnalyzeTranslation(
        trajs_motion,
        Path("Results", f"Combined_trel.png")
    )
    AnalyzeRotation(
        trajs_motion,
        Path("Results", f"Combined_rrel.png")
    )
    AnalyzeRTE_cdf(
        trajs_motion,
        None,
        Path("Results", f"Combined_RTEcdf.png")
    )
    AnalyzeROE_cdf(
        trajs_motion,
        None,
        Path("Results", f"Combined_ROEcdf.png")
    )


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
