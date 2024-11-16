import argparse
from statistics import mean

from Evaluation.MetricsSeq import evaluateATE, evaluateROE, evaluateRTE, evaluateRPE
from Utility.Plot import getColor
from Utility.PrettyPrint import ColoredTqdm, Logger, print_as_table, save_as_csv
from Utility.Sandbox import Sandbox
from Utility.Trajectory import Trajectory


NEED_ALIGN_SCALE = {"dpvo", "droid", "tartanvo_mono"}


def EvaluateSequences(spaces: list[str], correct_scale=False):
    eval_results = []
    DECORATOR = "[S]" if correct_scale else ""

    pb = ColoredTqdm(spaces, desc="Evaluating")
    for spaceid in pb:
        exp_space = Sandbox.load(spaceid)
        config = exp_space.config
        
        if not exp_space.is_finished:
            Logger.write(
                "error",
                f"Experiment space {spaceid} (project: {config.Project}) did not finish. Is this run crashed?",
            )
            eval_results.append([config.Project] + ([None] * 12))
            continue

        gt_traj, est_traj = Trajectory.from_sandbox(exp_space)
        
        if est_traj is None:
            Logger.write("error", f"Unable to retrieve estimated trajectory from {config.Project}")
            eval_results.append([config.Project] + ([None] * 12))
            continue
        
        est_traj.plot_kwargs |= dict(color=getColor("-", 4, 0))

        if any([ kw in est_traj.name.lower() for kw in NEED_ALIGN_SCALE]):
            est_traj = est_traj.apply(lambda traj : traj.align_scale(gt_traj.data))

        ate_res = evaluateATE(gt_traj.data.as_evo, est_traj.data.as_evo, correct_scale=correct_scale)
        rte_res = evaluateRTE(gt_traj.data.as_evo, est_traj.data.as_evo, correct_scale=correct_scale)
        roe_res = evaluateROE(gt_traj.data.as_evo, est_traj.data.as_evo, correct_scale=correct_scale)
        rpe_res = evaluateRPE(gt_traj.data.as_evo, est_traj.data.as_evo, correct_scale=correct_scale)
        eval_results.append(
            [
                est_traj.name,
                ate_res.stats["mean"], ate_res.stats["std"], ate_res.stats["rmse"],
                rte_res.stats["mean"], rte_res.stats["std"], rte_res.stats["rmse"],
                roe_res.stats["mean"], roe_res.stats["std"], roe_res.stats["rmse"],
                rpe_res.stats["mean"], rpe_res.stats["std"], rpe_res.stats["rmse"],
            ]
        )
    pb.close()
    
    eval_results.append(
        ["Average"] + list(map(mean, [
            [r[idx] for r in eval_results if r[idx] is not None] for idx in range(1, len(eval_results[0]))
        ])),
    )
    return [
        "Trajectory",
        f"μ_ATE{DECORATOR}", "σ_ATE", "RMSE_ATE",
        f"μ_RTE{DECORATOR}", "σ_RTE", "RMSE_RTE",
        f"μ_ROE{DECORATOR}", "σ_ROE", "RMSE_ROE",
        f"μ_RPE{DECORATOR}", "σ_RPE", "RMSE_RPE",
    ], eval_results


def EvaluateSequencesAvg(spaces: list[str]):
    _, metrics = EvaluateSequences(spaces)
    return ["μ_ATE", "μ_RTE", "μ_ROE", "μ_RPE"], [
        mean([m[1] for m in metrics]),
        mean([m[4] for m in metrics]),
        mean([m[7] for m in metrics]),
        mean([m[10] for m in metrics])
    ]


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--spaces", type=str, nargs="+", default=[])
    args.add_argument("--correctScale", action="store_true")
    args.add_argument("--recursive", action="store_true", help="Find and evaluate on leaf sandboxes only.")
    args.add_argument("--csv", type=str, default=None, required=False)
    args = args.parse_args()
    
    if args.recursive:
        spaces = []
        for space in args.spaces:
            spaces.extend([str(child.folder.absolute()) for child in Sandbox.load(space).get_leaves()])
        Logger.write("info", f"Found {len(spaces)} spaces to evaluate on.")
    else:
        spaces = args.spaces

    eval_header, eval_results = EvaluateSequences(
        spaces, correct_scale=args.correctScale
    )
    
    print_as_table(eval_header, eval_results, sort_rows=lambda row: row[0])

    if args.csv is not None:
        save_as_csv(eval_header, eval_results, args.csv, sort_rows=lambda row: row[0])
