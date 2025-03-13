import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

from Utility.PrettyPrint import Logger
from Utility.Sandbox import Sandbox
from Utility.Plot import plot_cumulative_density



def plot_compare_plot(spaces: list[str]):
    fig = plt.figure(figsize=(5, 5), dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    
    for space in spaces:
        box = Sandbox.load(space)
        if not box.path("time_elapsed.json").exists():
            Logger.write("warn", f"Sandbox at path {space} does not have 'time_elapsed.json', passed.")
            continue
        with box.open("time_elapsed.json", "r") as f:
            timestamps = json.load(f)
        timestamps = np.array(timestamps)
        elapsed = timestamps[1:] - timestamps[:-1]
        plot_cumulative_density(elapsed, label=box.folder.parent.name)(ax)
    
    ax.legend(frameon=False)
    ax.set_xlim(left=0.0, right=1.0)
    ax.set_ylabel("Proportion of frame")
    ax.set_xlabel("Time elapsed (sec)")
    plt.tight_layout()
    plt.savefig("output.png")
    plt.close()



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
    
    plot_compare_plot(spaces)
