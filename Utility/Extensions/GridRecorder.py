import numpy as np
import typing as T
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


class GridRecorder:
    def __init__(self, axis0: tuple[float, float, float], axis1: tuple[float, float, float]):
        self.axis0_scale, self.axis1_scale = 1/axis0[2], 1/axis1[2]
        self.axis0_start, self.axis0_end, self.axis0_step = axis0
        self.axis1_start, self.axis1_end, self.axis1_step = axis0

        self.get_axis0_index: T.Callable[[np.ndarray], np.ndarray] = lambda v: ((v - self.axis0_start) * self.axis0_scale).astype(int)
        self.get_axis1_index: T.Callable[[np.ndarray], np.ndarray] = lambda v: ((v - self.axis1_start) * self.axis1_scale).astype(int)

        self.grid = np.zeros((
            int((self.axis0_end - self.axis0_start) * self.axis0_scale), 
            int((self.axis1_end - self.axis1_start) * self.axis1_scale)
        ), dtype=np.uint32)
        self.grid_size0 = self.grid.shape[0]
        self.grid_size1 = self.grid.shape[1]

    def store(self, axis0_values: np.ndarray, axis1_values: np.ndarray):
        axis0_idx = self.get_axis0_index(axis0_values)
        axis1_idx = self.get_axis1_index(axis1_values)
        
        mask0 = (0 <= axis0_idx) & (axis0_idx < self.grid_size0)
        mask1 = (0 <= axis1_idx) & (axis1_idx < self.grid_size1)
        mask  = mask0 & mask1
        axis0_idx, axis1_idx = axis0_idx[mask], axis1_idx[mask]
        
        index = np.stack([axis0_idx.flatten(), axis1_idx.flatten()], axis=-1)
        index, index_cnt = np.unique(index, axis=0, return_counts=True)

        self.grid[index[:, 0], index[:, 1]] += index_cnt.astype(np.uint32)
    
    def plot(self, ax: Axes, axis0_name: str="Axis 0", axis1_name: str="Axis 1", reduction: T.Literal["None", "Log"] = "None") -> Axes:
        match reduction:
            case "None": ax.imshow(self.grid, cmap='plasma')
            case "Log" : ax.imshow(np.log10(self.grid), cmap='plasma')
        
        mult_factor0 = max(self.grid_size0 // 10, 1)
        mult_factor1 = max(self.grid_size1 // 10, 1)
        ax.set_yticks(np.arange(start=0, stop=self.grid_size0, step=mult_factor0))
        ax.set_xticks(np.arange(start=0, stop=self.grid_size1, step=mult_factor1))
        ax.set_yticklabels(map(lambda x: f"{x:.2f}", np.arange(start=self.axis0_start, stop=self.axis0_end, step=self.axis0_step * mult_factor0)))
        ax.set_xticklabels(map(lambda x: f"{x:.2f}", np.arange(start=self.axis1_start, stop=self.axis1_end, step=self.axis1_step * mult_factor1)), rotation=90)
        ax.set_ylabel(axis0_name)
        ax.set_xlabel(axis1_name)
        return ax
    
    def plot_figure(self, axis0_name: str="Axis 0", axis1_name: str="Axis 1", reduction: T.Literal["None", "Log"] = "None") -> Figure:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        self.plot(ax, axis0_name, axis1_name, reduction)
        fig.set_dpi(300)
        fig.tight_layout()
        return fig


if __name__ == "__main__":
    recorder = GridRecorder((0, 1, 0.05), (0, 1, 0.05))
    values_0 = np.random.rand(640, 640)
    values_1 = np.random.rand(640, 640)
    recorder.store(values_0, values_1)
    
    recorder.plot(plt.gca())
    plt.savefig("output.png")
