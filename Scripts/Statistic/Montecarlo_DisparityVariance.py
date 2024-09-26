import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def Montecarlo_sample_depth(disparity: float, gamma: float, bl: float, fx: float, num_sample: int) -> torch.Tensor:
    sample_disp = ((torch.randn((num_sample,), device="cuda") * gamma * disparity) + disparity).abs()
    depth = (bl * fx) / sample_disp
    return depth.cpu()


def plot_experiment(ax, disparity: float, gamma: float, bl: float, fx: float, num_sample: int):
    
    
    sample_depth = Montecarlo_sample_depth(disparity, gamma, bl, fx, num_sample)
    
    # Plot approximated normal distribution
    mean = (bl * fx) / disparity
    std  = (bl * fx * gamma) / disparity
    x = np.linspace(mean - 4*std, mean + 4*std, 1000)
    pdf = norm.pdf(x, mean, std)

    # Plot histogram using the axes object
    ax.hist(sample_depth.numpy(), bins=100, density=True, color=(53/255, 172/255, 164/255), label=f"Simulation\nDisp~N({disparity}, {round((disparity * gamma) ** 2, 3)})")
    ax.plot(x, pdf, label=f'Our Approximation', color="orange")

    # Customize the plot
    
    ax.set_xlabel('Depth')
    ax.set_ylabel('Probability Density')
    ax.legend(loc="upper right")
    

fig, axs = plt.subplots(1, 3, figsize=(12, 3))
# ax.set_title(f'Monte Carlo simulation to depth distribution, )')
plot_experiment(axs[0], 1, 0.25, 0.25, 320., 50000)
plot_experiment(axs[1], 3, 0.2, 0.25, 320.,  50000)
plot_experiment(axs[2], 5, 0.1, 0.25, 320.,  50000)
fig.tight_layout()
fig.savefig("output.pdf",)
