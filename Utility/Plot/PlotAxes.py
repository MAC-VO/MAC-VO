import torch
import numpy as np
import typing as T

from matplotlib.axes import Axes
from matplotlib.collections import EllipseCollection, LineCollection
from matplotlib.patches import Ellipse
from functools import wraps
from ..Utils import IgnoreException
from ..Extensions import Chain

# For plotting
from flow_vis import flow_to_color
from scipy.stats import chi2
# End


I = T.ParamSpec("I")
def as_chain_func(func: T.Callable[T.Concatenate[Axes, I], Axes | None]) -> T.Callable[I, Chain[Axes, Axes]]:
    """
    Allows us to chain plotting operations efficiently
    """
    @wraps(func)
    def implement(*args: I.args, **kwargs: I.kwargs) -> Chain[Axes, Axes]:
        return Chain[Axes, Axes](lambda ax: res if (res := func(ax, *args, **kwargs)) else ax)
    return implement


@as_chain_func
@IgnoreException
def plot_start(ax: Axes) -> Axes: 
    return ax

@as_chain_func
@IgnoreException
def plot_no_border(ax: Axes) -> Axes:
    ax.axis("off")
    return ax

@as_chain_func
@IgnoreException
def plot_image(ax: Axes, image: torch.Tensor | np.ndarray) -> Axes:
    """
    Expect Shape and range:
        Image - (H, W, 3) - Any range
    """
    if isinstance(image, torch.Tensor): image = image.cpu().numpy()
    ax.imshow(image)
    return ax

def plot_whiten_image(image: torch.Tensor | np.ndarray, whiten: float=0.75) -> Chain[Axes, Axes]:
    """
    Expect Shape and range:
        Image - (H, W, 3) - [0., 1.]
    """
    H, W, _ = image.shape
    return plot_image(image) >> plot_mask(np.ones((H, W)) * (1 - whiten))

@as_chain_func
@IgnoreException
def plot_scalarmap(ax: Axes, data: torch.Tensor | np.ndarray | None, vmin: float | None=None, vmax: float | None=None, alpha: float | None=None, colorbar: bool=False) -> Axes:
    """
    Expect Shape and range:
        Image - (H, W) - Any range
    """
    if data is None: return ax
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    ax.axis("off")
    cax = ax.imshow(data, cmap="plasma", vmin=vmin, vmax=vmax, alpha=alpha)
    if colorbar:
        fig = ax.get_figure()
        assert fig is not None
        fig.colorbar(cax, ax=ax)
    return ax

@as_chain_func
@IgnoreException
def plot_flow(ax: Axes, flow: torch.Tensor | None | np.ndarray) -> Axes:
    """
    Expect Shape and range:
        flow - (2, H, W) - Any range
    """
    if flow is None: return ax
    
    assert flow.shape[0] == 2    # 2 x H x W
    if isinstance(flow, torch.Tensor):
        color_map = flow_to_color(flow.permute(1, 2, 0).numpy())
    else:
        color_map = flow_to_color(flow.transpose((1, 2, 0)))
    return plot_image(color_map)(ax)

@as_chain_func
@IgnoreException
def plot_mask(ax: Axes, mask: torch.Tensor | np.ndarray | None) -> Axes:
    if mask is None: return ax
    if isinstance(mask, torch.Tensor):
        mask = mask.float().detach().cpu().numpy()
    
    H, W = mask.shape
    rgba_mask = np.zeros((H, W, 4))
    rgba_mask[..., 0:3] = 1
    rgba_mask[..., 3] = 1 - mask  # Alpha channel (0 = transparent, 1 = opaque)
    ax.imshow(rgba_mask)
    return ax

@as_chain_func
@IgnoreException
def plot_keypoints(ax: Axes, keypoints: torch.Tensor, depth_cov: torch.Tensor | None, **plot_kwargs) -> Axes:
    keypoints = keypoints.detach().cpu()
    if depth_cov is not None:
        ax.scatter(
            keypoints[..., 0], keypoints[..., 1],
            c=depth_cov.detach().cpu().numpy(),
            cmap="plasma", plotnonfinite=True, edgecolors='none',
            **plot_kwargs
        )
    else:
        ax.scatter(keypoints[..., 0], keypoints[..., 1], plotnonfinite=True,
                   **plot_kwargs)
    return ax

@as_chain_func
@IgnoreException
def plot_flow_cov(ax: Axes, keypoints: torch.Tensor, flow_cov: torch.Tensor | None, scale: float=3., colors=(0.2078431373, 0.6745098039, 0.6431372549, 0.5)) -> Axes:
    if flow_cov is None: return ax
    
    offsets = keypoints.detach().cpu()

    # Extract elements from the Nx3 covariance tensor
    sigma_uu = flow_cov[..., 0]  # Variance in u
    sigma_vv = flow_cov[..., 1]  # Variance in v
    sigma_uv = flow_cov[..., 2]  # Covariance term

    # Construct full covariance matrices in batch
    cov_matrices = torch.empty(flow_cov.size(0), 2, 2, device=flow_cov.device)
    cov_matrices[:, 0, 0] = sigma_uu
    cov_matrices[:, 1, 1] = sigma_vv
    cov_matrices[:, 0, 1] = sigma_uv
    cov_matrices[:, 1, 0] = sigma_uv

    # Compute eigenvalues and eigenvectors for all covariance matrices
    eigvals, eigvecs = torch.linalg.eigh(cov_matrices)  # Batched eigen decomposition

    # Eigenvalues correspond to semi-axis lengths (variance, so take sqrt for stddev)
    semi_axes = torch.sqrt(eigvals)  # Shape: (N, 2)
    major_axes = semi_axes[:, 1]     # Larger eigenvalue -> major axis
    minor_axes = semi_axes[:, 0]     # Smaller eigenvalue -> minor axis

    # Compute angles from eigenvectors
    angles = torch.atan2(eigvecs[:, 1, 1], eigvecs[:, 0, 1])  # Angle in radians
    angles = torch.rad2deg(angles)  # Convert to degrees

    # Prepare ellipse properties
    widths = (major_axes * scale).detach().cpu().numpy()
    heights = (minor_axes * scale).detach().cpu().numpy()
    angles = angles.detach().cpu().numpy()

    # Set face colors
    colors = colors

    # Create EllipseCollection
    ellipses = EllipseCollection(
        widths=widths,
        heights=heights,
        angles=angles,
        units="xy",
        offsets=offsets,
        transOffset=ax.transData,
        facecolors=colors,
        edgecolors="none",
    )

    # Add ellipses to the plot
    ax.add_collection(ellipses)
    return ax

@as_chain_func
@IgnoreException
def plot_histogram(ax: Axes, data: torch.Tensor | np.ndarray | list[float], bins: tuple[T.Literal["num", "width"], float], **hist_kwargs) -> Axes:
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    if isinstance(data, np.ndarray):
        data = data.flatten()
    
    match bins:
        case "num", bin_num:
            bin_num = int(bin_num)
            ax.hist(data, bins=bin_num, **hist_kwargs)
        case "width", bin_width:
            ax.hist(data, bins=np.arange(min(data), max(data) + bin_width, bin_width).tolist(), **hist_kwargs)
    
    return ax

@as_chain_func
@IgnoreException
def plot_eq_aspect(ax: Axes) -> Axes:
    ax.set_aspect('equal', adjustable='box')
    return ax

@as_chain_func
@IgnoreException
def plot_LinewithAlpha(ax: Axes, x: np.ndarray, y: np.ndarray, color, alpha: np.ndarray, linewidth: float=1., linestyle: str="-", label: str | None = None) -> Axes:
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
    ax.add_collection(line_collect)
    ax.set_xlim(min(x.min(), ax.get_xlim()[0]), max(x.max(), ax.get_xlim()[1]))
    ax.set_ylim(min(y.min(), ax.get_ylim()[0]), max(y.max(), ax.get_ylim()[1]))
    return ax

@as_chain_func
@IgnoreException
def plot_gaussian_conf(ax: Axes, mean: torch.Tensor | np.ndarray, cov_matrix: torch.Tensor | np.ndarray, confidence: float) -> Axes:
    """
    Given a 2x2 covariance matrix, plot the ellipse representing confidence interval of the designated confidence
    value. Confidence in range of [0, 1]
    """
    assert len(cov_matrix.shape) == 2 and cov_matrix.shape[0] == 2 and cov_matrix.shape[1] == 2
    assert mean.shape[0] == 2 and len(mean.shape) == 1
    
    if isinstance(mean, torch.Tensor):
        mean = mean.cpu().numpy()
    if isinstance(cov_matrix, torch.Tensor):
        cov_matrix = cov_matrix.cpu().numpy()
    
    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Sort eigenvalues and eigenvectors
    sort_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sort_indices]
    eigenvectors = eigenvectors[:, sort_indices]
    
    # Calculate scaling factor
    chisquare_val = chi2.ppf(confidence, df=2)
    scaling_factor = np.sqrt(chisquare_val)
    
    # Calculate ellipse parameters
    width, height = 2 * scaling_factor * np.sqrt(eigenvalues)
    angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
    
    # Create and plot ellipse
    ellipse = Ellipse((mean[0], mean[1]), width, height, angle=np.degrees(angle), 
                                  facecolor='none', edgecolor='red')
    
    ax.add_patch(ellipse)
    return ax

@as_chain_func
@IgnoreException
def plot_cumulative_density(ax: Axes, values: torch.Tensor | np.ndarray, **kwargs) -> Axes:
    if isinstance(values, torch.Tensor):
        values = values.cpu().numpy()
    
    ax.ecdf(values, **kwargs)
    return ax


@as_chain_func
@IgnoreException
def plot_kp_correspondence(ax: Axes, kp1: torch.Tensor | np.ndarray, kp2: torch.Tensor | np.ndarray, color='blue', linewidth=1.5) -> Axes:
    if isinstance(kp1, torch.Tensor):
        kp1 = kp1.cpu().detach().numpy()
    if isinstance(kp2, torch.Tensor):
        kp2 = kp2.cpu().detach().numpy()
    
    # Construct segments as (N, 2, 2), where each row is [[x1, y1], [x2, y2]]
    segments = np.stack((kp1, kp2), axis=1)  # Shape (N, 2, 2)

    # Create LineCollection
    lc = LineCollection(segments, color=color, linewidth=linewidth) #type: ignore

    ax.add_collection(lc)
    return ax
