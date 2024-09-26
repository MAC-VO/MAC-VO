import cv2
import torch
import matplotlib.cm as cm
import numpy as np
from typing import Any

from flow_vis import flow_to_color

from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from matplotlib.collections import EllipseCollection

from .Color import getColor
from Module.Map import BatchObservation


def plot_image(ax: Axes, image: torch.Tensor | np.ndarray) -> AxesImage:
    """
    Expect Shape and range:
        Image - (H, W, 3) - Any range
    """
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    ax.axis("off")
    cax = ax.imshow(image)
    return cax

def plot_whiten_image(ax: Axes, image: torch.Tensor | np.ndarray) -> AxesImage:
    """
    Expect Shape and range:
        Image - (H, W, 3) - [0., 1.]
    """
    if isinstance(image, torch.Tensor):
        image_np: np.ndarray = image.cpu().numpy()
    else:
        image_np = image
    ax.axis("off")
    gray_img = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    gray_img = gray_img * 0.25 + 0.75   #type: ignore
    cax = ax.imshow(gray_img, cmap="gray", vmin=0, vmax=1)
    return cax

def plot_scalarmap(ax: Axes, data: torch.Tensor | np.ndarray, vmin: float | None=None, vmax: float | None=None, alpha: float | None=None) -> AxesImage:
    """
    Expect Shape and range:
        Image - (H, W) - Any range
    """
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    ax.axis("off")
    cax = ax.imshow(data, cmap="plasma", vmin=vmin, vmax=vmax, alpha=alpha)
    return cax

def plot_flow(ax: Axes, flow: torch.Tensor) -> AxesImage:
    """
    Expect Shape and range:
        flow - (2, H, W) - Any range
    """
    assert flow.size(0) == 2    # 2 x H x W
    color_map = flow_to_color(flow.permute(1, 2, 0).numpy())
    return plot_image(ax, color_map)

def plot_keypoints(ax: Axes, keypoints: torch.Tensor, color=(0.0, 0.0, 1.0), s: float=1):
    ax.scatter(keypoints[..., 0], keypoints[..., 1], plotnonfinite=True, s=s, marker="x", color=color)

def plot_observations(
        ax: Axes, obs: BatchObservation,
        color_on_depthcov: bool = True,
        include_plot_kwargs: None | dict = None,
        exclude_plot_kwargs: None | dict = None,
        include_mask: torch.Tensor | None = None,
    ) -> None:
    default_include_kwargs: dict[str, Any] = dict(s=1, marker="x", color=(0.0, 0.0, 1.0))
    default_exclude_kwargs: dict[str, Any] = dict(s=1, marker="x", color=(1.0, 0.0, 0.0))
    
    default_include_kwargs |= dict() if include_plot_kwargs is None else include_plot_kwargs
    default_exclude_kwargs |= dict() if exclude_plot_kwargs is None else exclude_plot_kwargs

    if include_mask is None:
        include_mask = torch.tensor([True] * len(obs), dtype=torch.bool)
    
    excluded = obs[~include_mask]
    included = obs[include_mask]
    
    color_on_depthcov = color_on_depthcov and bool((obs.cov_pixel_d != -1).all().item())
    
    if len(excluded) > 0:
        excluded_uv = excluded.pixel_uv
        ax.scatter(excluded_uv[..., 0], excluded_uv[..., 1], plotnonfinite=True, **default_exclude_kwargs)
    
    if len(included) > 0:
        included_uv = included.pixel_uv
        if color_on_depthcov:
            del default_include_kwargs["color"]
            ax.scatter(included_uv[..., 0], included_uv[..., 1], 
                       c=included.pixel_d.numpy(), 
                       cmap="plasma", plotnonfinite=True, **default_include_kwargs)
        else:
            ax.scatter(included_uv[..., 0], included_uv[..., 1], plotnonfinite=True, **default_include_kwargs)

def plot_obs_flowcov(ax: Axes, obs: BatchObservation, color_on_depthcov: bool = True, scale_stdev: float = 3.):    
    color_on_depthcov = color_on_depthcov and bool((obs.cov_pixel_d != -1).all().item())
    
    offsets = obs.pixel_uv
    u_size = obs.cov_pixel_uv[..., 0].sqrt()
    v_size = obs.cov_pixel_uv[..., 1].sqrt()
    
    if color_on_depthcov:
        included_clr = obs.cov_pixel_d
        colors = cm.plasma(included_clr)    #type: ignore
        colors[..., -1] = 0.5
    else:
        colors = (0.2078431373, 0.6745098039, 0.6431372549, 0.5)

    ellipses = EllipseCollection(
        widths=u_size * scale_stdev,
        heights=v_size * scale_stdev,
        angles=torch.ones_like(u_size).numpy(),
        units="xy",
        offsets=offsets,
        transOffset=ax.transData,
        facecolors=colors,
        edgecolors="none",
    )
    ax.add_collection(ellipses)

def plot_sparsification(ax: Axes, est_curve, oracle_curve):
    x_est = np.linspace(0, 1, len(est_curve))
    x_orc = np.linspace(0, 1, len(oracle_curve))
    
    ax.plot(x_est, est_curve, label="Estimated Covariance", color=getColor("-", 4, 0))
    ax.plot(x_orc, oracle_curve, label="Oracle", color=getColor("-", 6, 0))
    ax.set_xlabel("Removed Points")
    ax.set_ylabel("AEPE (normalized)")
    ax.legend(frameon=False)
