import torch
import pypose as pp


def filterPointsInRange(pts1:torch.Tensor, u_range: tuple[int, int], v_range: tuple[int, int]) -> torch.Tensor:
    u_min, u_max = u_range
    v_min, v_max = v_range

    u_selector = torch.logical_and(pts1[..., 0] < u_max, pts1[..., 0] > u_min)
    v_selector = torch.logical_and(pts1[..., 1] < v_max, pts1[..., 1] > v_min)
    selector = torch.logical_and(u_selector, v_selector)

    return selector

def pixel2point_NED(pixels: torch.Tensor, depths: torch.Tensor, intrinsics: torch.Tensor):
    # pp.pixel2point will output points in EDN coordinate, we will convert it to NED coord.
    return pp.pixel2point(pixels, depths, intrinsics).roll(shifts=1, dims=-1)

def point2pixel_NED(points: torch.Tensor, intrinsics: torch.Tensor):
    # pp.pixel2point will output points in EDN coordinate, we will convert it to NED coord.
    return pp.point2pixel(points.roll(shifts=-1, dims=-1), intrinsics)
