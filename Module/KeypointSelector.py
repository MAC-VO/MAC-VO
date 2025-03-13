import math
from abc import ABC, abstractmethod
from types import SimpleNamespace

import torch


from .Frontend.StereoDepth import IStereoDepth
from .Frontend.Matching    import IMatcher
from DataLoader import StereoData

from Utility.Extensions import ConfigTestableSubclass
from Utility.Timer import Timer



class IKeypointSelector(ABC, ConfigTestableSubclass):
    """
    This module selects keypoint given current frame and (optionally) estimated depth, depth_cov, and flow_cov.
    
    The selector also receives an argument `numPoint` as hint to how many keypoints to select. This hint may *not* be 
    followed strictly.
    """
    def __init__(self, config: SimpleNamespace):
        self.config = config
    
    @abstractmethod
    def select_point(
        self,
        frame   : StereoData,
        numPoint: int,
        depth0_est: IStereoDepth.Output,
        depth1_est: IStereoDepth.Output,
        match_est: IMatcher.Output | None,
    ) -> torch.Tensor:
        """
        Select keypoint for tracking using given frame, (optionally) estimated depth, depth_cov, and flow_cov.
        
        Return keypoint as a FloatTensor with shape (N, 2) where keypoints are arranged in (u, v) format.
        
        ## NOTE
        
        this means that you need to output the index of keypoints in *different* coordinate system as pytorch.
        
        Use `image[kp[..., 1], kp[..., 0]]` to read value of image on all u-v coords of keypoints.
        The default output of this function is (0x2 torch.Tensor) which means no keypoints are selected.
        """
        return torch.zeros((0, 2), dtype=torch.long, device=self.config.device)
    

class SelectorCompose(IKeypointSelector):
    """
    Given multiple keypoint selectors and their weight, distribute keypoint selection 
    requirement to these according to the provided weight.
    """
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.selectors = [IKeypointSelector.instantiate(arg.type, arg.args) for arg in self.config.selector_args]

        self.weight = torch.tensor(self.config.weight)
        self.weight = self.weight / self.weight.sum()
    
    def select_point(self, frame: StereoData, numPoint: int, depth0_est: IStereoDepth.Output, depth1_est: IStereoDepth.Output, match_est: IMatcher.Output | None) -> torch.Tensor:
        keypoints = []
        for selector, weight in zip(self.selectors, self.weight):
            keypoints.append(selector.select_point(frame, int(numPoint * weight), depth0_est, depth1_est, match_est))
        return torch.cat(keypoints, dim=0)
    
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        assert config is not None
        for arg in config.selector_args:
            IKeypointSelector.is_valid_config(arg)
        assert isinstance(config.weight, list)
        for val in config.weight: assert isinstance(val, (int, float))


class MappingPointSelector(IKeypointSelector):
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        return cls._enforce_config_spec(config, {
            "max_depth": lambda v: isinstance(v, float),
            "max_depth_cov": lambda v: isinstance(v, float),
            "mask_width": lambda v: isinstance(v, int)
        })
    
    def select_point(self, frame: StereoData, numPoint: int, depth0_est: IStereoDepth.Output, depth1_est: IStereoDepth.Output, match_est: IMatcher.Output | None) -> torch.Tensor:
        assert depth0_est.cov is not None
        depth_mask     = depth0_est.depth < self.config.max_depth
        depth_cov_mask = depth0_est.cov   < self.config.max_depth_cov
        border_mask = torch.zeros_like(depth_mask, dtype=torch.bool)
        border_mask[
            ..., self.config.mask_width : -self.config.mask_width, self.config.mask_width : -self.config.mask_width
        ] = True
        
        candidates     = depth_mask & depth_cov_mask & border_mask
        selected_points = torch.nonzero(candidates, as_tuple=False)
        perm = torch.randperm(selected_points.size(0))[:numPoint]
        pixels = selected_points[perm][..., 2:].roll(shifts=1, dims=1)
        return pixels


class RandomSelector(IKeypointSelector):
    """
    Uniformly random select keypoints within the scope of [mask_width : -mask_width]
    """
    def select_point(self, frame: StereoData, numPoint: int, depth0_est: IStereoDepth.Output, depth1_est: IStereoDepth.Output, match_est: IMatcher.Output | None) -> torch.Tensor:
        h_indices = torch.randint(self.config.mask_width, frame.height - self.config.mask_width, (numPoint, 1), device=self.config.device)
        w_indices = torch.randint(self.config.mask_width, frame.width  - self.config.mask_width, (numPoint, 1), device=self.config.device)
        kps = torch.cat([w_indices, h_indices], dim=1)
        return kps
    
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "mask_width": lambda m: isinstance(m, int) and m >= 0,
            "device": lambda dev: isinstance(dev, str) and (("cuda" in dev) or (dev == "cpu"))
        })


class GradientSelector(IKeypointSelector):
    """
    Select keypoint based on gradient information. Will random select points with 
    local image gradient > config.grad_std.
    """
    def select_point(self, frame: StereoData, numPoint: int, depth0_est: IStereoDepth.Output, depth1_est: IStereoDepth.Output, match_est: IMatcher.Output | None) -> torch.Tensor:
        image = frame.imageL[0]

        image_grad = torch.nn.functional.conv2d(
            image.unsqueeze(0),
            torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
            .float()
            .expand((1, 3, 3, 3)),
            padding=1,
        )[0].abs()
        image_grad_avg = image_grad.mean(dim=(1, 2), keepdim=True)
        image_grad_std = image_grad.std(dim=(1, 2), keepdim=True)
        # Positions with sufficient gradient (feature) > +3std
        points = image_grad > (image_grad_avg + self.config.grad_std * image_grad_std)

        # Positions that are not too close to the edge of image
        border_mask = torch.zeros_like(points)
        border_mask[
            ..., self.config.mask_width : -self.config.mask_width, self.config.mask_width : -self.config.mask_width
        ] = 1.0
        points = points * border_mask
        selected_points = torch.nonzero(points, as_tuple=False)

        # Randomly select points
        perm = torch.randperm(selected_points.shape[0])[:numPoint]
        return selected_points[perm][..., 1:].roll(shifts=1, dims=1)

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "mask_width": lambda m: isinstance(m, int) and m >= 0,
            "grad_std"  : lambda g: isinstance(g, (int, float)) and g > 0.
        })


class SparseGradienSelector(IKeypointSelector):
    """
    Select keypoint based on gradient information. Will random select points with 
    local image gradient > config.grad_std.
    
    Ensured sparsity of keypoint by applying non-maximum suppresion (NMS) on image gradient
    of keypoint candidates.
    """
    def select_point(self, frame: StereoData, numPoint: int, depth0_est: IStereoDepth.Output, depth1_est: IStereoDepth.Output, match_est: IMatcher.Output | None) -> torch.Tensor:
        image = frame.imageL[0]

        image_grad = torch.nn.functional.conv2d(
            image.unsqueeze(0),
            torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
            .float()
            .expand((1, 3, 3, 3)),
            padding=1,
        )[0].abs()
        image_grad_avg = image_grad.mean(dim=(1, 2), keepdim=True)
        image_grad_std = image_grad.std(dim=(1, 2), keepdim=True)
        # Positions with sufficient gradient (feature) > +3std
        points = image_grad > (image_grad_avg + self.config.grad_std * image_grad_std)

        # Positions that are not too close to the edge of image
        border_mask = torch.zeros_like(points)
        border_mask[
            ..., self.config.mask_width : -self.config.mask_width, self.config.mask_width : -self.config.mask_width
        ] = 1.0
        points = points * border_mask

        # Positions that are sufficiently far away (sparse)
        image_grad_erode = torch.nn.functional.max_pool2d(
            image_grad,
            kernel_size=self.config.nms_size,
            stride=1,
            padding=(self.config.nms_size // 2),
        )
        image_grad_nms = image_grad == image_grad_erode
        points = points * image_grad_nms

        selected_points = torch.nonzero(points, as_tuple=False)

        # Randomly select points
        perm = torch.randperm(selected_points.shape[0])[:numPoint]
        return selected_points[perm][..., 1:].roll(shifts=1, dims=1)

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "mask_width": lambda m: isinstance(m, int) and m >= 0,
            "grad_std"  : lambda g: isinstance(g, (int, float)) and g > 0.,
            "nms_size"  : lambda k: isinstance(k, int) and k >= 0 and (k % 2 == 1),
        })


class GridSelector(IKeypointSelector):
    """
    Select keypoint following the grid - strictly uniform across the entire image.
    
    The requested `numPoint` will be used to estimate the spacing between keypoints, but the 
    selector may not generate exactly `numPoint` amount of keypoints.
    """
    def select_point(self, frame: StereoData, numPoint: int, depth0_est: IStereoDepth.Output, depth1_est: IStereoDepth.Output, match_est: IMatcher.Output | None) -> torch.Tensor:
        h, w = frame.height, frame.width
        h -= 2 * self.config.mask_width
        w -= 2 * self.config.mask_width

        unit = max(1, int(math.sqrt(numPoint // 2)))

        mesh_u, mesh_v = torch.meshgrid(
            torch.arange(0, h, h // unit, device=self.config.device),
            torch.arange(0, w, w // (unit * 2), device=self.config.device),
            indexing="ij",
        )
        mesh_u, mesh_v = mesh_u.flatten(), mesh_v.flatten()

        points = torch.stack([mesh_v, mesh_u], dim=1)
        points += self.config.mask_width

        return points

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "mask_width": lambda m: isinstance(m, int) and m >= 0,
            "device": lambda dev: isinstance(dev, str) and (("cuda" in dev) or (dev == "cpu"))
        })


class CovAwareSelector(IKeypointSelector):
    """
    The keypoint selector used by the MAC-VO.
    
    Selecting keypoints based on estimated depth, depth_cov, and flow_cov. See sect III.B 
    of paper for detail.
    """
    @Timer.cpu_timeit("KPSelector.select")
    @Timer.gpu_timeit("KPSelector.select")
    @torch.inference_mode()
    def select_point(self, frame: StereoData, numPoint: int, depth0_est: IStereoDepth.Output, depth1_est: IStereoDepth.Output, match_est: IMatcher.Output | None) -> torch.Tensor:
        assert depth0_est.cov is not None
        assert depth1_est.cov is not None
        if self.config.max_depth == "auto": self.config.max_depth = frame.fx * frame.frame_baseline

        depth0_map     = depth0_est.depth.to(self.config.device)
        depth0_cov_map = depth0_est.cov.to(self.config.device)
        depth1_map     = depth1_est.depth.to(self.config.device)
        depth1_cov_map = depth1_est.cov.to(self.config.device)

        if match_est is not None and match_est.cov is not None:
            flow_cov_map = match_est.cov.to(self.config.device)
        else:
            flow_cov_map = None

        # Derive quality map
        quality_map = depth0_cov_map + depth1_cov_map
        if flow_cov_map is not None:
            flow_cov_map = (flow_cov_map[:, 0] + flow_cov_map[:, 1] - 2 * flow_cov_map[:, 2]).unsqueeze(1)
            quality_map *= flow_cov_map
        
        # Apply NMS on quality map
        quality_map_erode = -torch.nn.functional.max_pool2d(
            -quality_map,
            kernel_size=self.config.kernel_size,
            stride=1,
            padding=(self.config.kernel_size // 2),
        )
        quality_nms = torch.logical_and(quality_map == quality_map_erode, ~quality_map.isnan())
        
        # Positions that are not too close to the edge of image
        border_mask = torch.zeros_like(quality_nms, dtype=torch.bool)
        border_mask[
            ..., self.config.mask_width : -self.config.mask_width, self.config.mask_width : -self.config.mask_width
        ] = True
        
        # Positions that are sufficiently close to camera.
        depth_mask = (depth0_map < self.config.max_depth) & (depth1_map < self.config.max_depth)

        depth0_cov_thresh = min(self.config.max_depth_cov, depth0_cov_map[quality_nms].nanmedian().item() * 1.5)
        # depth1_cov_thresh = min(self.config.max_depth_cov, depth1_cov_map[quality_nms].nanmedian().item() * 2.0)
        depth0_cov_mask = depth0_cov_map < depth0_cov_thresh
        # depth1_cov_mask = depth1_cov_map < depth1_cov_thresh

        # Positions that has sufficiently small flow_cov
        if flow_cov_map is not None:
            flow_cov_thresh = min(self.config.max_match_cov, flow_cov_map[quality_nms].nanmedian().item() * 1.5)
            flow_cov_mask = flow_cov_map < flow_cov_thresh
        else:
            flow_cov_mask = None

        point_mask = torch.logical_and(quality_nms, border_mask)
        point_mask = torch.logical_and(point_mask, depth_mask)
        point_mask = torch.logical_and(point_mask, depth0_cov_mask)
        
        if flow_cov_mask is not None:
            point_mask = torch.logical_and(point_mask, flow_cov_mask)
        
        if depth0_est.mask is not None:
            point_mask = torch.logical_and(point_mask, depth0_est.mask.to(point_mask.device))
        
        if match_est is not None and match_est.mask is not None:
            point_mask = torch.logical_and(point_mask, match_est.mask.to(point_mask.device))
        
        # Select points
        # NOTE: potential performance bottleneck
        # this will trigger host-device sync and hang the CPU until CUDA stream finishes.
        selected_points = torch.nonzero(point_mask, as_tuple=False)
        # end

        # Randomly select points
        perm = torch.randperm(selected_points.size(0))[:numPoint]
        pixels = selected_points[perm][..., 2:].roll(shifts=1, dims=1)

        return pixels

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        assert config is not None
        cls._enforce_config_spec(config, {
            "device"        : lambda dev: isinstance(dev, str) and (("cuda" in dev) or (dev == "cpu")),
            "mask_width"    : lambda m: isinstance(m, int) and m >= 0,
            "max_depth"     : lambda dist: (dist == "auto") or (isinstance(dist, (int, float)) and dist > 0.),
            "kernel_size"   : lambda k: isinstance(k, int) and k > 0 and (k % 2 == 1),
            "max_depth_cov" : lambda c: isinstance(c, (int, float)) and c > 0.,
            "max_match_cov" : lambda c: isinstance(c, (int, float)) and c > 0.
        })


class CovAwareSelector_NoDepth(IKeypointSelector):
    """
    Selecting keypoints based on estimated flow_cov. 
    
    The main difference with CovAwareSelector is dropping filters related with depth (i.e. max_depth and depth_cov).
    """
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.fallback_grid_selector = GridSelector(SimpleNamespace(mask_width = self.config.mask_width, device=self.config.device))
    
    @Timer.cpu_timeit("KPSelector.select")
    @Timer.gpu_timeit("KPSelector.select")
    @torch.inference_mode()
    def select_point(self, frame: StereoData, numPoint: int, depth0_est: IStereoDepth.Output, depth1_est: IStereoDepth.Output, match_est: IMatcher.Output | None) -> torch.Tensor:
        if match_est is None or match_est.cov is None:
            return self.fallback_grid_selector.select_point(frame, numPoint, depth0_est, depth1_est, match_est)
        else:
            flow_cov_map = match_est.cov.to(self.config.device)

        # Derive quality map
        quality_map = (flow_cov_map[:, 0] + flow_cov_map[:, 1] - 2 * flow_cov_map[:, 2]).unsqueeze(1)
        flow_cov_map = quality_map
        
        # Apply NMS on quality map
        quality_map_erode = -torch.nn.functional.max_pool2d(
            -quality_map,
            kernel_size=self.config.kernel_size,
            stride=1,
            padding=(self.config.kernel_size // 2),
        )
        quality_nms = torch.logical_and(quality_map == quality_map_erode, ~quality_map.isnan())
        
        # Positions that are not too close to the edge of image
        border_mask = torch.zeros_like(quality_nms, dtype=torch.bool)
        border_mask[
            ..., self.config.mask_width : -self.config.mask_width, self.config.mask_width : -self.config.mask_width
        ] = True

        # Positions that has sufficiently small flow_cov
        flow_cov_thresh = min(self.config.max_match_cov, flow_cov_map[quality_nms].median().item() * 1.5)
        flow_cov_mask = flow_cov_map < flow_cov_thresh

        point_mask = torch.logical_and(quality_nms, border_mask)
        point_mask = torch.logical_and(point_mask, flow_cov_mask)
        
        if match_est is not None and match_est.mask is not None:
            point_mask = torch.logical_and(point_mask, match_est.mask.to(point_mask.device))
        
        # Select points
        # NOTE: potential performance bottleneck
        # this will trigger host-device sync and hang the CPU until CUDA stream finishes.
        selected_points = torch.nonzero(point_mask, as_tuple=False)
        # end

        # Randomly select points
        perm = torch.randperm(selected_points.size(0))[:numPoint]
        pixels = selected_points[perm][..., 2:].roll(shifts=1, dims=1)

        return pixels

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "device"        : lambda dev: isinstance(dev, str) and (("cuda" in dev) or (dev == "cpu")),
            "mask_width"    : lambda m: isinstance(m, int) and m >= 0,
            "kernel_size"   : lambda k: isinstance(k, int) and k > 0 and (k % 2 == 1),
            "max_match_cov" : lambda c: isinstance(c, (int, float)) and c > 0.
        })
