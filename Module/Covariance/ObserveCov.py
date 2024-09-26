from abc import ABC, abstractmethod

import torch
from types import SimpleNamespace

from DataLoader import SourceDataFrame

from Utility.Math import gaussian_kernels, gaussian_mixture_mean_var
from Utility.PrettyPrint import Logger
from Utility.Visualizer import PLTVisualizer
from Utility.Extensions import ConfigTestableSubclass

# Interface #########################################################

class IObservationCov(ABC, ConfigTestableSubclass):
    def __init__(self, config: SimpleNamespace):
        self.config = config
    
    @abstractmethod
    def estimate(
        self,
        frame: SourceDataFrame,
        kp: torch.Tensor,
        depth_cov_map: torch.Tensor | None,
        depth_map: torch.Tensor | None,
        depth_cov: torch.Tensor | None,
        flow_cov: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Given N points, each with (u, v) coord, depth value, depth_cov, flow, and flow_cov, output
        a Nx3x3 tensor s.t. output[i] represents the 3x3 covariance matrix for the inverse-projected point
        in camera coordinate.

        Parameters:

        * frame     - SourceDataFrame from GenericSequence.
        * kp        - N * 2 FloatTensor, [[u0, v0], ...], uv coordinate of keypoints on current frame
        * depth_map - H * W FloatTensor, Dense depth map for current frame
        * depth_cov - N     FloatTensor, [cov0, ...], depth covariance for each keypoint on current frame
        * flow_cov  - N * 2 FloatTensor, [[σ²u, σ²v], ...], covariance of flow on u and v direction for keypoint on current frame
        """
        ...
    
    @staticmethod
    def _create_3x3_matrix(matrix: list[list[torch.Tensor | float]], n_sample: int):
        assert len(matrix) == 3 and len(matrix[0]) == 3 and len(matrix[1]) == 3 and len(matrix[2]) == 3
        mat = torch.empty((n_sample, 3, 3))
        
        for i in range(3):
            for j in range(3):
                mat[..., i, j] = matrix[i][j]
        return mat

# Implementation ####################################################

class NoCovariance(IObservationCov):
    """
    Returns identity covariance matrix for all observations.
    """
    def estimate(self, frame: SourceDataFrame, kp: torch.Tensor, depth_cov_map: torch.Tensor | None, 
                 depth_map: torch.Tensor | None, 
                 depth_cov: torch.Tensor | None, 
                 flow_cov: torch.Tensor | None) -> torch.Tensor:
        N = kp.size(0)
        return torch.eye(3).unsqueeze(0).repeat(N, 1, 1).double()
    
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None: return


class DepthCovariance(IObservationCov):
    """
    Returns covariance matrix estimated using only depth cov.
    
    Since this will be ill-defined (cov matrix won't be full-ranked since there is no uncertainty on xy direction in 3D space)
    we add a `regularization` term on the main diagonal to make it full-ranked.
    """
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        if self.config.regularization is None:
            self.config.regularization = 1e-5
            Logger.write(
                "info",
                f"DepthCovariance model set regularization constant to {self.config.regularization} by default"
            )

    def estimate(
        self,
        frame: SourceDataFrame,
        kp: torch.Tensor,
        depth_cov_map: torch.Tensor | None,
        depth_map: torch.Tensor | None,
        depth_cov: torch.Tensor | None,
        flow_cov: torch.Tensor | None,
    ) -> torch.Tensor:
        assert depth_cov is not None
        meta = frame.meta
        
        pixel_u, pixel_v = kp[..., 0], kp[..., 1]
        fx, fy, cx, cy = meta.fx, meta.fy, meta.cx, meta.cy

        factor_x = (pixel_u - cx) / fx
        factor_y = (pixel_v - cy) / fy

        var_z = depth_cov
        var_x = factor_x.square() * depth_cov
        var_y = factor_y.square() * depth_cov

        cov_xy = factor_x * factor_y * var_z
        cov_xz = factor_x * var_z
        cov_yz = factor_y * var_z

        cov = self._create_3x3_matrix([
            [var_z, cov_xz, cov_yz],
            [cov_xz, var_x, cov_xy],
            [cov_yz, cov_xy, var_y],
        ], n_sample=var_z.size(0)).double()

        # NOTE: This is for numerical stability. (cov must be positive semi-definite)
        # NOTE: This factor here is very vital for the entire covariance system to work when
        #       we only have the depth covariance. Otherwise, cov have rank 1 and numerical
        #       instability degrades the non-linear optimization process significantly.
        cov += (self.config.regularization * torch.eye(3)).unsqueeze(0).double()
        return cov
    
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "regularization": lambda r: (r is None) or (isinstance(r, (int, float)) and r > 0.)
        })


class MatchCovariance(IObservationCov):
    """
    Covariance model used by MAC-VO. See section III.C for detail.
    """
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.config.kernel_size_hlf = self.config.kernel_size // 2

    def estimate(
        self,
        frame: SourceDataFrame,
        kp: torch.Tensor,
        depth_cov_map: torch.Tensor | None,
        depth_map: torch.Tensor | None,
        depth_cov: torch.Tensor | None,
        flow_cov: torch.Tensor | None,
    ) -> torch.Tensor:
        assert depth_map is not None
        meta = frame.meta
        
        n_sample = kp.size(0)
        fx, fy, cx, cy = meta.fx, meta.fy, meta.cx, meta.cy
        has_flow_cov_flag = True

        kp_long = kp.clone().long()
        if flow_cov is None:
            has_flow_cov_flag = False
            flow_cov = torch.ones_like(kp, dtype=torch.float) * self.config.match_cov_default
            assert flow_cov is not None
            flow_std = flow_cov.sqrt()
        else:
            # Min clamp to 0.16 since camera plane is descretelized by pixels, which has at least an
            # uncertainty of 0.16
            flow_cov = flow_cov.clamp(min=self.config.min_flow_cov**2)
            flow_std = flow_cov.sqrt()

        var_u, var_v = flow_cov[..., 0], flow_cov[..., 1]
        std_u, std_v = flow_std[..., 0], flow_std[..., 1]
        kp_u, kp_v = kp[..., 0], kp[..., 1]

        # Get local depth average and variance
        u_indices = torch.arange(
            -self.config.kernel_size_hlf, self.config.kernel_size_hlf + 1, dtype=torch.long, device=self.config.device
        )
        v_indices = torch.arange(
            -self.config.kernel_size_hlf, self.config.kernel_size_hlf + 1, dtype=torch.long, device=self.config.device
        )
        uu, vv = torch.meshgrid(u_indices, v_indices, indexing="ij")

        all_u_indices = kp_long[:, 0].unsqueeze(-1) + uu.reshape(1, -1)
        all_v_indices = kp_long[:, 1].unsqueeze(-1) + vv.reshape(1, -1)

        local_filters = gaussian_kernels(std_u, std_v, kernel_size=self.config.kernel_size)
        patches = depth_map[..., all_v_indices, all_u_indices].view(
            n_sample, self.config.kernel_size, self.config.kernel_size
        )
        patches = patches.permute(0, 2, 1)

        # Weighted Average
        wavg_depth = (local_filters * patches).sum(dim=[1, 2])
        if has_flow_cov_flag or (depth_cov is None):    
            # Weighted Variance
            wvar_depth = torch.sum(
                local_filters * (patches - (wavg_depth.unsqueeze(1).unsqueeze(1))).square(),
                dim=[1, 2],
            )
        else:
            wvar_depth = depth_cov
        
        wvar_depth = wvar_depth.clamp(min=self.config.min_depth_cov)

        if has_flow_cov_flag:
            PLTVisualizer.visualize_dpatch(f"patch_depthmap", patches, local_filters, wvar_depth, flow_cov)
            PLTVisualizer.visualize_depth("depth", depth_map)
            if depth_cov_map is not None:
                PLTVisualizer.visualize_depthcov("depth_cov", depth_cov_map)
            
            if PLTVisualizer.isActive(PLTVisualizer.visualize_image_patches):
                PLTVisualizer.visualize_image_patches(
                    "image_patch", 
                    frame.imageL[0, ..., all_v_indices, all_u_indices].view(
                        3, n_sample, self.config.kernel_size, self.config.kernel_size
                    ).permute(1, 2, 3, 0))

        # Inverse project 2D keypoint to 3D space.
        var_x = (
            (var_u * wvar_depth)
            + (var_u * (wavg_depth**2))
            + (wvar_depth * ((kp_u - cx) ** 2))
        ) / (fx * fx)
        var_y = (
            (var_v * wvar_depth)
            + (var_v * (wavg_depth**2))
            + (wvar_depth * ((kp_v - cy) ** 2))
        ) / (fy * fy)
        var_z = wvar_depth

        cov_xy = (wvar_depth * (kp_u - cx) * (kp_v - cy)) / (fx * fy)
        cov_xz = (wvar_depth * (kp_u - cx)) / fx
        cov_yz = (wvar_depth * (kp_v - cy)) / fy
        
        cov = self._create_3x3_matrix([
            [var_z, cov_xz, cov_yz],
            [cov_xz, var_x, cov_xy],
            [cov_yz, cov_xy, var_y],
        ], n_sample=var_z.size(0)).double()

        return cov
    
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "kernel_size"       : lambda k: isinstance(k, int) and k > 0 and (k % 2 == 1),
            "match_cov_default" : lambda c: isinstance(c, (int, float)) and c > 0.,
            "min_flow_cov"      : lambda c: isinstance(c, (int, float)) and c > 0.,
            "min_depth_cov"     : lambda c: isinstance(c, (int, float)) and c > 0.,
        })


class GaussianMixtureCovariance(IObservationCov):
    """
    Using gaussian mixture to model the depth uncertainty in presence of flow uncertainty
    (while MAC-VO only uses weighted variance).
    """
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.config.kernel_size_hlf = self.config.kernel_size // 2

    def estimate(
        self,
        frame: SourceDataFrame,
        kp: torch.Tensor,
        depth_cov_map: torch.Tensor | None,
        depth_map: torch.Tensor | None,
        depth_cov: torch.Tensor | None,
        flow_cov: torch.Tensor | None,
    ) -> torch.Tensor:
        assert depth_map is not None
        assert depth_cov_map is not None
        meta = frame.meta
        
        n_sample = kp.size(0)
        fx, fy, cx, cy = meta.fx, meta.fy, meta.cx, meta.cy
        has_flow_cov_flag = True

        kp_long = kp.long()
        if flow_cov is None:
            has_flow_cov_flag = False
            flow_cov = torch.ones_like(kp, dtype=torch.float) * self.config.match_cov_default
        assert flow_cov is not None
        
        # Min clamp to 0.5 since camera plane is descretelized by pixels, which has at least an
        # uncertainty of 0.5
        flow_std = flow_cov.sqrt().clamp(min=self.config.min_flow_cov)

        var_u, var_v = flow_cov[..., 0], flow_cov[..., 1]
        std_u, std_v = flow_std[..., 0], flow_std[..., 1]
        kp_u, kp_v = kp[..., 0], kp[..., 1]

        # Get local depth average and variance
        u_indices = torch.arange(
            -self.config.kernel_size_hlf, self.config.kernel_size_hlf + 1, dtype=torch.long
        )
        v_indices = torch.arange(
            -self.config.kernel_size_hlf, self.config.kernel_size_hlf + 1, dtype=torch.long
        )
        uu, vv = torch.meshgrid(u_indices, v_indices, indexing="ij")

        all_u_indices = kp_long[:, 0].unsqueeze(-1) + uu.reshape(1, -1)
        all_v_indices = kp_long[:, 1].unsqueeze(-1) + vv.reshape(1, -1)

        local_filters = gaussian_kernels(std_u, std_v, kernel_size=self.config.kernel_size)
        patches = depth_map[..., all_v_indices, all_u_indices].view(
            n_sample, self.config.kernel_size, self.config.kernel_size
        )
        cov_patches = depth_cov_map[..., all_v_indices, all_u_indices]\
                      .view(n_sample, self.config.kernel_size, self.config.kernel_size)
        patches = patches.permute(0, 2, 1)
        cov_patches = cov_patches.permute(0, 2, 1)

        # Calculate average depth and depth covariance as a mixture of gaussian
        wavg_depth, wvar_depth = gaussian_mixture_mean_var(
            patches.flatten(1), cov_patches.flatten(1), local_filters.flatten(1)
        )
        if (not has_flow_cov_flag) and (depth_cov is not None):
            wvar_depth = depth_cov

        if has_flow_cov_flag:
            PLTVisualizer.visualize_dpatch(f"Fpatch_depthmap", patches, local_filters, wvar_depth, flow_cov)

        # Inverse project 2D keypoint to 3D space.
        var_x = (
            (var_u * wvar_depth)
            + (var_u * (wavg_depth**2))
            + (wvar_depth * ((kp_u - cx) ** 2))
        ) / (fx * fx)
        var_y = (
            (var_v * wvar_depth)
            + (var_v * (wavg_depth**2))
            + (wvar_depth * ((kp_v - cy) ** 2))
        ) / (fy * fy)
        var_z = wvar_depth

        cov_xy = (wvar_depth * (kp_u - cx) * (kp_v - cy)) / (fx * fy)
        cov_xz = (wvar_depth * (kp_u - cx)) / fx
        cov_yz = (wvar_depth * (kp_v - cy)) / fy

        cov = self._create_3x3_matrix([
            [var_z, cov_xz, cov_yz], 
            [cov_xz, var_x, cov_xy],
            [cov_yz, cov_xy, var_y],
        ], n_sample).double()
        return cov

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "kernel_size"       : lambda k: isinstance(k, int) and k > 0 and (k % 2 == 1),
            "match_cov_default" : lambda c: isinstance(c, (int, float)) and c > 0.,
            "min_flow_cov"      : lambda c: isinstance(c, (int, float)) and c > 0.,
            "min_depth_cov"     : lambda c: isinstance(c, (int, float)) and c > 0.,
        })


# Modifiers #########################################################
#
#  Modifier convert / tweak the input/output of a cov model to modify it.
#       Modifier(CovModel) -> CovModel'
#

class Modifier_Diagonalize(IObservationCov):
    """
    This class is a wrapper (modifier) to another IObservationCov model.
    
    On every call, it will forward everything to the internal model and diagonalize the 
    output covariance model. (by setting the off-diagonal terms to zero.)
    """
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.submodule = IObservationCov.instantiate(config.type, config.args)
    
    def estimate(self, frame: SourceDataFrame, kp: torch.Tensor, depth_cov_map: torch.Tensor | None, depth_map: torch.Tensor | None, depth_cov: torch.Tensor | None, flow_cov: torch.Tensor | None) -> torch.Tensor:
        covs = self.submodule.estimate(frame, kp, depth_cov_map, depth_map, depth_cov, flow_cov)
        for i in range(3):
            for j in range(3):
                if i == j: continue
                covs[..., i, j] = 0.
        return covs
    
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        IObservationCov.is_valid_config(config)


class Modifier_Normalize(IObservationCov):
    """
    This class is a wrapper (modifier) to another IObservationCov model.
    
    On every call, it will forward everything to the internal model and normalize the output
    covariance matrices (by setting average determinant of all points to 1).
    """
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.submodule = IObservationCov.instantiate(config.type, config.args)
    
    def estimate(self, frame: SourceDataFrame, kp: torch.Tensor, depth_cov_map: torch.Tensor | None, depth_map: torch.Tensor | None, depth_cov: torch.Tensor | None, flow_cov: torch.Tensor | None) -> torch.Tensor:
        covs = self.submodule.estimate(frame, kp, depth_cov_map, depth_map, depth_cov, flow_cov)
        covs /= torch.det(covs).unsqueeze(-1).unsqueeze(-1)
        return covs

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        IObservationCov.is_valid_config(config)
