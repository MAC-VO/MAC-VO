from abc import ABC, abstractmethod

import torch
from types import SimpleNamespace

from DataLoader import StereoData
from ..Frontend.StereoDepth import IStereoDepth

from Utility.Math import gaussian_mixture_mean_var, gaussain_full_kernels
from Utility.PrettyPrint import Logger
from Utility.Timer import Timer
from Utility.Extensions import ConfigTestableSubclass

# Interface #########################################################

class ICovariance2to3(ABC, ConfigTestableSubclass):
    def __init__(self, config: SimpleNamespace):
        self.config = config
    
    @abstractmethod
    def estimate(
        self,
        frame: StereoData,
        kp: torch.Tensor,
        depth_est: IStereoDepth.Output,
        depth_cov: torch.Tensor | None,
        flow_cov: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Given N points, each with (u, v) coord, depth value, depth_cov, flow, and flow_cov, output
        a Nx3x3 tensor s.t. output[i] represents the 3x3 covariance matrix for the inverse-projected point
        in camera coordinate.

        Parameters:

        * frame         - SourceDataFrame from GenericSequence.
        * kp            - N * 2 FloatTensor, [[u0, v0], ...], uv coordinate of keypoints on current frame
        * depth_cov_map - H * W FloatTensor, Dense depth covariance map for current frame
        * depth_map     - H * W FloatTensor, Dense depth map for current frame
        * depth_cov     - N     FloatTensor, [cov0, ...], depth covariance for each keypoint on current frame
        * flow_cov      - N * 2 FloatTensor, [[σ²u, σ²v], ...], covariance of flow on u and v direction for keypoint on current frame
        """
        ...


# Implementation ####################################################

class NoCovariance(ICovariance2to3):
    """
    Returns identity covariance matrix for all observations.
    """
    def estimate(self, frame: StereoData, kp: torch.Tensor, depth_est: IStereoDepth.Output, depth_cov: torch.Tensor | None, flow_cov: torch.Tensor | None) -> torch.Tensor:
        N = kp.size(0)
        return torch.eye(3).unsqueeze(0).repeat(N, 1, 1).double()
    
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None: return


class DepthCovariance(ICovariance2to3):
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

    def estimate(self, frame: StereoData, kp: torch.Tensor, depth_est: IStereoDepth.Output, depth_cov: torch.Tensor | None, flow_cov: torch.Tensor | None) -> torch.Tensor:
        assert depth_est.cov is not None
        assert depth_cov is not None
        
        pixel_u, pixel_v = kp[..., 0], kp[..., 1]
        fx, fy, cx, cy = frame.fx, frame.fy, frame.cx, frame.cy

        factor_x = (pixel_u - cx) / fx
        factor_y = (pixel_v - cy) / fy

        var_z = depth_est.cov
        var_x = factor_x.square() * depth_cov
        var_y = factor_y.square() * depth_cov

        cov_xy = factor_x * factor_y * var_z
        cov_xz = factor_x * var_z
        cov_yz = factor_y * var_z

        cov = create_3x3_matrix([
            [var_z, cov_xz, cov_yz],
            [cov_xz, var_x, cov_xy],
            [cov_yz, cov_xy, var_y],
        ], n_sample=var_z.size(0), device=var_z.device).double()

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


class MatchCovariance(ICovariance2to3):
    """
    Covariance model used by MAC-VO. See section III.C for detail.
    """
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.config.kernel_size_hlf = self.config.kernel_size // 2

    @Timer.cpu_timeit("Cov Model")
    @Timer.gpu_timeit("Cov Model")
    def estimate(self, frame: StereoData, kp: torch.Tensor, depth_est: IStereoDepth.Output, depth_cov: torch.Tensor | None, flow_cov: torch.Tensor | None) -> torch.Tensor:        
        n_sample = kp.size(0)

        kp_long = kp.clone().long()
        if (has_flow_cov_flag := flow_cov is not None):
            # Min clamp to 0.16 since camera plane is descretelized by pixels, which has at least an
            # uncertainty of 0.16
            flow_cov[..., :2].clamp_(min=self.config.min_flow_cov**2)
        else:
            flow_cov = torch.ones((n_sample, 3), device=torch.device(self.config.device), dtype=torch.float) * self.config.match_cov_default
            assert flow_cov is not None
            flow_cov[..., 2] = 0.

        var_u, var_v, var_uv = flow_cov[..., 0], flow_cov[..., 1], flow_cov[..., 2]
        kp_u, kp_v = kp[..., 0], kp[..., 1]

        # Get local depth average and variance
        u_indices = torch.arange(
            -self.config.kernel_size_hlf, self.config.kernel_size_hlf + 1, dtype=torch.long, device=torch.device(self.config.device)
        )
        v_indices = torch.arange(
            -self.config.kernel_size_hlf, self.config.kernel_size_hlf + 1, dtype=torch.long, device=torch.device(self.config.device)
        )
        uu, vv = torch.meshgrid(u_indices, v_indices, indexing="ij")

        all_u_indices = kp_long[:, 0].unsqueeze(-1) + uu.reshape(1, -1)
        all_v_indices = kp_long[:, 1].unsqueeze(-1) + vv.reshape(1, -1)

        cov_matrices = create_2x2_matrix([[var_u, var_uv], [var_uv, var_v]], n_sample=n_sample, device=torch.device(self.config.device))
        local_filters = gaussain_full_kernels(cov_matrices, kernel_size=self.config.kernel_size)
        
        patches = depth_est.depth[..., all_v_indices, all_u_indices].view(
            n_sample, self.config.kernel_size, self.config.kernel_size
        ).to(self.config.device)
        patches = patches.permute(0, 2, 1)

        # Weighted Average
        wavg_depth = (local_filters * patches).sum(dim=[1, 2])
        if (has_flow_cov_flag or (depth_cov is None)):
            # Weighted Variance
            wvar_depth = torch.sum(
                local_filters * (patches - (wavg_depth.unsqueeze(1).unsqueeze(1))).square(),
                dim=[1, 2],
            )
        else:
            assert depth_cov is not None
            wvar_depth = depth_cov
        
        wvar_depth = wvar_depth.clamp(min=self.config.min_depth_cov)

        # Inverse project 2D keypoint to 3D space.
        cov = Covariance_2to3_full(
            var_u, var_uv, var_v, wvar_depth,
            kp_u, kp_v, wavg_depth,
            frame.fx, frame.fy, frame.cx, frame.cy
        ).double()

        return cov
    
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "kernel_size"       : lambda k: isinstance(k, int) and k > 0 and (k % 2 == 1),
            "match_cov_default" : lambda c: isinstance(c, (int, float)) and c > 0.,
            "min_flow_cov"      : lambda c: isinstance(c, (int, float)) and c > 0.,
            "min_depth_cov"     : lambda c: isinstance(c, (int, float)) and c > 0.,
            "device"            : lambda dev: isinstance(dev, str) and (("cuda" in dev) or (dev == "cpu"))
        })


class GaussianMixtureCovariance(ICovariance2to3):
    """
    Using gaussian mixture to model the depth uncertainty in presence of flow uncertainty
    (while MAC-VO only uses weighted variance).
    """
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.config.kernel_size_hlf = self.config.kernel_size // 2

    @Timer.cpu_timeit("Cov Model")
    @Timer.gpu_timeit("Cov Model")
    def estimate(self, frame: StereoData, kp: torch.Tensor, depth_est: IStereoDepth.Output, depth_cov: torch.Tensor | None, flow_cov: torch.Tensor | None) -> torch.Tensor:
        assert depth_est.cov is not None
        
        n_sample = kp.size(0)
        kp_long = kp.long()
        
        if (has_flow_cov_flag := flow_cov is not None):
            # Min clamp to 0.16 since camera plane is descretelized by pixels, which has at least an
            # uncertainty of 0.16
            flow_cov[..., :2].clamp_(min=self.config.min_flow_cov**2)
        else:
            flow_cov = torch.ones((n_sample, 3), device=kp.device, dtype=torch.float) * self.config.match_cov_default
            assert flow_cov is not None
            flow_cov[..., 2] = 0.
        
        # Min clamp to 0.5 since camera plane is descretelized by pixels, which has at least an
        # uncertainty of 0.5

        var_u, var_v, var_uv = flow_cov[..., 0], flow_cov[..., 1], flow_cov[..., 2]
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

        cov_matrices = create_2x2_matrix([[var_u, var_uv], [var_uv, var_v]], n_sample=n_sample, device=var_u.device)
        local_filters = gaussain_full_kernels(cov_matrices, kernel_size=self.config.kernel_size)
        
        patches = depth_est.depth[..., all_v_indices, all_u_indices].view(
            n_sample, self.config.kernel_size, self.config.kernel_size
        )
        cov_patches = depth_est.cov[..., all_v_indices, all_u_indices].view(
            n_sample, self.config.kernel_size, self.config.kernel_size
        )
        patches = patches.permute(0, 2, 1)
        cov_patches = cov_patches.permute(0, 2, 1)

        # Calculate average depth and depth covariance as a mixture of gaussian
        wavg_depth, wvar_depth = gaussian_mixture_mean_var(
            patches.flatten(1), cov_patches.flatten(1), local_filters.flatten(1)
        )
        if (not has_flow_cov_flag) and (depth_cov is not None):
            wvar_depth = depth_cov

        # Inverse project 2D keypoint to 3D space.
        cov = Covariance_2to3_full(
            var_u, var_uv, var_v, wvar_depth,
            kp_u, kp_v, wavg_depth,
            frame.fx, frame.fy, frame.cx, frame.cy
        ).double()
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

class Modifier_Diagonalize(ICovariance2to3):
    """
    This class is a wrapper (modifier) to another IObservationCov model.
    
    On every call, it will forward everything to the internal model and diagonalize the 
    output covariance model. (by setting the off-diagonal terms to zero.)
    """
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.submodule = ICovariance2to3.instantiate(config.type, config.args)
    
    def estimate(self, frame: StereoData, kp: torch.Tensor, depth_est: IStereoDepth.Output, depth_cov: torch.Tensor | None, flow_cov: torch.Tensor | None) -> torch.Tensor:
        covs = self.submodule.estimate(frame, kp, depth_est, depth_cov, flow_cov)
        for i in range(3):
            for j in range(3):
                if i == j: continue
                covs[..., i, j] = 0.
        return covs
    
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        ICovariance2to3.is_valid_config(config)


class Modifier_Normalize(ICovariance2to3):
    """
    This class is a wrapper (modifier) to another IObservationCov model.
    
    On every call, it will forward everything to the internal model and normalize the output
    covariance matrices (by setting average determinant of all points to 1).
    """
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.submodule = ICovariance2to3.instantiate(config.type, config.args)
    
    def estimate(self, frame: StereoData, kp: torch.Tensor, depth_est: IStereoDepth.Output, depth_cov: torch.Tensor | None, flow_cov: torch.Tensor | None) -> torch.Tensor:
        covs = self.submodule.estimate(frame, kp, depth_est, depth_cov, flow_cov)
        covs /= torch.det(covs).unsqueeze(-1).unsqueeze(-1)
        return covs

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        ICovariance2to3.is_valid_config(config)


# Utility Functions #################################################
# 
# Common routeins shared across multiple modules
#

def Covariance_2to3_diag(
    sigma_uu: torch.Tensor, sigma_vv: torch.Tensor, sigma_dd: torch.Tensor, 
    u: torch.Tensor, v: torch.Tensor, d: torch.Tensor,
    fx: float, fy: float, cx: float, cy: float
) -> torch.Tensor:
    """
    Project the uncertainty on 2D image plane to 3D space (under camera coordinate)
    
    In
    * sigma_uu: torch.Tensor (N,)
        Variance of keypoints' u-coordinate on image plane.
    
    * sigma_vv: torch.Tensor (N,)
        Variance of keypoints' v-coordinate on image plane.
    
    * sigma_dd: torch.Tensor (N,)
        Variance of keypoints' depth on image plane.
    
    * u: torch.Tensor (N,) - keypoints' u-coordinate
    * v: torch.Tensor (N,) - keypoints' v-coordinate
    * d: torch.Tensor (N,) - keypoints' depth
    
    * meta: MetaInfo
        Camera meta information (camera intrinsic, mostly)
    
    Out
    * cov_3d: torch.Tensor(N, 3, 3)
        3x3 Covariance matrix of keypoints' distribution in 3D space (under camera coordinate)
    
    Ref: For deriviation, see Appendix A and Section III.C of MAC-VO's paper.
    """
    sigma_xx = (sigma_uu * sigma_dd + sigma_uu * d.square() + (u - cx).square() * sigma_dd) / (fx ** 2)
    sigma_yy = (sigma_vv * sigma_dd + sigma_vv * d.square() + (v - cy).square() * sigma_dd) / (fy ** 2)
    sigma_zz = sigma_dd
    
    sigma_xy = ((u - cx) * (v - cy) * sigma_dd) / (fx * fy)
    sigma_xz = ((u - cx) * sigma_dd) / fx
    sigma_yz = ((v - cy) * sigma_dd) / fy
    
    return create_3x3_matrix([
        [sigma_zz, sigma_xz, sigma_yz],
        [sigma_xz, sigma_xx, sigma_xy],
        [sigma_yz, sigma_xy, sigma_yy]
    ], n_sample=u.size(0), device=sigma_zz.device)


def Covariance_2to3_full(
    sigma_uu: torch.Tensor, sigma_uv: torch.Tensor, sigma_vv: torch.Tensor, sigma_dd: torch.Tensor,
    u: torch.Tensor, v: torch.Tensor, d: torch.Tensor,
    fx: float, fy: float, cx: float, cy: float
) -> torch.Tensor:
    """
    Project the uncertainty on 2D image plane to 3D space (under camera coordinate)
    
    In
    * sigma_uu: torch.Tensor (N,)
        Variance of keypoints' u-coordinate on image plane.
    
    * sigma_vv: torch.Tensor (N,)
        Variance of keypoints' v-coordinate on image plane.
    
    * sigma_uv: torch.Tensor (N,)
        Covariance of keypoints' u and v coordinate on image plane.
    
    * sigma_dd: torch.Tensor (N,)
        Variance of keypoints' depth on image plane.
    
    * u: torch.Tensor (N,) - keypoints' u-coordinate
    * v: torch.Tensor (N,) - keypoints' v-coordinate
    * d: torch.Tensor (N,) - keypoints' depth
    
    * meta: MetaInfo
        Camera meta information (camera intrinsic, mostly)
    
    Out
    * cov_3d: torch.Tensor(N, 3, 3)
        3x3 Covariance matrix of keypoints' distribution in 3D space (under camera coordinate)
    
    For deriviation, ask Yutian : )
    """
    sigma_xx = (((u - cx).square() * sigma_dd) + (d.square() * sigma_uu) + (sigma_uu * sigma_dd)) / (fx ** 2)
    sigma_yy = (((v - cy).square() * sigma_dd) + (d.square() * sigma_vv) + (sigma_vv * sigma_dd)) / (fy ** 2)
    sigma_zz = sigma_dd
    
    sigma_xy = (((u - cx) * (v - cy) * sigma_dd) + (d.square() + sigma_dd) * sigma_uv) / (fx * fy)
    sigma_xz = (sigma_dd * (u - cx)) / fx
    sigma_yz = (sigma_dd * (v - cy)) / fy
    
    return create_3x3_matrix([
        [sigma_zz, sigma_xz, sigma_yz],
        [sigma_xz, sigma_xx, sigma_xy],
        [sigma_yz, sigma_xy, sigma_yy]
    ], n_sample=u.size(0), device=sigma_zz.device)


def create_3x3_matrix(matrix: list[list[torch.Tensor | float]], n_sample: int, device: torch.device):
    assert len(matrix) == 3 and len(matrix[0]) == 3 and len(matrix[1]) == 3 and len(matrix[2]) == 3
    mat = torch.empty((n_sample, 3, 3))
    
    for i in range(3):
        for j in range(3):
            mat[..., i, j] = matrix[i][j]
    return mat


def create_2x2_matrix(matrix: list[list[torch.Tensor | float]], n_sample: int, device: torch.device):
    assert len(matrix) == 2 and len(matrix[0]) == 2 and len(matrix[1]) == 2
    mat = torch.empty((n_sample, 2, 2), device=device)
    
    for i in range(2):
        for j in range(2):
            mat[..., i, j] = matrix[i][j]
    return mat
