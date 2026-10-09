import math
import unittest

import pypose as pp
import torch

from DataLoader import StereoData
from Module.Frontend.StereoDepth import IStereoDepth
from Module.LoopClosure.SparseGeometry import (
    SparseGeometryConfig,
    conservative_sparse_factor,
    estimate_se3_ransac,
    validate_sparse_loop,
)


def synthetic_problem(seed=4, outliers=40):
    generator = torch.Generator().manual_seed(seed)
    points_target = torch.randn(100, 3, generator=generator, dtype=torch.float64)
    points_target[:, 0] += 4.0
    tangent = torch.tensor(
        [0.25, -0.12, 0.08, 0.04, -0.03, 0.05], dtype=torch.float64
    )
    truth = pp.se3(tangent).Exp()
    points_source = truth.Act(points_target)
    points_source += torch.randn(
        points_source.shape, generator=generator, dtype=torch.float64
    ) * 0.002
    if outliers:
        points_source[:outliers] = torch.randn(
            outliers, 3, generator=generator, dtype=torch.float64
        ) * 3.0
    covariance = torch.eye(3, dtype=torch.float64)[None].repeat(100, 1, 1)
    covariance *= 1e-4
    return points_source, points_target, covariance, covariance.clone(), truth


class SparseSE3RansacTests(unittest.TestCase):
    def test_recovers_rigid_transform_with_forty_percent_outliers(self):
        points_a, points_b, cov_a, cov_b, truth = synthetic_problem()

        result = estimate_se3_ransac(
            points_a,
            points_b,
            cov_a,
            cov_b,
            iterations=256,
            mahalanobis_threshold=3.5,
            min_geometry_ratio=1e-3,
            seed=17,
        )

        self.assertIsNotNone(result.measurement)
        self.assertGreaterEqual(result.inliers, 55)
        error = (pp.SE3(result.measurement).Inv() @ truth).Log().tensor()
        self.assertLess(float(error[:3].norm()), 0.02)
        self.assertLess(float(error[3:].norm()), float(torch.deg2rad(torch.tensor(1.0))))

    def test_repeated_seed_produces_identical_result(self):
        args = synthetic_problem(seed=9)

        first = estimate_se3_ransac(
            *args[:4], iterations=128, mahalanobis_threshold=3.5,
            min_geometry_ratio=1e-3, seed=31,
        )
        second = estimate_se3_ransac(
            *args[:4], iterations=128, mahalanobis_threshold=3.5,
            min_geometry_ratio=1e-3, seed=31,
        )

        self.assertTrue(torch.equal(first.measurement, second.measurement))
        self.assertTrue(torch.equal(first.mask, second.mask))

    def test_rejects_too_few_points(self):
        points = torch.zeros(2, 3, dtype=torch.float64)
        covariance = torch.eye(3, dtype=torch.float64)[None].repeat(2, 1, 1)

        result = estimate_se3_ransac(
            points, points, covariance, covariance,
            iterations=10, mahalanobis_threshold=3.5,
            min_geometry_ratio=1e-3, seed=1,
        )

        self.assertIsNone(result.measurement)
        self.assertEqual(result.reason, "too_few_correspondences")

    def test_rejects_collinear_geometry(self):
        x = torch.linspace(1, 5, 20, dtype=torch.float64)
        points = torch.stack((x, torch.zeros_like(x), torch.zeros_like(x)), dim=-1)
        covariance = torch.eye(3, dtype=torch.float64)[None].repeat(20, 1, 1) * 1e-4

        result = estimate_se3_ransac(
            points, points, covariance, covariance,
            iterations=64, mahalanobis_threshold=3.5,
            min_geometry_ratio=1e-3, seed=2,
        )

        self.assertIsNone(result.measurement)
        self.assertEqual(result.reason, "degenerate_geometry")

    def test_rejects_non_positive_definite_covariance(self):
        points_a, points_b, cov_a, cov_b, _ = synthetic_problem(outliers=0)
        cov_a[0, 0, 0] = -1

        result = estimate_se3_ransac(
            points_a, points_b, cov_a, cov_b,
            iterations=64, mahalanobis_threshold=3.5,
            min_geometry_ratio=1e-3, seed=3,
        )

        self.assertIsNone(result.measurement)
        self.assertEqual(result.reason, "invalid_covariance")


class IdentityCovariance:
    def estimate(self, frame, pixels, depth, depth_cov, pixel_cov):
        return torch.eye(3, device=pixels.device)[None].repeat(len(pixels), 1, 1) * 1e-4


def stereo_data(height=48, width=64):
    K = torch.tensor(
        [[[50.0, 0.0, width / 2], [0.0, 50.0, height / 2], [0.0, 0.0, 1.0]]]
    )
    return StereoData(
        T_BS=pp.identity_SE3(1),
        K=K,
        baseline=torch.tensor([0.2]),
        time_ns=[0],
        height=height,
        width=width,
        imageL=torch.zeros(1, 3, height, width),
        imageR=torch.zeros(1, 3, height, width),
    )


def depth_output(height=48, width=64, value=5.0):
    depth = torch.full((1, 1, height, width), value)
    covariance = torch.full_like(depth, 1e-4)
    return IStereoDepth.Output(depth=depth, cov=covariance)


def covered_matches():
    values = []
    for y in (8, 16, 24, 32, 40):
        for x in (8, 16, 24, 32, 40, 48, 56):
            values.append([x, y, x, y, 4])
    return torch.tensor(values, dtype=torch.float64).numpy()


class SparseLoopValidationTests(unittest.TestCase):
    def config(self, **overrides):
        values = dict(
            min_mutual_matches=20,
            min_valid_3d=20,
            min_ransac_inliers=20,
            min_ransac_ratio=0.5,
            grid_rows=4,
            grid_cols=6,
            min_grid_cells=6,
            ransac_iterations=64,
            mahalanobis_threshold=3.5,
            max_median_reprojection_px=1.0,
            max_p90_reprojection_px=2.0,
            min_geometry_ratio=1e-3,
            orb_pixel_variance=2.25,
        )
        values.update(overrides)
        return SparseGeometryConfig(**values)

    def test_validates_identity_loop_from_depth_and_pixels(self):
        result = validate_sparse_loop(
            source=10,
            target=100,
            matches=covered_matches(),
            source_stereo=stereo_data(),
            target_stereo=stereo_data(),
            source_depth=depth_output(),
            target_depth=depth_output(),
            covariance_model=IdentityCovariance(),
            config=self.config(),
            seed=7,
        )

        self.assertIsNone(result.reason)
        self.assertIsNotNone(result.measurement)
        self.assertGreaterEqual(result.metrics["ransac_inliers"], 20)
        self.assertLess(result.metrics["reprojection_forward_median_px"], 1e-6)

    def test_rejects_invalid_depth(self):
        invalid = depth_output(value=0.0)

        result = validate_sparse_loop(
            source=10,
            target=100,
            matches=covered_matches(),
            source_stereo=stereo_data(),
            target_stereo=stereo_data(),
            source_depth=invalid,
            target_depth=invalid,
            covariance_model=IdentityCovariance(),
            config=self.config(),
            seed=7,
        )

        self.assertEqual(result.reason, "insufficient_valid_depth")

    def test_rejects_weak_image_coverage(self):
        matches = covered_matches()
        matches[:, :4] = torch.tensor([8.0, 8.0, 8.0, 8.0]).numpy()

        result = validate_sparse_loop(
            source=10,
            target=100,
            matches=matches,
            source_stereo=stereo_data(),
            target_stereo=stereo_data(),
            source_depth=depth_output(),
            target_depth=depth_output(),
            covariance_model=IdentityCovariance(),
            config=self.config(),
            seed=7,
        )

        self.assertEqual(result.reason, "insufficient_grid_coverage")

    def test_builds_conservative_sparse_factor(self):
        measurement = pp.identity_SE3(dtype=torch.float64).tensor()

        factor = conservative_sparse_factor(
            source=10,
            target=100,
            measurement=measurement,
            inliers=60,
            inlier_ratio=0.6,
            translation_sigma_m=0.25,
            rotation_sigma_deg=10.0,
        )

        self.assertEqual(factor.kind, "loop")
        self.assertEqual(factor.observation_count, 60)
        self.assertAlmostEqual(factor.confidence, 0.6)
        self.assertAlmostEqual(float(factor.information[0, 0]), 16.0)
        expected_rotation = 1 / float(torch.deg2rad(torch.tensor(10.0))) ** 2
        self.assertAlmostEqual(float(factor.information[3, 3]), expected_rotation, places=4)

    def test_scales_conservative_sparse_factor_information(self):
        factor = conservative_sparse_factor(
            source=10,
            target=100,
            measurement=pp.identity_SE3(dtype=torch.float64).tensor(),
            inliers=60,
            inlier_ratio=0.6,
            translation_sigma_m=0.25,
            rotation_sigma_deg=10.0,
            information_scale=10000.0,
        )

        self.assertAlmostEqual(float(factor.information[0, 0]), 160000.0)
        expected_rotation = 10000.0 / math.radians(10.0) ** 2
        self.assertAlmostEqual(
            float(factor.information[3, 3]), expected_rotation, places=3
        )
        expected = torch.tensor(
            [160000.0] * 3 + [expected_rotation] * 3,
            dtype=torch.float64,
        )
        self.assertTrue(torch.allclose(
            torch.linalg.eigvalsh(factor.information), expected
        ))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_depth_maps_accept_cpu_orb_matches(self):
        source_depth = depth_output()
        target_depth = depth_output()
        source_depth.depth = source_depth.depth.cuda()
        source_depth.cov = source_depth.cov.cuda()
        target_depth.depth = target_depth.depth.cuda()
        target_depth.cov = target_depth.cov.cuda()

        result = validate_sparse_loop(
            source=10,
            target=100,
            matches=covered_matches(),
            source_stereo=stereo_data(),
            target_stereo=stereo_data(),
            source_depth=source_depth,
            target_depth=target_depth,
            covariance_model=IdentityCovariance(),
            config=self.config(),
            seed=7,
        )

        self.assertIsNone(result.reason)


if __name__ == "__main__":
    unittest.main()
