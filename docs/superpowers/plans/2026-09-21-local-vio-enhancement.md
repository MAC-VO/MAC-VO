# Local Tightly Coupled VIO Enhancement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional five-frame GTSAM stereo-inertial optimizer to the existing t030 WindowICP and sparse-loop system without changing the visual-only pipeline.

**Architecture:** EuRoC IMU intervals are normalized to exact camera boundaries and preintegrated with GTSAM `PreintegratedCombinedMeasurements`. Existing covariance-aware adjacent and skip ICP edges are compressed into relative body-pose factors, then jointly optimized with pose, velocity, and bias states in a bounded graph rebuilt on every camera frame. The existing sparse loop detector and pose-only global graph remain unchanged; global writeback invalidates and reanchors the local VIO state.

**Tech Stack:** Python 3.12, PyTorch 2.4, PyPose 0.9.5, GTSAM 4.3.0, NumPy, existing MAC-VO/WindowICP modules, `unittest`.

---

## File Structure

- Create `requirements-vio.txt`: optional pinned GTSAM dependency.
- Modify `DataLoader/Interface.py`: typed IMU sensor metadata carried with every inertial frame.
- Modify `DataLoader/Dataset/EuRoC.py`: parse IMU metadata and return exact camera-bounded IMU intervals.
- Create `Module/Optimization/GTSAMBridge.py`: pose, tangent-order, information, covariance, and camera/body conversion.
- Create `Module/Optimization/IMUPreintegration.py`: GTSAM parameter construction and interval preintegration.
- Create `Module/Optimization/LocalVIO.py`: bounded batch graph, state initialization, optimization, validation, and diagnostics.
- Create `Odometry/OnlineLoopVIOWindowMACVO.py`: optional VIO layer around the existing t030 online-loop odometry.
- Modify `MACVO.py`: register the new odometry type.
- Create `Config/Experiment/MACVO/MACVO_Fast_WindowVIO_ORBLoop_Sparse_t030.yaml`: isolated VIO configuration.
- Modify `Scripts/Experiment/CompareOnlineORBLoop.py`: add a VIO comparison mode.
- Create `Scripts/run_euroc_sparse_t030_vio_gate.sh`: run MH04, MH05, V103, and V203.
- Create focused unit tests under `Scripts/UnitTest/`.

### Task 1: Create an isolated VIO execution environment

**Files:**
- Create: `requirements-vio.txt`
- Create: `Scripts/UnitTest/test_gtsam_dependency.py`

- [ ] **Step 1: Write the failing dependency test**

Create:

```python
import unittest


class GTSAMDependencyTests(unittest.TestCase):
    def test_required_vio_symbols_are_available(self):
        import gtsam

        self.assertTrue(hasattr(gtsam, "Pose3"))
        self.assertTrue(hasattr(gtsam, "CombinedImuFactor"))
        self.assertTrue(
            hasattr(gtsam, "PreintegratedCombinedMeasurements")
        )
        self.assertTrue(hasattr(gtsam, "LevenbergMarquardtOptimizer"))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Verify the test fails in an environment without GTSAM**

Run:

```bash
/home/zzh/MACVO/.venv/bin/python -m unittest   Scripts.UnitTest.test_gtsam_dependency -v
```

Expected before installation: `ModuleNotFoundError: No module named 'gtsam'`.

- [ ] **Step 3: Add the optional dependency file**

Create:

```text
gtsam==4.3.0
```

Do not add GTSAM to the base requirements because the visual-only system must
remain installable without it.

- [ ] **Step 4: Install the optional dependency in the execution worktree environment**

Run:

```bash
/home/zzh/MACVO/.venv/bin/python -m pip install -r requirements-vio.txt
```

Expected: installation of the CPython 3.12 GTSAM 4.3.0 wheel.

- [ ] **Step 5: Run the dependency test**

Run:

```bash
/home/zzh/MACVO/.venv/bin/python -m unittest   Scripts.UnitTest.test_gtsam_dependency -v
```

Expected: one passing test.

- [ ] **Step 6: Commit**

```bash
git add requirements-vio.txt Scripts/UnitTest/test_gtsam_dependency.py
git commit -m "build: add optional GTSAM VIO dependency"
```

### Task 2: Parse and carry EuRoC IMU sensor metadata

**Files:**
- Modify: `DataLoader/Interface.py`
- Modify: `DataLoader/Dataset/EuRoC.py`
- Create: `Scripts/UnitTest/test_euroc_imu_metadata.py`

- [ ] **Step 1: Write failing metadata tests**

Create tests that build a temporary `imu0/sensor.yaml` and assert exact
parsing:

```python
import tempfile
from pathlib import Path
import unittest

import torch
import yaml

from DataLoader.Dataset.EuRoC import load_euroc_imu_sensor


class EuRoCIMUMetadataTests(unittest.TestCase):
    def test_parses_extrinsic_rate_and_noise(self):
        payload = {
            "T_BS": {
                "rows": 4,
                "cols": 4,
                "data": [
                    1.0, 0.0, 0.0, 0.1,
                    0.0, 1.0, 0.0, 0.2,
                    0.0, 0.0, 1.0, 0.3,
                    0.0, 0.0, 0.0, 1.0,
                ],
            },
            "rate_hz": 200,
            "gyroscope_noise_density": 1.6968e-4,
            "gyroscope_random_walk": 1.9393e-5,
            "accelerometer_noise_density": 2.0e-3,
            "accelerometer_random_walk": 3.0e-3,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "sensor.yaml")
            path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            sensor = load_euroc_imu_sensor(path)

        self.assertEqual(sensor.rate_hz.item(), 200.0)
        self.assertAlmostEqual(sensor.gyro_noise_density.item(), 1.6968e-4)
        self.assertAlmostEqual(sensor.gyro_random_walk.item(), 1.9393e-5)
        self.assertAlmostEqual(sensor.accel_noise_density.item(), 2.0e-3)
        self.assertAlmostEqual(sensor.accel_random_walk.item(), 3.0e-3)
        self.assertTrue(torch.allclose(
            sensor.T_BS.translation(),
            torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float64),
        ))
```

Add a second test asserting that a real `EuRoC_Sequence` frame carries a
sensor model and that estimator input does not require `gt_attitude`.

- [ ] **Step 2: Run the tests and verify the import fails**

Run:

```bash
/home/zzh/MACVO/.venv/bin/python -m unittest   Scripts.UnitTest.test_euroc_imu_metadata -v
```

Expected: failure because `load_euroc_imu_sensor` and the sensor dataclass do
not exist.

- [ ] **Step 3: Add the sensor dataclass**

Add to `DataLoader/Interface.py`:

```python
@dataclass(kw_only=True)
class IMUSensorModel(Collatable):
    T_BS: pp.LieTensor
    rate_hz: torch.Tensor
    gyro_noise_density: torch.Tensor
    gyro_random_walk: torch.Tensor
    accel_noise_density: torch.Tensor
    accel_random_walk: torch.Tensor


@dataclass(kw_only=True)
class IMUData(Collatable):
    T_BS: pp.LieTensor
    time_ns: torch.Tensor
    gravity: list[float]
    acc: torch.Tensor
    gyro: torch.Tensor
    sensor: IMUSensorModel | None = None
```

Keep the existing IMUData properties unchanged.

- [ ] **Step 4: Parse the EuRoC sensor file**

Add:

```python
def load_euroc_imu_sensor(path: Path) -> IMUSensorModel:
    config, _ = load_config(path)
    matrix = torch.tensor(
        config.T_BS.data, dtype=torch.float64
    ).reshape(1, 4, 4)
    return IMUSensorModel(
        T_BS=pp.from_matrix(matrix, pp.SE3_type),
        rate_hz=torch.tensor([float(config.rate_hz)], dtype=torch.float64),
        gyro_noise_density=torch.tensor(
            [float(config.gyroscope_noise_density)], dtype=torch.float64
        ),
        gyro_random_walk=torch.tensor(
            [float(config.gyroscope_random_walk)], dtype=torch.float64
        ),
        accel_noise_density=torch.tensor(
            [float(config.accelerometer_noise_density)], dtype=torch.float64
        ),
        accel_random_walk=torch.tensor(
            [float(config.accelerometer_random_walk)], dtype=torch.float64
        ),
    )
```

Load it once in `EurocIMULoader.__init__` and attach it to every returned
`IMUData`. Replace `pp.identity_SE3(0)` with the parsed one-element
extrinsic.

- [ ] **Step 5: Run metadata and existing sequence tests**

Run:

```bash
/home/zzh/MACVO/.venv/bin/python -m unittest   Scripts.UnitTest.test_euroc_imu_metadata   Scripts.UnitTest.test_config_sequence -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add DataLoader/Interface.py DataLoader/Dataset/EuRoC.py   Scripts/UnitTest/test_euroc_imu_metadata.py
git commit -m "feat: expose EuRoC IMU sensor metadata"
```

### Task 3: Return exact camera-bounded IMU intervals

**Files:**
- Modify: `DataLoader/Dataset/EuRoC.py`
- Create: `Scripts/UnitTest/test_euroc_imu_intervals.py`

- [ ] **Step 1: Write boundary interpolation tests**

Use synthetic samples at 0, 5, 10, and 15 ms, then query 2–12 ms:

```python
def test_interval_interpolates_camera_boundaries(self):
    time_ns = torch.tensor([0, 5, 10, 15], dtype=torch.int64) * 1_000_000
    acc = torch.tensor([
        [0.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [15.0, 0.0, 0.0],
    ], dtype=torch.float64)
    gyro = acc.clone()

    interval = interpolate_imu_interval(
        time_ns, acc, gyro, 2_000_000, 12_000_000
    )

    self.assertEqual(interval.time_ns.tolist(), [2_000_000, 5_000_000,
                                                 10_000_000, 12_000_000])
    self.assertTrue(torch.allclose(
        interval.acc[:, 0],
        torch.tensor([2.0, 5.0, 10.0, 12.0], dtype=torch.float64),
    ))
```

Also test exact endpoints, non-overlapping intervals, and insufficient sample
failure.

- [ ] **Step 2: Verify tests fail**

Run:

```bash
/home/zzh/MACVO/.venv/bin/python -m unittest   Scripts.UnitTest.test_euroc_imu_intervals -v
```

Expected: import failure for `interpolate_imu_interval`.

- [ ] **Step 3: Implement a pure interval helper**

Add a small result dataclass and helper that:

1. validates strictly increasing IMU timestamps;
2. finds samples surrounding both requested boundaries;
3. linearly interpolates accelerometer and gyro values at each boundary;
4. returns every interior sample once;
5. guarantees first and last timestamps equal the camera timestamps;
6. rejects intervals with fewer than two output timestamps.

Use this helper in `frameRangeQuery` instead of slicing directly by
`cam2imuIdx`.

- [ ] **Step 4: Run interval and EuRoC loader tests**

Run:

```bash
/home/zzh/MACVO/.venv/bin/python -m unittest   Scripts.UnitTest.test_euroc_imu_intervals   Scripts.UnitTest.test_euroc_imu_metadata -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add DataLoader/Dataset/EuRoC.py   Scripts/UnitTest/test_euroc_imu_intervals.py
git commit -m "fix: align IMU intervals to camera timestamps"
```

### Task 4: Implement the PyPose–GTSAM bridge

**Files:**
- Create: `Module/Optimization/GTSAMBridge.py`
- Create: `Scripts/UnitTest/test_gtsam_bridge.py`

- [ ] **Step 1: Write failing conversion tests**

Cover:

- SE3 matrix round trip;
- PyPose tangent order `[tx,ty,tz,rx,ry,rz]` to GTSAM order
  `[rx,ry,rz,tx,ty,tz]`;
- information/covariance inverse consistency;
- camera-relative to body-relative conversion.

Example:

```python
def test_information_permutation_preserves_named_axes(self):
    information = torch.diag(torch.tensor(
        [1., 2., 3., 4., 5., 6.], dtype=torch.float64
    ))
    converted = information_pypose_to_gtsam(information)
    self.assertTrue(torch.equal(
        converted.diagonal(),
        torch.tensor([4., 5., 6., 1., 2., 3.], dtype=torch.float64),
    ))
```

For the extrinsic test, construct non-commuting rotations and translations and
compare against direct homogeneous matrix multiplication.

- [ ] **Step 2: Verify tests fail**

Run:

```bash
/home/zzh/MACVO/.venv/bin/python -m unittest   Scripts.UnitTest.test_gtsam_bridge -v
```

Expected: import failure.

- [ ] **Step 3: Implement the bridge**

Provide these exact public functions:

```python
PYPOSE_TO_GTSAM = (3, 4, 5, 0, 1, 2)


def pypose_se3_to_pose3(pose: torch.Tensor | pp.LieTensor) -> gtsam.Pose3:
    matrix = pp.SE3(torch.as_tensor(pose).double()).matrix().cpu().numpy()
    return gtsam.Pose3(gtsam.Rot3(matrix[:3, :3]), matrix[:3, 3])


def pose3_to_pypose_se3(pose: gtsam.Pose3) -> torch.Tensor:
    matrix = torch.from_numpy(pose.matrix()).double()
    return pp.from_matrix(matrix, pp.SE3_type).tensor()


def information_pypose_to_gtsam(information: torch.Tensor) -> np.ndarray:
    info = torch.as_tensor(information).double()
    order = torch.tensor(PYPOSE_TO_GTSAM)
    return info[order][:, order].cpu().numpy()


def gtsam_noise_from_pypose_information(information: torch.Tensor):
    info = information_pypose_to_gtsam(information)
    covariance = np.linalg.inv(info)
    covariance = 0.5 * (covariance + covariance.T)
    return gtsam.noiseModel.Gaussian.Covariance(covariance)


def camera_factor_to_body(measurement, body_T_camera):
    camera = pp.SE3(torch.as_tensor(measurement).double())
    T_BC = pp.SE3(torch.as_tensor(body_T_camera).double())
    return (T_BC @ camera @ T_BC.Inv()).tensor()
```

Use explicit shape, finiteness, symmetry, and Cholesky validation.

- [ ] **Step 4: Run bridge tests**

Run:

```bash
/home/zzh/MACVO/.venv/bin/python -m unittest   Scripts.UnitTest.test_gtsam_bridge -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add Module/Optimization/GTSAMBridge.py   Scripts/UnitTest/test_gtsam_bridge.py
git commit -m "feat: add PyPose GTSAM conversion bridge"
```

### Task 5: Implement combined IMU preintegration

**Files:**
- Create: `Module/Optimization/IMUPreintegration.py`
- Create: `Scripts/UnitTest/test_imu_preintegration.py`

- [ ] **Step 1: Write failing stationary and constant-rotation tests**

The stationary test uses gravity-aligned specific force and zero angular rate.
The constant-rotation test integrates a known yaw rate for 0.1 seconds.

Assert:

- integrated duration exactly equals the camera interval;
- stationary delta rotation and horizontal delta velocity are near zero;
- constant yaw matches the analytical rotation;
- sensor noise and random-walk covariances equal squared metadata values.
- stationary initialization estimates the expected visual-world gravity vector
  and gyroscope bias without reading ground truth;
- a moving initialization interval selects the documented NED fallback.

- [ ] **Step 2: Verify tests fail**

Run:

```bash
/home/zzh/MACVO/.venv/bin/python -m unittest   Scripts.UnitTest.test_imu_preintegration -v
```

Expected: import failure.

- [ ] **Step 3: Implement parameter construction**

Provide:

```python
def build_combined_params(sensor, gravity_vector):
    gravity = np.asarray(gravity_vector, dtype=np.float64).reshape(3)
    params = gtsam.PreintegrationCombinedParams(gravity)
    params.setAccelerometerCovariance(
        np.eye(3) * float(sensor.accel_noise_density.item()) ** 2
    )
    params.setGyroscopeCovariance(
        np.eye(3) * float(sensor.gyro_noise_density.item()) ** 2
    )
    params.setIntegrationCovariance(np.eye(3) * 1e-8)
    params.setBiasAccCovariance(
        np.eye(3) * float(sensor.accel_random_walk.item()) ** 2
    )
    params.setBiasOmegaCovariance(
        np.eye(3) * float(sensor.gyro_random_walk.item()) ** 2
    )
    return params
```

Set `body_P_sensor` when the IMU extrinsic is non-identity.

- [ ] **Step 4: Implement interval integration**

Provide:

```python
def preintegrate_interval(imu, params, bias):
    pim = gtsam.PreintegratedCombinedMeasurements(params, bias)
    time = imu.time_ns[0, :, 0].double() * 1e-9
    acc = imu.acc[0].double()
    gyro = imu.gyro[0].double()
    for index in range(len(time) - 1):
        dt = float(time[index + 1] - time[index])
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("IMU timestamps must increase")
        measured_acc = 0.5 * (acc[index] + acc[index + 1])
        measured_gyro = 0.5 * (gyro[index] + gyro[index + 1])
        pim.integrateMeasurement(
            measured_acc.cpu().numpy(),
            measured_gyro.cpu().numpy(),
            dt,
        )
    return pim
```

Validate finite measurements and a minimum of two samples.

Also implement:

```python
def estimate_initial_imu_state(
    initial_body_pose,
    imu_intervals,
    *,
    gravity_mps2,
    initialization_seconds,
    stationary_gyro_threshold,
    stationary_acc_threshold,
):
    """Return gravity_W, gyro_bias, accel_bias, stationary diagnostics."""
```

The stationary path computes `gravity_W = -R_WB @ mean_acc_B`, normalizes it
to `gravity_mps2`, sets gyro bias to the mean angular rate, and initializes
accelerometer bias to zero. The moving fallback returns project-NED gravity
`[0, 0, +gravity_mps2]` and zero biases. The return diagnostics records which
path was used.

- [ ] **Step 5: Run preintegration tests**

Run:

```bash
/home/zzh/MACVO/.venv/bin/python -m unittest   Scripts.UnitTest.test_imu_preintegration -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add Module/Optimization/IMUPreintegration.py   Scripts/UnitTest/test_imu_preintegration.py
git commit -m "feat: add GTSAM combined IMU preintegration"
```

### Task 6: Build the bounded local VIO graph

**Files:**
- Create: `Module/Optimization/LocalVIO.py`
- Create: `Scripts/UnitTest/test_local_vio.py`

- [ ] **Step 1: Write failing synthetic graph tests**

Create deterministic five-state tests:

1. Constant velocity with IMU factors and noisy visual factors.
2. Constant yaw rotation.
3. One missing visual adjacent factor while IMU remains valid.
4. Degenerate visual information rejected without aborting optimization.
5. No ground-truth object accepted by the public API.

Assert finite poses, velocities, and biases and lower final graph error.

- [ ] **Step 2: Verify tests fail**

Run:

```bash
/home/zzh/MACVO/.venv/bin/python -m unittest   Scripts.UnitTest.test_local_vio -v
```

Expected: import failure.

- [ ] **Step 3: Define focused state and result types**

Implement:

```python
@dataclass(frozen=True)
class LocalVIOConfig:
    window_size: int = 5
    optimizer_iterations: int = 10
    gravity_mps2: float = 9.81007
    visual_huber_delta: float = 3.0
    initialization_seconds: float = 0.5
    stationary_gyro_threshold: float = 0.05
    stationary_acc_threshold: float = 0.5
    pose_prior_rotation_sigma_deg: float = 0.1
    pose_prior_translation_sigma_m: float = 0.01
    velocity_prior_sigma_mps: float = 1.0
    accel_bias_prior_sigma_mps2: float = 0.1
    gyro_bias_prior_sigma_radps: float = 0.01
    max_velocity_mps: float = 30.0
    max_accel_bias_norm_mps2: float = 2.0
    max_gyro_bias_norm_radps: float = 0.5


@dataclass(frozen=True)
class LocalVIOState:
    frame_id: int
    pose: torch.Tensor
    velocity: torch.Tensor
    accel_bias: torch.Tensor
    gyro_bias: torch.Tensor


@dataclass(frozen=True)
class LocalVIOResult:
    states: tuple[LocalVIOState, ...]  # variadic tuple type, not a placeholder
    diagnostics: dict
```

- [ ] **Step 4: Implement graph construction**

Implement `LocalVIOOptimizer.optimize(self, frame_ids, initial_states,
visual_factors, imu_intervals)` so that it:

- accepts cached frame IDs, visual `PoseGraphFactor` objects, and IMU
  intervals only;
- creates `X/V/B` keys;
- adds oldest-state priors;
- adds `CombinedImuFactor` objects;
- adds adjacent/skip `BetweenFactorPose3` objects with Gaussian noise
  wrapped in a GTSAM Huber robust model using `visual_huber_delta`;
- initializes later states from PIM prediction when available;
- configures LM maximum iterations from the config;
- validates finite and bounded outputs;
- returns diagnostics with factor counts, initial error, final error, and timing.

No argument or field named `gt_*` is permitted in this module.

- [ ] **Step 5: Run local VIO tests**

Run:

```bash
/home/zzh/MACVO/.venv/bin/python -m unittest   Scripts.UnitTest.test_local_vio -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add Module/Optimization/LocalVIO.py   Scripts/UnitTest/test_local_vio.py
git commit -m "feat: add bounded local stereo inertial graph"
```

### Task 7: Integrate local VIO into online loop odometry

**Files:**
- Create: `Odometry/OnlineLoopVIOWindowMACVO.py`
- Modify: `MACVO.py:150-160`
- Create: `Scripts/UnitTest/test_online_loop_vio_window_macvo.py`

- [ ] **Step 1: Write failing integration tests**

Build a small fake graph and fake `LocalVIOOptimizer`. Verify:

- initialization caches frame-zero body state;
- `_after_window_step` compresses current adjacent/skip factors, runs local
  VIO, writes optimized poses, then invokes the existing loop step;
- pose writeback reanchors map points;
- missing IMU on a frame records `imu_missing` and preserves visual poses;
- no access to `frame.gt_attitude`.

- [ ] **Step 2: Verify tests fail**

Run:

```bash
/home/zzh/MACVO/.venv/bin/python -m unittest   Scripts.UnitTest.test_online_loop_vio_window_macvo -v
```

Expected: import failure.

- [ ] **Step 3: Implement the new odometry class**

Create:

```python
class OnlineLoopVIOWindowMACVO(OnlineLoopWindowMACVO):
    def __init__(self, *, local_vio, **kwargs):
        super().__init__(**kwargs)
        self.local_vio_config = LocalVIOConfig(**vars(local_vio))
        self.local_vio = LocalVIOOptimizer(self.local_vio_config)
        self.local_vio_states = {}
        self.local_vio_imu = {}
        self.local_vio_records = []
        self.local_vio_rebuilds = 0

    def initialize(self, frame0):
        if not hasattr(frame0, "imu"):
            raise ValueError("local VIO requires StereoInertialFrame")
        super().initialize(frame0)
        self.local_vio.initialize_from_visual(
            frame_id=0,
            pose=self.graph.frames.data["pose"][0].double(),
            imu=frame0.imu,
            body_T_camera=frame0.stereo.T_BS[0].double(),
        )

    def _after_window_step(self, frame, depth, adjacent, skip):
        self._run_local_vio(frame, adjacent, skip)
        super()._after_window_step(frame, depth, adjacent, skip)
```

Use `compress_edge_to_pose_factor` for local visual factors so the relative
measurement is estimated from the raw covariance-aware 3D correspondences.
Do not use `linearize_edge_to_pose_factor` for the local VIO measurement and
do not add duplicate entries to the global `pose_factors` dictionary before
the parent method runs.

- [ ] **Step 4: Register the odometry type**

Add to `MACVO.py`:

```python
elif system_type == "OnlineLoopVIOWindowMACVO":
    from Odometry.OnlineLoopVIOWindowMACVO import OnlineLoopVIOWindowMACVO
    system_class = OnlineLoopVIOWindowMACVO
```

- [ ] **Step 5: Run integration and existing online-loop tests**

Run:

```bash
/home/zzh/MACVO/.venv/bin/python -m unittest   Scripts.UnitTest.test_online_loop_vio_window_macvo   Scripts.UnitTest.test_online_loop_window_macvo -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add Odometry/OnlineLoopVIOWindowMACVO.py MACVO.py   Scripts/UnitTest/test_online_loop_vio_window_macvo.py
git commit -m "feat: integrate local VIO with online loop odometry"
```

### Task 8: Handle global loop writeback and save VIO diagnostics

**Files:**
- Modify: `Odometry/OnlineLoopVIOWindowMACVO.py`
- Modify: `Scripts/UnitTest/test_online_loop_vio_window_macvo.py`

- [ ] **Step 1: Write failing global-correction tests**

Create a local state with nonzero velocity and biases, apply a known SE3 global
correction, and assert:

- local poses receive the correction;
- velocity is multiplied by the correction rotation only;
- both biases are unchanged;
- the next local optimization rebuilds from cached IMU and visual factors;
- rebuild count increments exactly once.

Add a diagnostics test checking `local_vio_diagnostics.json`.

- [ ] **Step 2: Verify tests fail**

Run:

```bash
/home/zzh/MACVO/.venv/bin/python -m unittest   Scripts.UnitTest.test_online_loop_vio_window_macvo -v
```

Expected: failures in writeback and diagnostics assertions.

- [ ] **Step 3: Override global result application**

Wrap the parent method:

```python
def _apply_pose_graph_result(self, solved):
    before = self.graph.frames.data["pose"].tensor.double().clone()
    applied = super()._apply_pose_graph_result(solved)
    if not applied:
        return False
    after = self.graph.frames.data["pose"].tensor.double()
    self.local_vio.apply_global_pose_corrections(before, after)
    self.local_vio.invalidate_graph()
    self.local_vio_rebuilds += 1
    return True
```

Only active-window states are updated.

- [ ] **Step 4: Save diagnostics**

Extend `save_window_diagnostics` to write:

```python
payload = {
    "config": asdict(self.local_vio_config),
    "records": self.local_vio_records,
    "rebuilds": self.local_vio_rebuilds,
    "states": self.local_vio.serialize_states(),
}
(folder / "local_vio_diagnostics.json").write_text(
    json.dumps(payload, indent=2, allow_nan=False),
    encoding="utf-8",
)
```

Do not serialize ground truth.

- [ ] **Step 5: Run integration tests**

Run:

```bash
/home/zzh/MACVO/.venv/bin/python -m unittest   Scripts.UnitTest.test_online_loop_vio_window_macvo   Scripts.UnitTest.test_online_loop_window_macvo -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add Odometry/OnlineLoopVIOWindowMACVO.py   Scripts/UnitTest/test_online_loop_vio_window_macvo.py
git commit -m "fix: rebuild local VIO after loop writeback"
```

### Task 9: Add isolated VIO configuration and comparison mode

**Files:**
- Create: `Config/Experiment/MACVO/MACVO_Fast_WindowVIO_ORBLoop_Sparse_t030.yaml`
- Modify: `Scripts/Experiment/CompareOnlineORBLoop.py`
- Modify: `Scripts/UnitTest/test_compare_online_orb_loop.py`
- Create: `Scripts/run_euroc_sparse_t030_vio_gate.sh`
- Create: `Scripts/UnitTest/test_run_euroc_sparse_t030_vio_gate.py`

- [ ] **Step 1: Write failing configuration-isolation tests**

Assert that the new config differs from the t030 visual-only config only by:

- odometry type and name;
- one `local_vio` configuration block.

Assert that the comparison parser accepts
`window_orb_loop_sparse_t030_vio` and treats it as an online diagnostic mode.

- [ ] **Step 2: Verify tests fail**

Run:

```bash
/home/zzh/MACVO/.venv/bin/python -m unittest   Scripts.UnitTest.test_compare_online_orb_loop -v
```

Expected: missing mode/config failure.

- [ ] **Step 3: Create the VIO config**

Start from the tracked t030 YAML. Change:

```yaml
Odometry:
  type: OnlineLoopVIOWindowMACVO
  name: MACVO-Fast-WindowVIO5-ORBLoop-Sparse-t030
  args:
    local_vio:
      enabled: true
      window_size: 5
      optimizer_iterations: 10
      gravity_mps2: 9.81007
      visual_huber_delta: 3.0
      initialization_seconds: 0.5
      stationary_gyro_threshold: 0.05
      stationary_acc_threshold: 0.5
      pose_prior_rotation_sigma_deg: 0.1
      pose_prior_translation_sigma_m: 0.01
      velocity_prior_sigma_mps: 1.0
      accel_bias_prior_sigma_mps2: 0.1
      gyro_bias_prior_sigma_radps: 0.01
      max_velocity_mps: 30.0
      max_accel_bias_norm_mps2: 2.0
      max_gyro_bias_norm_radps: 0.5
```

All visual, loop, information-scale, and switch settings remain identical.

- [ ] **Step 4: Register the comparison mode**

Map `window_orb_loop_sparse_t030_vio` to the new config and add it to
`ONLINE_DIAGNOSTIC_MODES`.

- [ ] **Step 5: Write the four-sequence gate runner test**

The script defaults must be:

```text
MH04 MH05 V103 V203
```

Its dry run must select only
`window_orb_loop_sparse_t030_vio` and use a new result root:

```text
Results/SparseORBLoop_EuRoC_alpha10000_t030_vio_gate
```

- [ ] **Step 6: Implement the gate runner**

Reuse the existing resumable batch runner through `MODE`, `ODOM_CONFIG`, and
`RESULT_ROOT`, following the t030 wrapper pattern.

- [ ] **Step 7: Run config and runner tests**

Run:

```bash
/home/zzh/MACVO/.venv/bin/python -m unittest   Scripts.UnitTest.test_compare_online_orb_loop   Scripts.UnitTest.test_run_euroc_sparse_t030_vio_gate   Scripts.UnitTest.test_run_euroc_sparse_t030_batch   Scripts.UnitTest.test_run_euroc_sparse_batch -v
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add   Config/Experiment/MACVO/MACVO_Fast_WindowVIO_ORBLoop_Sparse_t030.yaml   Scripts/Experiment/CompareOnlineORBLoop.py   Scripts/UnitTest/test_compare_online_orb_loop.py   Scripts/run_euroc_sparse_t030_vio_gate.sh   Scripts/UnitTest/test_run_euroc_sparse_t030_vio_gate.py
git commit -m "test: add local VIO EuRoC gate experiment"
```

### Task 10: Run regression, smoke, and four-sequence acceptance gates

**Files:**
- Verify all files above.
- Record results only after successful runs.

- [ ] **Step 1: Run the focused CPU suite**

Run:

```bash
/home/zzh/MACVO/.venv/bin/python -m unittest   Scripts.UnitTest.test_gtsam_dependency   Scripts.UnitTest.test_euroc_imu_metadata   Scripts.UnitTest.test_euroc_imu_intervals   Scripts.UnitTest.test_gtsam_bridge   Scripts.UnitTest.test_imu_preintegration   Scripts.UnitTest.test_local_vio   Scripts.UnitTest.test_online_loop_vio_window_macvo   Scripts.UnitTest.test_online_loop_window_macvo   Scripts.UnitTest.test_compare_online_orb_loop   Scripts.UnitTest.test_run_euroc_sparse_t030_vio_gate -v
```

Expected: zero failures.

- [ ] **Step 2: Run visual-only regression tests**

Run:

```bash
/home/zzh/MACVO/.venv/bin/python -m unittest   Scripts.UnitTest.test_window_icp   Scripts.UnitTest.test_window_map   Scripts.UnitTest.test_loop_hypothesis   Scripts.UnitTest.test_sparse_geometry   Scripts.UnitTest.test_run_euroc_sparse_batch   Scripts.UnitTest.test_run_euroc_sparse_t030_batch -v
```

Expected: zero failures and no changed visual-only configuration.

- [ ] **Step 3: Run a 30-frame real-GPU smoke test**

Run V203 frames 0–29 with the VIO comparison mode:

```bash
/home/zzh/MACVO/.venv/bin/python   Scripts/Experiment/CompareOnlineORBLoop.py   --sequence V203 --seq-from 0 --seq-to 30 --seed 0   --modes window_orb_loop_sparse_t030_vio   --result-root /home/zzh/MACVO/Results/VIO_smoke
```

Expected:

- 30 finite output poses;
- `local_vio_diagnostics.json` exists;
- every optimized state is finite;
- no ground-truth initialization marker;
- no visual-only result directory is modified.

- [ ] **Step 4: Run the four-sequence gate**

Run:

```bash
/home/zzh/.config/superpowers/worktrees/MACVO/window-icp-global-v03/Scripts/run_euroc_sparse_t030_vio_gate.sh
```

Expected result root:

```text
/home/zzh/MACVO/Results/SparseORBLoop_EuRoC_alpha10000_t030_vio_gate
```

- [ ] **Step 5: Evaluate acceptance criteria**

Compare against the t030 visual-only values for MH04, MH05, V103, and V203.
Create a table with:

```text
sequence, ATE, RTE, ROE, RPE, runtime, VIO failures,
loop factors, effective long loops
```

Accept the enhancement for the paper only if:

- mean ATE degradation is at most 5%;
- mean RTE or mean ROE improves at least 5%;
- at least two gate sequences improve;
- VIO overhead is at most 50 ms/frame;
- no obvious false-loop increase occurs.

- [ ] **Step 6: Run all 11 sequences only after the gate passes**

Create the full runner by reusing the four-sequence wrapper pattern, with
default sequences:

```text
MH01 MH02 MH03 MH04 MH05 V101 V102 V103 V201 V202 V203
```

Do not start the full batch when the gate fails.

- [ ] **Step 7: Final verification and version-control audit**

Run:

```bash
git diff --check
git status --short
git log --oneline --decorate -12
```

Expected: only the user's pre-existing `docs/WindowICP.md` modification may
remain uncommitted; all VIO implementation files are committed separately.
