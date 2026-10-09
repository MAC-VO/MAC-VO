# Local Tightly Coupled VIO Enhancement Design

## Status and Paper Role

This feature is an optional enhancement experiment for the CCC paper. The
paper's primary method remains covariance-aware multi-frame sparse loop closure.
The VIO result is reported as an additional row, not as a prerequisite for the
main loop-closure claims.

The implementation baseline is commit `4af8251` on
`experiment/window-icp-global-v03`, with the t030 loop configuration and its
11-sequence EuRoC result preserved unchanged.

## Goal

Add a local tightly coupled stereo-inertial optimizer that combines existing
covariance-aware visual relative-pose factors with EuRoC IMU preintegration.
Keep the existing sparse loop detector and pose-only global graph unchanged.

The first version uses a bounded batch graph rebuilt over the newest five
camera frames. It does not implement a persistent incremental marginalization
prior.

## Non-Goals

The first version does not:

- add velocity or IMU bias variables to the global loop graph;
- perform global visual-inertial bundle adjustment;
- estimate camera-IMU extrinsics or temporal offset online;
- integrate OpenVINS;
- replace the ORB sparse loop-closure pipeline;
- add TensorRT;
- use ground-truth pose, velocity, attitude, gravity, or bias for estimator
  initialization;
- alter the existing t030 visual-only configuration or results.

## Architecture

```text
EuRoC stereo frames ──> FlowFormerCov ──> covariance-aware ICP edges
       │                                      │
       └── 200 Hz IMU ──> preintegration ─────┤
                                              ↓
                                  five-frame GTSAM batch graph
                                  pose + velocity + IMU bias
                                              ↓
                                    local optimized body states
                                              │
              ORB sparse loop closure ────────┤
                                              ↓
                                  existing pose-only global graph
```

The GTSAM graph is local and rebuilt whenever a new camera frame arrives. The
existing asynchronous global pose graph remains the owner of loop-closure
corrections.

## Coordinate Convention

GTSAM state poses are body/IMU poses `T_WB`. Existing visual factors are
camera-frame relative poses. The EuRoC camera extrinsic already stored as
`T_BS` must be normalized into one explicitly named transform convention and
covered by unit tests.

For a camera relative measurement `Z_Cij`, the body relative measurement is:

```text
Z_Bij = T_BC * Z_Cij * T_CB
```

The exact multiplication order must be verified with synthetic transforms and
the project's NED/EDN conversion. No coordinate conversion is accepted based
only on visual inspection of trajectories.

The existing PyPose factor tangent order is translation followed by rotation,
while GTSAM Pose3 noise uses rotation followed by translation. Every 6x6
information or covariance matrix crossing the bridge must be permuted with an
explicit tested permutation; directly passing the matrix is invalid.

## Local State

For every camera frame `i` in the local window, create:

- `X(i)`: GTSAM `Pose3`, body pose in the world frame;
- `V(i)`: three-dimensional world-frame velocity;
- `B(i)`: `imuBias.ConstantBias`, accelerometer and gyroscope bias.

The oldest state in each rebuilt window receives pose, velocity, and bias
priors derived from the previous optimized local state. This bounds the graph
without introducing a persistent marginalization prior.

## IMU Data and Preintegration

Extend the EuRoC IMU loader to expose:

- body-to-IMU extrinsic from `imu0/sensor.yaml`;
- gyroscope noise density;
- gyroscope random walk;
- accelerometer noise density;
- accelerometer random walk;
- IMU rate.

Camera interval queries must include interpolated boundary samples so the
preintegration interval exactly matches adjacent camera timestamps.

Use GTSAM `PreintegratedCombinedMeasurements` and `CombinedImuFactor`.
The factor connects:

```text
X(i), V(i), B(i), X(i+1), V(i+1), B(i+1)
```

The implementation must convert continuous-time sensor noise values into the
covariance/PSD convention expected by GTSAM and test the conversion explicitly.

## Visual Factors

Reuse the existing covariance-aware visual measurements rather than
reimplementing dense point residuals in GTSAM.

Each accepted adjacent or skip edge is compressed into the existing
`PoseGraphFactor` representation:

- six-degree-of-freedom relative SE(3) measurement;
- positive-definite 6x6 information matrix;
- observation count and factor kind.

Convert the camera measurement into a body measurement and add a
`BetweenFactorPose3` with a Gaussian covariance obtained from the inverse of
the information matrix.

Adjacent factors connect `X(i)` and `X(i+1)`. Skip factors connect `X(i)`
and `X(i+2)`. A rejected or degenerate visual compression must not abort IMU
propagation; it is recorded in diagnostics and the graph continues with the
remaining factors.

## Initialization

Initialization must not use `gt_attitude` or other ground-truth fields.

The first implementation uses:

1. First body pose from the visual estimator.
2. Gravity vector in the existing visual world frame from the mean
   accelerometer measurement, the initial visual body orientation, and a
   configurable initial interval, subject to a stationarity check.
3. Gyroscope bias from the initial mean angular velocity when stationary.
4. Accelerometer bias initialized to zero after gravity alignment.
5. Initial velocity from finite differences over the first valid visual body
   poses.
6. Explicit priors on pose, velocity, accelerometer bias, and gyroscope bias.

If the stationarity test fails, the system falls back to configurable zero
bias and the project's NED gravity vector `[0, 0, +g]` and records the
fallback in diagnostics. GTSAM preintegration receives this explicit gravity
vector; it must not silently assume its default Z-up convention.

## Optimization and Writeback

At every new frame:

1. Cache the frame's raw IMU interval and visual factors.
2. Select the newest at most five camera states.
3. Rebuild a GTSAM nonlinear factor graph for those states.
4. Insert oldest-state priors.
5. Insert Combined IMU factors.
6. Insert adjacent and skip visual factors.
7. Initialize new states from IMU prediction, falling back to visual pose and
   finite-difference velocity when prediction is unavailable.
8. Optimize with Levenberg-Marquardt.
9. Validate all optimized poses, velocities, and biases for finiteness and
   bounded increments.
10. Write body poses back through the existing map re-anchoring path.

The visual-only WindowICP path remains available under its existing
configuration.

## Global Loop Interaction

The global loop graph remains pose-only.

When a global writeback changes poses inside the active local window:

1. Apply the global pose correction to local body poses.
2. Rotate world-frame velocities with the correction rotation.
3. Keep accelerometer and gyroscope biases unchanged.
4. Discard the current local GTSAM graph.
5. Rebuild it from cached raw IMU intervals and visual factors at the next
   optimization step.

This avoids attempting to transform a stale nonlinear marginalization prior.

## Components

New focused components are expected:

- `Module/Optimization/GTSAMBridge.py`: SE(3), Pose3, information, covariance,
  and extrinsic conversions.
- `Module/Optimization/IMUPreintegration.py`: sensor parameters, interval
  normalization, and GTSAM preintegration construction.
- `Module/Optimization/LocalVIO.py`: bounded graph state, factor insertion,
  optimization, validation, and diagnostics.
- `Odometry/OnlineLoopVIOWindowMACVO.py`: integration with existing
  WindowICP and sparse loop closure.
- `Config/Experiment/MACVO/MACVO_Fast_WindowVIO_ORBLoop_Sparse_t030.yaml`:
  isolated experimental configuration.

Existing visual-only files must not change semantics.

## Configuration

The VIO configuration includes:

```yaml
local_vio:
  enabled: true
  window_size: 5
  optimizer_iterations: 10
  visual_huber_delta: 3.0
  gravity_mps2: 9.81007
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

Noise densities and random walks come from the dataset sensor metadata rather
than being duplicated in the odometry configuration.

The optional runtime dependency is pinned to `gtsam==4.3.0`, which provides a
CPython 3.12 Linux wheel compatible with the current development environment.

## Diagnostics

Write `local_vio_diagnostics.json` containing:

- initialization mode and stationarity metrics;
- IMU sample count and integrated duration per interval;
- graph state/factor counts;
- optimization status and cost;
- pose, velocity, and bias increments;
- missing or rejected visual factors;
- global-writeback rebuild count;
- per-frame VIO optimization time;
- fallback and failure reasons.

No ground-truth values are written into estimator diagnostics.

## Testing

Required tests include:

- EuRoC sensor-noise and extrinsic parsing;
- exact IMU interval boundary handling;
- stationary preintegration;
- constant-velocity and constant-rotation synthetic trajectories;
- bias random-walk factor construction;
- PyPose SE3 to GTSAM Pose3 round trips;
- camera-to-body relative measurement conversion;
- visual information-to-covariance conversion;
- five-frame graph recovery with visual and IMU factors;
- missing visual factor fallback;
- global writeback pose/velocity transformation;
- graph rebuild after loop correction;
- no access to `gt_attitude` during estimation;
- existing visual-only loop and batch tests remain green.

## Experiment Plan

The main paper table remains:

```text
Original MAC-VO
WindowICP
WindowICP + sparse loop t030
```

The enhancement table adds:

```text
WindowICP + sparse loop t030 + local VIO
```

Run short development experiments on MH04, MH05, V103, and V203. Only run all
11 EuRoC sequences after the four-sequence gate passes.

Report ATE, RTE, ROE, RPE, runtime, peak memory, loop counts, and VIO failure
counts.

## Acceptance Criteria

The VIO enhancement is paper-worthy only if:

- all produced trajectories are complete and finite;
- estimator code never reads ground-truth initialization fields;
- mean ATE on the four-sequence gate degrades by no more than 5%;
- mean RTE or mean ROE improves by at least 5%;
- at least two of MH04, MH05, V103, and V203 improve;
- no obvious false-loop increase is introduced;
- local VIO adds at most 50 ms per camera frame on the current machine;
- all visual-only regression tests pass.

If these criteria are not met, the VIO work remains an engineering experiment
or future-work result and is not included in the CCC paper.
