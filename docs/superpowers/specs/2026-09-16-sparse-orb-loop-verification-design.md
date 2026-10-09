# Sparse ORB Loop Verification Design

## Goal

Replace the current single-frame `ORB-BoW -> long-range FlowFormerCov -> ICP`
acceptance path with a sparse, metric, multi-keyframe loop verifier. The first
milestone ends when a confirmed ORB hypothesis produces one conservative SE(3)
loop factor without using long-range dense flow as correspondence evidence.

Dense FlowFormerCov refinement, information calibration, shared-switch loop
bundles, and pose-depth bundle adjustment remain separate later milestones.

## Existing Behavior and Required Correction

The ORB sidecar currently discards keypoints and descriptors after constructing
a BoW vector. Python receives only `(source, score)` candidates, so the online
path immediately invokes bidirectional long-range FlowFormerCov.

The current bidirectional frontend also routes its first batch slot as target
stereo depth while the caller consumes the result as `source_depth`. That means
the loop validator samples a target depth map at source-frame pixel locations.
The sparse implementation must add a regression test and correct this ownership
before any SE(3) geometry is accepted.

## Scope

This milestone implements:

- ORB feature retention and descriptor matching inside the existing C++
  sidecar;
- protocol version 2 with bounded pixel correspondences per candidate;
- a fixed-batch historical stereo-depth frontend call;
- covariance-aware fixed-scale SE(3) RANSAC;
- reprojection, coverage, depth, and degeneracy validation;
- a two-of-three temporal/covisibility hypothesis tracker;
- direct conservative sparse loop factors;
- complete per-candidate diagnostics and focused unit/integration tests.

This milestone does not:

- use temporal FlowFormerCov flow to accept or refine a loop;
- add multiple neighborhood loop factors;
- change the closed-form loop switch implementation;
- optimize inverse depth or map structure;
- replace the current ORB vocabulary or resolve distribution licensing.

## Chosen Architecture

### ORB sidecar owns feature matching

The sidecar stores, for each inserted loop keyframe:

```text
StoredORBFrame
  frame_id
  keypoints
  descriptors
  bow_vector
```

For each eligible historical frame it still computes the BoW score. After
sorting and truncating to `top_k`, it performs Hamming KNN matching for each
candidate. A match survives only when it passes both:

1. Lowe ratio test with configurable ratio, initially `0.80`;
2. reverse best-match agreement.

Matches are sorted by descriptor distance and truncated to a configured maximum,
initially 300. The sidecar returns source and target pixel coordinates; it never
uses depth, CUDA, trajectory poses, or MACVO map state.

Keeping descriptor matching in the sidecar avoids transferring complete ORB
descriptor matrices over a text protocol and guarantees that retrieval and
geometric matching use the same extracted features.

### Protocol version 2

The startup record becomes:

```text
READY\t2\t<vocabulary_sha256>\t<opencv_version>
```

The query remains a single line and gains matching settings:

```text
QUERY\t<request_id>\t<target_id>\t<image_path>\t<min_gap>\t<top_k>\t<ratio>\t<max_matches>
```

The result remains one line. Each candidate is encoded as one field:

```text
<source>:<bow_score>:<raw_knn_count>:<ratio_count>:<mutual_count>:<matches>
```

`<matches>` is a semicolon-separated list of:

```text
source_x,source_y,target_x,target_y,hamming_distance
```

Coordinates and scores use decimal text with deterministic precision. Empty
matches use an empty final component. The Python parser validates candidate
counts, match counts, finite coordinates, nonnegative distances, request IDs,
and target IDs before publishing a response.

Protocol v1 is intentionally rejected instead of silently dropping sparse
geometry. A sidecar/protocol mismatch disables loop closing while local
WindowICP continues.

## Historical Depth Ownership

Target depth already comes from normal tracking and remains in
`PendingLoopTargetStore` until sparse validation finishes.

Historical source depth is computed on demand after an ORB candidate passes the
cheap image-space match-count and grid-coverage gates. A new frontend method:

```python
estimate_loop_depth(source: StereoData) -> IStereoDepth.Output
```

uses the existing batch-three CUDA graph by duplicating `source.left ->
source.right` into all three slots and returns only slot zero. This preserves the
one-model, one-graph, no-concurrent-GPU-call invariants without computing a
semantically misleading long-range flow measurement.

The serialized frontend records this call as a loop call. No dense historical
depth cache is created.

The existing `build_bidirectional_inputs` path is corrected separately so that
slot zero is source stereo whenever `estimate_bidirectional(source, target)` is
used by later dense refinement. Its return value is explicitly named
`source_depth` in tests and callers.

## Sparse Observation Construction

`Module/LoopClosure/SparseGeometry.py` owns all sparse geometry. It accepts:

- source and target frame IDs;
- matched source/target pixels;
- source and target stereo metadata;
- source and target depth outputs;
- the MACVO covariance model;
- validation configuration and deterministic seed.

It performs these gates in order:

1. finite, in-bounds pixel coordinates;
2. minimum raw mutual matches;
3. minimum occupied cells in a 4x6 source grid;
4. finite positive depth at both endpoints;
5. finite positive-definite 3D endpoint covariance;
6. minimum valid 3D correspondence count;
7. fixed-scale SE(3) RANSAC;
8. minimum RANSAC inlier count and ratio;
9. nondegenerate inlier geometry;
10. bidirectional median and percentile reprojection error;
11. final grid coverage after RANSAC.

Initial defaults are:

```yaml
min_mutual_matches: 40
min_valid_3d: 30
min_ransac_inliers: 25
min_ransac_ratio: 0.35
grid_rows: 4
grid_cols: 6
min_grid_cells: 6
ransac_iterations: 256
mahalanobis_threshold: 3.5
max_median_reprojection_px: 3.0
max_p90_reprojection_px: 6.0
min_geometry_ratio: 1.0e-3
orb_pixel_variance: 2.25
```

All thresholds are configuration values and are recorded in run diagnostics.

## Fixed-Scale SE(3) RANSAC

Each hypothesis samples three correspondences. Collinear or nearly collinear
samples are discarded. Weighted Kabsch estimates a rigid transform from target
camera points into source camera coordinates:

```text
Z_it * X_t ~= X_i
Z_it ~= T_i.Inv() @ T_t
```

Scale is fixed to one. For a candidate rotation `R`, an observation covariance
is:

```text
Sigma_k = Sigma_source_k + R Sigma_target_k R^T
```

An inlier satisfies:

```text
sqrt(r_k^T Sigma_k^-1 r_k) <= mahalanobis_threshold
```

The best model is selected by inlier count, then median Mahalanobis residual,
then deterministic sample order. The final transform is refit using every
inlier. The random generator seed is derived from `(source, target)` so repeated
runs produce identical factors.

Geometry is rejected when the centered inlier point cloud has insufficient
second singular value relative to its first. This rejects line-like and tiny
clusters before information is assigned.

## Sparse Measurement Diagnostics

Every evaluated candidate produces a serializable record containing:

```text
source, target, gap, bow_score, rank
raw_knn_matches, ratio_matches, mutual_matches
initial_grid_cells, valid_depth_pairs
ransac_iterations, ransac_inliers, ransac_ratio
mahalanobis_median, mahalanobis_p90
reprojection_forward_median_px, reprojection_backward_median_px
reprojection_forward_p90_px, reprojection_backward_p90_px
final_grid_cells, geometry_singular_values
measurement_se3, information_eigenvalues
status, rejection_reason
```

Failure records retain every metric available before the failed gate.

## Loop Hypothesis State Machine

`Module/LoopClosure/LoopHypothesis.py` owns hypothesis persistence independently
from ORB retrieval and pose-graph insertion.

A sparse support contains its source/target pair, SE(3) measurement, quality
metrics, and a correction estimate computed from the current trajectory:

```text
C = (T_source @ Z_source_target) @ T_target.Inv()
```

Supports belong to the same hypothesis when:

- their source frame IDs differ by at most `source_cluster_frames`, initially
  10;
- their target IDs increase and differ by at most `target_support_frames`,
  initially 10;
- the relative correction transforms differ by at most 0.25 m and 10 degrees.

States are:

```text
tentative: one support
confirmed: two compatible supports from different target keyframes
strong: three compatible supports from different target keyframes
expired: no compatible support within the target window
rejected: incompatible or duplicate support
```

Only transition into `confirmed` or `strong` may emit a loop factor. A
hypothesis emits at most one factor in this milestone. The emitted support is
the member with the highest ordered quality tuple:

```text
(ransac_inlier_count, ransac_ratio, final_grid_cells,
 -max_forward_backward_reprojection_p90, bow_score)
```

This avoids adding a correlated edge bundle before group-level switch support
exists.

## Conservative Sparse Loop Factor

The selected SE(3) measurement is inserted directly as a `PoseGraphFactor`.
This path does not run pairwise dense ICP compression.

The initial information matrix is deliberately conservative and diagonal:

```text
sigma_translation = 0.25 m
sigma_rotation = 10 degrees
Omega = diag(1/sigma_translation^2 repeated 3,
             1/sigma_rotation_rad^2 repeated 3)
```

The factor confidence is the clipped RANSAC inlier ratio. The observation count
is the final inlier count. These values are temporary safe defaults; empirical
calibration and switch-prior tuning are a later milestone.

## Online Data Flow

```text
tracking depth for target t
  -> save target packet and ORB image
  -> sidecar BoW top-k + descriptor matches
  -> cheap match-count/grid gate
  -> estimate_loop_depth(source i)
  -> sparse 3D observations
  -> fixed-scale SE(3) RANSAC and reprojection validation
  -> add support to hypothesis tracker
  -> tentative: retain only diagnostics
  -> confirmed/strong: emit one conservative sparse loop factor
  -> asynchronous pose graph
```

ORB responses and pending target packets remain bounded. An expired target or
source image produces an explicit failure record and cannot partially update a
hypothesis.

## Configuration

The online-loop configuration gains:

```yaml
orb_ratio_test: 0.80
orb_max_matches: 300
sparse_geometry:
  min_mutual_matches: 40
  min_valid_3d: 30
  min_ransac_inliers: 25
  min_ransac_ratio: 0.35
  grid_rows: 4
  grid_cols: 6
  min_grid_cells: 6
  ransac_iterations: 256
  mahalanobis_threshold: 3.5
  max_median_reprojection_px: 3.0
  max_p90_reprojection_px: 6.0
  min_geometry_ratio: 1.0e-3
  orb_pixel_variance: 2.25
hypothesis:
  min_supports: 2
  strong_supports: 3
  source_cluster_frames: 10
  target_support_frames: 10
  max_correction_translation_m: 0.25
  max_correction_rotation_deg: 10.0
sparse_factor:
  translation_sigma_m: 0.25
  rotation_sigma_deg: 10.0
```

The old long-range-flow acceptance path is disabled when sparse loop mode is
enabled. It remains available only through a separately named legacy
configuration for reproducible ablation.

## Error Handling

- Protocol parse errors disable the provider and record the malformed line
  reason.
- Missing historical images or target depths reject only the affected support.
- CUDA depth failures disable loop processing for that candidate without
  interrupting tracking.
- RANSAC never emits a transform for nonfinite, underconstrained, or degenerate
  inputs.
- A hypothesis cannot emit twice.
- A factor cannot replace an existing factor with the same endpoints and kind.
- Local tracking and termination continue when every loop component is disabled.

## Testing Strategy

### Frontend regression

- Verify bidirectional slot zero is source stereo.
- Verify `estimate_bidirectional` returns source depth.
- Verify `estimate_loop_depth` uses three source-stereo slots and returns slot
  zero.
- Verify serialized frontend call ownership and concurrency remain one.

### Sidecar and protocol

- Fake protocol-v2 provider test with multiple candidates and match payloads.
- Reject protocol-v1 handshake.
- C++ sidecar test on a deterministic textured image and its translated copy.
- Verify temporal gap, top-k order, match counts, finite coordinates, and maximum
  match truncation.

### Sparse geometry

- Recover a known SE(3) transform with Gaussian noise and at least 40 percent
  outliers.
- Produce identical output for repeated seeds.
- Reject insufficient points, collinear geometry, invalid covariance, poor
  coverage, and excessive reprojection error.
- Verify returned measurement convention is `T_source.Inv() @ T_target`.

### Hypothesis tracker

- One support remains tentative and emits nothing.
- Two compatible target keyframes confirm and emit exactly once.
- Three compatible supports transition to strong without a second emission.
- Inconsistent correction transforms form separate hypotheses or reject support.
- Stale hypotheses expire deterministically.

### Online integration

- ORB candidate with insufficient matches never calls the GPU frontend.
- Valid sparse support calls historical depth once.
- No pose factor is stored after the first support.
- The second compatible support stores one conservative sparse loop factor.
- Dense pairwise compression is not invoked for sparse loop factors.
- Diagnostics contain rejection, tentative, confirmed, and factor records.

### Regression and experiment gates

Run all existing online-loop, frontend, pose-graph, and WindowICP unit tests.
Then run:

1. the ORB sidecar integration test;
2. the synthetic sparse geometry suite;
3. the 120-frame V203 loop fixture or selected revisit segment;
4. a V203 segment containing at least one confirmed long-range hypothesis.

The milestone passes when:

- local tracking remains finite with zero unrefined windows introduced;
- no single-frame candidate emits a factor;
- every emitted loop has at least two compatible supports;
- emitted factor information eigenvalues remain below the configured
  conservative bounds;
- diagnostics can reproduce every accept/reject decision;
- when GT is available for evaluation only, confirmed measurement errors are
  materially below the legacy long-range median of 0.80 m and 15 degrees.

## Version-Control Boundaries

- This design, its implementation plan, and each implementation component use
  separate commits.
- Existing modified files unrelated to this milestone are never staged.
- Existing design documents and legacy loop configurations are preserved.
- The legacy online ORB path remains runnable under its existing configuration
  until the sparse configuration passes the experiment gates.
