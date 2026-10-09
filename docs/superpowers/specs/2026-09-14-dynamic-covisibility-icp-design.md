# Dynamic Covisibility Proximity ICP Design

## Goal

Add a true geometry-driven proximity graph experiment to MACVO. Preserve the
existing temporal skeleton (`t-1` and `t-2` factors), choose at most one
historical frame per scheduled target using current pose/depth overlap, verify
the candidate with real bidirectional FlowFormerCov matching, and add only
validated proximity factors to the termination-time global pose-only ICP.

The first experiment reuses the existing V203 `[1095, 1215)` source result and
does not rerun online odometry. After that result is recorded, a separate tool
uses ground truth only to locate another difficult 120-frame evaluation segment
that contains genuine long-baseline revisits. Ground truth is never available
to the covisibility selector or matcher.

## Terminology correction

The completed `gap5/gap10` experiment is a fixed temporal-gap ablation:

```text
target t -> predetermined sources t-5 and t-10
```

It is not a dynamic covisibility graph. The new experiment selects sources from
the complete eligible history using current geometry:

```text
target t -> source argmin_i covisibility_score(i, t)
```

The existing isolated branch already permits arbitrary forward `Edge`
endpoints, but `LongRangeICP.long_range_targets()` remains a fixed temporal
schedule. The new selector and validation pipeline are separate components and
do not redefine fixed-gap factors as proximity factors.

## Scope

This version adds:

- a reusable depth/covariance cache for scheduled keyframes;
- sparse geometry proxies for inexpensive historical candidate search;
- bidirectional projection overlap and depth-consistency scoring;
- pair-space temporal NMS and a maximum of one candidate per target;
- fixed batch-three forward/backward FlowFormerCov verification;
- spatial coverage, forward-backward consistency, and covariance-aware ICP
  residual validation;
- persistent proximity factor records with score, validation statistics, age,
  and state;
- strict same-source comparison against short-only and fixed gap-5 factors;
- a later ground-truth-only evaluation-segment mining tool.

This version does not add online proximity updates, appearance retrieval,
learned place recognition, loop-closure pose verification, joint depth
optimization, DROID update operators, or periodic online global solves.

## Approaches considered

### Offline two-pass geometry selection — selected

First cache scheduled-frame depth and sparse geometry, then choose candidates,
then run real bidirectional matching. This reuses the saved pre-global
trajectory, is deterministic, and avoids changing online odometry while the
selection and validation rules are still experimental.

### Online dynamic covisibility

Select and match proximity candidates while tracking. This is closer to the
DROID frontend lifecycle, but it couples candidate quality to changing online
poses, complicates CUDA Graph scheduling, and makes every threshold change
require another complete odometry run.

### Appearance retrieval before geometry

Use DINO, NetVLAD, or BoW to retrieve candidates and then verify them
geometrically. This is required eventually for loop closures after large drift,
but introduces a second learned model and confounds the first geometry-only
experiment.

## Input contract

The dynamic experiment consumes one complete result leaf containing:

```text
config.yaml
run_provenance.json
poses_before_global.npy
global_factors.npz
ref_poses.npy
```

For comparison with the previous temporal experiment it also consumes, when
available:

```text
long_factors_gap5_10.npz
```

The saved configuration reconstructs the original sequence range and image
preprocessing. Pose count, timestamps, source identity, factor SHA-256, and
preprocessing configuration are validated before the network is loaded.

## Scheduled keyframes

Depth and proximity targets use a stride of five frames:

```text
0, 5, 10, 15, ...
```

For target `t`, historical candidates satisfy:

```text
i % 5 == 0
t - i >= 15
i < t
```

The 15-frame minimum distinguishes proximity edges from the previously tested
gap-5 and gap-10 temporal edges. No edge is added when no historical candidate
passes the geometry gates.

## Pass A: depth and geometry cache

### Fixed batch-three depth inference

Each scheduled keyframe uses the existing fixed batch-three frontend. Slot zero
computes stereo depth. Both temporal slots use deterministic filler pairs and
their outputs are discarded. This produces one CUDAGraph capture and one
inference call per scheduled keyframe.

For the current 120-frame segment there are 24 scheduled keyframes and 24 Pass A
calls. Complete V203 would contain approximately 373 scheduled keyframes, but a
complete-sequence experiment is outside this version's validation scope.

### Cache representation

The cache is published as a temporary directory followed by an atomic rename:

```text
covisibility_cache/
  metadata.json
  frame_ids.npy
  time_ns.npy
  depth.npy
  depth_cov.npy
  proxy_uv.npy
  proxy_depth.npy
  proxy_depth_cov.npy
```

`depth.npy` and `depth_cov.npy` are float32 memory-mapped arrays with shape
`[K, H, W]`. Full-resolution values are retained because a selected historical
source must later construct the same covariance-aware 3D observations as the
online pipeline.

Each keyframe also stores up to 1024 deterministic grid-distributed proxy
samples. Invalid or non-positive depth samples are represented by NaN and a
valid-count field in `metadata.json`. Sparse proxies are used for candidate
scoring so historical search does not scan full images.

For 120 V203 frames, full depth plus depth covariance is approximately 69 MiB.
For complete V203 it would be approximately 1.0 GiB. Cache files remain outside
Git and can be deleted and regenerated from the source result.

## Covisibility candidate scoring

### Relative geometry

The saved pose convention is sensor-to-world. Source proxy points are projected
into the target using:

```text
T_target_source = T_target.Inv() @ T_source
```

The reverse direction uses the inverse relationship. Projection and depth
coordinates use the repository's existing NED camera convention and camera
intrinsics; no OpenCV optical-axis convention is introduced into the scoring
path.

### Measurements

For each candidate `(i, t)`, compute:

- `overlap_forward`: fraction of valid source proxies that project in front of
  and inside the target image;
- `overlap_backward`: equivalent target-to-source fraction;
- `mean_overlap`: mean of the two overlap fractions;
- `depth_consistency`: fraction of in-bounds projections whose projected depth
  agrees with sampled target depth within three combined depth standard
  deviations;
- `median_motion_px`: median source-to-target projected pixel displacement.

Depth consistency is a cheap candidate-ranking approximation. It combines the
stored scalar endpoint depth variances and does not replace the later full 3D
covariance validation.

### Gates and score

Initial configurable gates are:

```yaml
keyframe_stride: 5
min_temporal_gap: 15
proxy_points: 1024
min_directional_overlap: 0.15
min_mean_overlap: 0.25
min_depth_consistency: 0.30
min_median_motion_px: 8.0
max_median_motion_px: 240.0
candidate_nms_frames: 5
max_candidates_per_target: 1
```

Candidates passing every gate are ranked by:

```text
score = (1 - mean_overlap)
      + 0.5 * (1 - depth_consistency)
      + 0.1 * median_motion_px / image_diagonal
```

Lower is better. All candidate terms and gate failures are serialized; a
threshold change can rerun selection without repeating Pass A.

### Pair-space NMS

Process targets in chronological order. A previously accepted proximity pair
`(i, t)` suppresses a new pair `(j, u)` only when both endpoints are within
`candidate_nms_frames`:

```text
abs(i - j) <= nms and abs(t - u) <= nms
```

This mirrors the purpose of DROID's factor-pair NMS: avoid adding many nearly
identical constraints from neighboring source and target frames. A candidate
that later fails real matching does not suppress future pairs. NMS never
removes the short temporal skeleton.

## Pass B: real bidirectional matching

For every selected target/source pair, run one fixed batch-three inference:

```text
slot 0: target left -> target right   (target depth refresh)
slot 1: source left -> target left    (forward match)
slot 2: target left -> source left    (backward match)
```

The refreshed target depth is used for factor construction. The source depth
comes from Pass A. The third slot is not a second candidate; it verifies the
single selected candidate.

At most 23 Pass B calls occur on the original 120-frame segment. Pass A plus
Pass B therefore uses at most 47 offline batch-three calls and no online
odometry rerun.

## Match validation and factor construction

### Forward-backward consistency

Use the forward match to produce target correspondences. Sample backward flow
at those target positions and retain observations satisfying:

```text
norm(flow_forward + sampled_flow_backward) <= 2.0 pixels
```

At least 30 observations and 50% of inbound forward correspondences must pass.

### Depth and spatial coverage

After forward-backward filtering:

- at least 50% of observations must have finite positive source and target
  depth;
- accepted observations must occupy at least six cells of a `4 x 6` image grid.

This rejects concentrated matches on one small object or texture patch.

### Full covariance ICP validation

Construct the candidate `Edge` with the existing depth covariance,
`MatchCovariance`, full `3 x 3` observation covariance, camera convention, and
deterministic selector. Before adding it to the graph, evaluate each observation
under the saved pre-global poses:

```text
r_k = T_i p_i,k - T_t p_t,k
S_k = R_i Cov_i,k R_i^T + R_t Cov_t,k R_t^T
d_k = sqrt(r_k^T S_k^-1 r_k)
```

An observation is an inlier when `d_k <= 3.5`. The edge is accepted only when
at least 30 observations and 50% of its valid observations are inliers. The
stored edge contains only these validated inliers.

Initial configurable match gates are:

```yaml
max_forward_backward_error_px: 2.0
min_forward_backward_inliers: 30
min_forward_backward_ratio: 0.50
min_depth_valid_ratio: 0.50
grid_rows: 4
grid_cols: 6
min_occupied_grid_cells: 6
mahalanobis_threshold: 3.5
min_mahalanobis_inliers: 30
min_mahalanobis_inlier_ratio: 0.50
```

Only the forward `i -> t` factor is stored. The backward inference is validation
evidence, not a second highly correlated graph constraint.

## Component boundaries

### `CovisibilityDepthCache`

Owns Pass A scheduling, atomic cache publication, cache validation, memory-map
loading, and sparse proxy extraction. It depends on the sequence and frontend,
but not on global optimization.

### `CovisibilitySelector`

Consumes cached proxies, poses, and intrinsics and returns zero or one ranked
candidate for each target. It contains no neural-network calls and can be
rerun cheaply for threshold experiments.

### `BidirectionalMatchValidator`

Consumes one candidate plus forward/backward frontend outputs and produces
validated correspondences and diagnostic statistics. It does not own candidate
selection or global solving.

### `PersistentFactorStore`

Stores arbitrary-span accepted factors and aligned records:

```text
edge
kind = proximity
score
confidence
age
state
candidate_metrics
validation_metrics
```

The first offline version sets `age=0` and `state=accepted`. The fields establish
an interface for later online active/inactive lifecycle work without adding
online removal behavior now.

Recorded confidence is diagnostic only:

```text
confidence = 0.25 * (
    mean_overlap
  + depth_consistency
  + forward_backward_inlier_ratio
  + mahalanobis_inlier_ratio)
```

It is clipped to `[0, 1]` and is not applied to the solver Hessian or covariance.

### Global solver integration

The existing `GlobalPoseICP` consumes plain `Edge` objects. It receives:

```text
short archive edges + accepted proximity edges
```

The robust objective, Huber threshold, covariance model, and pose-only sparse
solver remain unchanged. Proximity confidence is recorded but is not used as an
extra numerical weight in this version.

## Serialization and outputs

Extend the known factor kind set with `proximity`. Continue using factor archive
schema version 1 because numeric solver inputs are unchanged. Aligned per-edge
records are stored in the archive's JSON metadata and validated by the
proximity-store loader.

Outputs are:

```text
covisibility_cache/
covisibility_candidates.json
proximity_factors.npz
proximity_generation.json
poses_global_proximity.npy
global_proximity_diagnostics.json
global_proximity_metrics.json
dynamic_covisibility_metrics.csv
dynamic_covisibility_metrics.json
dynamic_covisibility_gate.json
```

`covisibility_candidates.json` contains all evaluated candidates, not only the
winner, so failed gates and score distributions remain available for offline
analysis. Large cache and factor files remain in the result directory and are
not committed.

## Failure handling

- Missing or mismatched source artifacts: stop before loading the network.
- Incomplete cache directory: reject it and regenerate from a new temporary
  directory; never mix old and new arrays.
- Non-finite pose, depth, covariance, projection, or score: reject the affected
  candidate with an explicit reason.
- No geometry candidate for a target: record `no_candidate` and continue.
- Frontend failure for one candidate: record `inference_failure` and continue.
- Failed forward-backward, depth, grid, or Mahalanobis gate: store diagnostics
  but no factor.
- No accepted proximity factors: do not run a misleading proximity solve;
  report the graph as `no_proximity_edges`.
- Disconnected or failed global solve: preserve all source outputs and omit the
  tagged proximity pose file.
- Ground truth unavailable: preserve candidate, factor, and pose artifacts and
  record metrics as unavailable.

No command overwrites `poses.npy`, `poses_before_global.npy`,
`global_factors.npz`, or `long_factors_gap5_10.npz`.

## Stage 1: original difficult V203 segment

Reuse the existing result:

```text
V203 [1095, 1215), 120 frames, seed 0
```

The source factor archive and pre-global trajectory remain identical across all
variants. Compare:

1. `before_global`;
2. `short`: adjacent plus `t-2` factors;
3. `gap5`: short plus exactly the accepted fixed gap-5 factors;
4. `proximity`: short plus dynamically selected proximity factors.

A proximity-only solve is intentionally excluded because at most one historical
edge per target does not guarantee a connected pose graph. Every valid global
solve retains the short temporal skeleton.

### Stage 1 gates

Basic dynamic-covisibility success requires:

- finite output;
- exact fixed anchor;
- lower robust objective within the proximity graph;
- proximity ATE RMSE strictly below short-only ATE RMSE using default
  `huber_delta=3.0`.

Selector superiority additionally requires:

- accepted proximity edge count no greater than the 23 accepted gap-5 edges;
- proximity ATE RMSE no greater than fixed gap-5 ATE RMSE.

RTE, ROE, RPE, edge count, observation count, candidate count, rejection
reasons, cache time, inference time, and solver time are reported but are not
hard gates.

If the original segment contains no accepted proximity factors, that is a valid
result rather than a reason to lower thresholds automatically.

## Stage 2: find another difficult revisit segment

After Stage 1 is recorded, scan EuRoC ground truth only to choose an evaluation
segment. Candidate 120-frame windows must contain at least three frame pairs
with:

```text
temporal gap >= 30 frames
translation distance <= 2.0 metres
relative viewing-angle difference <= 30 degrees
```

Viewing-angle difference is the SO(3) geodesic angle between the two ground
truth camera orientations.

Among qualifying windows, select the one with the highest combined rolling RTE
and ROE from the existing baseline trajectory. Ground truth chooses the test
window only; the dynamic selector still receives saved estimated poses and
depths exclusively.

Run one online source experiment for that selected window, then repeat the same
four offline variants and gates. Complete V203 and appearance-retrieval loop
closure remain later decisions.

## Testing strategy

All behavior changes use test-first development.

### Depth cache tests

- schedule contains exactly every fifth frame;
- fixed batch-three calls use deterministic filler slots;
- arrays preserve depth/covariance values, IDs, timestamps, shape, and dtype;
- proxy sampling is deterministic and grid-distributed;
- interrupted publication leaves no accepted cache directory;
- corrupt metadata, partial arrays, and source identity mismatch are rejected.

### Projection and selector tests

- identity poses produce maximum bidirectional overlap and zero motion;
- translated/rotated synthetic cameras produce expected projected pixels;
- behind-camera and out-of-bounds points are excluded;
- overlap, depth consistency, motion, and score match hand-computed fixtures;
- every geometry gate records the correct rejection reason;
- pair-space NMS suppresses nearby factor pairs but not distant pairs;
- no more than one candidate is returned per target;
- ground-truth data is absent from the selector API.

### Bidirectional validation tests

- inverse synthetic flows pass exact forward-backward consistency;
- inconsistent backward flow is rejected;
- grid coverage rejects concentrated observations;
- invalid depth ratios are counted correctly;
- full covariance Mahalanobis filtering retains known inliers and rejects
  synthetic outliers;
- only one forward factor is emitted.

### Persistent store and archive tests

- arbitrary-span proximity edges round-trip through schema version 1;
- aligned edge records retain score, confidence, age, state, and metrics;
- record/edge count mismatch and unknown state are rejected;
- proximity archive merges with short factors without duplicate keys;
- existing adjacent, skip2, gap5, and gap10 archives remain loadable.

### End-to-end offline tests

- a synthetic revisit selects a non-temporal-neighbor source;
- a time-near but non-overlapping frame is rejected;
- a time-distant overlapping frame becomes a validated factor;
- source pose and factor files remain unchanged;
- comparison gates distinguish basic success from selector superiority;
- segment mining uses ground truth only in the benchmark-selection module.

## Expected interpretation

If proximity beats both short-only and the equal-or-larger fixed gap-5 edge
budget, geometry-driven candidate selection is supported. If it improves
short-only but not gap-5, the new information is useful but the selector is not
yet superior to a simple temporal heuristic. If no candidates pass, the
original difficult segment lacks usable revisit structure under current
estimates or the geometry gates are too conservative; Stage 2 then tests a
known-revisit segment without changing Stage 1 thresholds.
