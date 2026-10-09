# Offline Long-Range ICP Factor Generation Design

## Goal

Make global pose refinement reproducible and useful for drift correction by:

1. saving the exact pre-global trajectory and all short-range ICP factors from
   one online odometry run;
2. generating a small number of real long-range factors offline, initially
   direct `t-5 -> t` and `t-10 -> t` matches;
3. evaluating whether those added observations reduce ATE before considering a
   more complex DROID-style pose-and-depth Schur backend.

The first validation target remains the difficult EuRoC V203 interval
`[1095, 1215)`. A complete V203 run is performed only after the short interval
shows an ATE improvement under a controlled comparison.

## Scope

This version adds:

- durable pre-global trajectory and short-range factor artifacts;
- an offline global-refinement command that consumes saved factors;
- an offline long-range factor generator using the existing frontend,
  selector, depth covariance, and match covariance models;
- deterministic `t-5` and `t-10` edge scheduling;
- comparison and metrics artifacts for short-only, `+t-5`, and
  `+t-5+t-10` graphs.

This version does not add image retrieval, loop-closure detection, learned
place recognition, joint depth optimization, online long-range inference, or a
marginalization prior. Those remain later experiments after direct long-range
edges establish whether additional graph observability improves ATE.

## Architecture

The workflow is split into three independently testable stages:

```text
online WindowMACVO run
  -> poses_before_global.npy
  -> global_factors.npz
  -> global_refinement.json

offline long-range generator
  + original sequence/config/checkpoint
  -> long_factors_gap5_10.npz

offline GlobalPoseICP refinement
  + pre-global poses
  + short factors
  + selected long factors
  -> refined pose file
  -> diagnostics JSON
  -> evaluation metrics JSON
```

The factor archive is the interface between inference and optimization. The
global solver does not know whether a factor was generated online or offline;
all factor sources use the same serialized representation.

## Online reproducibility artifacts

When `global_refine=true`, `WindowMACVO` retains a termination snapshot until
`save_window_diagnostics(folder)` has written all artifacts. The snapshot is
created after normal interpolation and before global pose refinement.

### `poses_before_global.npy`

This file uses the same public eight-column representation as `poses.npy`:

```text
timestamp_ns tx ty tz qx qy qz qw
```

Stored sensor poses are converted to public body poses with the graph's
extrinsic transform:

```text
T_WB = T_BS @ T_WS @ T_BS.Inv()
```

The timestamps and row order exactly match the standard output trajectory.
This makes before/after ATE comparisons possible within the same run and avoids
comparisons across different commits, random seeds, or frontend executions.

### `global_factors.npz`

The uncompressed NumPy archive uses integer schema version `1` and stores exact
CPU float64 factor data:

```text
schema_version
edge_a[E]
edge_b[E]
edge_offsets[E + 1]
points_a[N, 3]
points_b[N, 3]
cov_a[N, 3, 3]
cov_b[N, 3, 3]
initial_sensor_poses[F, 7]
time_ns[F]
T_BS[7]
edge_kind[E]
metadata_json
```

`edge_offsets[i]:edge_offsets[i+1]` selects observations for edge `i`.
`edge_kind` identifies `adjacent`, `skip2`, `gap5`, or `gap10` without changing
the solver input contract. Online archives contain only `adjacent` and `skip2`
factors.

Pose arrays use `[tx, ty, tz, qx, qy, qz, qw]`. `metadata_json` is one UTF-8
JSON scalar for provenance and generation settings; numeric solver inputs stay
in dedicated typed arrays rather than being encoded in JSON.

The archive is written to a temporary file in the result directory and then
atomically renamed. Serialization failure is recorded in diagnostics and does
not delete the in-memory snapshot before the failure has been reported.

### `global_refinement.json`

This standalone file copies the global solve diagnostics currently embedded in
`window_diagnostics.json` and adds artifact metadata:

- schema version;
- factor archive filename and byte size;
- pre-global pose filename;
- source configuration and commit when available;
- pose, edge, and observation counts;
- initial/final objective and solver timing.

The existing `window_diagnostics.json` remains backward compatible.

## Shared factor archive API

A small serialization module owns archive construction, validation, loading,
and merging. It exposes typed operations conceptually equivalent to:

```text
save_factor_archive(path, poses, factors, metadata)
load_factor_archive(path) -> FactorArchive
merge_factor_archives(short, long) -> FactorArchive
```

Validation requires:

- supported `schema_version`;
- monotonically nondecreasing offsets starting at zero and ending at `N`;
- unique directed `(a, b)` edge keys within one archive;
- endpoints within the pose range and `a < b`;
- finite points, poses, and covariance entries;
- symmetric positive-definite covariance matrices;
- matching array sizes and known edge kinds.

Merging rejects duplicate keys by default. This prevents a long-range archive
from silently replacing short-range measurements.

## Offline long-range factor generation

The generator reads the same sequence configuration and model checkpoint as the
source run. It does not change the online odometry path.

### Deterministic schedule

For the initial experiment, targets occur every five frames:

```text
t = 5, 10, 15, ...
```

At the first sequence frame `t=0`, one initialization batch computes stereo
depth for frame zero. Both temporal slots use the identity pair `0 -> 0` and
their outputs are discarded. This preserves the fixed batch-three CUDAGraph
contract and initializes the source-depth cache.

At `t=5`, one batch computes current stereo depth and the `t-5 -> t` temporal
match. The unused third batch slot duplicates `t-5 -> t` and is discarded.

At `t>=10`, one fixed batch-three frontend inference computes:

```text
current left -> current right : depth for t
t-5 -> t                    : gap-5 flow/covariance
t-10 -> t                   : gap-10 flow/covariance
```

Every scheduled target becomes a later scheduled source, so its depth remains
in the cache until all dependent edges have been constructed. Complete V203
therefore requires one initialization call plus one batch-three inference per
five target frames, rather than one inference per online frame.

### Match-to-factor construction

Long-range edges use the existing measurement pipeline:

- `CovAwareSelector_NoDepth` for point selection;
- the existing depth estimate and depth covariance for both endpoints;
- `MatchCovariance` for flow-conditioned correspondence covariance;
- full `3 x 3` covariance matrices;
- maximum 200 selected observations per edge;
- minimum 10 valid observations for accepting an edge.

The current skip-edge construction is factored into a reusable match-to-edge
function so online `t-2` and offline `t-5/t-10` edges share identical geometry,
validity filters, coordinate conventions, and covariance handling.

Rejected edges are counted by reason, including too few valid observations,
invalid depth, invalid covariance, or out-of-range correspondence. Rejection of
one edge does not stop generation of later targets.

### Long-range archive

The generator writes `long_factors_gap5_10.npz` with only accepted `gap5` and
`gap10` factors plus metadata containing:

- source result directory and short-factor archive identity;
- sequence name and frame range;
- frontend and covariance configuration identity;
- schedule and thresholds;
- attempted, accepted, and rejected counts per gap;
- total observations and elapsed inference time.

The long archive uses the same pose indexing and timestamps as the short-factor
archive. A mismatch is a hard error.

## Offline global refinement command

`Scripts/Experiment/RefineGlobalPoseICP.py` loads a source result directory and
runs the existing sparse `GlobalPoseICP` solver without rerunning odometry.
Conceptual usage is:

```bash
python Scripts/Experiment/RefineGlobalPoseICP.py \
  --space <result-leaf> \
  --long-factors <optional-long-archive> \
  --iterations 5 \
  --huber 3.0 \
  --output-tag gap5_10
```

With no long archive, it reproduces the short-only global solve from
`poses_before_global.npy` and `global_factors.npz`. With a long archive, it
validates and merges the graphs before solving.

The command never overwrites `poses.npy`, `poses_before_global.npy`, or the
input factor files. It writes tagged outputs such as:

```text
poses_global_short.npy
poses_global_gap5.npy
poses_global_gap5_10.npy
global_short_diagnostics.json
global_gap5_diagnostics.json
global_gap5_10_diagnostics.json
global_gap5_10_metrics.json
```

The public body-pose conversion is identical to the online writer. Solver input
remains sensor poses, preserving the current ICP coordinate convention.

## Evaluation protocol

### Stage 1: difficult V203 short sequence

Run one online artifact-producing experiment on V203 `[1095, 1215)` with the
same seed, configuration, and checkpoint used by all comparisons. From that
single artifact set, evaluate:

1. pre-global trajectory;
2. short-only graph (`t-1` and `t-2`);
3. short graph plus `t-5` factors;
4. short graph plus both `t-5` and `t-10` factors.

All variants start from the exact same `poses_before_global.npy`; only the
factor set changes. Record ATE, RTE, ROE, RPE, objective, accepted/rejected
steps, edge counts, observation counts, solver time, and long-edge generation
time.

The experiment succeeds when `+t-5+t-10` has finite output, preserves the
anchor, lowers the robust objective, and produces lower ATE RMSE than the
short-only graph. Relative metrics are reported but are not required to improve
for this gate.

### Stage 2: complete V203

Only after Stage 1 passes, run one complete V203 online experiment to produce
the durable artifacts. Generate full-sequence long factors offline, then
evaluate short-only, `+t-5`, and `+t-5+t-10` from the same pre-global trajectory.

Results are recorded in a validation CSV and a concise Markdown report. No
claim that long-range edges improve ATE is made unless the complete-sequence
short-only and long-range variants use identical saved inputs.

## Failure handling

- Missing online artifacts: stop before inference and report the exact file.
- Archive schema or shape mismatch: reject the archive without partial merge.
- Sequence timestamps or pose counts differ: reject the long archive.
- Frontend inference failure for one scheduled target: record it and continue
  only when the failure is local and subsequent inputs remain valid.
- No accepted long edges: write diagnostics, do not invoke a misleading
  long-range solve, and return a non-success experiment status.
- Sparse solver failure or non-finite result: preserve all inputs and omit the
  tagged refined pose file.
- Metrics unavailable because ground truth is missing: preserve pose and solve
  outputs and explicitly record metrics status as unavailable.

## Testing strategy

All implementation changes use test-first development.

### Serialization tests

- Factor archives round-trip points, covariances, poses, timestamps, edge kinds,
  and metadata exactly.
- Ragged edge offsets reconstruct every edge without reordering observations.
- Temporary-file publication leaves no partial destination after failure.
- Unsupported schema, duplicate keys, invalid endpoints, malformed offsets,
  non-finite values, and invalid covariance are rejected.
- Merging short and long archives preserves order and rejects duplicates.

### Online artifact tests

- `poses_before_global.npy` is captured before any global writeback.
- Its timestamps and body-pose convention match `poses.npy`.
- Saving diagnostics emits all three artifacts and retains factors until the
  write completes.
- Disabled global refinement does not retain or write factor archives.
- A serialization failure is visible in diagnostics and does not corrupt
  standard trajectory output.

### Long-range generation tests

- The scheduler produces exactly the expected `gap5` and `gap10` pairs.
- Each target uses at most one batch-three frontend call after required source
  depths are cached.
- Duplicate filler output at `t=5` is ignored.
- Reusable match-to-edge construction matches existing online `t-2` edge output
  for identical synthetic inputs.
- Acceptance thresholds and rejection reasons are deterministic.
- The generated archive matches the source pose indexing and timestamps.

### Offline refinement tests

- A short-only offline solve reproduces the online global result within numeric
  tolerance.
- Adding a synthetic valid long edge changes the merged graph and improves a
  controlled drifting trajectory.
- Input files are never overwritten.
- Anchor preservation, finite output, objective decrease, and tagged output
  naming are verified.

### Real-data validation

- Run V203 `[1095, 1215)` once to create source artifacts.
- Generate `gap5` and `gap10` factors offline.
- Compare all four trajectories from the same initial poses.
- Proceed to complete V203 only if the short-sequence ATE gate passes.

## Expected outcome and next decision

This experiment tests the missing-observability hypothesis directly. If true
long-range measurements reduce ATE, the next step is to improve candidate
selection with co-visibility or loop closure and then consider joint
pose-and-depth Schur optimization. If ATE does not improve, diagnostics from the
saved archives allow covariance weighting, robust loss, edge density, and
outlier filtering to be adjusted offline before paying the cost of another
full network run.
