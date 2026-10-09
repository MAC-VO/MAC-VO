# Global Pose-Only ICP v0.3 Design

## Goal

Extend `WindowMACVO` with an opt-in experimental backend that retains every
valid adjacent and two-step ICP factor produced online, then performs one
global pose-only refinement when the sequence terminates. Existing v0.2
configuration and online behavior remain unchanged unless the new global
refinement option is enabled.

The first validation target is the difficult 120-frame EuRoC V203 segment
`[1095, 1215)`. This range was selected from the completed V203 trajectory
because it maximizes the combined rolling relative translation and rotation
error among 120-frame windows: approximately `0.0192 m/frame` RTE RMSE and
`0.407 deg/frame` ROE RMSE.

## Scope

This version adds:

- active and inactive ICP factor lifecycle management;
- retention of existing adjacent and direct `t-2 -> t` factors;
- a block-sparse CPU float64 global pose-only solver;
- end-of-sequence pose and map-point writeback;
- configuration, diagnostics, tests, and a controlled V203 comparison.

This version does not add loop closure, new image matching, joint depth
optimization, landmark bundle adjustment, or a marginalization prior.

## Configuration and compatibility

The existing `MACVO_Fast_WindowICP.yaml` remains unchanged. A new configuration
`Config/Experiment/MACVO/MACVO_Fast_WindowICP_Global.yaml` enables the backend:

```yaml
Odometry:
  type: WindowMACVO
  name: MACVO-Fast-WindowICP5-Global
  args:
    window_size: 5
    skip_matching: true
    window_iterations: 10
    window_huber_delta: 3.0
    global_refine: true
    global_iterations: 5
    global_huber_delta: 3.0
```

`WindowMACVO.__init__` provides defaults `global_refine=false`,
`global_iterations=5`, and `global_huber_delta=3.0`. This ensures old saved
configurations and v0.2 callers continue to execute without retaining inactive
factors or invoking the global solver.

Configuration validation requires:

- `global_refine` is boolean;
- `global_iterations` is a positive integer;
- `global_huber_delta` is finite and positive.

## Factor lifecycle

The online solver continues to use an active five-frame `EdgeWindow`. Its
`advance(current)` operation is changed to return the list of edges that leave
the active window while preserving the existing active-edge result.

When `global_refine=false`, the returned edges are ignored and memory behavior
matches v0.2.

When `global_refine=true`, `WindowMACVO` moves each returned edge into an
inactive dictionary keyed by `(edge.a, edge.b)`. The edge object itself is
reused; point and covariance tensors are not cloned again. At termination the
global factor set is:

```text
all_edges = inactive_edges + current_active_edges
```

Duplicate keys are rejected or replaced deterministically so that exactly one
factor exists for each directed frame pair.

The retained payload stays in CPU float64 because `Edge` already validates and
normalizes observations into that representation. Based on existing runs, the
raw tensor payload is approximately 132 MiB for complete V203 and 261 MiB for
complete MH01. Python and allocator overhead will make actual resident memory
higher. v0.3 records an estimated tensor byte count in diagnostics.

Inactive raw observations are not serialized in v0.3. They exist only for the
current process and are released after global refinement. Re-running global
optimization from a saved v0.3 result therefore still requires rerunning the
odometry sequence.

## Global objective

The global backend minimizes the same pose-only full-covariance ICP objective
as the local window solver. For each retained edge `(a, b)` and observation
`k`:

```text
r_k = T_a p_a,k - T_b p_b,k
S_k = R_a Cov_a,k R_a^T + R_b Cov_b,k R_b^T
```

Residuals and Jacobians are whitened with the Cholesky factor of `S_k`. The
same vector Huber loss is applied to the whitened residual norm using
`global_huber_delta`.

All poses are transformed into the first pose's local coordinate system before
optimization. Pose zero is excluded from the variable vector and must be copied
back bit-for-bit unchanged. Other poses use the existing left-multiplicative
SE(3) retraction and quaternion normalization.

## Block-sparse linearization

The existing `factor_system()` creates a dense Jacobian and is suitable only
for at most five poses. It must not be called over the full trajectory.

For each edge, the global solver computes only its local blocks:

```text
J_a = [ I, -skew(T_a p_a) ]
J_b = [-I,  skew(T_b p_b) ]
```

After whitening and Huber weighting, it accumulates:

```text
H_aa += A_a^T A_a
H_ab += A_a^T A_b
H_ba += A_b^T A_a
H_bb += A_b^T A_b
g_a  += A_a^T e
g_b  += A_b^T e
```

Each pose contributes a 6x6 diagonal block and each edge contributes at most
two off-diagonal blocks. Because retained factors connect only adjacent or
two-step frames, the Hessian is block-banded, but the implementation uses a
general sparse representation so later edge types can be added without
changing the solver contract.

The accumulated blocks are expanded into a SciPy COO matrix, duplicate scalar
entries are summed, and the matrix is converted to CSC for
`scipy.sparse.linalg.spsolve`. SciPy 1.13.1 is already an explicit project
dependency.

Levenberg-Marquardt damping is added using the clamped Hessian diagonal. Trial
steps recompute the complete robust objective. A step is accepted only when the
objective is finite and lower; otherwise damping is increased and the trial is
retried. The solver stops on a small gradient, a small step, a small objective
decrease, a rejected outer iteration, or `global_iterations`.

The implementation must never allocate a dense matrix of shape
`6 * (num_poses - 1)` squared.

## Termination and map consistency

For `global_refine=false`, termination follows the current v0.2 implementation
exactly.

For `global_refine=true`, termination performs:

1. Stop and clear the synchronous two-frame initializer.
2. Run existing motion interpolation when at least three poses exist.
3. Reanchor owned map points for any interpolation changes.
4. Gather inactive and active factors and validate global connectivity.
5. Run block-sparse global pose-only ICP.
6. Reanchor all owned map points and covariances from pre-global to post-global
   poses.
7. Write refined poses into the visual map.
8. Clear frame, active-edge, and inactive-edge caches.

The global solver does not modify point coordinates stored inside retained
`Edge` observations. They remain camera-frame measurements. Only global map
points owned by source frames are reanchored using the existing
`reanchor_points()` operation.

If global refinement cannot run, the interpolated v0.2 trajectory is retained.
Failure must not leave a partially updated pose or point map.

## Connectivity and failure handling

The global graph must contain every pose reachable from pose zero. Connectivity
is checked before linearization.

- Fewer than two poses or no factors: record `skipped`.
- Disconnected factor graph: record `not_refined` with component information.
- Invalid edge endpoint, non-finite pose, non-unit quaternion, non-positive
  definite covariance, non-finite sparse system, or sparse solver failure:
  record `not_refined` and preserve the input trajectory.
- Non-finite or non-decreasing trial objective: reject the trial and increase
  damping.
- Global writeback occurs only after a complete finite result is returned.

## Diagnostics

`window_diagnostics.json` retains its current top-level fields and `windows`
array. It gains:

```json
{
  "global_refine": true,
  "inactive_edges": 0,
  "retained_tensor_bytes": 0,
  "global_refinement": {
    "status": "refined",
    "poses": 120,
    "edges": 237,
    "observations": 0,
    "initial_cost": 0.0,
    "final_cost": 0.0,
    "iterations": 0,
    "accepted_steps": 0,
    "rejected_steps": 0,
    "seconds": 0.0,
    "anchor_preserved": true
  }
}
```

Counts and costs contain actual values. When disabled, the file records
`global_refine=false`, zero inactive edges, and a global status of `disabled`.

Global solve time is also recorded through the existing timing mechanism under
`GlobalPoseICP`; it is termination-only time and is not included in online
per-frame `Odom_Runtime`.

## Testing strategy

All behavior changes use test-first development.

### Factor lifecycle tests

- `EdgeWindow.advance()` returns exactly the evicted edges.
- Active edges remain identical to v0.2 after advance.
- Re-adding an existing key replaces deterministically without duplication.
- Disabled global refinement retains no inactive edges.
- Enabled refinement stores each evicted edge exactly once.

### Sparse solver tests

- A synthetic 5-pose problem matches the current dense window solver within
  tolerance.
- A synthetic graph larger than five poses recovers all poses while preserving
  the exact first pose.
- The sparse matrix dimension and nonzero pattern are checked without allowing
  a dense global Jacobian.
- Disconnected, invalid-endpoint, non-finite, and singular cases preserve the
  input trajectory and return explicit diagnostics.
- Initial cost is finite and final cost is lower for a successful solve.

### Map and termination tests

- Global writeback reanchors owned points and rotates covariances exactly once.
- Failed global refinement leaves poses and map points unchanged after the
  normal interpolation stage.
- v0.2 configuration produces identical active-window and termination behavior.
- v0.3 configuration is loadable and records complete diagnostics.

### Real-data validation

Run v0.2 and v0.3 sequentially on V203 `[1095, 1215)`, seed 0, with the same
code commit and separate result directories. Verify equal frame timestamps and
identical online configuration apart from the global options.

Report:

- ATE, RTE, ROE, and RPE RMSE before and after global refinement;
- initial and final global objective;
- active, inactive, and total factors;
- retained tensor bytes and process memory;
- online mean `Odom_Runtime` and termination-only global solve time;
- unrefined local and global status counts.

## Acceptance criteria

- Existing v0.2 tests and configuration remain green.
- The V203 test has 120 finite output poses with identical timestamps.
- All 119 local windows retain their previous success status.
- Exactly one copy of every valid adjacent and skip edge is present globally.
- A successful global solve preserves pose zero exactly and decreases the
  robust objective.
- Quaternion norm error remains below `1e-6`.
- Online mean `Odom_Runtime` regression is below 5% relative to v0.2.
- No primary metric may regress by more than 2% if v0.3 is proposed as a
  preferred configuration. If this threshold is exceeded, the implementation
  remains available as an experimental configuration and the result is
  documented without replacing v0.2.

## Expected limitations

The retained graph contains only adjacent and two-step constraints already used
online. Global refinement can distribute accumulated inconsistency across the
trajectory, but it introduces no new information and cannot guarantee reduced
absolute drift. Large ATE improvement generally requires long-range factors or
loop closure, which are explicitly deferred.

The SciPy sparse solver is an intentionally low-risk v0.3 implementation. If
full-sequence termination time later becomes excessive, block-banded or CUDA
assembly and solve can be evaluated without changing the factor lifecycle or
global optimizer interface.
