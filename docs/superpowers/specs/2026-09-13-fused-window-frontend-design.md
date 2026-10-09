# Fused Batch-3 Window Frontend Design

## Goal

Reduce `WindowMACVO` latency without changing its geometric constraints,
keypoint selection, covariance model, or window optimizer. Replace the two
batch-2 FlowFormerCov calls made for each new keyframe with one fixed batch-3
CUDA Graph replay that produces the current stereo depth, the adjacent temporal
match, and the two-step temporal match.

## Current bottleneck

For a current frame `t`, the implementation performs:

1. `estimate_pair(t-1, t)`, whose batch contains `(t_left, t_right)` for stereo
   depth and `(t-1_left, t_left)` for adjacent flow.
2. `estimate_pair(t-2, t)`, whose batch repeats `(t_left, t_right)` and computes
   `(t-2_left, t_left)` for the skip edge. The repeated stereo output is
   discarded.

On the complete MH01 run, the frontend is called 7,229 times for 3,616 frames.
Each call averages about 488 ms, while the window solve averages about 62 ms.
The redundant second frontend call is therefore the primary performance cost.

## Selected approach

Use one fixed batch-3 inference per temporal step:

| Batch slot | Input A | Input B | Consumed output |
|---|---|---|---|
| 0 | `t_left` | `t_right` | current depth and depth covariance |
| 1 | `t-1_left` | `t_left` | adjacent flow and flow covariance |
| 2 | `t-2_left` | `t_left` | skip flow and flow covariance |

This preserves the three pairwise network inputs used by the existing
algorithm. It removes only the duplicate current-frame stereo pair and one CUDA
Graph replay.

Two alternatives were rejected for this version:

- A separate batch-1 skip graph is a smaller refactor, but still requires two
  launches and retains a second CUDA Graph memory pool.
- Composing adjacent flows avoids direct skip inference, but changes the skip
  observation and covariance semantics and therefore is not numerically
  equivalent to the current algorithm.

## Frontend interface

Add a frontend operation with the following semantics:

```python
estimate_window(
    frame_t2: StereoData | None,
    frame_t1: StereoData,
    frame_t: StereoData,
) -> tuple[IStereoDepth.Output, IMatcher.Output, IMatcher.Output | None]
```

The outputs are current depth, `t-1 -> t` matching, and optional `t-2 -> t`
matching.

`IFrontend` supplies a compatibility implementation using existing public
operations. It calls `estimate_pair(frame_t1, frame_t)` for current depth and
adjacent matching, then calls `estimate_pair(frame_t2, frame_t)` only when
`frame_t2` is present. This fallback preserves correctness for frontend types
that do not implement fused execution.

`CUDAGraph_FlowFormerCovFrontend` overrides the operation and performs the
single batch-3 execution. The third slot is filled with the adjacent temporal
pair when `frame_t2` is absent, and the corresponding output is returned as
`None`. This makes the first temporal step use the same batch shape as all later
steps, so the class captures exactly one fixed batch-3 CUDA Graph.

The existing `estimate_pair` API remains unchanged for ordinary `MACVO` and
other callers.

## MACVO factor-construction boundary

Split `MACVO.run_pair()` at the existing frontend boundary:

```python
def run_pair(frame0, frame1):
    depth1, match01 = Frontend.estimate_pair(frame0.stereo, frame1.stereo)
    run_pair_from_estimate(frame0, frame1, depth1, match01)
```

`run_pair_from_estimate()` owns all existing behavior after inference:
optimizer writeback, motion initialization, keypoint selection, covariance
projection, outlier filtering, map registration, visualization, context update,
two-frame optimization, and optional mapping. Moving this code must not change
its order or data conversions.

This boundary lets `WindowMACVO` provide fused results without duplicating the
base factor-building implementation.

## WindowMACVO data flow

`WindowMACVO` keeps the most recent frame needed for fused input separately
from its bounded window cache:

1. For frame `t`, identify `t-1` from the existing previous-keyframe context and
   `t-2` from the frame cache when available.
2. Call `Frontend.estimate_window(t-2, t-1, t)` exactly once.
3. Pass current depth and adjacent matching to
   `MACVO.run_pair_from_estimate()`.
4. Preserve the current synchronous two-frame ICP initializer and writeback.
5. Build the adjacent edge from the newly registered observations as before.
6. Build the skip edge from the already-produced skip match. It must continue
   to use cached depths, independent deterministic keypoint sampling,
   `MatchCovariance`, validity filtering, and the same minimum-point rule.
7. Run the unchanged five-pose CPU float64 window optimizer and write back the
   result.

The old `_skip_edge()` inference responsibility is separated from its
observation-building responsibility. No second call to `estimate_pair()` is
allowed on the optimized CUDA Graph path.

## CUDA Graph ownership

The existing CUDA Graph handler remains responsible for static inputs and
outputs. Fused window inference captures batch size 3 from its first call and
replays that same graph on later calls.

Ordinary `estimate_pair()` still captures batch size 2 when used by ordinary
`MACVO`. A single frontend instance must not mix batch-2 and batch-3 capture
modes. `WindowMACVO` routes skip-enabled execution exclusively through
`estimate_window()`, while the base class continues to use `estimate_pair()`.
Generic frontends inherit a correct two-call fallback; the CUDA Graph frontend
overrides it with fused execution. The existing capture-shape assertion reports
both requested and captured batch sizes if a caller mixes the two modes.

No concurrent graph replay or additional CUDA stream is introduced in this
version. This keeps output-buffer lifetime and synchronization equivalent to
the existing sequential frontend.

## Correctness requirements

- Batch slots must follow the table above exactly.
- Current depth must be derived only from slot 0.
- Adjacent flow and covariance must be derived only from slot 1.
- Skip flow and covariance must be derived only from slot 2.
- The first temporal step returns `skip_match=None` and creates no skip edge.
- All subsequent eligible steps create the same direct `t-2 -> t` constraint as
  the current implementation.
- Adjacent and skip keypoint random-number behavior remains deterministic for a
  fixed seed.
- No change is made to `num_point`, covariance thresholds, Huber delta, window
  size, optimizer iterations, pose anchoring, or map-point reanchoring.
- Ordinary `MACVO`, baseline DISP, and two-frame ICP continue to use batch-2
  `estimate_pair()` and retain their behavior.

## Error handling

- Validate the three images have compatible batch, channel, height, width,
  device, and dtype before CUDA Graph replay.
- Retain the CUDA Graph input-shape assertion and include the selected frontend
  mode in shape-mismatch messages.
- If no skip match is available, preserve the initializer and adjacent window
  path rather than constructing an invalid edge.
- Existing insufficient-match, invalid-depth, non-finite covariance, and
  disconnected-window handling remains unchanged.

## Testing strategy

Tests are written before production changes.

1. A CPU unit test exercises a pure batch-construction helper with distinct
   sentinel images and verifies all three slot pairings and the first-step
   filler behavior.
2. A frontend routing test uses deterministic synthetic FlowFormer outputs and
   verifies depth, adjacent match, and skip match are sliced from slots 0, 1,
   and 2 respectively.
3. A `WindowMACVO` orchestration test uses a lightweight recording frontend and
   verifies one `estimate_window()` call and zero skip-time `estimate_pair()`
   calls per frame.
4. Existing window ICP, map consistency, provenance, config, and frontend tests
   remain green.
5. A real-GPU smoke test processes the bundled short stereo sequence and
   verifies finite poses, one frontend call per temporal frame, fixed batch 3,
   bounded window state, and zero new unrefined windows.

## Performance validation

Performance is measured only after the currently running EuRoC batch releases
the GPU.

First run the controlled MH01 range `[1500, 1620)` with seed 0 and the same Git
commit for old and new implementations. Record:

- mean, median, and p95 `Odom_Runtime`;
- frontend call count and mean GPU time;
- peak and reserved CUDA memory;
- ATE, RTE, ROE, and RPE RMSE;
- unrefined-window count and mean window-solver time.

Acceptance criteria:

- exactly one fused frontend call for every temporal frame;
- no duplicate stereo slot in the logical three-pair batch;
- ATE, RTE, ROE, and RPE differ only within floating-point replay tolerance;
- zero additional unrefined windows;
- mean `Odom_Runtime` is at most 850 ms/frame on the current RTX 4060 Laptop GPU;
- peak CUDA memory remains below the device's 8 GiB capacity with adequate
  headroom to complete the controlled run.

After the controlled run passes, run complete MH01 and compare against
`Results/MACVO-Fast-WindowICP5@MH01/09_12_220737`.

## Non-goals

- Reducing decoder depth or image resolution.
- Adaptive or intermittent skip matching.
- Approximate flow composition.
- Replacing FlowFormerCov or changing its weights.
- Moving the CPU window solver to CUDA.
- Adding marginalization, bundle adjustment, or new window constraints.
