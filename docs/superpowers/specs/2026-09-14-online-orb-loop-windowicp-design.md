# Online ORB-BoW Loop Closing for WindowICP Design

## Goal

Extend `WindowMACVO` into a single-run online SLAM pipeline. Preserve the
existing normal batch-three frontend and five-frame WindowICP, retrieve loop
candidates asynchronously with ORB-BoW, validate at most one queued candidate
at a time with an additional serialized MACVO batch-three call, and optimize
accepted relative-pose loop constraints in a CPU background pose graph.

The production path must not replay the sequence or build an offline depth
cache.

## Hard invariants

```text
one FlowFormerCov model instance
one captured batch=3 CUDA Graph
all frontend calls occur on the WindowMACVO main thread
normal tracking inference always runs before optional loop inference
no concurrent GPU frontend inference
```

Normal and loop calls reuse the same captured graph because both have batch
size three:

```text
normal: current stereo, t-1 -> t, t-2 -> t
loop:   source stereo, source -> target, target -> source
```

The loop call occurs only when a candidate has arrived. A frame without a loop
candidate executes exactly the current WindowICP frontend path.

## Scope

This version adds:

- an external persistent ORB-BoW sidecar built from the existing local
  `/home/zzh/ORB_SLAM3/Thirdparty/DBoW2` source;
- asynchronous CPU candidate requests and responses;
- bounded lossless historical stereo-image storage;
- bounded pending target depth storage;
- serialized online bidirectional loop matching;
- independent pairwise covariance-aware ICP measurements;
- compact SE(3) pose-graph factors with `6 x 6` information matrices;
- switchable loop constraints;
- a single CPU background pose-graph worker;
- main-thread-only safe pose and map writeback;
- V203 short, revisit, and complete-sequence evaluation.

This version does not adopt ORB-SLAM3 `MapPoint`, LocalMapping, Atlas, local
bundle adjustment, IMU states, or multi-map merging. Offline proximity tools
remain as evaluation utilities but are never called by online odometry.

## ORB-BoW sidecar

### Build boundary

Create a small C++ executable inside MACVO. Its CMake build accepts:

```text
ORB_SLAM3_ROOT=/home/zzh/ORB_SLAM3
```

It compiles the required DBoW2 sources directly into the executable and links
system OpenCV. It does not modify the user's ORB-SLAM3 checkout.

The vocabulary is extracted from:

```text
/home/zzh/ORB_SLAM3/Vocabulary/ORBvoc.txt.tar.gz
```

into an ignored MACVO cache path. The sidecar loads it once at startup.

ORB-SLAM3 and its DBoW2 copy are GPLv3. The generated integration remains an
experimental local branch; distribution requires a separate license review.

### Protocol

Use one persistent subprocess with tab-delimited line messages.

Startup output:

```text
READY <protocol_version> <vocabulary_sha256> <opencv_version>
```

Request:

```text
QUERY <request_id> <frame_id> <left_png_path> <min_temporal_gap> <top_k>
```

Response:

```text
RESULT <request_id> <target_id> <count> <source_1>:<score_1> [additional source:score fields]
```

The sidecar uses OpenCV ORB to compute descriptors, transforms them with the
loaded ORB vocabulary, scores every eligible historical BoW vector, returns a
deterministically sorted top-k list, and inserts the current vector after the
query.

Malformed, duplicate, unknown, or stale request IDs are rejected by the Python
adapter. Sidecar failure disables loop closing while WindowICP continues.

## Runtime components

### `WindowMACVO`

Owns the live `VisualMap`, the only frontend model, all GPU calls, local
WindowICP, accepted factor insertion, and final writeback.

### `ORBLoopCandidateProvider`

Owns the sidecar process and one CPU reader thread. Input and output queues are
bounded. It never accesses CUDA or mutates `VisualMap`.

### `LoopKeyframeStore`

Stores every fifth keyframe as lossless PNG left/right images plus timestamp,
intrinsics, extrinsic, and pose metadata. It stores files in the experiment
space, not CUDA memory.

### `PendingLoopTargetStore`

Stores at most eight unresolved target packets. Each packet contains the target
depth/covariance moved to CPU, target frame metadata, and image-store reference.
Expired targets are recorded and never reconstructed through sequence replay.

### `AsyncPoseGraphBackend`

Owns one CPU worker, immutable snapshots, and result messages. It never writes
the live map.

## Per-frame online flow

For each current frame `t`:

1. Run the existing normal batch-three inference.
2. Build adjacent `t-1 -> t` and skip `t-2 -> t` raw ICP edges.
3. Run five-frame WindowICP and write local results.
4. Compress newly accepted local edges into global pose-graph factors.
5. Every five frames, save a loop keyframe and submit an ORB query.
6. Poll ORB responses without blocking.
7. If no normal frontend call is pending, process at most one loop candidate
   synchronously on the main thread.
8. Poll completed pose-graph results and safely write them back.

The optional loop call is deliberately placed after local tracking. It can add
latency to that frame but cannot run concurrently with or ahead of tracking.

## Loop candidate policy

Initial settings:

```yaml
keyframe_stride: 5
min_temporal_gap: 30
bow_top_k: 3
max_candidates_per_target: 1
candidate_nms_frames: 10
max_candidate_queue: 8
max_pending_targets: 8
```

The Python adapter filters existing, pending, recently rejected, and NMS-near
pairs. Only the highest remaining BoW score is queued for MACVO validation.

## Serialized loop batch-three validation

For source `i` and target `t`, the main thread calls the same frontend object:

```text
slot 0: source left -> source right
slot 1: source left -> target left
slot 2: target left -> source left
```

Source depth comes from slot zero. Target depth comes from the pending target
packet created during normal tracking.

After the call, temporary flow/depth CUDA tensors are consumed immediately.
The final raw loop edge is cloned to CPU float64 and GPU references are released.

Validation requires:

- forward-backward flow consistency;
- positive finite endpoint depth;
- image-grid coverage;
- full observation covariance construction;
- independent two-frame weighted ICP;
- pairwise ICP inlier count and ratio;
- finite, well-conditioned information matrix.

The current global pose prediction is diagnostic only. A large disagreement is
not a hard rejection because it may be the drift the loop must correct.

## Pairwise measurement compression

Every adjacent, skip, and loop raw edge is independently reduced to:

```text
PoseGraphFactor
  a, b
  measurement Z_ab[7]
  information Omega_ab[6,6]
  kind: adjacent | skip2 | loop
  confidence
  observation_count
```

`Z_ab` maps points from camera `b` into camera `a` and should measure:

```text
Z_ab ~= T_a.Inv() @ T_b
```

It is estimated solely from edge point correspondences using the existing full
covariance and vector Huber model. The converged Gauss-Newton Hessian becomes
`Omega_ab` after symmetrization and eigenvalue clamping.

Reject factors with non-finite values, insufficient rank, a minimum eigenvalue
below `1e-6`, or condition number above `1e8`.

## Global pose graph

All frames are nodes. Adjacent factors form the mandatory connected skeleton,
skip factors strengthen local geometry, and loop factors add nonlocal
observability.

For sensor-to-world poses:

```text
r_ab = Log(Z_ab.Inv() @ (T_a.Inv() @ T_b))
```

Pose zero is fixed exactly.

Factor treatment:

- adjacent: information matrix plus Huber loss;
- skip2: information matrix, Huber loss, and configurable information cap;
- loop: information matrix plus one switch variable.

Loop objective:

```text
E_loop = s_ab^2 * r_ab^T Omega_ab r_ab
       + lambda_switch * (1 - s_ab)^2
```

Bad loops may converge toward switch zero without disconnecting the trajectory.

## Background policy

- No confirmed loop means no periodic global solve.
- The first accepted loop requests a background solve when idle.
- While busy, new factors set one dirty flag instead of creating more jobs.
- Completion followed by a dirty graph triggers one newest-snapshot rerun.
- Successful writebacks are throttled to 50 frames apart, except loops spanning
  at least 100 frames may request immediate correction.
- Termination drains ORB responses and one active loop validation, waits for the
  backend, and runs one mandatory final pose-graph solve.

Only one pose-graph solve runs at a time.

## Safe writeback

The worker optimizes an immutable snapshot containing graph version, prefix
length, poses, factors, and switch states. The main thread accepts a result only
when it is finite, preserves pose zero, lowers the objective, and refers to a
compatible live-map prefix.

For the optimized prefix, reanchor owned points and rotate covariances using the
pose corrections. Frames appended during the solve receive the rigid correction
of the last optimized prefix pose:

```text
C = T_new[N-1] @ T_old[N-1].Inv()
T_new[k] = C @ T_old[k], k >= N
```

Update the motion model after writeback. Failed or stale results never partially
modify the map.

## Configuration

```yaml
online_loop:
  enabled: true
  keyframe_stride: 5
  min_temporal_gap: 30
  bow_top_k: 3
  max_candidates_per_target: 1
  candidate_nms_frames: 10
  max_candidate_queue: 8
  max_pending_targets: 8
  orb_sidecar: ./build/orb_bow/macvo_orb_bow
  vocabulary: ./cache/ORBvoc.txt

pairwise_icp:
  iterations: 10
  huber_delta: 3.0
  min_information_eigenvalue: 1.0e-6
  max_information_eigenvalue: 1.0e6
  max_information_condition: 1.0e8

pose_graph:
  iterations: 10
  huber_delta: 3.0
  skip_information_cap: 0.5
  switch_prior: 1.0
  switch_disable_threshold: 0.25
  min_frames_between_updates: 50
  immediate_loop_gap: 100
```

Missing sidecar or vocabulary disables loop closing with diagnostics and leaves
normal WindowICP unchanged.

## Memory and performance requirements

- Exactly one FlowFormerCov object and CUDA Graph are constructed.
- Maximum simultaneous frontend inference count is one.
- Peak reserved CUDA memory on the current 8 GiB GPU remains below 6 GiB.
- No-loop frame latency regresses by at most 3%.
- Average complete-V203 online overhead remains below 5%.
- Historical image-store disk use remains below 512 MiB on V203.
- Pending target depth count never exceeds eight.
- No offline depth cache or post-run inference is invoked.

## Diagnostics

Record:

- model construction and CUDA Graph capture counts;
- normal and loop frontend call counts, latency, and CUDA memory;
- ORB requests, results, scores, queue drops, expirations, and NMS rejection;
- loop match validation statistics and pairwise ICP information spectrum;
- accepted loop spans, switch values, and disabled loops;
- pose-graph solve count, dirty coalescing, latency, stale results, and writeback
  correction size;
- complete online per-frame latency percentiles.

## Testing

All behavior changes use test-first development.

Required tests include:

- sidecar build, vocabulary load, protocol handshake, deterministic top-k, and
  process-failure fallback;
- bounded keyframe/pending-target stores and lossless image reconstruction;
- proof that normal and loop calls share the same frontend and CUDA Graph;
- forced contention proving no concurrent frontend calls;
- tracking-before-loop scheduling;
- independent pairwise SE(3) recovery and information-matrix degeneracy checks;
- switchable pose graph correction and false-loop suppression;
- snapshot versioning, suffix correction, point/covariance reanchoring, and stale
  result rejection;
- loop-disabled WindowICP numerical regression;
- termination queue draining;
- V203 short, revisit, and complete-sequence runtime/accuracy evaluation.

## Validation

1. Confirm loop-disabled behavior on V203 `[1095,1215)` matches WindowICP.
2. Run online loop closing on V203 `[1733,1853)` and compare same-run trajectory
   before and after online pose-graph writeback.
3. Run complete V203 once.

Complete-V203 success requires:

- lower ATE than the same-run loop-disabled WindowICP trajectory;
- RTE, ROE, and RPE not worse by more than 2%;
- one model, one graph, and concurrency one proven by diagnostics;
- peak reserved CUDA memory below 6 GiB;
- no missing frames or failed local windows;
- no post-run network inference.

Secondary success is matching the offline dynamic proximity ATE of approximately
`0.57753 m` with lower total runtime and without its 1.08 GiB depth cache.
