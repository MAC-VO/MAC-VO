# Online ORB-BoW Loop WindowICP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add single-run online ORB-BoW loop retrieval, serialized MACVO batch-three validation, and asynchronous sparse pose-graph correction to WindowICP without duplicating the frontend model or CUDA Graph.

**Architecture:** A persistent C++ DBoW2 sidecar retrieves appearance candidates on CPU. `OnlineLoopWindowMACVO` keeps the normal local batch-three path unchanged, processes at most one loop batch-three call after tracking on the same frontend object, compresses raw ICP edges into relative SE(3) factors, and sends immutable snapshots to one CPU pose-graph worker for safe main-thread writeback.

**Tech Stack:** Python 3.12, PyTorch/pypose, SciPy sparse solve, OpenCV 4, C++17, CMake, ORB-SLAM3 DBoW2 sources and ORB vocabulary, unittest/pytest.

---

### Task 1: Build and verify the ORB-BoW sidecar

**Files:**
- Create: `Tools/ORBBoW/CMakeLists.txt`
- Create: `Tools/ORBBoW/main.cpp`
- Create: `Scripts/Build/build_orb_bow_sidecar.sh`
- Create: `Scripts/UnitTest/test_orb_bow_sidecar.py`

- [ ] Write failing tests that build the executable with `ORB_SLAM3_ROOT=/home/zzh/ORB_SLAM3`, start it with a tiny generated DBoW2 vocabulary fixture, verify the `READY` handshake, insert/query deterministic synthetic images, enforce temporal exclusion, and reject malformed commands.
- [ ] Run the focused test and confirm the executable is absent.
- [ ] Implement a C++17 persistent process using OpenCV ORB and DBoW2 `OrbVocabulary`. Maintain `frame_id -> BowVector`, score all eligible history, sort by `(-score, frame_id)`, return top-k, then insert the target.
- [ ] Implement tab-delimited `PING`, `QUERY`, and `STOP` commands with flushed single-line responses.
- [ ] Add a build script that extracts `ORBvoc.txt` into ignored `cache/ORBvoc.txt` when absent and builds into ignored `build/orb_bow/` without modifying `/home/zzh/ORB_SLAM3`.
- [ ] Run sidecar tests and commit.

Verification command:

```bash
PYTHONPATH=. /home/zzh/MACVO/.venv/bin/python -m pytest -o addopts='' Scripts/UnitTest/test_orb_bow_sidecar.py -q
```

Commit message: `feat: add ORB BoW candidate sidecar`

### Task 2: Add asynchronous candidate provider and bounded stores

**Files:**
- Create: `Module/LoopClosure/__init__.py`
- Create: `Module/LoopClosure/ORBBoW.py`
- Create: `Module/LoopClosure/OnlineLoopStore.py`
- Create: `Scripts/UnitTest/test_orb_bow_provider.py`
- Create: `Scripts/UnitTest/test_online_loop_store.py`

- [ ] Write failing tests for handshake validation, chronological request IDs, asynchronous response delivery, sidecar crash fallback, queue bounds, target expiration, candidate NMS, and lossless PNG stereo reconstruction.
- [ ] Implement `ORBLoopCandidateProvider` with one subprocess and one reader thread. Its public API is `submit(packet)`, `poll()`, `enabled`, and `close()`.
- [ ] Implement `LoopKeyframeStore` using a temporary directory, PNG left/right files, and metadata. Convert `[0,1]` RGB tensors to uint8 without CUDA retention.
- [ ] Implement `PendingLoopTargetStore(capacity=8)` storing CPU depth/covariance and target metadata. Oldest unresolved targets expire deterministically.
- [ ] Run focused tests and commit.

Verification:

```bash
PYTHONPATH=. /home/zzh/MACVO/.venv/bin/python -m pytest -o addopts='' Scripts/UnitTest/test_orb_bow_provider.py Scripts/UnitTest/test_online_loop_store.py -q
```

Commit: `feat: add asynchronous ORB loop candidates`

### Task 3: Prove single-model single-graph serialized inference

**Files:**
- Modify: `Module/Frontend/Frontend.py`
- Create: `Module/Frontend/SerializedFrontend.py`
- Create: `Scripts/UnitTest/test_serialized_frontend.py`

- [ ] Write failing tests using forced thread contention that assert maximum simultaneous call count is one, tracking calls execute before a waiting loop call, both call types expose the same wrapped frontend identity and CUDA Graph identity, and no second graph capture occurs.
- [ ] Implement `SerializedFrontend` as a lock-owning wrapper with synchronous `estimate_window()` and `estimate_bidirectional()` methods, tracking/loop counters, active-call count, maximum concurrency, and graph identity diagnostics.
- [ ] The wrapper must not create a worker thread or frontend copy. Both methods call the same wrapped object under one reentrant lock. `estimate_window()` is tracking priority because the main thread invokes loop work only after tracking completes.
- [ ] Add construction/capture counters to `CUDAGraph_FlowFormerCovFrontend` diagnostics without changing inference output.
- [ ] Run frontend regression tests and commit.

Verification:

```bash
PYTHONPATH=. /home/zzh/MACVO/.venv/bin/python -m pytest -o addopts='' Scripts/UnitTest/test_serialized_frontend.py Scripts/UnitTest/test_window_frontend.py -q
```

Commit: `feat: serialize tracking and loop inference`

### Task 4: Compress raw ICP edges into relative pose factors

**Files:**
- Create: `Module/Optimization/PairwiseICP.py`
- Create: `Module/Optimization/PoseGraph.py`
- Create: `Scripts/UnitTest/test_pairwise_icp.py`

- [ ] Write failing tests for weighted Kabsch initialization, known SE(3) recovery, independence from current global poses, full-covariance objective decrease, SPD information output, planar/collinear degeneracy rejection, and serialization.
- [ ] Implement `PoseGraphFactor(a,b,measurement,information,kind,confidence,observation_count)` with strict finite/unit-quaternion/SPD validation.
- [ ] Implement weighted Kabsch initialization mapping `points_b -> points_a`, then refine with a remapped two-frame `optimize_window()` problem anchored at identity.
- [ ] Re-linearize the converged pairwise problem, apply Huber weights, form `Omega=J.T@J`, symmetrize, eigen-clamp to `[1e-6,1e6]`, and reject condition number above `1e8`.
- [ ] Add NPZ save/load for pose-graph factors and source poses/timestamps.
- [ ] Run tests and commit.

Verification:

```bash
PYTHONPATH=. /home/zzh/MACVO/.venv/bin/python -m pytest -o addopts='' Scripts/UnitTest/test_pairwise_icp.py -q
```

Commit: `feat: compress ICP edges into pose factors`

### Task 5: Implement sparse switchable pose-graph optimization

**Files:**
- Modify: `Module/Optimization/PoseGraph.py`
- Create: `Scripts/UnitTest/test_pose_graph.py`

- [ ] Write failing synthetic tests: chain recovery, loop drift correction across intermediate poses, exact anchor, false-loop suppression, skip information cap, sparse system dimensions, and objective decrease.
- [ ] Implement vectorized residuals `Log(Z_ab.Inv() @ (T_a.Inv() @ T_b))`.
- [ ] Compute factor-local `6x6` endpoint Jacobians with central finite differences: six batched perturbations for endpoint `a` and six for `b`, never a dense global Jacobian.
- [ ] Whiten with Cholesky of each information matrix, apply vector Huber weights, and apply closed-form loop switch `s=lambda/(lambda+chi2)` as the pose weight.
- [ ] Accumulate sparse `6x6` blocks and solve with SciPy CSC `spsolve`, using the existing LM accept/reject policy.
- [ ] Return poses, switch states, factor counts, objective, iterations, timing, and anchor diagnostics.
- [ ] Run tests and commit.

Verification:

```bash
PYTHONPATH=. /home/zzh/MACVO/.venv/bin/python -m pytest -o addopts='' Scripts/UnitTest/test_pose_graph.py -q
```

Commit: `feat: optimize switchable sparse pose graph`

### Task 6: Add one-worker asynchronous pose-graph backend

**Files:**
- Create: `Module/Optimization/AsyncPoseGraph.py`
- Create: `Scripts/UnitTest/test_async_pose_graph.py`

- [ ] Write failing tests for one active solve, dirty coalescing, newest snapshot rerun, result polling, clean shutdown, exception preservation, and stale version metadata.
- [ ] Implement immutable `PoseGraphSnapshot` and `PoseGraphAsyncResult` dataclasses.
- [ ] Implement one daemon worker with one pending/latest snapshot slot and one dirty flag. Submitting while busy replaces only the pending snapshot.
- [ ] Worker exceptions become failed results and never escape into tracking.
- [ ] `terminate(final_snapshot)` waits for the active job and ensures the final newest snapshot is solved once.
- [ ] Run tests and commit.

Verification:

```bash
PYTHONPATH=. /home/zzh/MACVO/.venv/bin/python -m pytest -o addopts='' Scripts/UnitTest/test_async_pose_graph.py -q
```

Commit: `feat: add asynchronous pose graph backend`

### Task 7: Add online loop WindowMACVO integration

**Files:**
- Modify: `Odometry/WindowMACVO.py`
- Create: `Odometry/OnlineLoopWindowMACVO.py`
- Modify: `MACVO.py`
- Create: `Config/Experiment/MACVO/MACVO_Fast_WindowICP_ORBLoop.yaml`
- Create: `Scripts/UnitTest/test_online_loop_window_macvo.py`

- [ ] Write failing tests for loop-disabled numerical parity, every-fifth-frame ORB submission, normal inference before loop inference, at-most-one loop per frame, target expiration, accepted factor insertion, background trigger, and sidecar failure fallback.
- [ ] Add a protected no-op hook to `WindowMACVO` after each successful/failed window step, passing current frame, depth, adjacent edge, and skip edge. Existing behavior remains unchanged.
- [ ] Implement `OnlineLoopWindowMACVO` that wraps its existing frontend with `SerializedFrontend`, owns the provider/stores/factor dictionary/backend, compresses adjacent/skip edges, and handles loop messages after local optimization.
- [ ] Reconstruct source/target `StereoData` from lossless PNG packets; use pending target depth and the same `estimate_bidirectional()` frontend object.
- [ ] Replace current-global-pose Mahalanobis hard rejection with pairwise ICP factor validation. Retain flow/depth/grid checks.
- [ ] Add `OnlineLoopWindowMACVO` dispatch to `MACVO.py` and strict configuration validation.
- [ ] Run tests and commit.

Verification:

```bash
PYTHONPATH=. /home/zzh/MACVO/.venv/bin/python -m pytest -o addopts='' Scripts/UnitTest/test_online_loop_window_macvo.py Scripts/UnitTest/test_window_frontend_routing.py Scripts/UnitTest/test_window_map.py -q
```

Commit: `feat: integrate online ORB loop WindowICP`

### Task 8: Implement safe asynchronous writeback and termination

**Files:**
- Modify: `Odometry/OnlineLoopWindowMACVO.py`
- Modify: `Module/Optimization/AsyncPoseGraph.py`
- Create: `Scripts/UnitTest/test_online_loop_writeback.py`

- [ ] Write failing tests for graph-version validation, optimized-prefix writeback, suffix rigid correction, point/covariance reanchoring, motion-model update, stale result rejection, failed-result preservation, and termination drain order.
- [ ] Implement main-thread polling and `_apply_pose_graph_result()`. Reuse `reanchor_points()` for the optimized prefix and apply the last-prefix rigid correction to appended suffix frames and owned points.
- [ ] Increment graph version after accepted factor insertion and successful writeback. Reset invalidates every outstanding snapshot.
- [ ] Termination stops ORB submission, drains responses, processes one final valid loop, closes the sidecar, drains the backend, applies its final result, then follows standard trajectory serialization.
- [ ] Save `online_loop_diagnostics.json`, `pose_graph_factors.npz`, loop switches, frontend identity/concurrency counts, latency, memory, and failure states.
- [ ] Run tests and commit.

Verification:

```bash
PYTHONPATH=. /home/zzh/MACVO/.venv/bin/python -m pytest -o addopts='' Scripts/UnitTest/test_online_loop_writeback.py Scripts/UnitTest/test_window_global_termination.py -q
```

Commit: `feat: safely write online pose graph results`

### Task 9: Add online-loop experiment and evaluation tooling

**Files:**
- Create: `Scripts/Experiment/CompareOnlineORBLoop.py`
- Create: `Scripts/UnitTest/test_compare_online_orb_loop.py`
- Modify: `docs/WindowICP.md`

- [ ] Write failing tests for strict same-timestamp rows, loop-disabled/loop-enabled modes, source commit/config identity, completeness, online-only inference accounting, memory gates, and CSV/JSON consistency.
- [ ] Implement a sequential comparison runner for `window_skip` and `window_orb_loop`. It must reject results reporting post-run frontend calls or more than one model/graph/concurrent call.
- [ ] Record ATE/RTE/ROE/RPE, mean/median/p95 frame latency, normal/loop frontend calls, candidate/accepted loops, loop spans, pose-graph timing, peak CUDA memory, and factor counts.
- [ ] Add documented commands for short, revisit, and complete V203.
- [ ] Run tests and commit.

Verification:

```bash
PYTHONPATH=. /home/zzh/MACVO/.venv/bin/python -m pytest -o addopts='' Scripts/UnitTest/test_compare_online_orb_loop.py -q
```

Commit: `feat: compare online ORB loop WindowICP`

### Task 10: Verification and V203 evaluation

**Files:**
- Create: `docs/validation/online_orb_loop_v203_1095_1215.csv`
- Create: `docs/validation/online_orb_loop_v203_1733_1853.csv`
- Create: `docs/validation/online_orb_loop_v203_full.csv`
- Modify: `docs/WindowICP.md`

- [ ] Run all focused tests, then the complete non-local suite.
- [ ] Build the release sidecar and extract the official vocabulary into ignored cache.
- [ ] Run loop-disabled regression on `[1095,1215)` and confirm normal WindowICP metrics/latency within tolerance.
- [ ] Run loop-enabled `[1733,1853)` and inspect accepted ORB loop spans before complete evaluation.
- [ ] Run complete V203 `window_skip` and `window_orb_loop` sequentially with seed zero.
- [ ] Verify one model, one CUDA Graph, concurrency one, peak reserved VRAM below 6 GiB, no post-run inference, no missing frames, and no failed local windows.
- [ ] Record results and commit documentation only; keep large images, vocabulary, factors, and result directories outside Git.

Commands:

```bash
PYTHONPATH=. /home/zzh/MACVO/.venv/bin/python -m pytest -o addopts='' Scripts/UnitTest -m 'not local' -q

ORB_SLAM3_ROOT=/home/zzh/ORB_SLAM3 ./Scripts/Build/build_orb_bow_sidecar.sh

PYTHONPATH=. /home/zzh/MACVO/.venv/bin/python Scripts/Experiment/CompareOnlineORBLoop.py \
  --sequence V203 --seq-from 1095 --seq-to 1215 --seed 0 \
  --result-root /home/zzh/MACVO/Results/OnlineORBLoop_V203_short

PYTHONPATH=. /home/zzh/MACVO/.venv/bin/python Scripts/Experiment/CompareOnlineORBLoop.py \
  --sequence V203 --seq-from 1733 --seq-to 1853 --seed 0 \
  --result-root /home/zzh/MACVO/Results/OnlineORBLoop_V203_revisit

PYTHONPATH=. /home/zzh/MACVO/.venv/bin/python Scripts/Experiment/CompareOnlineORBLoop.py \
  --sequence V203 --seq-from 0 --seed 0 \
  --result-root /home/zzh/MACVO/Results/OnlineORBLoop_V203_full
```

Final commit: `docs: validate online ORB loop closing on V203`
