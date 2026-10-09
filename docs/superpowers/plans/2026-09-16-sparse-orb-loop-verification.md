# Sparse ORB Loop Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Confirm ORB loop hypotheses with fixed-scale sparse SE(3) geometry across multiple keyframes and insert one conservative sparse loop factor without using long-range dense flow as correspondence evidence.

**Architecture:** The C++ ORB-BoW sidecar retains features and returns bounded mutual descriptor matches through protocol v2. Python obtains historical stereo depth through the existing batch-three CUDA graph, estimates a covariance-aware rigid transform, accumulates compatible supports in a two-of-three hypothesis tracker, and emits a directly constructed conservative `PoseGraphFactor` only after confirmation.

**Tech Stack:** C++17, OpenCV ORB/BFMatcher, DBoW2, Python 3.12, PyTorch, PyPose, NumPy, unittest, YAML.

---

## File Map

- Modify `Module/Frontend/Frontend.py`: correct depth ownership and add sparse-loop depth inference.
- Modify `Module/Frontend/SerializedFrontend.py`: serialize sparse-loop depth calls.
- Modify `Tools/ORBBoW/main.cpp`: retain features, match descriptors, emit protocol v2.
- Modify `Module/LoopClosure/ORBBoW.py`: parse validated sparse correspondences.
- Create `Module/LoopClosure/SparseGeometry.py`: observation construction, RANSAC, validation, and conservative factors.
- Create `Module/LoopClosure/LoopHypothesis.py`: confirmation state machine.
- Modify `Odometry/OnlineLoopWindowMACVO.py`: integrate sparse confirmation.
- Create `Config/Experiment/MACVO/MACVO_Fast_WindowICP_ORBLoop_Sparse.yaml`: preserve legacy config and expose sparse thresholds.

### Task 1: Correct Depth Ownership and Add Sparse Loop Depth Inference

**Files:**
- Modify: `Scripts/UnitTest/test_window_frontend.py`
- Modify: `Scripts/UnitTest/test_serialized_frontend.py`
- Modify: `Module/Frontend/Frontend.py`
- Modify: `Module/Frontend/SerializedFrontend.py`

- [ ] **Step 1: Write failing routing tests**

Add these expected APIs and slot semantics:

```python
def test_builds_source_stereo_forward_and_backward_slots(self):
    input_a, input_b = build_bidirectional_inputs(stereo(2, 20), stereo(0, 30))
    self.assertEqual(input_a[:, 0, 0, 0].tolist(), [2.0, 2.0, 0.0])
    self.assertEqual(input_b[:, 0, 0, 0].tolist(), [20.0, 0.0, 2.0])

def test_sparse_loop_depth_duplicates_source_stereo(self):
    depth = frontend.estimate_loop_depth(stereo(2, 20))
    self.assertEqual(captured_a[:, 0, 0, 0].tolist(), [2.0, 2.0, 2.0])
    self.assertEqual(captured_b[:, 0, 0, 0].tolist(), [20.0, 20.0, 20.0])
    self.assertTrue(torch.allclose(depth.depth, torch.ones_like(depth.depth)))
```

Add a serialized wrapper test proving `estimate_loop_depth` increments loop calls.

- [ ] **Step 2: Verify RED**

```bash
.venv/bin/python -m unittest Scripts.UnitTest.test_window_frontend Scripts.UnitTest.test_serialized_frontend -v
```

Expected: slot assertion failure and missing `estimate_loop_depth`.

- [ ] **Step 3: Implement minimal behavior**

```python
def build_bidirectional_inputs(source, target):
    return (
        torch.cat([source.imageL, source.imageL, target.imageL], dim=0),
        torch.cat([source.imageR, target.imageL, source.imageL], dim=0),
    )

def build_loop_depth_inputs(source):
    return source.imageL.repeat(3, 1, 1, 1), source.imageR.repeat(3, 1, 1, 1)
```

Validate inputs before concatenation. Add `estimate_loop_depth` to `IFrontend`,
the CUDA-graph frontend, and `SerializedFrontend`; return only slot-zero depth.

- [ ] **Step 4: Verify GREEN**

Run the Step 2 command. Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add Module/Frontend/Frontend.py Module/Frontend/SerializedFrontend.py Scripts/UnitTest/test_window_frontend.py Scripts/UnitTest/test_serialized_frontend.py
git commit -m "fix: route sparse loop source depth correctly"
```

### Task 2: Define and Parse ORB Protocol Version 2

**Files:**
- Modify: `Scripts/UnitTest/test_orb_bow_provider.py`
- Modify: `Module/LoopClosure/ORBBoW.py`

- [ ] **Step 1: Write failing parser tests**

Use handshake `READY\t2` and response:

```text
RESULT\t7\t40\t1\t0:0.9:120:80:2:10,20,12,22,4;30,40,31,41,8
```

Assert candidate source, raw/ratio/mutual counts, and a `(2, 5)` match array.
Add rejection tests for protocol v1, count mismatch, nonfinite coordinates, and
negative Hamming distance.

- [ ] **Step 2: Verify RED**

```bash
.venv/bin/python -m unittest Scripts.UnitTest.test_orb_bow_provider -v
```

Expected: old protocol/API assertions fail.

- [ ] **Step 3: Implement parser and immutable records**

```python
@dataclass(frozen=True)
class ORBLoopCandidate:
    source: int
    score: float
    rank: int
    raw_knn_matches: int
    ratio_matches: int
    matches: np.ndarray

    @property
    def mutual_matches(self):
        return len(self.matches)
```

Require protocol 2. Extend query submission with ratio and maximum match count.
Preserve complete candidate records through `filter_loop_candidates`.

- [ ] **Step 4: Verify GREEN**

Run the Step 2 command. Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add Module/LoopClosure/ORBBoW.py Scripts/UnitTest/test_orb_bow_provider.py
git commit -m "feat: parse ORB sparse match protocol"
```

### Task 3: Match ORB Descriptors in the Sidecar

**Files:**
- Modify: `Scripts/UnitTest/test_orb_bow_sidecar.py`
- Modify: `Tools/ORBBoW/main.cpp`

- [ ] **Step 1: Write a failing translated-image test**

Generate a deterministic textured image and a `(5, 3)` translated copy. Query
frames 0 and 40 with ratio `0.8` and maximum 50 matches. Assert protocol 2,
at least 20 mutual matches, at most 50 returned matches, and median displacement
within 1.5 pixels of `(5, 3)`.

- [ ] **Step 2: Verify RED**

```bash
.venv/bin/python -m unittest Scripts.UnitTest.test_orb_bow_sidecar -v
```

Expected: query arity or protocol version failure.

- [ ] **Step 3: Implement retained features and matching**

```cpp
struct StoredORBFrame {
    DBoW2::BowVector bow;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
};
```

For each top-k candidate run Hamming KNN, Lowe ratio filtering, reverse
best-match agreement, stable distance/index sorting, truncation, and coordinate
serialization. Clone descriptors before storing the current frame.

- [ ] **Step 4: Verify GREEN**

Run the Step 2 command. Expected: sidecar rebuilds and tests pass.

- [ ] **Step 5: Commit**

```bash
git add Tools/ORBBoW/main.cpp Scripts/UnitTest/test_orb_bow_sidecar.py
git commit -m "feat: return mutual ORB matches from sidecar"
```

### Task 4: Implement Deterministic Fixed-Scale SE(3) RANSAC

**Files:**
- Create: `Scripts/UnitTest/test_sparse_loop_geometry.py`
- Create: `Module/LoopClosure/SparseGeometry.py`

- [ ] **Step 1: Write failing synthetic tests**

Create 100 noncoplanar points, transform them with known SE(3), add Gaussian
noise and 40 percent outliers. Assert at least 55 inliers, less than 2 cm
translation error, less than 1 degree rotation error, and identical repeated
results. Add insufficient-point, collinear, and invalid-covariance cases.

```python
result = estimate_se3_ransac(points_a, points_b, cov_a, cov_b,
                             iterations=256,
                             mahalanobis_threshold=3.5,
                             seed=17)
error = (pp.SE3(result.measurement).Inv() @ truth).Log().tensor()
```

- [ ] **Step 2: Verify RED**

```bash
.venv/bin/python -m unittest Scripts.UnitTest.test_sparse_loop_geometry -v
```

Expected: module import fails.

- [ ] **Step 3: Implement minimal RANSAC core**

Create `SE3RansacResult` and implement deterministic three-point sampling,
unit-scale weighted Kabsch, rotation-aware combined covariance, Cholesky
whitening, lexicographic model selection, all-inlier refit, and singular-value
geometry rejection.

- [ ] **Step 4: Verify GREEN**

Run the Step 2 command. Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add Module/LoopClosure/SparseGeometry.py Scripts/UnitTest/test_sparse_loop_geometry.py
git commit -m "feat: add fixed-scale sparse SE3 RANSAC"
```

### Task 5: Validate Pixel/Depth Observations and Build Conservative Factors

**Files:**
- Modify: `Scripts/UnitTest/test_sparse_loop_geometry.py`
- Modify: `Module/LoopClosure/SparseGeometry.py`

- [ ] **Step 1: Write failing validation tests**

Use synthetic depth maps and intrinsics to verify expected 3D points. Test
explicit rejection reasons for invalid depth, weak coverage, too few inliers,
and excessive bidirectional reprojection error. Test conservative factor values:

```python
factor = conservative_sparse_factor(10, 100, truth.tensor(), 60, 0.6,
                                    translation_sigma_m=0.25,
                                    rotation_sigma_deg=10.0)
self.assertAlmostEqual(float(factor.information[0, 0]), 16.0)
self.assertEqual(factor.observation_count, 60)
self.assertAlmostEqual(factor.confidence, 0.6)
```

- [ ] **Step 2: Verify RED**

Run the Task 4 test command. Expected: missing validation/factor APIs.

- [ ] **Step 3: Implement validation pipeline**

Add `SparseGeometryConfig`, `SparseLoopResult`, grid coverage, bilinear depth
sampling, covariance projection through the existing covariance model,
bidirectional reprojection metrics, ordered gates, JSON-safe metric output, and
separate translational/rotational diagonal precision.

- [ ] **Step 4: Verify GREEN**

Run the Task 4 test command. Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add Module/LoopClosure/SparseGeometry.py Scripts/UnitTest/test_sparse_loop_geometry.py
git commit -m "feat: validate sparse loop geometry"
```

### Task 6: Implement the Multi-Keyframe Hypothesis Tracker

**Files:**
- Create: `Scripts/UnitTest/test_loop_hypothesis.py`
- Create: `Module/LoopClosure/LoopHypothesis.py`

- [ ] **Step 1: Write failing state-machine tests**

Verify one support is tentative, two compatible target keyframes confirm and
emit once, a third transitions to strong without another emission, duplicate
targets reject, incompatible corrections separate, best quality wins, and stale
hypotheses expire.

```python
first = tracker.add(support(source=100, target=500, correction=a))
second = tracker.add(support(source=105, target=505, correction=b))
self.assertEqual(first.state, "tentative")
self.assertEqual(second.state, "confirmed")
self.assertIsNotNone(second.emitted)
```

- [ ] **Step 2: Verify RED**

```bash
.venv/bin/python -m unittest Scripts.UnitTest.test_loop_hypothesis -v
```

Expected: module import fails.

- [ ] **Step 3: Implement tracker**

Create immutable `LoopSupport`, `LoopHypothesis`, and `HypothesisUpdate`. Cluster
by source interval, increasing target interval, and SE(3) correction distance.
Store one support per target and emit the highest quality support exactly once.

- [ ] **Step 4: Verify GREEN**

Run the Step 2 command. Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add Module/LoopClosure/LoopHypothesis.py Scripts/UnitTest/test_loop_hypothesis.py
git commit -m "feat: confirm loop hypotheses across keyframes"
```

### Task 7: Integrate Sparse Confirmation into Online Odometry

**Files:**
- Modify: `Scripts/UnitTest/test_online_loop_window_macvo.py`
- Modify: `Module/LoopClosure/__init__.py`
- Modify: `Odometry/OnlineLoopWindowMACVO.py`
- Create: `Config/Experiment/MACVO/MACVO_Fast_WindowICP_ORBLoop_Sparse.yaml`

- [ ] **Step 1: Write failing integration tests**

Prove insufficient matches reject before GPU depth, the first valid support
stores no factor, the second compatible support stores one direct loop factor,
dense pairwise compression is not called, and detailed rejection reasons are
recorded. Inject sparse validator, tracker, and factor builder at test seams.

- [ ] **Step 2: Verify RED**

```bash
.venv/bin/python -m unittest Scripts.UnitTest.test_online_loop_window_macvo -v
```

Expected: sparse path and injection seams are missing.

- [ ] **Step 3: Implement sparse path and preserve legacy behavior**

Add `_validate_sparse_candidate`, `_register_sparse_support`, and
`_store_sparse_loop_factor`. Only a confirmed emitted support inserts a direct
factor, increments graph version, records endpoints, and submits the backend.
Keep current dense validation behind `validation_mode: legacy_dense`. Create a
new sparse YAML; do not edit the legacy YAML.

- [ ] **Step 4: Verify GREEN**

Run the Step 2 command. Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add Odometry/OnlineLoopWindowMACVO.py Module/LoopClosure/__init__.py Config/Experiment/MACVO/MACVO_Fast_WindowICP_ORBLoop_Sparse.yaml Scripts/UnitTest/test_online_loop_window_macvo.py
git commit -m "feat: add sparse confirmed online loop factors"
```

### Task 8: Complete Diagnostics and Experiment Routing

**Files:**
- Modify: `Scripts/UnitTest/test_online_loop_window_macvo.py`
- Modify: `Scripts/UnitTest/test_compare_online_orb_loop.py`
- Modify: `Odometry/OnlineLoopWindowMACVO.py`
- Modify: `Scripts/Experiment/CompareOnlineORBLoop.py`

- [ ] **Step 1: Write failing diagnostics tests**

Assert saved output contains protocol version, threshold snapshot, ORB match
counts, geometry metrics, hypothesis transitions, emitted factor eigenvalues,
and aggregate rejection counters. Add `window_orb_loop_sparse` mode routing.

- [ ] **Step 2: Verify RED**

```bash
.venv/bin/python -m unittest Scripts.UnitTest.test_online_loop_window_macvo Scripts.UnitTest.test_compare_online_orb_loop -v
```

Expected: diagnostic keys and sparse mode are missing.

- [ ] **Step 3: Implement JSON-safe diagnostics and comparison summaries**

Report tentative, confirmed, strong, emitted sparse factors, median RANSAC
ratio, median reprojection error, maximum information eigenvalue, and effective
long-loop switch counts.

- [ ] **Step 4: Verify GREEN**

Run the Step 2 command. Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add Odometry/OnlineLoopWindowMACVO.py Scripts/Experiment/CompareOnlineORBLoop.py Scripts/UnitTest/test_online_loop_window_macvo.py Scripts/UnitTest/test_compare_online_orb_loop.py
git commit -m "feat: report sparse loop verification diagnostics"
```

### Task 9: Full Verification and V203 Test Gates

**Files:**
- Create: `docs/validation/sparse_orb_loop_v203_short.csv` after a successful experiment.
- Preserve: existing unstaged `docs/WindowICP.md` changes.

- [ ] **Step 1: Run focused suite**

```bash
.venv/bin/python -m unittest Scripts.UnitTest.test_window_frontend Scripts.UnitTest.test_serialized_frontend Scripts.UnitTest.test_orb_bow_provider Scripts.UnitTest.test_orb_bow_sidecar Scripts.UnitTest.test_sparse_loop_geometry Scripts.UnitTest.test_loop_hypothesis Scripts.UnitTest.test_online_loop_window_macvo Scripts.UnitTest.test_pose_graph -v
```

Expected: zero failures and errors.

- [ ] **Step 2: Run all unit tests**

```bash
.venv/bin/python -m unittest discover -s Scripts/UnitTest -p 'test_*.py' -v
```

Expected: zero failures and errors; environment-dependent skips recorded.

- [ ] **Step 3: Run V203 short experiment**

```bash
.venv/bin/python Scripts/Experiment/CompareOnlineORBLoop.py --sequence V203 --seq-from 1095 --seq-to 1215 --seed 0 --modes window_orb_loop_sparse --result-root /home/zzh/MACVO/Results/SparseORBLoop_V203_short
```

Expected: 120 finite poses, no newly unrefined windows, protocol 2, and no
single-support factor.

- [ ] **Step 4: Run V203 revisit experiment if short gate passes**

```bash
.venv/bin/python Scripts/Experiment/CompareOnlineORBLoop.py --sequence V203 --seq-from 1733 --seq-to 1853 --seed 0 --modes window_orb_loop_sparse --result-root /home/zzh/MACVO/Results/SparseORBLoop_V203_revisit
```

Expected: every emitted factor has at least two supports and conservative
information eigenvalues.

- [ ] **Step 5: Verify version-control boundaries**

```bash
git status --short
git diff --check
git log --oneline --decorate -12
```

Expected: pre-existing `docs/WindowICP.md` remains unstaged and unchanged;
component commits remain separate.

- [ ] **Step 6: Commit validation record separately**

```bash
git add docs/validation/sparse_orb_loop_v203_short.csv
git commit -m "docs: record sparse ORB loop validation"
```
