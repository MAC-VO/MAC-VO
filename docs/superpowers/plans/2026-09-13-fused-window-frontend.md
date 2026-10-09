# Fused Batch-3 Window Frontend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace WindowMACVO's two batch-2 FlowFormerCov calls per temporal frame with one fixed batch-3 CUDA Graph call while preserving the existing depth, flow, covariance, and optimization semantics.

**Architecture:** Add a three-output window operation to the frontend boundary, split MACVO's factor construction from its inference call, and inject the fused results from WindowMACVO. The CUDA Graph batch contains current stereo, adjacent temporal, and two-step temporal pairs; before `t-2` exists, the third slot duplicates the adjacent pair and its result is ignored.

**Tech Stack:** Python 3.12, PyTorch 2.4/CUDA Graphs, PyPose, unittest/pytest, MAC-VO evaluation utilities.

---

### Task 1: Add the fused frontend contract

**Files:**
- Create: `Scripts/UnitTest/test_window_frontend.py`
- Modify: `Module/Frontend/Frontend.py:45-104`
- Modify: `Module/Frontend/Frontend.py:264-353`

- [ ] **Step 1: Write failing tests for batch construction and output routing**

Create sentinel `StereoData` values and test the desired API:

```python
import unittest
from types import SimpleNamespace
import pypose as pp
import torch
from DataLoader import StereoData
from Module.Frontend.Frontend import CUDAGraph_FlowFormerCovFrontend, build_window_inputs


def stereo(left, right):
    K = torch.eye(3).unsqueeze(0)
    K[0, 0, 0] = K[0, 1, 1] = 10
    return StereoData(
        T_BS=pp.identity_SE3(1), K=K, baseline=torch.tensor([0.2]),
        time_ns=[0], height=2, width=2,
        imageL=torch.full((1, 3, 2, 2), float(left)),
        imageR=torch.full((1, 3, 2, 2), float(right)))


class FusedWindowFrontendTests(unittest.TestCase):
    def test_builds_stereo_adjacent_and_skip_slots(self):
        a, b, has_skip = build_window_inputs(stereo(2, 20), stereo(1, 10), stereo(0, 30))
        self.assertTrue(has_skip)
        self.assertEqual(a[:, 0, 0, 0].tolist(), [0., 1., 2.])
        self.assertEqual(b[:, 0, 0, 0].tolist(), [30., 0., 0.])

    def test_first_step_uses_fixed_batch3_without_skip_output(self):
        a, b, has_skip = build_window_inputs(None, stereo(1, 10), stereo(0, 30))
        self.assertFalse(has_skip)
        self.assertEqual(a[:, 0, 0, 0].tolist(), [0., 1., 1.])
        self.assertEqual(b[:, 0, 0, 0].tolist(), [30., 0., 0.])

    def test_routes_slots_zero_one_two(self):
        frontend = CUDAGraph_FlowFormerCovFrontend.__new__(CUDAGraph_FlowFormerCovFrontend)
        frontend.config = SimpleNamespace(device="cpu", enforce_positive_disparity=False)
        flow, cov = torch.zeros(3, 2, 2, 2), torch.ones(3, 2, 2, 2)
        flow[0, 0], flow[1], flow[2] = 2, 11, 22
        frontend.cuda_graph_estimate = lambda *_: (flow, cov)
        depth, adjacent, skip = frontend.estimate_window(stereo(2, 20), stereo(1, 10), stereo(0, 30))
        self.assertTrue(torch.allclose(depth.depth, torch.ones_like(depth.depth)))
        self.assertTrue(torch.equal(adjacent.flow, flow[1:2]))
        self.assertTrue(torch.equal(skip.flow, flow[2:3]))
```

- [ ] **Step 2: Run the new test and verify RED**

Run:

```bash
PYTHONPATH=. /home/zzh/MACVO/.venv/bin/python -m unittest Scripts.UnitTest.test_window_frontend -v
```

Expected: import or attribute failure because `build_window_inputs` and `estimate_window` do not exist.

- [ ] **Step 3: Implement the generic contract and batch helper**

Add after `CUDAGraphHandler`:

```python
def build_window_inputs(frame_t2, frame_t1, frame_t):
    skip_source = frame_t1 if frame_t2 is None else frame_t2
    input_a = torch.cat([frame_t.imageL, frame_t1.imageL, skip_source.imageL], dim=0)
    input_b = torch.cat([frame_t.imageR, frame_t.imageL, frame_t.imageL], dim=0)
    return input_a, input_b, frame_t2 is not None
```

Add to `IFrontend`:

```python
def estimate_window(self, frame_t2, frame_t1, frame_t):
    depth, adjacent = self.estimate_pair(frame_t1, frame_t)
    if frame_t2 is None:
        return depth, adjacent, None
    _, skip = self.estimate_pair(frame_t2, frame_t)
    return depth, adjacent, skip
```

- [ ] **Step 4: Implement the CUDA Graph batch-3 override**

Add to `CUDAGraph_FlowFormerCovFrontend`:

```python
@Timer.cpu_timeit("Frontend.estimate")
@Timer.gpu_timeit("Frontend.estimate")
@torch.inference_mode()
def estimate_window(self, frame_t2, frame_t1, frame_t):
    input_a, input_b, has_skip = build_window_inputs(frame_t2, frame_t1, frame_t)
    flow, cov = self.cuda_graph_estimate(
        input_a.to(self.config.device), input_b.to(self.config.device))
    flow, cov = flow.float(), cov.float()
    depth = self.inference_2_depth(flow[0:1], cov[0:1], frame_t,
                                   self.config.enforce_positive_disparity)
    adjacent = self.inference_2_match(flow[1:2], cov[1:2])
    skip = self.inference_2_match(flow[2:3], cov[2:3]) if has_skip else None
    return depth, adjacent, skip
```

Change the shape assertion to include requested and captured batch sizes.

- [ ] **Step 5: Verify GREEN and commit**

Run:

```bash
PYTHONPATH=. /home/zzh/MACVO/.venv/bin/python -m unittest Scripts.UnitTest.test_window_frontend -v
```

Expected: 3 tests pass.

Commit:

```bash
git add Module/Frontend/Frontend.py Scripts/UnitTest/test_window_frontend.py
git commit -m "feat: add fused batch3 window frontend"
```

### Task 2: Split MACVO inference from factor construction

**Files:**
- Create: `Scripts/UnitTest/test_macvo_inference_boundary.py`
- Modify: `Odometry/MACVO.py:162-348`

- [ ] **Step 1: Write the failing delegation test**

```python
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from Odometry.MACVO import MACVO


class MACVOInferenceBoundaryTests(unittest.TestCase):
    def test_run_pair_delegates_precomputed_outputs(self):
        system = MACVO.__new__(MACVO)
        depth, match = object(), object()
        frame0, frame1 = SimpleNamespace(stereo=object()), SimpleNamespace(stereo=object())
        system.prev_keyframe = (frame0, 0, object())
        system.Frontend = SimpleNamespace(estimate_pair=Mock(return_value=(depth, match)))
        system.run_pair_from_estimate = Mock()
        system.run_pair(frame0, frame1)
        system.Frontend.estimate_pair.assert_called_once_with(frame0.stereo, frame1.stereo)
        system.run_pair_from_estimate.assert_called_once_with(frame0, frame1, depth, match)
```

- [ ] **Step 2: Run the test and verify RED**

```bash
PYTHONPATH=. /home/zzh/MACVO/.venv/bin/python -m unittest Scripts.UnitTest.test_macvo_inference_boundary -v
```

Expected: failure because the current `run_pair()` does not delegate.

- [ ] **Step 3: Extract the post-inference method**

Replace the beginning of `run_pair()` with:

```python
def run_pair(self, frame0, frame1):
    assert self.prev_keyframe is not None
    depth1, match01 = self.Frontend.estimate_pair(frame0.stereo, frame1.stereo)
    self.run_pair_from_estimate(frame0, frame1, depth1, match01)

def run_pair_from_estimate(self, frame0, frame1, depth1, match01):
    assert self.prev_keyframe is not None
    depth0 = self.prev_keyframe[2]
```

Move the existing code after inference through optional mapping under the new method without changing statement order or data conversions.

- [ ] **Step 4: Verify and commit**

```bash
PYTHONPATH=. /home/zzh/MACVO/.venv/bin/python -m unittest \
  Scripts.UnitTest.test_macvo_inference_boundary \
  Scripts.UnitTest.test_config_macvo Scripts.UnitTest.test_config_modules -v
git add Odometry/MACVO.py Scripts/UnitTest/test_macvo_inference_boundary.py
git commit -m "refactor: separate MACVO inference from factors"
```

Expected: all selected tests pass and the commit succeeds.

### Task 3: Route WindowMACVO through the fused operation

**Files:**
- Create: `Scripts/UnitTest/test_window_frontend_routing.py`
- Modify: `Odometry/WindowMACVO.py:82-164`

- [ ] **Step 1: Write failing routing tests**

```python
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from Odometry.WindowMACVO import WindowMACVO


class WindowFrontendRoutingTests(unittest.TestCase):
    def make_system(self, skip):
        system = WindowMACVO.__new__(WindowMACVO)
        system.skip_matching = skip
        system.frame_cache = {3: (SimpleNamespace(stereo="t2"), object())}
        system.prev_keyframe = (SimpleNamespace(stereo="t1"), 4, object())
        system.Frontend = SimpleNamespace(
            estimate_window=Mock(return_value=("depth", "adjacent", "skip")),
            estimate_pair=Mock(return_value=("depth", "adjacent")))
        return system

    def test_skip_mode_calls_fused_frontend_once(self):
        system = self.make_system(True)
        result = system._estimate_window_inputs(
            SimpleNamespace(stereo="t1"), SimpleNamespace(stereo="t"))
        self.assertEqual(result, ("depth", "adjacent", "skip"))
        system.Frontend.estimate_window.assert_called_once_with("t2", "t1", "t")
        system.Frontend.estimate_pair.assert_not_called()

    def test_adjacent_mode_keeps_batch2_path(self):
        system = self.make_system(False)
        result = system._estimate_window_inputs(
            SimpleNamespace(stereo="t1"), SimpleNamespace(stereo="t"))
        self.assertEqual(result, ("depth", "adjacent", None))
        system.Frontend.estimate_pair.assert_called_once_with("t1", "t")
        system.Frontend.estimate_window.assert_not_called()
```

- [ ] **Step 2: Run the test and verify RED**

```bash
PYTHONPATH=. /home/zzh/MACVO/.venv/bin/python -m unittest Scripts.UnitTest.test_window_frontend_routing -v
```

Expected: failure because `_estimate_window_inputs` does not exist.

- [ ] **Step 3: Add routing and remove skip-time inference**

Add:

```python
def _estimate_window_inputs(self, frame0, frame1):
    if not self.skip_matching:
        depth1, adjacent = self.Frontend.estimate_pair(frame0.stereo, frame1.stereo)
        return depth1, adjacent, None
    prev2 = self.frame_cache.get(self.prev_keyframe[1] - 1)
    frame_t2 = None if prev2 is None else prev2[0].stereo
    return self.Frontend.estimate_window(frame_t2, frame0.stereo, frame1.stereo)
```

Rename `_skip_edge(a, b)` to `_skip_edge_from_match(a, b, match)` and delete its `Frontend.estimate_pair()` call. In `run_pair()` obtain fused results first, call `super().run_pair_from_estimate(...)`, then pass `skip_match` into `_skip_edge_from_match()`.

The beginning of `run_pair()` becomes:

```python
before_match = len(self.graph.match)
a = self.prev_keyframe[1]
depth1, adjacent_match, skip_match = self._estimate_window_inputs(frame0, frame1)
super().run_pair_from_estimate(frame0, frame1, depth1, adjacent_match)
b = self.prev_keyframe[1]
```

The skip block becomes:

```python
skip_start = time.perf_counter()
skip = None
if skip_match is not None and b - 2 in self.frame_cache:
    skip = self._skip_edge_from_match(b - 2, b, skip_match)
    if skip is not None:
        self.edge_window.add(skip)
skip_seconds = time.perf_counter() - skip_start
```

- [ ] **Step 4: Verify focused tests and commit**

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. \
  /home/zzh/MACVO/.venv/bin/python -m unittest \
  Scripts.UnitTest.test_window_frontend_routing \
  Scripts.UnitTest.test_window_frontend \
  Scripts.UnitTest.test_window_icp \
  Scripts.UnitTest.test_window_map \
  Scripts.UnitTest.test_window_provenance -v
git add Odometry/WindowMACVO.py Scripts/UnitTest/test_window_frontend_routing.py
git commit -m "perf: fuse WindowMACVO frontend inference"
```

Expected: all selected tests pass and the commit succeeds.

### Task 4: Regression, GPU, and performance verification

**Files:**
- Modify: `docs/WindowICP.md`
- Test: `Scripts/UnitTest/test_frontend.py`
- Test: `Scripts/Experiment/CompareWindowICP.py`

- [ ] **Step 1: Run the complete relevant CPU regression set**

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. \
  /home/zzh/MACVO/.venv/bin/python -m unittest \
  Scripts.UnitTest.test_window_frontend \
  Scripts.UnitTest.test_macvo_inference_boundary \
  Scripts.UnitTest.test_window_frontend_routing \
  Scripts.UnitTest.test_window_icp \
  Scripts.UnitTest.test_window_map \
  Scripts.UnitTest.test_window_provenance \
  Scripts.UnitTest.test_config_macvo \
  Scripts.UnitTest.test_config_modules -v
```

Expected: zero failures.

- [ ] **Step 2: Wait until the main worktree releases the GPU**

```bash
ps -eo pid,args | rg 'run_euroc_window_icp|CompareWindowICP.py|MACVO.py' | rg -v 'rg '
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
```

Expected before benchmarking: no `/home/zzh/MACVO/MACVO.py` process remains.

- [ ] **Step 3: Run the real-GPU frontend and 10-frame smoke tests**

```bash
PYTHONPATH=. /home/zzh/MACVO/.venv/bin/python -m pytest Scripts/UnitTest/test_frontend.py -m local -q
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
PYTHONPATH=. /home/zzh/MACVO/.venv/bin/python MACVO.py \
  --odom Config/Experiment/MACVO/MACVO_Fast_WindowICP.yaml \
  --data Config/Sequence/TartanAir2_Test.yaml --seq_from 0 --seq_to 10 --seed 0 \
  --resultRoot /tmp/macvo-fused-batch3-smoke --noeval --timing
```

Expected: frontend tests pass; the smoke run creates 10 finite poses, one batch-3 graph, 9 frontend timing samples, bounded window state, and no new unrefined windows.

- [ ] **Step 4: Run controlled MH01 `[1500, 1620)` validation**

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
PYTHONPATH=. /home/zzh/MACVO/.venv/bin/python MACVO.py \
  --odom Config/Experiment/MACVO/MACVO_Fast_WindowICP.yaml \
  --data Config/Sequence/EuRoC_MH01_local.yaml --seq_from 1500 --seq_to 1620 --seed 0 \
  --resultRoot /tmp/macvo-fused-batch3-mh01 --noeval --timing
```

Evaluate the produced leaf sandbox with `python -m Evaluation.EvalSeq --spaces <leaf>`. Expected: 119 frontend calls, zero unrefined windows, metrics within floating-point replay tolerance of the historical 120-frame window-skip result, mean `Odom_Runtime <= 850 ms`, and peak memory below 8 GiB.

- [ ] **Step 5: Document measured results and run final verification**

Append the fused batch-3 data path, commit, GPU, call count, timing, memory, metrics, and unrefined-window count to `docs/WindowICP.md`, preserving historical v0.1 results.

Run:

```bash
git diff --check
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. \
  /home/zzh/MACVO/.venv/bin/python -m unittest discover Scripts/UnitTest -v
```

Expected: no whitespace errors and all non-local unit tests pass.

- [ ] **Step 6: Commit verification documentation**

```bash
git add docs/WindowICP.md
git commit -m "docs: record fused batch3 frontend validation"
```
