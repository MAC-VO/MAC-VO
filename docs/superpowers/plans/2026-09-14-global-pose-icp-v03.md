# Global Pose-Only ICP v0.3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retain every adjacent and two-step WindowICP factor and optionally perform a block-sparse global pose-only ICP refinement at sequence termination, without changing v0.2 defaults.

**Architecture:** `EdgeWindow.advance()` exposes evicted factors, `WindowMACVO` stores them only when global refinement is enabled, and a new `GlobalPoseICP` module accumulates 6x6 pose blocks into a SciPy sparse system. A separate v0.3 config enables the backend; termination applies a successful result atomically and records global diagnostics.

**Tech Stack:** Python 3.12, PyTorch/PyPose CPU float64 geometry, SciPy 1.13 sparse matrices and solvers, unittest/pytest, existing MAC-VO evaluation scripts.

---

### Task 1: Expose evicted factors and account for retained memory

**Files:**
- Modify: `Module/Optimization/WindowICP.py:10-54`
- Modify: `Scripts/UnitTest/test_window_icp.py:64-77`

- [ ] **Step 1: Write failing lifecycle and byte-accounting tests**

Extend `WindowICPTests`:

```python
from Module.Optimization.WindowICP import edge_tensor_bytes

def test_window_advance_returns_evicted_edges(self):
    _, _, edges = fixture()
    window = EdgeWindow(5)
    for edge in edges:
        window.add(edge)
    evicted = window.advance(5)
    self.assertEqual([(e.a, e.b) for e in evicted], [(0, 1), (0, 2)])
    self.assertTrue(all(edge.a >= 1 for edge in window.edges))

def test_edge_tensor_bytes_counts_only_observation_payload(self):
    _, _, edges = fixture()
    edge = edges[0]
    expected = sum(value.numel() * value.element_size() for value in (
        edge.points_a, edge.points_b, edge.cov_a, edge.cov_b))
    self.assertEqual(edge_tensor_bytes(edge), expected)
```

- [ ] **Step 2: Run focused tests and verify RED**

```bash
PYTHONPATH=. /home/zzh/MACVO/.venv/bin/python -m unittest \
  Scripts.UnitTest.test_window_icp.WindowICPTests.test_window_advance_returns_evicted_edges \
  Scripts.UnitTest.test_window_icp.WindowICPTests.test_edge_tensor_bytes_counts_only_observation_payload -v
```

Expected: import failure for `edge_tensor_bytes` or assertion failure because `advance()` returns `None`.

- [ ] **Step 3: Implement deterministic eviction return and byte counting**

Add:

```python
def edge_tensor_bytes(edge):
    return sum(value.numel() * value.element_size() for value in (
        edge.points_a, edge.points_b, edge.cov_a, edge.cov_b))
```

Change `EdgeWindow.advance()` to:

```python
def advance(self, current):
    first = current - self.size + 1
    evicted = [
        edge for key, edge in sorted(self._edges.items())
        if not (first <= edge.a < edge.b <= current)
    ]
    self._edges = {
        key: edge for key, edge in self._edges.items()
        if first <= edge.a < edge.b <= current
    }
    return evicted
```

- [ ] **Step 4: Run the complete window solver tests and verify GREEN**

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. \
  /home/zzh/MACVO/.venv/bin/python -m unittest Scripts.UnitTest.test_window_icp -v
```

Expected: all WindowICP tests pass.

- [ ] **Step 5: Commit factor lifecycle support**

```bash
git add Module/Optimization/WindowICP.py Scripts/UnitTest/test_window_icp.py
git commit -m "feat: expose evicted WindowICP factors"
```

### Task 2: Implement the block-sparse global pose solver

**Files:**
- Create: `Module/Optimization/GlobalPoseICP.py`
- Create: `Scripts/UnitTest/test_global_pose_icp.py`

- [ ] **Step 1: Write failing synthetic recovery and failure-atomicity tests**

Create `Scripts/UnitTest/test_global_pose_icp.py` with a 12-pose graph:

```python
import unittest
import pypose as pp
import torch

from Module.Optimization.WindowICP import Edge
from Module.Optimization.GlobalPoseICP import optimize_global_pose_graph


def global_fixture(count=12):
    torch.manual_seed(23)
    tangent = torch.zeros(count, 6, dtype=torch.float64)
    tangent[:, 0] = torch.arange(count) * 0.08
    tangent[:, 4] = torch.arange(count) * 0.012
    truth = pp.se3(tangent).Exp()
    world = torch.randn(40, 3, dtype=torch.float64)
    world[:, 0] += 4
    cov = torch.eye(3, dtype=torch.float64).expand(40, 3, 3).clone() * 0.001
    edges = []
    for b in range(1, count):
        for gap in (1, 2):
            a = b - gap
            if a >= 0:
                edges.append(Edge(
                    a, b, truth[a].Inv().Act(world), truth[b].Inv().Act(world), cov, cov))
    noise = torch.randn(count, 6, dtype=torch.float64) * 0.01
    noise[0] = 0
    initial = pp.se3(noise).Exp() @ truth
    return truth, initial.tensor(), edges


class GlobalPoseICPTests(unittest.TestCase):
    def test_recovers_more_than_five_poses_and_preserves_anchor(self):
        truth, initial, edges = global_fixture()
        result = optimize_global_pose_graph(initial, edges, max_iters=10)
        self.assertEqual(result.diagnostics["status"], "refined")
        self.assertTrue(torch.equal(result.poses[0], initial[0]))
        error = (pp.SE3(result.poses).Inv() @ truth).Log().tensor().norm(dim=-1)
        self.assertLess(error.max().item(), 1e-5)
        self.assertLess(result.diagnostics["final_cost"], result.diagnostics["initial_cost"])

    def test_matches_dense_solver_for_five_pose_graph(self):
        from Module.Optimization.WindowICP import optimize_window
        _, initial, edges = global_fixture(5)
        sparse = optimize_global_pose_graph(initial, edges, max_iters=10)
        dense = optimize_window(initial, list(range(5)), edges, max_iters=10)
        self.assertTrue(torch.allclose(sparse.poses, dense.poses, atol=1e-8, rtol=1e-7))

    def test_disconnected_graph_preserves_input(self):
        _, initial, edges = global_fixture()
        result = optimize_global_pose_graph(initial, edges[:1], max_iters=5)
        self.assertEqual(result.diagnostics["status"], "not_refined")
        self.assertTrue(torch.equal(result.poses, initial))
        self.assertIn("connected", result.diagnostics["reason"])
```

- [ ] **Step 2: Run the new test and verify RED**

```bash
PYTHONPATH=. /home/zzh/MACVO/.venv/bin/python -m unittest Scripts.UnitTest.test_global_pose_icp -v
```

Expected: import failure because `GlobalPoseICP.py` does not exist.

- [ ] **Step 3: Implement edge-local linearization and objective evaluation**

Create `Module/Optimization/GlobalPoseICP.py` with:

```python
from dataclasses import dataclass
import math
import time
import warnings

import numpy as np
import pypose as pp
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import MatrixRankWarning, spsolve
import torch

from Module.Optimization.WindowICP import Edge, huber_cost, normalized_pose, skew


@dataclass
class GlobalPoseResult:
    poses: torch.Tensor
    diagnostics: dict


def edge_linearization(poses, edge):
    pose = pp.SE3(poses)
    qa = pose[edge.a].Act(edge.points_a)
    qb = pose[edge.b].Act(edge.points_b)
    Ra = pose[edge.a].rotation().matrix()
    Rb = pose[edge.b].rotation().matrix()
    covariance = Ra @ edge.cov_a @ Ra.T + Rb @ edge.cov_b @ Rb.T
    covariance = (covariance + covariance.transpose(-1, -2)) * 0.5
    eye = torch.eye(3, dtype=poses.dtype)
    Ja = torch.cat((eye.expand(len(qa), 3, 3), -skew(qa)), dim=-1)
    Jb = torch.cat((-eye.expand(len(qb), 3, 3), skew(qb)), dim=-1)
    residual = qa - qb
    L = torch.linalg.cholesky(covariance)
    rw = torch.linalg.solve_triangular(L, residual[..., None], upper=False).squeeze(-1)
    Jaw = torch.linalg.solve_triangular(L, Ja, upper=False)
    Jbw = torch.linalg.solve_triangular(L, Jb, upper=False)
    return rw, Jaw, Jbw


def global_cost(poses, edges, huber_delta):
    return sum(float(huber_cost(edge_linearization(poses, edge)[0], huber_delta))
               for edge in edges)
```

- [ ] **Step 4: Implement sparse block accumulation and solve**

Add helpers that never form a global dense Jacobian:

```python
def accumulate_blocks(poses, edges, huber_delta):
    blocks, gradient = {}, torch.zeros((len(poses) - 1, 6), dtype=torch.float64)
    for edge in edges:
        rw, Ja, Jb = edge_linearization(poses, edge)
        weight = torch.clamp(
            huber_delta / rw.norm(dim=-1).clamp_min(1e-15), max=1.0).sqrt()
        Aa = (Ja * weight[:, None, None]).reshape(-1, 6)
        Ab = (Jb * weight[:, None, None]).reshape(-1, 6)
        e = (rw * weight[:, None]).reshape(-1)
        local = {
            (edge.a, edge.a): Aa.T @ Aa,
            (edge.a, edge.b): Aa.T @ Ab,
            (edge.b, edge.a): Ab.T @ Aa,
            (edge.b, edge.b): Ab.T @ Ab,
        }
        for (a, b), value in local.items():
            if a == 0 or b == 0:
                continue
            key = (a - 1, b - 1)
            blocks[key] = blocks.get(key, torch.zeros_like(value)) + value
        if edge.a:
            gradient[edge.a - 1] += Aa.T @ e
        if edge.b:
            gradient[edge.b - 1] += Ab.T @ e
    return blocks, gradient


def sparse_system(blocks, pose_variables):
    rows, cols, values = [], [], []
    for (bi, bj), block in sorted(blocks.items()):
        base_i, base_j = 6 * bi, 6 * bj
        for i in range(6):
            for j in range(6):
                rows.append(base_i + i)
                cols.append(base_j + j)
                values.append(float(block[i, j]))
    size = pose_variables * 6
    return coo_matrix((values, (rows, cols)), shape=(size, size)).tocsc()
```

Implement `optimize_global_pose_graph()` with input validation, connectivity
checking, anchor-local coordinates, LM damping, `MatrixRankWarning` promoted to
an exception, finite checks, objective-based trial acceptance, atomic failure
results, and diagnostics required by the design.

- [ ] **Step 5: Run solver tests and verify GREEN**

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. \
  /home/zzh/MACVO/.venv/bin/python -m unittest Scripts.UnitTest.test_global_pose_icp -v
```

Expected: all global solver tests pass.

- [ ] **Step 6: Commit the global solver**

```bash
git add Module/Optimization/GlobalPoseICP.py Scripts/UnitTest/test_global_pose_icp.py
git commit -m "feat: add sparse global pose ICP solver"
```

### Task 3: Add opt-in factor retention and global configuration

**Files:**
- Create: `Config/Experiment/MACVO/MACVO_Fast_WindowICP_Global.yaml`
- Create: `Scripts/UnitTest/test_window_global_config.py`
- Modify: `Odometry/WindowMACVO.py:37-88`
- Modify: `Odometry/WindowMACVO.py:151-205`

- [ ] **Step 1: Write failing configuration and retention tests**

Create `Scripts/UnitTest/test_window_global_config.py`:

```python
import unittest
from types import SimpleNamespace

from Module.Optimization.WindowICP import EdgeWindow
from Odometry.WindowMACVO import WindowMACVO
from Utility.Config import load_config


class WindowGlobalConfigTests(unittest.TestCase):
    def test_v03_config_enables_global_refinement(self):
        config, _ = load_config("Config/Experiment/MACVO/MACVO_Fast_WindowICP_Global.yaml")
        self.assertTrue(config.Odometry.args.global_refine)
        self.assertEqual(config.Odometry.args.global_iterations, 5)
        self.assertEqual(config.Odometry.name, "MACVO-Fast-WindowICP5-Global")
        WindowMACVO.is_valid_config(config.Odometry)

    def test_disabled_mode_does_not_retain_evicted_edges(self):
        system = WindowMACVO.__new__(WindowMACVO)
        system.global_refine = False
        system.inactive_edges = {}
        system._retain_evicted_edges([object()])
        self.assertEqual(system.inactive_edges, {})
```

Extend the test after RED with an enabled-mode test using real `Edge` values
from the WindowICP synthetic fixture and assert dictionary keys are unique.

- [ ] **Step 2: Run tests and verify RED**

```bash
PYTHONPATH=. /home/zzh/MACVO/.venv/bin/python -m unittest Scripts.UnitTest.test_window_global_config -v
```

Expected: missing config file and missing `_retain_evicted_edges` failure.

- [ ] **Step 3: Add constructor defaults and optional config validation**

Change the constructor signature to:

```python
def __init__(self, *, window_size=5, skip_matching=True,
             window_iterations=10, window_huber_delta=3.0,
             global_refine=False, global_iterations=5,
             global_huber_delta=3.0, **kwargs):
```

Initialize:

```python
self.global_refine = global_refine
self.global_iterations = global_iterations
self.global_huber_delta = global_huber_delta
self.inactive_edges = {}
self.global_record = {"status": "disabled" if not global_refine else "pending"}
self.retained_tensor_bytes = 0
```

In `is_valid_config()`, copy the argument namespace, inject the three defaults
when absent, and validate all thirteen keys. This keeps the exact v0.2 YAML
loadable while rejecting unknown or invalid v0.3 values.

- [ ] **Step 4: Implement opt-in retention**

Add:

```python
def _retain_evicted_edges(self, edges):
    if not self.global_refine:
        return
    for edge in edges:
        self.inactive_edges[(edge.a, edge.b)] = edge

def _global_edges(self):
    edges = dict(self.inactive_edges)
    edges.update({(edge.a, edge.b): edge for edge in self.edge_window.edges})
    return [edges[key] for key in sorted(edges)]
```

Change the online advance call to:

```python
evicted = self.edge_window.advance(b)
self._retain_evicted_edges(evicted)
```

- [ ] **Step 5: Create the v0.3 config**

Copy the v0.2 configuration and change only:

```yaml
Odometry:
  name: MACVO-Fast-WindowICP5-Global
  args:
    global_refine: true
    global_iterations: 5
    global_huber_delta: 3.0
```

- [ ] **Step 6: Verify configuration, retention, and v0.2 regression**

```bash
PYTHONPATH=. /home/zzh/MACVO/.venv/bin/python -m unittest \
  Scripts.UnitTest.test_window_global_config \
  Scripts.UnitTest.test_window_frontend_routing \
  Scripts.UnitTest.test_window_icp -v
PYTHONPATH=. /home/zzh/MACVO/.venv/bin/python -m pytest -o addopts='' \
  Scripts/UnitTest/test_config_macvo.py -q
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit configuration and retention**

```bash
git add Config/Experiment/MACVO/MACVO_Fast_WindowICP_Global.yaml \
  Odometry/WindowMACVO.py Scripts/UnitTest/test_window_global_config.py
git commit -m "feat: retain inactive WindowICP factors"
```

### Task 4: Integrate atomic termination writeback and diagnostics

**Files:**
- Modify: `Odometry/WindowMACVO.py:209-240`
- Modify: `Scripts/UnitTest/test_window_map.py`
- Create: `Scripts/UnitTest/test_window_global_termination.py`

- [ ] **Step 1: Write failing successful and failed writeback tests**

Create tests that construct a small `VisualMap`, retain synthetic edges, and
patch only the solver boundary:

```python
from unittest.mock import patch
from Module.Optimization.GlobalPoseICP import GlobalPoseResult

def test_successful_global_result_updates_poses_and_owned_points(self):
    system, before, target = make_global_system_fixture()
    result = GlobalPoseResult(target, {"status": "refined", "anchor_preserved": True})
    with patch("Odometry.WindowMACVO.optimize_global_pose_graph", return_value=result):
        system._run_global_refinement(before)
    self.assertTrue(torch.equal(system.graph.frames.data["pose"].tensor[0], before[0].float()))
    self.assertTrue(torch.allclose(system.graph.frames.data["pose"].tensor.double(), target))
    assert_owned_points_follow_pose_corrections(self, system.graph, before, target)

def test_failed_global_result_preserves_interpolated_state(self):
    system, before, _ = make_global_system_fixture()
    result = GlobalPoseResult(before.clone(), {"status": "not_refined", "reason": "singular"})
    with patch("Odometry.WindowMACVO.optimize_global_pose_graph", return_value=result):
        system._run_global_refinement(before)
    self.assertTrue(torch.equal(system.graph.frames.data["pose"].tensor.double(), before))
```

The fixture reuses the point ownership setup already present in
`test_window_map.py`; move shared construction into a local helper rather than
duplicating assertions.

- [ ] **Step 2: Run tests and verify RED**

```bash
PYTHONPATH=. /home/zzh/MACVO/.venv/bin/python -m unittest \
  Scripts.UnitTest.test_window_global_termination -v
```

Expected: missing `_run_global_refinement` failure.

- [ ] **Step 3: Implement atomic global refinement**

Import `optimize_global_pose_graph` and `edge_tensor_bytes`. Add:

```python
def _run_global_refinement(self, before):
    edges = self._global_edges()
    self.retained_tensor_bytes = sum(edge_tensor_bytes(edge) for edge in edges)
    if not self.global_refine:
        self.global_record = {"status": "disabled"}
        return
    result = optimize_global_pose_graph(
        before, edges, max_iters=self.global_iterations,
        huber_delta=self.global_huber_delta)
    self.global_record = result.diagnostics
    if result.diagnostics["status"] != "refined":
        return
    reanchor_points(self.graph, list(range(len(before))), before, result.poses)
    self.graph.frames.data["pose"].tensor[:] = result.poses.float()
```

Call it in `terminate()` after interpolation reanchoring and before clearing
edge caches. Capture `inactive_edge_count` before clearing.

- [ ] **Step 4: Extend diagnostics without breaking existing readers**

Add the following keys beside the existing `windows` array:

```python
"global_refine": self.global_refine,
"inactive_edges": self.inactive_edge_count,
"retained_tensor_bytes": self.retained_tensor_bytes,
"global_refinement": self.global_record,
```

Existing code that reads `payload["windows"]` must continue to work.

- [ ] **Step 5: Run map, termination, and provenance tests**

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. \
  /home/zzh/MACVO/.venv/bin/python -m unittest \
  Scripts.UnitTest.test_window_map \
  Scripts.UnitTest.test_window_global_termination \
  Scripts.UnitTest.test_window_provenance -v
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit termination integration**

```bash
git add Odometry/WindowMACVO.py Scripts/UnitTest/test_window_map.py \
  Scripts/UnitTest/test_window_global_termination.py
git commit -m "feat: run global pose ICP at termination"
```

### Task 5: Add controlled V203 comparison and document results

**Files:**
- Modify: `Scripts/Experiment/CompareWindowICP.py:55-130`
- Modify: `Scripts/UnitTest/test_compare_window_icp.py`
- Modify: `docs/WindowICP.md`

- [ ] **Step 1: Write failing comparison-mode tests**

Add tests that parse the comparison script's mode table and verify
`window_global` selects `MACVO_Fast_WindowICP_Global.yaml`, does not mutate the
v0.2 config, and exports these columns when diagnostics exist:

```python
expected = {
    "inactive_edges", "retained_tensor_bytes", "global_status",
    "global_seconds", "global_initial_cost", "global_final_cost",
}
self.assertTrue(expected.issubset(row))
```

- [ ] **Step 2: Run comparison tests and verify RED**

```bash
PYTHONPATH=. /home/zzh/MACVO/.venv/bin/python -m unittest \
  Scripts.UnitTest.test_compare_window_icp -v
```

Expected: `window_global` is not an accepted mode.

- [ ] **Step 3: Extend comparison mode and metrics extraction**

Add `window_global` to argparse choices and config mapping. For this mode,
parse `window_diagnostics.json["global_refinement"]` and append:

```python
row.update({
    "inactive_edges": diagnostics_payload["inactive_edges"],
    "retained_tensor_bytes": diagnostics_payload["retained_tensor_bytes"],
    "global_status": global_record["status"],
    "global_seconds": global_record.get("seconds"),
    "global_initial_cost": global_record.get("initial_cost"),
    "global_final_cost": global_record.get("final_cost"),
})
```

- [ ] **Step 4: Run complete CPU regression tests**

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. \
  /home/zzh/MACVO/.venv/bin/python -m pytest -o addopts='' \
  -m 'not local and not trt and not vio' -q
```

Expected: all non-local tests pass.

- [ ] **Step 5: Prepare the isolated worktree for GPU validation**

Initialize only the FlowFormer submodule and link the ignored model directory:

```bash
git submodule update --init Module/Network/FlowFormer
ln -s /home/zzh/MACVO/Model Model
```

Verify no other MACVO process is using the GPU before benchmarking.

- [ ] **Step 6: Run v0.2 and v0.3 on difficult V203 `[1095,1215)`**

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
PYTHONPATH=. /home/zzh/MACVO/.venv/bin/python \
  Scripts/Experiment/CompareWindowICP.py \
  --sequences V203 --seq_from 1095 --seq_to 1215 --seed 0 \
  --modes window_skip window_global \
  --resultRoot /tmp/macvo-window-global-v03
```

Expected: two complete 120-pose results with identical timestamps; each has
119 refined local windows; the global result has a finite decreasing objective
and exact anchor preservation.

- [ ] **Step 7: Evaluate acceptance criteria**

Compare the generated `metrics.csv` rows:

- compute percentage changes for ATE/RTE/ROE/RPE;
- confirm online mean `Odom_Runtime` regression is below 5%;
- confirm quaternion norm error below `1e-6`;
- record global solve seconds and retained bytes;
- classify v0.3 as preferred only if no primary metric regresses by more than
  2%, otherwise keep it experimental.

- [ ] **Step 8: Document the implementation and measured result**

Append a v0.3 section to `docs/WindowICP.md` containing configuration, factor
lifecycle, sparse solver, V203 range selection, exact metrics, objective change,
memory, runtime, status, limitations, and the experimental/preferred decision.

- [ ] **Step 9: Run final verification and commit documentation**

```bash
git diff --check
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. \
  /home/zzh/MACVO/.venv/bin/python -m pytest -o addopts='' \
  -m 'not local and not trt and not vio' -q
```

Expected: no whitespace errors and all non-local tests pass.

Commit:

```bash
git add Scripts/Experiment/CompareWindowICP.py \
  Scripts/UnitTest/test_compare_window_icp.py docs/WindowICP.md
git commit -m "docs: validate global pose ICP v0.3 on V203"
```
