# Sparse Loop Weight Scale Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply the approved `alpha = 10000` sparse-loop information and switch-prior scaling, then validate it on complete EuRoC MH04.

**Architecture:** `OnlineLoopWindowMACVO` reads one positive `loop_information_scale`, passes it into direct sparse factor construction, and multiplies the base switch prior by the same value for every pose-graph solve. Legacy configurations default to one; the sparse configuration explicitly selects 10000 and records all effective values in diagnostics.

**Tech Stack:** Python 3.12, PyTorch, PyPose, YAML, unittest, existing asynchronous pose graph and experiment runner.

---

### Task 1: Scale Sparse Factor Information

**Files:**
- Modify: `Scripts/UnitTest/test_sparse_loop_geometry.py`
- Modify: `Module/LoopClosure/SparseGeometry.py`

- [ ] **Step 1: Write the failing factor-scale test**

```python
def test_scales_conservative_sparse_factor_information(self):
    factor = conservative_sparse_factor(
        source=10,
        target=100,
        measurement=pp.identity_SE3(dtype=torch.float64).tensor(),
        inliers=60,
        inlier_ratio=0.6,
        translation_sigma_m=0.25,
        rotation_sigma_deg=10.0,
        information_scale=10000.0,
    )
    self.assertAlmostEqual(float(factor.information[0, 0]), 160000.0)
    self.assertAlmostEqual(
        float(factor.information[3, 3]),
        10000.0 / float(torch.deg2rad(torch.tensor(10.0))) ** 2,
        places=3,
    )
```

- [ ] **Step 2: Verify RED**

```bash
/home/zzh/MACVO/.venv/bin/python -m unittest \
  Scripts.UnitTest.test_sparse_loop_geometry.SparseLoopValidationTests.test_scales_conservative_sparse_factor_information -v
```

Expected: `conservative_sparse_factor` rejects the unknown argument.

- [ ] **Step 3: Implement the optional scale**

Add `information_scale=1.0`, reject nonfinite or nonpositive values, and multiply
the complete diagonal information matrix by the scale. Existing callers retain
identical output.

- [ ] **Step 4: Verify GREEN**

Run the Step 2 command and the complete sparse geometry test module. Expected:
all tests pass.

- [ ] **Step 5: Commit**

```bash
git add Module/LoopClosure/SparseGeometry.py Scripts/UnitTest/test_sparse_loop_geometry.py
git commit -m "feat: scale sparse loop information"
```

### Task 2: Couple Switch Prior, Configuration, and Diagnostics

**Files:**
- Modify: `Scripts/UnitTest/test_online_loop_window_macvo.py`
- Modify: `Scripts/UnitTest/test_compare_online_orb_loop.py`
- Modify: `Odometry/OnlineLoopWindowMACVO.py`
- Modify: `Config/Experiment/MACVO/MACVO_Fast_WindowICP_ORBLoop_Sparse.yaml`

- [ ] **Step 1: Write failing online scaling tests**

Extend the direct-factor test with:

```python
system.loop_information_scale = 10000.0
...
self.assertAlmostEqual(float(stored.information[0, 0]), 160000.0)
```

Add a helper/API test:

```python
system.switch_prior = 1.0
system.loop_information_scale = 10000.0
self.assertEqual(system._effective_switch_prior(), 10000.0)
```

Extend diagnostic assertions:

```python
self.assertEqual(payload["loop_information_scale"], 10000.0)
self.assertEqual(payload["switch_prior_base"], 1.0)
self.assertEqual(payload["switch_prior_effective"], 10000.0)
```

Add a configuration test proving the sparse YAML selects 10000 while the legacy
YAML defaults to one through `getattr` semantics.

- [ ] **Step 2: Verify RED**

```bash
/home/zzh/MACVO/.venv/bin/python -m unittest \
  Scripts.UnitTest.test_online_loop_window_macvo \
  Scripts.UnitTest.test_compare_online_orb_loop -v
```

Expected: missing scale/effective-prior assertions fail.

- [ ] **Step 3: Implement coupled online scaling**

In `__init__`:

```python
self.loop_information_scale = float(
    getattr(pose_graph, "loop_information_scale", 1.0)
)
```

Reject values that are nonfinite or nonpositive. Add:

```python
def _effective_switch_prior(self):
    return self.switch_prior * self.loop_information_scale
```

Use this helper in the asynchronous solver closure. Pass
`information_scale=self.loop_information_scale` to
`conservative_sparse_factor`. Record base, scale, and effective values in the
diagnostic payload. Set the sparse YAML value to `10000.0`; do not edit the
legacy YAML.

- [ ] **Step 4: Verify GREEN**

Run the Step 2 command, direct configuration loading, and pose-graph tests.
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add Odometry/OnlineLoopWindowMACVO.py \
  Config/Experiment/MACVO/MACVO_Fast_WindowICP_ORBLoop_Sparse.yaml \
  Scripts/UnitTest/test_online_loop_window_macvo.py \
  Scripts/UnitTest/test_compare_online_orb_loop.py
git commit -m "feat: couple sparse loop and switch scaling"
```

### Task 3: Verify and Run Complete MH04

**Files:**
- Create: `docs/validation/sparse_orb_loop_mh04_full.csv`
- Preserve: existing unstaged `docs/WindowICP.md`

- [ ] **Step 1: Run all unit tests**

```bash
/home/zzh/MACVO/.venv/bin/python -m unittest discover \
  -s Scripts/UnitTest -p 'test_*.py' -q
```

Expected: zero failures and errors.

- [ ] **Step 2: Run complete MH04 persistently**

Start a user systemd service with:

```bash
systemd-run --user --unit=macvo-sparse-mh04-alpha10000 \
  --collect \
  --working-directory=/home/zzh/.config/superpowers/worktrees/MACVO/window-icp-global-v03 \
  /home/zzh/MACVO/.venv/bin/python \
  Scripts/Experiment/CompareOnlineORBLoop.py \
  --sequence MH04 --seq-from 0 --seed 0 \
  --modes window_orb_loop_sparse \
  --result-root /home/zzh/MACVO/Results/SparseORBLoop_MH04_alpha10000
```

Monitor the service and result log until metrics are written.

- [ ] **Step 3: Audit result invariants and metrics**

Verify finite poses, quaternion norms, zero newly unrefined windows, protocol 2,
one model/graph, concurrency one, VRAM below 6 GiB, factor counts, switches,
candidate rejection counts, and pose-backend failures.

- [ ] **Step 4: Run same-run remove-loop ablation**

Load `pose_graph_factors.npz`, remove all loop factors, reoptimize adjacent and
skip factors, convert sensor poses through saved `T_BS`, and evaluate the same
reference trajectory. Report loop contribution separately from the online total.

- [ ] **Step 5: Record and commit MH04 validation**

Write online and remove-loop rows to
`docs/validation/sparse_orb_loop_mh04_full.csv`, then:

```bash
git add docs/validation/sparse_orb_loop_mh04_full.csv
git commit -m "docs: record MH04 alpha10000 validation"
```

- [ ] **Step 6: Final verification**

```bash
git diff --check
git status --short --branch
```

Expected: only the pre-existing `docs/WindowICP.md` modification remains
unstaged.
