# WindowICP No-Skip V203 Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a complete no-skip online WindowICP variant and measure its short-sequence and full-V203 accuracy, runtime, memory, and backend behavior against the existing online ORB result.

**Architecture:** Preserve `OnlineLoopWindowMACVO` and select `skip_matching: false` through a dedicated experiment configuration. Route no-skip tracking through `estimate_window(None, ...)` so the shared frontend retains fixed batch=3 for later bidirectional loop calls, while discarding the padded third-slot output and producing no `skip2` factors.

**Tech Stack:** Python, PyTorch, PyPose, pytest, YAML experiment configs, existing MACVO comparison/evaluation tools.

---

### Task 1: Register the no-skip experiment variant

**Files:**
- Create: `Config/Experiment/MACVO/MACVO_Fast_WindowICP_ORBLoop_NoSkip.yaml`
- Modify: `Scripts/Experiment/CompareOnlineORBLoop.py`
- Modify: `Scripts/UnitTest/test_compare_online_orb_loop.py`
- Modify: `Odometry/WindowMACVO.py`
- Modify: `Scripts/UnitTest/test_window_frontend_routing.py`

- [x] **Step 1: Write the failing configuration test**

Add a test that asserts `MODE_CONFIGS["window_orb_loop_no_skip"]` loads an `OnlineLoopWindowMACVO` configuration with `skip_matching == false` and `online_loop.enabled == true`.

- [x] **Step 2: Run the test and verify RED**

Run:

```bash
PYTHONPATH=. /home/zzh/MACVO/.venv/bin/python -m pytest -q -o addopts='' \
  Scripts/UnitTest/test_compare_online_orb_loop.py
```

Expected: failure because `window_orb_loop_no_skip` is not registered.

- [x] **Step 3: Add the dedicated configuration and mode**

Copy the current online ORB configuration values exactly, changing only:

```yaml
Odometry:
  name: MACVO-Fast-WindowICP5-ORBLoop-NoSkip
  args:
    skip_matching: false
```

Register:

```python
"window_orb_loop_no_skip": (
    ROOT / "Config/Experiment/MACVO/"
    "MACVO_Fast_WindowICP_ORBLoop_NoSkip.yaml"
)
```

Do not add this ablation mode to the normal three-mode default.

Route `skip_matching: false` through:

```python
depth, adjacent, _ = self.Frontend.estimate_window(
    None, frame0.stereo, frame1.stereo
)
return depth, adjacent, None
```

For the CUDA Graph frontend this captures and reuses batch=3; generic frontends retain their pair-estimation fallback.

- [x] **Step 4: Run the targeted tests and verify GREEN**

Run the command from Step 2. Expected: all tests pass.

- [x] **Step 5: Commit the experiment registration**

```bash
git add Config/Experiment/MACVO/MACVO_Fast_WindowICP_ORBLoop_NoSkip.yaml \
  Scripts/Experiment/CompareOnlineORBLoop.py \
  Scripts/UnitTest/test_compare_online_orb_loop.py
git commit -m "exp: add online WindowICP no-skip variant"
```

### Task 2: Run the difficult 120-frame stability gate

**Files:**
- Generated results: `/home/zzh/MACVO/Results/OnlineORBLoop_NoSkip_V203_short/`

- [x] **Step 1: Run no-skip on V203 `[1095, 1215)`**

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. \
  /home/zzh/MACVO/.venv/bin/python \
  Scripts/Experiment/CompareOnlineORBLoop.py \
  --sequence V203 --seq-from 1095 --seq-to 1215 --seed 0 \
  --modes window_orb_loop_no_skip \
  --result-root /home/zzh/MACVO/Results/OnlineORBLoop_NoSkip_V203_short
```

- [x] **Step 2: Verify the stability gate**

Require 120 finite poses, zero `skip2` factors, zero unrefined windows, one model, one CUDA Graph, maximum GPU concurrency one, and peak reserved VRAM below 6 GiB.

- [x] **Step 3: Compare short metrics**

Compare against the existing skip-enabled `[1095,1215)` result:

```text
/home/zzh/MACVO/Results/OnlineORBLoop_V203_short_v2/20260914_230552_38b2db
```

Proceed to the full sequence only if the stability gate passes. Accuracy is recorded but is not used to retune parameters.

### Task 3: Run and evaluate complete V203

**Files:**
- Generated results: `/home/zzh/MACVO/Results/OnlineORBLoop_NoSkip_V203_full/`

- [x] **Step 1: Run all 1865 frames**

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. \
  /home/zzh/MACVO/.venv/bin/python \
  Scripts/Experiment/CompareOnlineORBLoop.py \
  --sequence V203 --seq-from 0 --seed 0 \
  --modes window_orb_loop_no_skip \
  --result-root /home/zzh/MACVO/Results/OnlineORBLoop_NoSkip_V203_full
```

- [x] **Step 2: Verify complete result integrity**

Require provenance status `complete`, exactly 1865 timestamps matching the skip-enabled baseline, finite poses, normalized quaternions, zero `skip2` factors, passing online invariants, and finite metrics.

- [x] **Step 3: Calculate changes against skip-enabled online ORB**

Use the existing baseline:

```text
ATE RMSE: 0.465288823 m
mean runtime: 1127.039 ms/frame
P95 runtime: 1911.363 ms/frame
peak CUDA reserved memory: 4.048828 GiB
```

Report absolute and percentage changes for ATE/RTE/ROE/RPE and runtime.

### Task 4: Version the evaluation and verify the branch

**Files:**
- Create: `docs/validation/online_orb_loop_no_skip_v203.csv`
- Modify: `docs/WindowICP.md`

- [x] **Step 1: Add exact short and full metrics**

Record result paths, commit, factor counts, loop switch counts, runtime, memory, and interpretation. Explicitly state that this measures the total local-plus-global contribution of skip constraints.

- [x] **Step 2: Run all non-local tests and sidecar test**

```bash
PYTHONPATH=. /home/zzh/MACVO/.venv/bin/python -m pytest -q \
  -o addopts='' -m 'not local' Scripts/UnitTest
PYTHONPATH=. /home/zzh/MACVO/.venv/bin/python -m pytest -q \
  -o addopts='' Scripts/UnitTest/test_orb_bow_sidecar.py
git diff --check
```

Expected: all tests pass and no whitespace errors.

- [x] **Step 3: Commit the evaluation**

```bash
git add docs/WindowICP.md \
  docs/validation/online_orb_loop_no_skip_v203.csv
git commit -m "eval: compare no-skip online WindowICP on V203"
```

- [x] **Step 4: Confirm isolation**

Require a clean `experiment/window-icp-global-v03` worktree. Do not merge or push automatically.
