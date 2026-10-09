# Async Pose Graph Shutdown Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the production 30-second pose-graph shutdown deadline while preserving an explicit timeout option for tests and diagnostics.

**Architecture:** Extend `AsyncPoseGraphBackend.close()` and `terminate()` with an optional timeout. The default blocks until the one worker drains its active solve and newest pending final snapshot; finite caller-provided timeouts retain the existing failure signal.

**Tech Stack:** Python threading, condition variables, unittest/pytest, existing sparse PyPose backend.

---

### Task 1: Specify blocking and finite-timeout shutdown behavior

**Files:**
- Modify: `Scripts/UnitTest/test_async_pose_graph.py`

- [x] **Step 1: Add a failing default-drain test**

Create a two-call blocking solver. Start snapshot 1, call `terminate(snapshot(9), timeout=None)` on another thread, verify termination remains blocked until call 1 is released, then verify snapshot 9 is also solved and the worker exits.

- [x] **Step 2: Add a failing explicit-timeout test**

Start a blocked solve and call `close(timeout=0.01)`. Require `TimeoutError`; release the solver and require a later `close(timeout=1)` to succeed.

- [x] **Step 3: Run RED**

```bash
PYTHONPATH=. /home/zzh/MACVO/.venv/bin/python -m pytest -q -o addopts='' \
  Scripts/UnitTest/test_async_pose_graph.py
```

Expected: failure because `close` and `terminate` do not accept `timeout`.

### Task 2: Implement optional shutdown timeout

**Files:**
- Modify: `Module/Optimization/AsyncPoseGraph.py`

- [x] **Step 1: Extend `close`**

Implement:

```python
def close(self, timeout=None):
    with self._condition:
        self._stopping = True
        self._condition.notify_all()
    self._thread.join(timeout=timeout)
    if self._thread.is_alive():
        raise TimeoutError("pose graph backend did not stop")
```

- [x] **Step 2: Extend `terminate`**

Implement:

```python
def terminate(self, final_snapshot=None, timeout=None):
    if final_snapshot is not None:
        self.submit(final_snapshot)
    self.close(timeout=timeout)
```

- [x] **Step 3: Run GREEN**

Run the Task 1 test command. Expected: all async backend tests pass.

- [x] **Step 4: Run online termination regression tests**

```bash
PYTHONPATH=. /home/zzh/MACVO/.venv/bin/python -m pytest -q -o addopts='' \
  Scripts/UnitTest/test_online_loop_writeback.py \
  Scripts/UnitTest/test_online_loop_window_macvo.py
```

- [x] **Step 5: Commit**

```bash
git add Module/Optimization/AsyncPoseGraph.py \
  Scripts/UnitTest/test_async_pose_graph.py
git commit -m "fix: drain pose graph backend on shutdown"
```

### Task 3: Validate and document MH05 recovery

**Files:**
- Modify: `docs/WindowICP.md` only if it does not overlap the existing user edit

- [x] **Step 1: Run all non-local tests**

```bash
PYTHONPATH=. /home/zzh/MACVO/.venv/bin/python -m pytest -q \
  -o addopts='' -m 'not local' Scripts/UnitTest
PYTHONPATH=. /home/zzh/MACVO/.venv/bin/python -m pytest -q \
  -o addopts='' Scripts/UnitTest/test_orb_bow_sidecar.py
```

- [x] **Step 2: Replay the MH05 archive solver**

Load the existing 2192-pose, 4463-factor archive and solve ten iterations. Require a refined finite result and record the solve duration. This verifies the workload that previously participated in the two-solve shutdown sequence.

- [x] **Step 3: Preserve user changes**

Do not stage or modify the existing uncommitted MH04 command in `docs/WindowICP.md`. If documenting the shutdown fix would overlap that file, report the result without editing it.

- [x] **Step 4: Confirm isolation**

Keep `experiment/window-icp-global-v03` and its worktree. Do not merge, push, or clean up automatically.
