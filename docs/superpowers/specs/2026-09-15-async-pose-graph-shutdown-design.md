# Async Pose Graph Shutdown Design

## Objective

Ensure long sequences always finish their final pose-graph optimization instead of failing because active and final solves exceed a fixed 30-second shutdown timeout.

## Current Failure

`AsyncPoseGraphBackend.terminate(final_snapshot)` submits the final snapshot and immediately calls `close()`. If a solve is already active, the worker must finish that solve and then process the final pending snapshot before exiting. `close()` currently applies one 30-second join timeout to both operations.

MH05 reproduced the failure with an active solve of about 20 seconds followed by a final solve of about 21 seconds. Tracking, local windows, and the complete factor archive were valid; only the shutdown join timed out.

## Selected Behavior

Change the API to:

```python
def close(self, timeout=None): ...
def terminate(self, final_snapshot=None, timeout=None): ...
```

Production callers use the default `timeout=None`, which waits until the worker drains the active solve and newest pending snapshot and exits. Tests and diagnostic callers may pass a finite timeout; if the worker remains alive after that interval, `TimeoutError` is raised as before.

The final snapshot remains the newest pending snapshot. Existing coalescing behavior is preserved: an older pending snapshot may be replaced, but the already-active solve is allowed to finish safely. No solver thread is cancelled and no second synchronous solver is started.

## State and Error Semantics

- `close()` sets `_stopping` and wakes the worker.
- If work is active or pending, the worker finishes it before exiting.
- A finite explicit timeout raises `TimeoutError` without corrupting the worker; callers may release the solver and call `close()` again.
- Once stopping begins, new `submit()` calls remain rejected.
- Solver exceptions continue to become failed asynchronous results and do not prevent worker shutdown.
- Existing odometry termination remains idempotent through its `self.terminated` guard.

## Tests

Add tests that verify:

1. `terminate(final_snapshot, timeout=None)` waits for a blocked active solve and then processes the final snapshot.
2. The final result is the submitted final graph version and the worker exits.
3. `close(timeout=<small value>)` still raises for a deliberately blocked solver.
4. After the solver is released, a second blocking close succeeds.
5. Existing newest-pending coalescing and exception conversion tests continue to pass.

## Validation

Run all non-local unit tests and the ORB sidecar test. Then replay an MH05-sized archive termination scenario using the existing 2192-pose, 4463-factor archive and confirm the full solve duration can exceed the old fixed deadline without causing a shutdown error.

The previous failed MH05 run will not be rewritten or marked complete. Its recovered accuracy remains explicitly identified as archive-based evaluation.

## Scope

Only asynchronous backend shutdown semantics and tests change. Pose-graph mathematics, factor selection, ORB matching, GPU scheduling, and trajectory writeback are unchanged. Work remains on the isolated `experiment/window-icp-global-v03` branch.
