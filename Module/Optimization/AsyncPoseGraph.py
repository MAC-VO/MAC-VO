"""One-worker asynchronous pose-graph execution with newest-job coalescing."""

from dataclasses import dataclass
import queue
import threading
import time

import torch

from Module.Optimization.PoseGraph import optimize_pose_graph


@dataclass(frozen=True)
class PoseGraphSnapshot:
    graph_version: int
    frame_count: int
    poses: torch.Tensor
    factors: tuple

    def __post_init__(self):
        object.__setattr__(
            self,
            "poses",
            torch.as_tensor(self.poses).detach().cpu().double().clone(),
        )
        object.__setattr__(self, "factors", tuple(self.factors))


@dataclass(frozen=True)
class PoseGraphAsyncResult:
    graph_version: int
    frame_count: int
    status: str
    result: object | None
    reason: str | None
    seconds: float


class AsyncPoseGraphBackend:
    def __init__(self, solver=None):
        self.solver = optimize_pose_graph if solver is None else solver
        self._condition = threading.Condition()
        self._pending = None
        self._active = False
        self._stopping = False
        self._results = queue.Queue()
        self._submitted = 0
        self._completed = 0
        self._coalesced = 0
        self._failures = 0
        self._thread = threading.Thread(
            target=self._run,
            name="AsyncPoseGraph",
            daemon=True,
        )
        self._thread.start()

    @property
    def is_alive(self):
        return self._thread.is_alive()

    @property
    def diagnostics(self):
        return {
            "submitted": self._submitted,
            "completed": self._completed,
            "coalesced_submissions": self._coalesced,
            "failures": self._failures,
            "active": self._active,
            "pending": self._pending is not None,
        }

    def submit(self, snapshot):
        if not isinstance(snapshot, PoseGraphSnapshot):
            raise TypeError("snapshot must be a PoseGraphSnapshot")
        with self._condition:
            if self._stopping:
                raise RuntimeError("pose graph backend is stopping")
            if self._pending is not None:
                self._coalesced += 1
            self._pending = snapshot
            self._submitted += 1
            self._condition.notify_all()

    def _run(self):
        while True:
            with self._condition:
                while self._pending is None and not self._stopping:
                    self._condition.wait()
                if self._pending is None and self._stopping:
                    return
                snapshot = self._pending
                self._pending = None
                self._active = True
            start = time.perf_counter()
            try:
                solved = self.solver(snapshot.poses, snapshot.factors)
                status = (
                    solved.get("status", "refined")
                    if isinstance(solved, dict)
                    else solved.diagnostics.get("status", "refined")
                )
                result = PoseGraphAsyncResult(
                    snapshot.graph_version,
                    snapshot.frame_count,
                    status,
                    solved,
                    None,
                    time.perf_counter() - start,
                )
            except Exception as error:
                self._failures += 1
                result = PoseGraphAsyncResult(
                    snapshot.graph_version,
                    snapshot.frame_count,
                    "failed",
                    None,
                    str(error),
                    time.perf_counter() - start,
                )
            self._results.put(result)
            with self._condition:
                self._completed += 1
                self._active = False
                self._condition.notify_all()

    def poll(self):
        results = []
        while True:
            try:
                results.append(self._results.get_nowait())
            except queue.Empty:
                return results

    def close(self, timeout=None):
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            raise TimeoutError("pose graph backend did not stop")

    def terminate(self, final_snapshot=None, timeout=None):
        if final_snapshot is not None:
            self.submit(final_snapshot)
        self.close(timeout=timeout)
