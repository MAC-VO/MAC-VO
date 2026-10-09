import threading
import time
import unittest

import pypose as pp

from Module.Optimization.AsyncPoseGraph import (
    AsyncPoseGraphBackend,
    PoseGraphSnapshot,
)


def snapshot(version):
    return PoseGraphSnapshot(
        graph_version=version,
        frame_count=2,
        poses=pp.identity_SE3(2).tensor(),
        factors=(),
    )


class BlockingSolver:
    def __init__(self):
        self.calls = []
        self.started = threading.Event()
        self.release = threading.Event()

    def __call__(self, poses, factors):
        self.calls.append(len(self.calls))
        self.started.set()
        self.release.wait(timeout=2)
        return {"poses": poses, "status": "refined"}


class SequencedBlockingSolver:
    def __init__(self, calls=2):
        self.started = [threading.Event() for _ in range(calls)]
        self.release = [threading.Event() for _ in range(calls)]
        self._lock = threading.Lock()
        self.versions = []

    def __call__(self, poses, factors):
        with self._lock:
            index = len(self.versions)
            self.versions.append(len(poses))
        self.started[index].set()
        self.release[index].wait(timeout=2)
        return {"poses": poses, "status": "refined"}


class AsyncPoseGraphTests(unittest.TestCase):
    def test_busy_backend_coalesces_to_newest_pending_snapshot(self):
        solver = BlockingSolver()
        backend = AsyncPoseGraphBackend(solver=solver)
        backend.submit(snapshot(1))
        self.assertTrue(solver.started.wait(timeout=1))
        backend.submit(snapshot(2))
        backend.submit(snapshot(3))
        solver.release.set()
        deadline = time.time() + 2
        results = []
        while len(results) < 2 and time.time() < deadline:
            results.extend(backend.poll())
            time.sleep(0.01)
        backend.close()

        self.assertEqual([result.graph_version for result in results], [1, 3])
        self.assertEqual(backend.diagnostics["coalesced_submissions"], 1)

    def test_solver_exception_becomes_failed_result(self):
        backend = AsyncPoseGraphBackend(
            solver=lambda *_: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        backend.submit(snapshot(4))
        deadline = time.time() + 2
        results = []
        while not results and time.time() < deadline:
            results = backend.poll()
            time.sleep(0.01)
        backend.close()

        self.assertEqual(results[0].status, "failed")
        self.assertIn("boom", results[0].reason)

    def test_terminate_solves_final_newest_snapshot(self):
        versions = []

        def solver(poses, factors):
            versions.append(len(poses))
            return {"poses": poses, "status": "refined"}

        backend = AsyncPoseGraphBackend(solver=solver)
        backend.submit(snapshot(1))
        backend.terminate(snapshot(9))
        results = backend.poll()

        self.assertEqual(results[-1].graph_version, 9)
        self.assertFalse(backend.is_alive)

    def test_default_terminate_waits_for_active_and_final_snapshot(self):
        solver = SequencedBlockingSolver()
        backend = AsyncPoseGraphBackend(solver=solver)
        backend.submit(snapshot(1))
        self.assertTrue(solver.started[0].wait(timeout=1))
        errors = []

        def terminate():
            try:
                backend.terminate(snapshot(9), timeout=None)
            except Exception as error:
                errors.append(error)

        thread = threading.Thread(target=terminate)
        thread.start()
        time.sleep(0.02)
        self.assertTrue(thread.is_alive())
        solver.release[0].set()
        self.assertTrue(solver.started[1].wait(timeout=1))
        self.assertTrue(thread.is_alive())
        solver.release[1].set()
        thread.join(timeout=1)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(
            [result.graph_version for result in backend.poll()], [1, 9]
        )
        self.assertFalse(backend.is_alive)

    def test_explicit_close_timeout_remains_available(self):
        solver = BlockingSolver()
        backend = AsyncPoseGraphBackend(solver=solver)
        backend.submit(snapshot(1))
        self.assertTrue(solver.started.wait(timeout=1))

        with self.assertRaisesRegex(TimeoutError, "did not stop"):
            backend.close(timeout=0.01)

        solver.release.set()
        backend.close(timeout=1)
        self.assertFalse(backend.is_alive)


if __name__ == "__main__":
    unittest.main()
