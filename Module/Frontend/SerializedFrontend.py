"""Single-owner synchronization for tracking and loop frontend calls."""

from contextlib import contextmanager
import threading
import time


class SerializedFrontend:
    def __init__(self, frontend):
        self.frontend = frontend
        self._condition = threading.Condition(threading.RLock())
        self._active = False
        self._waiting_tracking = 0
        self._active_calls = 0
        self._max_concurrent = 0
        self._tracking_calls = 0
        self._loop_calls = 0
        self._tracking_seconds = 0.0
        self._loop_seconds = 0.0

    def __getattr__(self, name):
        return getattr(self.frontend, name)

    @property
    def model_identity(self):
        return id(getattr(self.frontend, "model", self.frontend))

    @property
    def graph_identity(self):
        graph = getattr(self.frontend, "cuda_graph", None)
        return None if graph is None else id(graph)

    @property
    def diagnostics(self):
        return {
            "model_identity": self.model_identity,
            "graph_identity": self.graph_identity,
            "tracking_calls": self._tracking_calls,
            "loop_calls": self._loop_calls,
            "tracking_seconds": self._tracking_seconds,
            "loop_seconds": self._loop_seconds,
            "max_concurrent_calls": self._max_concurrent,
            "wrapped_instances_created": getattr(
                self.frontend.__class__, "instances_created", None
            ),
            "wrapped_graphs_captured": getattr(
                self.frontend.__class__, "graphs_captured", None
            ),
        }

    @contextmanager
    def _call_slot(self, kind):
        tracking = kind == "tracking"
        with self._condition:
            if tracking:
                self._waiting_tracking += 1
            try:
                while self._active or (
                    not tracking and self._waiting_tracking > 0
                ):
                    self._condition.wait()
                self._active = True
                self._active_calls += 1
                self._max_concurrent = max(
                    self._max_concurrent, self._active_calls
                )
            finally:
                if tracking:
                    self._waiting_tracking -= 1
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            with self._condition:
                if tracking:
                    self._tracking_calls += 1
                    self._tracking_seconds += elapsed
                else:
                    self._loop_calls += 1
                    self._loop_seconds += elapsed
                self._active_calls -= 1
                self._active = False
                self._condition.notify_all()

    def estimate_depth(self, *args, **kwargs):
        with self._call_slot("tracking"):
            return self.frontend.estimate_depth(*args, **kwargs)

    def estimate_pair(self, *args, **kwargs):
        with self._call_slot("tracking"):
            return self.frontend.estimate_pair(*args, **kwargs)

    def estimate_window(self, *args, **kwargs):
        with self._call_slot("tracking"):
            return self.frontend.estimate_window(*args, **kwargs)

    def estimate_bidirectional(self, *args, **kwargs):
        with self._call_slot("loop"):
            return self.frontend.estimate_bidirectional(*args, **kwargs)

    def estimate_loop_depth(self, *args, **kwargs):
        with self._call_slot("loop"):
            return self.frontend.estimate_loop_depth(*args, **kwargs)
