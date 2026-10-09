import threading
import time
import unittest

from Module.Frontend.SerializedFrontend import SerializedFrontend


class FakeFrontend:
    def __init__(self):
        self.model = object()
        self.cuda_graph = object()
        self.active = 0
        self.max_active = 0
        self.order = []
        self.started = threading.Event()
        self.release = threading.Event()

    @property
    def provide_cov(self):
        return True, True

    def _call(self, name, block=False):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.order.append(name)
        self.started.set()
        if block:
            self.release.wait(timeout=2)
        else:
            time.sleep(0.02)
        self.active -= 1
        return name

    def estimate_depth(self, *_):
        return self._call("depth")

    def estimate_pair(self, *_):
        return self._call("pair")

    def estimate_window(self, *args):
        return self._call(args[0], block=args[0] == "block")

    def estimate_bidirectional(self, *args):
        return self._call(args[0])

    def estimate_loop_depth(self, *args):
        return self._call(args[0])


class SerializedFrontendTests(unittest.TestCase):
    def test_forced_contention_never_runs_two_frontend_calls(self):
        frontend = FakeFrontend()
        wrapper = SerializedFrontend(frontend)
        threads = [
            threading.Thread(
                target=wrapper.estimate_bidirectional,
                args=(f"loop-{index}", "target"),
            )
            for index in range(4)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(frontend.max_active, 1)
        self.assertEqual(wrapper.diagnostics["max_concurrent_calls"], 1)

    def test_waiting_tracking_call_runs_before_waiting_loop_call(self):
        frontend = FakeFrontend()
        wrapper = SerializedFrontend(frontend)
        blocker = threading.Thread(
            target=wrapper.estimate_window,
            args=("block", None, None),
        )
        blocker.start()
        self.assertTrue(frontend.started.wait(timeout=1))
        loop = threading.Thread(
            target=wrapper.estimate_bidirectional,
            args=("loop", "target"),
        )
        tracking = threading.Thread(
            target=wrapper.estimate_window,
            args=("tracking", None, None),
        )
        loop.start()
        time.sleep(0.02)
        tracking.start()
        time.sleep(0.02)
        frontend.release.set()
        blocker.join()
        tracking.join()
        loop.join()

        self.assertEqual(frontend.order, ["block", "tracking", "loop"])

    def test_wrapper_preserves_model_and_graph_identity(self):
        frontend = FakeFrontend()
        wrapper = SerializedFrontend(frontend)

        wrapper.estimate_window("tracking", None, None)
        wrapper.estimate_bidirectional("loop", "target")

        self.assertEqual(wrapper.model_identity, id(frontend.model))
        self.assertEqual(wrapper.graph_identity, id(frontend.cuda_graph))
        self.assertEqual(wrapper.diagnostics["tracking_calls"], 1)
        self.assertEqual(wrapper.diagnostics["loop_calls"], 1)

    def test_sparse_loop_depth_is_serialized_as_loop_work(self):
        frontend = FakeFrontend()
        wrapper = SerializedFrontend(frontend)

        result = wrapper.estimate_loop_depth("sparse-depth")

        self.assertEqual(result, "sparse-depth")
        self.assertEqual(wrapper.diagnostics["tracking_calls"], 0)
        self.assertEqual(wrapper.diagnostics["loop_calls"], 1)


if __name__ == "__main__":
    unittest.main()
