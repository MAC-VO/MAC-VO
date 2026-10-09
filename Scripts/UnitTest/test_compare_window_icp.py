import io
from pathlib import Path
import sys
import tempfile
import unittest
import json

from Scripts.Experiment.CompareWindowICP import (
    MODE_CONFIGS,
    append_window_diagnostics,
    run_and_tee,
)


class LiveOutputTests(unittest.TestCase):
    def test_subprocess_output_is_written_to_terminal_and_log(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory, "run.log")
            terminal = io.BytesIO()
            run_and_tee(
                [sys.executable, "-c", "import sys; sys.stdout.write('frame 1\\rframe 2\\n'); sys.stdout.flush()"],
                Path(directory), None, log, terminal,
            )
            expected = b"frame 1\rframe 2\n"
            self.assertEqual(terminal.getvalue(), expected)
            self.assertEqual(log.read_bytes(), expected)

    def test_global_mode_selects_global_config(self):
        self.assertEqual(
            MODE_CONFIGS["window_global"].name,
            "MACVO_Fast_WindowICP_Global.yaml",
        )

    def test_global_diagnostics_are_exported(self):
        with tempfile.TemporaryDirectory() as directory:
            space = Path(directory)
            payload = {
                "windows": [
                    {
                        "status": "refined",
                        "seconds": 0.02,
                        "skip_seconds": 0.01,
                        "cached_frames": 5,
                        "cached_edges": 7,
                    }
                ],
                "inactive_edges": 12,
                "retained_tensor_bytes": 3456,
                "global_refinement": {
                    "status": "refined",
                    "seconds": 0.5,
                    "initial_cost": 20.0,
                    "final_cost": 10.0,
                },
            }
            (space / "window_diagnostics.json").write_text(json.dumps(payload))

            row = append_window_diagnostics({}, space, "window_global")

        self.assertEqual(row["inactive_edges"], 12)
        self.assertEqual(row["retained_tensor_bytes"], 3456)
        self.assertEqual(row["global_status"], "refined")
        self.assertEqual(row["global_seconds"], 0.5)
        self.assertEqual(row["global_initial_cost"], 20.0)
        self.assertEqual(row["global_final_cost"], 10.0)


if __name__ == "__main__":
    unittest.main()
