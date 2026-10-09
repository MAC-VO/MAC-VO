import os
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "Scripts/run_v203_sparse_alpha10000_t030.sh"


class V203SparseT030RunnerTests(unittest.TestCase):
    def test_dry_run_is_isolated_and_selects_t030_mode(self):
        environment = os.environ.copy()
        environment["WORKTREE"] = str(ROOT)
        result = subprocess.run(
            [str(SCRIPT), "--dry-run"],
            cwd=ROOT,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("--sequence V203", result.stdout)
        self.assertIn("--seq-from 0", result.stdout)
        self.assertIn("--seed 0", result.stdout)
        self.assertIn(
            "--modes window_orb_loop_sparse_t030",
            result.stdout,
        )
        self.assertIn(
            "Results/SparseORBLoop_V203_alpha10000_t030",
            result.stdout,
        )
        self.assertNotIn(
            "Results/SparseORBLoop_EuRoC_alpha10000 ",
            result.stdout,
        )


if __name__ == "__main__":
    unittest.main()
