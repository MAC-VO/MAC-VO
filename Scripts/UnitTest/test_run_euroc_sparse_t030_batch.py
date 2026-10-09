import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "Scripts/run_euroc_sparse_alpha10000_t030_remaining.sh"


def make_fixture(root, sequences):
    required = [
        "Scripts/Experiment/CompareOnlineORBLoop.py",
        "Config/Experiment/MACVO/"
        "MACVO_Fast_WindowICP_ORBLoop_Sparse_t030.yaml",
        "build/orb_bow/macvo_orb_bow",
        "cache/ORBvoc.txt",
        "Model/MACVO_FrontendCov.pth",
    ]
    for relative in required:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture", encoding="utf-8")
    sidecar = root / "build/orb_bow/macvo_orb_bow"
    sidecar.chmod(sidecar.stat().st_mode | stat.S_IXUSR)
    for sequence in sequences:
        data_root = root / "data" / sequence
        data_root.mkdir(parents=True)
        config = root / f"Config/Sequence/EuRoC_{sequence}_local.yaml"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(
            f"type: EuRoC\nname: {sequence}\nargs:\n  root: {data_root}\n",
            encoding="utf-8",
        )


def run_script(*arguments, worktree=None, result_root=None):
    environment = os.environ.copy()
    environment["PYTHON"] = sys.executable
    if worktree is not None:
        environment["WORKTREE"] = str(worktree)
    if result_root is not None:
        environment["RESULT_ROOT"] = str(result_root)
    return subprocess.run(
        [str(SCRIPT), *arguments],
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


class SparseEuRoCT030BatchTests(unittest.TestCase):
    def test_lists_all_non_v203_sequences(self):
        result = run_script("--list-defaults")

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(
            result.stdout.strip(),
            "MH01 MH02 MH03 MH04 MH05 V101 V102 V103 V201 V202",
        )

    def test_dry_run_uses_t030_mode_and_isolated_result_root(self):
        with tempfile.TemporaryDirectory() as directory:
            worktree = Path(directory, "worktree")
            result_root = Path(directory, "results-t030")
            make_fixture(worktree, ("MH04",))

            result = run_script(
                "--dry-run",
                "MH04",
                worktree=worktree,
                result_root=result_root,
            )

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("RUN\tMH04", result.stdout)
        self.assertIn(
            "--modes window_orb_loop_sparse_t030",
            result.stdout,
        )
        self.assertIn(str(result_root), result.stdout)
        self.assertNotIn("--modes window_orb_loop_sparse ", result.stdout)


if __name__ == "__main__":
    unittest.main()
