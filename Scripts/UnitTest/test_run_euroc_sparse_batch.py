import json
import os
from pathlib import Path
import fcntl
import stat
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "Scripts/run_euroc_sparse_alpha10000_remaining.sh"


def make_fixture(root, sequences=("MH01",)):
    required = [
        "Scripts/Experiment/CompareOnlineORBLoop.py",
        "Config/Experiment/MACVO/MACVO_Fast_WindowICP_ORBLoop_Sparse.yaml",
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


def run_script(*arguments, worktree=None, result_root=None, updates=None):
    environment = os.environ.copy()
    if worktree is not None:
        environment["WORKTREE"] = str(worktree)
    if result_root is not None:
        environment["RESULT_ROOT"] = str(result_root)
    environment["PYTHON"] = sys.executable
    if updates:
        environment.update(updates)
    return subprocess.run(
        [str(SCRIPT), *arguments],
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def make_success_runner(worktree):
    runner = worktree / "Scripts/Experiment/CompareOnlineORBLoop.py"
    runner.write_text(
        """\
import argparse, json
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--sequence')
p.add_argument('--seq-from')
p.add_argument('--seed')
p.add_argument('--modes', nargs='+')
p.add_argument('--result-root', type=Path)
a=p.parse_args()
folder=a.result_root / ('fixture-' + a.sequence)
folder.mkdir(parents=True, exist_ok=True)
(folder/'metrics.json').write_text(json.dumps([{
    'sequence': a.sequence, 'mode': 'window_orb_loop_sparse',
    'RMSE_ATE': 1, 'RMSE_RTE': 2, 'RMSE_ROE': 3, 'RMSE_RPE': 4,
    'runtime_mean_ms': 5, 'accepted_loops': 6,
    'effective_long_loops_001': 7, 'peak_vram_bytes': 8,
}]))
""",
        encoding="utf-8",
    )


class SparseEuRoCBatchTests(unittest.TestCase):
    def test_lists_default_remaining_sequences_in_order(self):
        result = run_script("--list-defaults")

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(
            result.stdout.strip(),
            "MH01 MH02 MH03 MH05 V101 V102 V103 V201 V202",
        )

    def test_positional_sequences_override_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            worktree = Path(directory, "worktree")
            result_root = Path(directory, "results")
            make_fixture(worktree, ("V101", "V202"))

            result = run_script(
                "--dry-run", "V101", "V202",
                worktree=worktree, result_root=result_root,
            )

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("RUN\tV101", result.stdout)
        self.assertIn("RUN\tV202", result.stdout)
        self.assertNotIn("MH01", result.stdout)

    def test_missing_required_file_fails_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            worktree = Path(directory, "worktree")
            result_root = Path(directory, "results")
            make_fixture(worktree)
            (worktree / "Model/MACVO_FrontendCov.pth").unlink()

            result = run_script(
                "--dry-run", "MH01",
                worktree=worktree, result_root=result_root,
            )

        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("MISSING", result.stdout)

    def test_real_preflight_logs_missing_file_in_fresh_result_root(self):
        with tempfile.TemporaryDirectory() as directory:
            worktree = Path(directory, "worktree")
            result_root = Path(directory, "new-results")
            make_fixture(worktree)
            (worktree / "Model/MACVO_FrontendCov.pth").unlink()

            result = run_script(
                "MH01", worktree=worktree, result_root=result_root,
            )

            batch_log = result_root / "batch.log"
            self.assertTrue(batch_log.exists())
            self.assertIn("MISSING", batch_log.read_text(encoding="utf-8"))
        self.assertEqual(result.returncode, 2, result.stdout)

    def test_marker_and_matching_metrics_skip_sequence(self):
        with tempfile.TemporaryDirectory() as directory:
            worktree = Path(directory, "worktree")
            result_root = Path(directory, "results")
            make_fixture(worktree)
            state = result_root / ".batch_state"
            state.mkdir(parents=True)
            (state / "MH01.complete").write_text("complete\n")
            run = result_root / "run-one"
            run.mkdir()
            (run / "metrics.json").write_text(json.dumps([{
                "sequence": "MH01", "mode": "window_orb_loop_sparse"
            }]))

            result = run_script(
                "--dry-run", "MH01",
                worktree=worktree, result_root=result_root,
            )

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("SKIP\tMH01", result.stdout)
        self.assertNotIn("RUN\tMH01", result.stdout)

    def test_stale_marker_without_metrics_does_not_skip(self):
        with tempfile.TemporaryDirectory() as directory:
            worktree = Path(directory, "worktree")
            result_root = Path(directory, "results")
            make_fixture(worktree)
            state = result_root / ".batch_state"
            state.mkdir(parents=True)
            (state / "MH01.complete").write_text("stale\n")

            result = run_script(
                "--dry-run", "MH01",
                worktree=worktree, result_root=result_root,
            )

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("RUN\tMH01", result.stdout)
        self.assertNotIn("SKIP\tMH01", result.stdout)

    def test_dry_run_failure_continues_to_next_sequence(self):
        with tempfile.TemporaryDirectory() as directory:
            worktree = Path(directory, "worktree")
            result_root = Path(directory, "results")
            make_fixture(worktree, ("MH01", "MH02"))

            result = run_script(
                "--dry-run", "MH01", "MH02",
                worktree=worktree,
                result_root=result_root,
                updates={"DRY_RUN_FAIL_SEQUENCE": "MH01"},
            )

        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("FAIL\tMH01", result.stdout)
        self.assertIn("RUN\tMH02", result.stdout)

    def test_dry_run_does_not_modify_existing_summary_or_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            worktree = Path(directory, "worktree")
            result_root = Path(directory, "results")
            make_fixture(worktree)
            result_root.mkdir()
            summary = result_root / "summary.tsv"
            summary.write_text("preserved\n", encoding="utf-8")

            result = run_script(
                "--dry-run", "MH01",
                worktree=worktree, result_root=result_root,
            )

            self.assertEqual(summary.read_text(encoding="utf-8"), "preserved\n")
            self.assertFalse((result_root / "batch.log").exists())
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_malformed_metrics_do_not_abort_completion_scan(self):
        with tempfile.TemporaryDirectory() as directory:
            worktree = Path(directory, "worktree")
            result_root = Path(directory, "results")
            make_fixture(worktree)
            state = result_root / ".batch_state"
            state.mkdir(parents=True)
            (state / "MH01.complete").write_text("stale\n")
            run = result_root / "unrelated"
            run.mkdir()
            (run / "metrics.json").write_text('{"not": "a list"}')

            result = run_script(
                "--dry-run", "MH01",
                worktree=worktree, result_root=result_root,
            )

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("RUN\tMH01", result.stdout)

    def test_failed_rerun_removes_stale_complete_and_writes_failed_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            worktree = Path(directory, "worktree")
            result_root = Path(directory, "results")
            make_fixture(worktree)
            state = result_root / ".batch_state"
            state.mkdir(parents=True)
            complete = state / "MH01.complete"
            complete.write_text("stale\n")

            result = run_script(
                "MH01", worktree=worktree, result_root=result_root,
            )

            self.assertFalse(complete.exists())
            self.assertTrue((state / "MH01.failed").exists())
        self.assertEqual(result.returncode, 1, result.stdout)

    def test_success_without_metrics_is_marked_failed(self):
        with tempfile.TemporaryDirectory() as directory:
            worktree = Path(directory, "worktree")
            result_root = Path(directory, "results")
            make_fixture(worktree)
            runner = worktree / "Scripts/Experiment/CompareOnlineORBLoop.py"
            runner.write_text("raise SystemExit(0)\n", encoding="utf-8")

            result = run_script(
                "MH01", worktree=worktree, result_root=result_root,
            )

            state = result_root / ".batch_state"
            self.assertFalse((state / "MH01.complete").exists())
            self.assertTrue((state / "MH01.failed").exists())
        self.assertEqual(result.returncode, 1, result.stdout)

    def test_success_appends_complete_summary_row_and_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            worktree = Path(directory, "worktree")
            result_root = Path(directory, "results")
            make_fixture(worktree)
            make_success_runner(worktree)

            result = run_script(
                "MH01", worktree=worktree, result_root=result_root,
            )

            lines = (result_root / "summary.tsv").read_text().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(len(lines[1].split("\t")), 11)
            self.assertTrue((result_root / ".batch_state/MH01.complete").exists())
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_exclusive_lock_rejects_second_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            worktree = Path(directory, "worktree")
            result_root = Path(directory, "results")
            make_fixture(worktree)
            result_root.mkdir()
            lock_stream = (result_root / ".batch.lock").open("w")
            fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                result = run_script(
                    "MH01", worktree=worktree, result_root=result_root,
                )
            finally:
                lock_stream.close()

        self.assertEqual(result.returncode, 3, result.stdout)
        self.assertIn("LOCKED", result.stdout)

    def test_summary_write_failure_does_not_leave_complete_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            worktree = Path(directory, "worktree")
            result_root = Path(directory, "results")
            make_fixture(worktree)
            make_success_runner(worktree)
            result_root.mkdir()
            (result_root / "summary.tsv").mkdir()

            result = run_script(
                "MH01", worktree=worktree, result_root=result_root,
            )

            self.assertFalse(
                (result_root / ".batch_state/MH01.complete").exists()
            )
        self.assertNotEqual(result.returncode, 0, result.stdout)


if __name__ == "__main__":
    unittest.main()
