# EuRoC Sparse Alpha10000 Batch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create, test, and launch a resumable single-GPU shell batch for the nine remaining EuRoC sequences using sparse ORB loops with alpha 10000.

**Architecture:** A Bash driver owns preflight, locking, sequence scheduling, marker validation, per-sequence logs, and TSV summaries. It delegates each actual run to the existing Python comparison script and is launched by a persistent user systemd service.

**Tech Stack:** Bash, flock, systemd user services, Python unittest/subprocess fixtures, existing MACVO experiment runner.

---

### Task 1: Specify Script Behavior with Failing Tests

**Files:**
- Create: `Scripts/UnitTest/test_run_euroc_sparse_batch.py`
- Create: `Scripts/run_euroc_sparse_alpha10000_remaining.sh`

- [ ] **Step 1: Write failing subprocess tests**

Create temporary fixture roots and assert:

```python
result = run_script("--list-defaults")
self.assertEqual(
    result.stdout.strip(),
    "MH01 MH02 MH03 MH05 V101 V102 V103 V201 V202",
)
```

Run `DRY_RUN=1` with positional `V101 V202` and assert only those `RUN` lines
appear. Add fixture tests for missing required files, valid completion marker plus
matching metrics, stale marker without metrics, and `DRY_RUN_FAIL_SEQUENCE=MH01`
continuing to schedule MH02.

- [ ] **Step 2: Verify RED**

```bash
/home/zzh/MACVO/.venv/bin/python -m unittest \
  Scripts.UnitTest.test_run_euroc_sparse_batch -v
```

Expected: script-not-found failures.

- [ ] **Step 3: Create the minimal script skeleton**

Implement argument parsing for `--list-defaults`, `--dry-run`, and positional
sequence names; environment overrides for `WORKTREE`, `PYTHON`, and
`RESULT_ROOT`; deterministic state/log/summary paths; and default sequence
order.

- [ ] **Step 4: Verify the default and override tests**

Run the Step 2 command. Expected: scheduling tests pass while remaining
preflight/resume tests guide Task 2.

### Task 2: Implement Preflight, Resume, Failure Continuation, and Summary

**Files:**
- Modify: `Scripts/run_euroc_sparse_alpha10000_remaining.sh`
- Modify: `Scripts/UnitTest/test_run_euroc_sparse_batch.py`

- [ ] **Step 1: Complete the production script**

Add:

```bash
exec 9>"${RESULT_ROOT}/.batch.lock"
flock -n 9 || exit 3
```

Validate Python, comparison script, sparse configuration, sidecar, vocabulary,
model, sequence configs, and configured data roots. Implement
`has_completed_result` with a Python JSON scan requiring matching sequence and
mode. Skip only with marker plus metrics. Run each sequence through
`CompareOnlineORBLoop.py`, write atomic success/failure markers, continue after
failures, and generate `summary.tsv` plus final array summaries.

Dry-run mode prints commands without GPU work. `DRY_RUN_FAIL_SEQUENCE` simulates
one failure exclusively for tests.

- [ ] **Step 2: Verify GREEN**

```bash
/home/zzh/MACVO/.venv/bin/python -m unittest \
  Scripts.UnitTest.test_run_euroc_sparse_batch -v
bash -n Scripts/run_euroc_sparse_alpha10000_remaining.sh
```

Expected: all script tests pass and shell syntax is valid.

- [ ] **Step 3: Commit**

```bash
git add Scripts/run_euroc_sparse_alpha10000_remaining.sh \
  Scripts/UnitTest/test_run_euroc_sparse_batch.py
git commit -m "feat: add resumable sparse EuRoC batch"
```

### Task 3: Verify and Launch Persistent Batch

**Files:**
- Preserve: `docs/WindowICP.md`
- Results: `/home/zzh/MACVO/Results/SparseORBLoop_EuRoC_alpha10000`

- [ ] **Step 1: Run complete tests**

```bash
/home/zzh/MACVO/.venv/bin/python -m unittest discover \
  -s Scripts/UnitTest -p 'test_*.py' -q
```

Expected: zero failures and errors.

- [ ] **Step 2: Run real preflight in dry-run mode**

```bash
DRY_RUN=1 Scripts/run_euroc_sparse_alpha10000_remaining.sh
```

Expected: all nine sequence commands are scheduled with no missing assets.

- [ ] **Step 3: Launch persistent systemd batch**

```bash
systemd-run --user \
  --unit=macvo-sparse-euroc-alpha10000-remaining \
  --collect \
  --working-directory=/home/zzh/.config/superpowers/worktrees/MACVO/window-icp-global-v03 \
  Scripts/run_euroc_sparse_alpha10000_remaining.sh
```

- [ ] **Step 4: Confirm persistence and first sequence startup**

Verify active service state, batch lock, batch log, state directory, summary
header, and an active first sequence process.

- [ ] **Step 5: Final version-control check**

```bash
git diff --check
git status --short --branch
```

Expected: only the pre-existing `docs/WindowICP.md` modification remains
unstaged.
