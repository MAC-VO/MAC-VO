# V203 t030 Isolated Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an isolated `0.30 m` loop-hypothesis experiment and a dedicated script for rerunning full V203 without changing or reusing the existing `0.25 m` baseline.

**Architecture:** Duplicate the proven sparse-loop YAML as an explicitly named t030 variant, register it as a separate comparison mode, and invoke that mode from a small dedicated shell runner with a distinct result root. Unit tests compare the two parsed configurations structurally and verify the runner's dry-run command, so accidental changes beyond the threshold and experiment name are rejected.

**Tech Stack:** Python 3.12, `unittest`, YAML configuration, Bash, existing `CompareOnlineORBLoop.py` experiment harness.

---

## File Structure

- Create `Config/Experiment/MACVO/MACVO_Fast_WindowICP_ORBLoop_Sparse_t030.yaml`: isolated t030 configuration; differs from the baseline only in odometry name and hypothesis translation threshold.
- Modify `Scripts/Experiment/CompareOnlineORBLoop.py`: register the t030 mode and mark it as producing online-loop diagnostics.
- Modify `Scripts/UnitTest/test_compare_online_orb_loop.py`: enforce exact structural equivalence with the baseline after normalizing the two intended differences.
- Create `Scripts/run_v203_sparse_alpha10000_t030.sh`: dedicated full-sequence V203 runner with a separate result root and dry-run support.
- Create `Scripts/UnitTest/test_run_v203_sparse_t030.py`: verify that the runner selects V203, frame zero, seed zero, and the t030 mode/result root.

### Task 1: Register an isolated t030 sparse-loop mode

**Files:**
- Create: `Config/Experiment/MACVO/MACVO_Fast_WindowICP_ORBLoop_Sparse_t030.yaml`
- Modify: `Scripts/Experiment/CompareOnlineORBLoop.py:24-42`
- Modify: `Scripts/UnitTest/test_compare_online_orb_loop.py:84-102`

- [ ] **Step 1: Write the failing configuration-isolation test**

Add `copy` to the imports and add this test to `CompareOnlineORBLoopTests`:

```python
import copy


def test_t030_sparse_mode_changes_only_name_and_translation_threshold(self):
    _, baseline = load_config(MODE_CONFIGS["window_orb_loop_sparse"])
    _, t030 = load_config(MODE_CONFIGS["window_orb_loop_sparse_t030"])

    self.assertEqual(
        t030["Odometry"]["args"]["online_loop"]["hypothesis"]
        ["max_correction_translation_m"],
        0.30,
    )
    self.assertTrue(uses_online_diagnostics("window_orb_loop_sparse_t030"))

    normalized = copy.deepcopy(t030)
    normalized["Odometry"]["name"] = baseline["Odometry"]["name"]
    normalized["Odometry"]["args"]["online_loop"]["hypothesis"] \
        ["max_correction_translation_m"] = 0.25
    self.assertEqual(normalized, baseline)
```

- [ ] **Step 2: Run the test and verify the missing mode fails**

Run:

```bash
/home/zzh/MACVO/.venv/bin/python -m unittest \
  Scripts.UnitTest.test_compare_online_orb_loop.CompareOnlineORBLoopTests.test_t030_sparse_mode_changes_only_name_and_translation_threshold -v
```

Expected: `ERROR` with `KeyError: 'window_orb_loop_sparse_t030'`.

- [ ] **Step 3: Create the isolated configuration from the baseline**

Create the new file from the tracked baseline, then change exactly these two lines:

```bash
cp Config/Experiment/MACVO/MACVO_Fast_WindowICP_ORBLoop_Sparse.yaml \
  Config/Experiment/MACVO/MACVO_Fast_WindowICP_ORBLoop_Sparse_t030.yaml
```

```diff
-  name: MACVO-Fast-WindowICP5-ORBLoop-Sparse
+  name: MACVO-Fast-WindowICP5-ORBLoop-Sparse-t030
@@
-        max_correction_translation_m: 0.25
+        max_correction_translation_m: 0.30
```

Do not change `loop_information_scale`, `switch_prior`, sparse geometry, frontend, or any other field.

- [ ] **Step 4: Register the mode and diagnostics**

Add the mode beside the existing sparse mode:

```python
MODE_CONFIGS = {
    "window_skip": ROOT / "Config/Experiment/MACVO/MACVO_Fast_WindowICP.yaml",
    "window_pose_graph": (
        ROOT / "Config/Experiment/MACVO/MACVO_Fast_WindowICP_ORBLoop.yaml"
    ),
    "window_orb_loop": (
        ROOT / "Config/Experiment/MACVO/MACVO_Fast_WindowICP_ORBLoop.yaml"
    ),
    "window_orb_loop_sparse": (
        ROOT / "Config/Experiment/MACVO/"
        "MACVO_Fast_WindowICP_ORBLoop_Sparse.yaml"
    ),
    "window_orb_loop_sparse_t030": (
        ROOT / "Config/Experiment/MACVO/"
        "MACVO_Fast_WindowICP_ORBLoop_Sparse_t030.yaml"
    ),
    "window_orb_loop_no_skip": (
        ROOT / "Config/Experiment/MACVO/"
        "MACVO_Fast_WindowICP_ORBLoop_NoSkip.yaml"
    ),
}

ONLINE_DIAGNOSTIC_MODES = frozenset({
    "window_pose_graph",
    "window_orb_loop",
    "window_orb_loop_sparse",
    "window_orb_loop_sparse_t030",
    "window_orb_loop_no_skip",
})
```

- [ ] **Step 5: Run the focused comparison tests**

Run:

```bash
/home/zzh/MACVO/.venv/bin/python -m unittest \
  Scripts.UnitTest.test_compare_online_orb_loop -v
```

Expected: all tests pass, including the structural equality assertion.

- [ ] **Step 6: Commit the isolated mode**

```bash
git add \
  Config/Experiment/MACVO/MACVO_Fast_WindowICP_ORBLoop_Sparse_t030.yaml \
  Scripts/Experiment/CompareOnlineORBLoop.py \
  Scripts/UnitTest/test_compare_online_orb_loop.py
git commit -m "test: add isolated sparse loop t030 mode"
```

### Task 2: Add a dedicated V203 t030 runner

**Files:**
- Create: `Scripts/run_v203_sparse_alpha10000_t030.sh`
- Create: `Scripts/UnitTest/test_run_v203_sparse_t030.py`

- [ ] **Step 1: Write the failing runner test**

Create `Scripts/UnitTest/test_run_v203_sparse_t030.py`:

```python
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
        self.assertIn("--modes window_orb_loop_sparse_t030", result.stdout)
        self.assertIn(
            "Results/SparseORBLoop_V203_alpha10000_t030",
            result.stdout,
        )
        self.assertNotIn("SparseORBLoop_EuRoC_alpha10000 ", result.stdout)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test and verify the missing script fails**

Run:

```bash
/home/zzh/MACVO/.venv/bin/python -m unittest \
  Scripts.UnitTest.test_run_v203_sparse_t030 -v
```

Expected: `ERROR` with `FileNotFoundError` for `run_v203_sparse_alpha10000_t030.sh`.

- [ ] **Step 3: Implement the dedicated runner**

Create `Scripts/run_v203_sparse_alpha10000_t030.sh`:

```bash
#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKTREE="${WORKTREE:-$SCRIPT_ROOT}"
PYTHON="${PYTHON:-/home/zzh/MACVO/.venv/bin/python}"
RESULT_ROOT="${RESULT_ROOT:-/home/zzh/MACVO/Results/SparseORBLoop_V203_alpha10000_t030}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    shift
fi
if (( $# )); then
    echo "Usage: $(basename "$0") [--dry-run]" >&2
    exit 2
fi

command=(
    "$PYTHON" Scripts/Experiment/CompareOnlineORBLoop.py
    --sequence V203
    --seq-from 0
    --seed 0
    --modes window_orb_loop_sparse_t030
    --result-root "$RESULT_ROOT"
)

if [[ "$DRY_RUN" == "1" ]]; then
    printf 'WORKTREE %s\n' "$WORKTREE"
    printf 'RESULT_ROOT %s\n' "$RESULT_ROOT"
    printf 'COMMAND'
    printf ' %s' "${command[@]}"
    printf '\n'
    exit 0
fi

required=(
    "$PYTHON"
    "$WORKTREE/Scripts/Experiment/CompareOnlineORBLoop.py"
    "$WORKTREE/Config/Experiment/MACVO/MACVO_Fast_WindowICP_ORBLoop_Sparse_t030.yaml"
    "$WORKTREE/Config/Sequence/EuRoC_V203_local.yaml"
    "$WORKTREE/build/orb_bow/macvo_orb_bow"
    "$WORKTREE/cache/ORBvoc.txt"
    "$WORKTREE/Model/MACVO_FrontendCov.pth"
)
for path in "${required[@]}"; do
    if [[ ! -e "$path" ]]; then
        echo "MISSING $path" >&2
        exit 2
    fi
done
if [[ ! -x "$PYTHON" || ! -x "$WORKTREE/build/orb_bow/macvo_orb_bow" ]]; then
    echo "MISSING executable Python or ORB sidecar" >&2
    exit 2
fi

cd "$WORKTREE"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="$WORKTREE"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
exec "${command[@]}"
```

Make it executable:

```bash
chmod +x Scripts/run_v203_sparse_alpha10000_t030.sh
```

- [ ] **Step 4: Run the runner unit test**

Run:

```bash
/home/zzh/MACVO/.venv/bin/python -m unittest \
  Scripts.UnitTest.test_run_v203_sparse_t030 -v
```

Expected: one test passes and the printed command contains only the t030 mode.

- [ ] **Step 5: Commit the runner**

```bash
git add \
  Scripts/run_v203_sparse_alpha10000_t030.sh \
  Scripts/UnitTest/test_run_v203_sparse_t030.py
git commit -m "test: add isolated V203 t030 runner"
```

### Task 3: Verify the experiment package and hand off the full run

**Files:**
- Verify: `Config/Experiment/MACVO/MACVO_Fast_WindowICP_ORBLoop_Sparse.yaml`
- Verify: `Config/Experiment/MACVO/MACVO_Fast_WindowICP_ORBLoop_Sparse_t030.yaml`
- Verify: `Scripts/run_v203_sparse_alpha10000_t030.sh`

- [ ] **Step 1: Run both focused test modules**

Run:

```bash
/home/zzh/MACVO/.venv/bin/python -m unittest \
  Scripts.UnitTest.test_compare_online_orb_loop \
  Scripts.UnitTest.test_run_v203_sparse_t030 -v
```

Expected: all tests pass.

- [ ] **Step 2: Validate both YAML files and print the isolated values**

Run:

```bash
/home/zzh/MACVO/.venv/bin/python - <<'PY'
from pathlib import Path
from Utility.Config import load_config

root = Path("Config/Experiment/MACVO")
for name in (
    "MACVO_Fast_WindowICP_ORBLoop_Sparse.yaml",
    "MACVO_Fast_WindowICP_ORBLoop_Sparse_t030.yaml",
):
    _, data = load_config(root / name)
    hypothesis = data["Odometry"]["args"]["online_loop"]["hypothesis"]
    pose_graph = data["Odometry"]["args"]["pose_graph"]
    print(
        name,
        data["Odometry"]["name"],
        hypothesis["max_correction_translation_m"],
        pose_graph["loop_information_scale"],
        pose_graph["switch_prior"],
    )
PY
```

Expected:

```text
MACVO_Fast_WindowICP_ORBLoop_Sparse.yaml MACVO-Fast-WindowICP5-ORBLoop-Sparse 0.25 10000.0 1.0
MACVO_Fast_WindowICP_ORBLoop_Sparse_t030.yaml MACVO-Fast-WindowICP5-ORBLoop-Sparse-t030 0.3 10000.0 1.0
```

- [ ] **Step 3: Verify the exact full-run command without starting the 30-minute run**

Run:

```bash
Scripts/run_v203_sparse_alpha10000_t030.sh --dry-run
```

Expected: V203, `--seq-from 0`, seed zero, mode `window_orb_loop_sparse_t030`, and result root `/home/zzh/MACVO/Results/SparseORBLoop_V203_alpha10000_t030`.

- [ ] **Step 4: Confirm version-control isolation**

Run:

```bash
git diff --exit-code HEAD^ -- Config/Experiment/MACVO/MACVO_Fast_WindowICP_ORBLoop_Sparse.yaml
git status --short
```

Expected: the baseline YAML command exits zero. `docs/WindowICP.md` may remain modified as the user's preserved pre-existing change; no implementation commit includes it.

- [ ] **Step 5: Run V203 when ready**

Run:

```bash
/home/zzh/.config/superpowers/worktrees/MACVO/window-icp-global-v03/Scripts/run_v203_sparse_alpha10000_t030.sh
```

Expected: a new timestamped result appears only beneath `/home/zzh/MACVO/Results/SparseORBLoop_V203_alpha10000_t030`.

After completion, inspect `metrics.json` and `online_loop_diagnostics.json` for the success criteria in the design specification. Do not widen the threshold again if `510 -> 1355` is still absent; first extract the processing-time hypothesis corrections.
