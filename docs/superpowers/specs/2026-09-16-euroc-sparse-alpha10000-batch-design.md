# EuRoC Sparse Alpha10000 Batch Design

## Goal

Run the remaining EuRoC sequences with the verified sparse ORB loop pipeline and
`loop_information_scale = 10000`, while making the batch resumable, observable,
and safe for a single 8 GiB GPU.

MH04 and V203 are already complete. The default remaining set is:

```text
MH01 MH02 MH03 MH05 V101 V102 V103 V201 V202
```

## Script

Create:

```text
Scripts/run_euroc_sparse_alpha10000_remaining.sh
```

The script runs from any working directory by resolving the repository root
from `BASH_SOURCE`. It uses the shared virtual environment at
`/home/zzh/MACVO/.venv/bin/python` by default and permits environment-variable
overrides.

## Execution Policy

- Run one sequence at a time on one GPU.
- Use `CompareOnlineORBLoop.py` with mode `window_orb_loop_sparse`.
- Use seed zero and the complete sequence starting at index zero.
- Store every invocation below one common result root.
- Continue after an individual sequence failure.
- Exit nonzero after the batch if any sequence failed.
- Accept optional positional sequence names to run or retry a subset.

No sequence processes run concurrently.

## Result and State Layout

Default result root:

```text
/home/zzh/MACVO/Results/SparseORBLoop_EuRoC_alpha10000
```

State and logs:

```text
<result-root>/.batch_state/<sequence>.complete
<result-root>/.batch_state/<sequence>.failed
<result-root>/logs/<sequence>.log
<result-root>/batch.log
<result-root>/summary.tsv
```

A sequence is skipped only when both conditions hold:

1. its completion marker exists;
2. a result under the common root contains `metrics.json` with a row for that
   sequence and mode `window_orb_loop_sparse`.

This prevents stale marker files from hiding incomplete results.

Before retrying a failed sequence, remove its `.failed` marker. A successful
retry removes the failure marker and writes the completion marker atomically.

## Preflight

Before starting, require:

- Python executable;
- comparison script;
- sparse alpha10000 experiment configuration;
- ORB sidecar executable;
- ORB vocabulary;
- frontend model;
- every requested local sequence configuration;
- every sequence data root referenced by its configuration.

Acquire an exclusive batch lock under the result root with `flock`. A second
batch process exits without starting work.

## Logging and Summary

Each sequence has a dedicated log and also appends to the batch log. After a
successful sequence, parse its newest metrics row and append one TSV summary
row containing:

```text
sequence status result_dir RMSE_ATE RMSE_RTE RMSE_ROE RMSE_RPE
runtime_mean_ms loop_factors effective_long_loops_001 peak_vram_bytes
```

Failed sequences receive a summary row with status `failed` and empty metrics.
Skipped sequences receive status `skipped`.

The final terminal and batch-log summary lists successful, skipped, and failed
sequences.

## Persistent Launch

After the script passes shell syntax and dry-run tests, launch it as a user
systemd transient service:

```text
macvo-sparse-euroc-alpha10000-remaining.service
```

This avoids losing the batch when the interactive tool session or chat turn
ends. Progress is monitored through `systemctl`, the batch log, and per-sequence
logs.

## Testing

Add a shell-focused unit test that runs the script in dry-run mode with fixture
paths. It must prove:

- default sequence order excludes MH04 and V203;
- positional sequence overrides work;
- missing required files fail preflight;
- valid completion marker plus metrics skips a sequence;
- stale marker without metrics does not skip;
- one failed sequence does not prevent the next from being scheduled;
- summary and marker paths are deterministic.

Also run `bash -n` and the complete Python unit-test suite.

## Version Control

- Commit the specification, plan, script/tests, and final batch summary
  separately.
- Do not modify the existing legacy batch scripts.
- Preserve the unstaged `docs/WindowICP.md` modification.
