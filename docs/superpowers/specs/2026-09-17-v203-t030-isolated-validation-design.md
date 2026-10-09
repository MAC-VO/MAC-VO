# V203 t030 Isolated Validation Design

## Goal

Validate whether increasing the loop-hypothesis correction translation threshold
from `0.25 m` to `0.30 m` restores the missing `510 -> 1355` loop in the V203
online `alpha=10000` run and reduces ATE toward the fixed-graph offline result.

## Isolation

- Keep `MACVO_Fast_WindowICP_ORBLoop_Sparse.yaml` unchanged as the `0.25 m`
  baseline.
- Add a separate sparse-loop experiment configuration whose only semantic
  difference is `max_correction_translation_m: 0.30`.
- Expose the new configuration through a distinct comparison mode so the
  diagnostics and metrics retain an unambiguous experiment identity.
- Add a dedicated V203 runner that writes to
  `Results/SparseORBLoop_V203_alpha10000_t030`; it must not reuse the baseline
  result root or completion markers.

## Execution

The runner executes the full V203 sequence from frame zero with seed zero and
the t030 mode only. Existing model, ORB vocabulary, sidecar, sequence data, loop
weight (`10000`), base switch prior (`1.0`, effective value `10000` after loop
scaling), and all sparse-geometry thresholds remain unchanged.

## Validation

Automated tests verify that:

1. The new mode resolves to the t030 configuration.
2. The t030 configuration differs from the baseline only at the hypothesis
   translation threshold and experiment name.
3. The runner uses a distinct result root and selects only the t030 mode.

After the full run, inspect diagnostics and report:

- V203 ATE, RTE, ROE, and RPE;
- emitted loop-factor count and effective long-loop count;
- whether `510 -> 1355` was emitted;
- whether `520 -> 1350` and `510 -> 1355` merged into one confirmed hypothesis;
- comparison against the baseline online ATE `0.327074` and offline fixed-graph
  ATE `0.147434`.

## Success Criterion

The hypothesis is supported if the t030 run emits `510 -> 1355`, produces 13
loop factors, and materially approaches the offline ATE without introducing new
obviously inconsistent long-loop factors. If it does not, retain the isolated
result as evidence and investigate the processing-time correction values rather
than widening the threshold again.
