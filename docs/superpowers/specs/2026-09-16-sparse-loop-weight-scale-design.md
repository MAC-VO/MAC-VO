# Sparse Loop Weight Scale Design

## Goal

Make the sparse SE(3) loop factors strong enough to correct the global trajectory
while preserving the switch values established by the verified sparse
measurements. Use the V203 offline sweep selection `alpha = 10000` and validate
the resulting online behavior on the independent EuRoC MH04 sequence.

## Configuration

Add one pose-graph configuration value:

```yaml
pose_graph:
  loop_information_scale: 10000.0
```

The sparse configuration sets this value to `10000.0`. Existing legacy
configurations omit it and therefore retain the default value `1.0`.

The value must be finite and positive.

## Optimization Semantics

For every sparse loop factor:

```text
Omega_scaled = loop_information_scale * Omega_base
```

The effective switch prior passed to the pose-graph optimizer is:

```text
switch_prior_effective = loop_information_scale * switch_prior
```

Scaling both quantities by the same factor preserves the initial ratio:

```text
s = lambda / (lambda + chi_squared)
```

while increasing the loop contribution relative to adjacent and skip factors.

Adjacent and skip factors are unchanged. Dense legacy loop compression is
unchanged unless its configuration explicitly sets a scale.

## Ownership

`OnlineLoopWindowMACVO` owns the configured scale. It applies the scale when
constructing a direct sparse `PoseGraphFactor` and uses the coupled effective
switch prior in every asynchronous and final pose-graph solve.

`conservative_sparse_factor` accepts an optional `information_scale` argument
with default `1.0`, so existing callers and tests retain their current values.

## Diagnostics

`online_loop_diagnostics.json` records:

```text
loop_information_scale
switch_prior_base
switch_prior_effective
```

Each accepted sparse factor continues to record its final scaled information
eigenvalues. With the approved V203 base factor, the expected maximum eigenvalue
is approximately `328280.635`, which is comparable to the short-factor
information range without saturating the `1e6` cap.

## Testing

Unit tests must prove:

- `information_scale=10000` multiplies all six sparse information eigenvalues;
- default scale `1.0` preserves existing behavior;
- the online sparse factor path passes the configured scale to the builder;
- effective switch prior equals base prior times scale;
- diagnostics contain base and effective values;
- legacy configurations remain valid with the default scale.

Run the complete unit-test suite after focused tests.

## MH04 Validation

Run complete EuRoC MH04 with seed zero and the sparse online configuration.

Required safety gates:

- all trajectory values finite;
- zero newly unrefined windows;
- one frontend model, one CUDA graph, and concurrency one;
- peak reserved VRAM below 6 GiB;
- provider remains protocol-v2 ready;
- no failed pose-graph solves.

Report:

- ATE, RTE, ROE, and RPE;
- runtime mean, median, and P95;
- loop candidate rejection reasons;
- tentative, confirmed, and strong hypotheses;
- loop-factor count and long-loop count;
- switch distribution and effective long loops;
- maximum trajectory correction and information eigenvalue;
- same-run remove-loop reoptimization ablation.

The MH04 result is an independent-sequence validation. It is not used to change
the selected scale during this task.

## Version Control

- Commit the design, implementation, and MH04 validation separately.
- Preserve the existing unstaged `docs/WindowICP.md` modification.
- Write MH04 results to a new result root and a new validation CSV.
