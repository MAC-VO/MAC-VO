# Five-frame sliding-window ICP implementation plan

**Goal:** Add opt-in five-frame pose-only optimization with adjacent and independently inferred two-step matches, retaining Fast FlowFormerCov and full 3D covariances.

**Architecture:** Reuse synchronous two-frame ICP as the current-frame initializer and adjacent observation builder. A MACVO subclass retains five frame/depth entries, builds direct t-2→t matches, then jointly refines poses with the oldest fixed. A CPU float64 block-Jacobian LM solver solves at most 24 unknowns. Re-anchor world points after historical pose updates.

**Tech stack:** Existing Python 3.12 / torch 2.4.1+cu124 / pypose 0.9.5. No dependency installation or BAE in v1.

## Version management

- Branch: experiment/sliding-window-icp-v1.
- Baseline commit: 5542d91; tag: baseline/fast-icp-euroc-20260912.
- Stage only feature files; existing papers, translations and reports stay uncommitted.
- Separate solver, integration and validation commits; tag the verified version.

## Scope and limitations

- Fixed oldest pose; no marginalized prior or landmark BA.
- Adjacent two-frame initializer adds CPU work before joint refinement.
- Skip matching reuses the two-sample CUDA graph, recomputing its stereo half to avoid a second graph and memory pool.
- Skip factors live in a bounded side cache, not existing adjacency-only map indices.
- Cross-factor observation correlations are approximated as independent.
- Whitened vector Huber threshold is explicit; new solver settings differ from the old optimizer.
- Synchronous window pose writes and consistent owned-point/covariance updates.

## Tasks and checks

- [x] Write Scripts/UnitTest/test_window_icp.py first: synthetic pose recovery, finite-difference Jacobian, off-diagonal whitening, fixed anchor, eviction, disconnected graph and NaN rejection.
- [x] Run .venv/bin/python -m unittest Scripts.UnitTest.test_window_icp and record expected missing-feature failure.
- [x] Implement Module/Optimization/WindowICP.py: edge data, whitening, Jacobians, damped normal equations, acceptance, bounded storage.
- [x] Pass CPU tests and commit solver.
- [x] Add Odometry/WindowMACVO.py: adjacent initializer, direct skip match, bounded frame/depth cache, joint refinement, re-anchoring, telemetry.
- [x] Add opt-in routing plus seed/provenance to MACVO.py while preserving default two-frame mode.
- [x] Add Config/Experiment/MACVO/MACVO_Fast_WindowICP.yaml and sequential run/compare script.
- [x] Test map re-anchoring and config/mode invariants.
- [x] Run baseline and window on the same MH01 crop and seed. Check at most five poses, both edge distances, finite trajectories and successful evaluation.
- [x] Review solver math, eviction, map writes, completion handling and baseline behavior.
- [x] Write docs/WindowICP.md with commands, limitations, metrics and versions; commit and tag after validation.

## Validation outcome

11 CPU tests passed, including regressions for float32 quaternion drift over 40 frames,
recovered-frame interpolation and constructor failure provenance. Real built-in
TartanAir 10-frame pipeline passed. Final MH01 range [1500,1620), seed 0, ran three
modes at a0f49b4; both window modes refined all 119 windows without fallback.
Full results: docs/WindowICP.md and docs/validation/window_icp_mh01_1500_1620.csv.
Independent code review has no outstanding findings. Retain the development branch;
do not merge or push as part of this task.
