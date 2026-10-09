# WindowICP No-Skip V203 Ablation Design

## Objective

Measure the accuracy, runtime, memory, and backend behavior change caused by completely removing the `t-2 -> t` constraint from the current online ORB plus background pose-graph architecture.

## Experimental Change

The no-skip variant will set `skip_matching: false`. Consequently:

- the tracking frontend will retain fixed batch=3, using a duplicated adjacent input in the third slot and discarding that output;
- only current stereo depth and `t-1 -> t` matching will become observations;
- the five-frame WindowICP will contain only adjacent edges;
- no `skip2` pose factors will be compressed or stored;
- the persistent pose graph will contain adjacent and validated loop factors only.

All other parameters remain identical to `MACVO_Fast_WindowICP_ORBLoop.yaml`, including the five-frame window, ORB keyframe stride, candidate thresholds, pairwise ICP, pose-graph robust settings, random seed, sequence preprocessing, serialized GPU access, and background writeback policy.

## Comparison

Baseline: complete V203 online ORB result at commit `ca88f45`:

- 1865 frames;
- ATE RMSE 0.465288823 m;
- mean runtime 1127.039 ms/frame;
- P95 runtime 1911.363 ms/frame;
- peak CUDA reserved memory 4.048828 GiB;
- 1864 adjacent, 1863 skip2, and 51 loop factors.

The no-skip run must use V203, seed 0, and the same `[0, 1865)` timestamp range. A preliminary `[1095, 1215)` run will verify stability before the complete run.

## Measurements

Record:

- ATE, RTE, ROE, and RPE mean/std/RMSE;
- mean, median, and P95 odometry runtime;
- normal and loop frontend call counts;
- peak CUDA reserved memory;
- adjacent, skip2, and loop factor counts;
- pose-graph submissions, writebacks, and final solve time;
- maximum loop switch, effective loop count, and effective long-loop count;
- output completeness, finite poses, unit quaternions, and timestamp equality.

## Interpretation

The comparison measures the total contribution of `t-2 -> t`, including both local WindowICP stabilization and persistent global skip factors. It is not a single-variable measurement of only the global skip factors.

Expected outcomes:

- GPU inference runtime and memory should remain comparable because fixed batch=3 is preserved; only skip edge construction, local residuals, factor compression, and global sparse blocks are removed;
- accuracy may degrade because adjacent-only chains have weaker rotational and translational redundancy;
- any accuracy improvement would indicate that current skip measurements or their information scaling are harmful.

No parameter will be retuned after observing the short result. The full run will use the same no-skip configuration if the short run completes without non-finite poses, crashes, or unrefined windows caused by graph disconnection.

## Artifacts

The experiment will produce a dedicated no-skip configuration, comparison metrics, online diagnostics, a versioned validation CSV, and a result section in `docs/WindowICP.md`. Work remains on the isolated `experiment/window-icp-global-v03` branch and will not be merged automatically.
