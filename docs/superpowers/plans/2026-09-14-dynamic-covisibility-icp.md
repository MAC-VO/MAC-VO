# Dynamic Covisibility Proximity ICP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Select one geometry-driven historical proximity candidate per scheduled target, validate it with real bidirectional FlowFormerCov matching, and compare short-plus-proximity against short-only and fixed gap-5 ICP from identical saved inputs.

**Architecture:** Add a fixed-batch bidirectional frontend route, build an atomic scheduled-keyframe depth cache, rank historical pairs using current pre-global pose/depth overlap, and validate selected pairs before serializing `proximity` factors. Keep global pose-only ICP unchanged and orchestrate both the existing difficult V203 segment and a later ground-truth-selected difficult revisit segment entirely from reproducible result artifacts.

**Tech Stack:** Python 3.12 project virtualenv, PyTorch/pypose, NumPy memory maps and NPZ, existing FlowFormerCov CUDA Graph frontend, SciPy sparse global ICP, evo metrics, unittest/pytest.

---

## File structure

- Modify `Module/Frontend/Frontend.py`: fixed batch-three target stereo plus forward/backward temporal matching.
- Modify `Module/Optimization/FactorArchive.py`: add `proximity` as a schema-v1 edge kind.
- Create `Module/Optimization/ProximityFactorStore.py`: aligned proximity factor records and archive conversion.
- Create `Module/Optimization/CovisibilityCache.py`: scheduled depth/covariance memory maps and sparse proxy generation.
- Create `Module/Optimization/CovisibilitySelector.py`: NED projection, bidirectional overlap/depth score, gates, and pair-space NMS.
- Modify `Module/Optimization/MatchICP.py`: construct factors from prevalidated correspondences without changing the existing online wrapper.
- Create `Module/Optimization/ProximityICP.py`: forward/backward validation, spatial coverage, Mahalanobis filtering, and sequential offline generation.
- Create `Scripts/Experiment/GenerateProximityICP.py`: cache, candidate, matching, factor, and diagnostics CLI.
- Modify `Scripts/Experiment/RefineGlobalPoseICP.py`: allow proximity factor selection.
- Create `Scripts/Experiment/CompareCovisibilityICP.py`: strict four-mode comparison and two-level gate.
- Create `Scripts/Experiment/FindCovisibilitySegments.py`: ground-truth-only benchmark segment mining.
- Create focused tests under `Scripts/UnitTest/`.
- Modify `docs/WindowICP.md` and add validation CSV files after real-data runs.

### Task 1: Add fixed batch-three bidirectional frontend inference

**Files:**
- Modify: `Module/Frontend/Frontend.py`
- Modify: `Scripts/UnitTest/test_window_frontend.py`

- [ ] **Step 1: Write failing slot-construction and routing tests**

Add:

```python
from Module.Frontend.Frontend import build_bidirectional_inputs

def test_builds_target_stereo_forward_and_backward_slots(self):
    input_a, input_b = build_bidirectional_inputs(stereo(2, 20), stereo(0, 30))
    self.assertEqual(input_a[:, 0, 0, 0].tolist(), [0.0, 2.0, 0.0])
    self.assertEqual(input_b[:, 0, 0, 0].tolist(), [30.0, 0.0, 2.0])

def test_cuda_frontend_routes_bidirectional_slots(self):
    frontend = CUDAGraph_FlowFormerCovFrontend.__new__(CUDAGraph_FlowFormerCovFrontend)
    frontend.config = SimpleNamespace(device="cpu", enforce_positive_disparity=False)
    flow = torch.zeros(3, 2, 2, 2)
    covariance = torch.ones(3, 2, 2, 2)
    flow[0, 0] = 2
    flow[1] = 11
    flow[2] = 22
    frontend.cuda_graph_estimate = lambda *_: (flow, covariance)
    depth, forward, backward = frontend.estimate_bidirectional(
        stereo(2, 20), stereo(0, 30))
    self.assertTrue(torch.equal(forward.flow, flow[1:2]))
    self.assertTrue(torch.equal(backward.flow, flow[2:3]))
```

Also test input shape/device/dtype rejection through the same validation helper used by `build_window_inputs`.

- [ ] **Step 2: Run the tests and verify the API is absent**

Run:

```bash
PYTHONPATH=. /home/zzh/MACVO/.venv/bin/python -m pytest -o addopts='' Scripts/UnitTest/test_window_frontend.py -q
```

Expected: import or attribute failure for the bidirectional API.

- [ ] **Step 3: Implement generic fallback and CUDA Graph route**

Add to `IFrontend`:

```python
def estimate_bidirectional(self, source, target):
    depth, forward = self.estimate_pair(source, target)
    _, backward = self.estimate_pair(target, source)
    return depth, forward, backward
```

Add the fixed input builder:

```python
def build_bidirectional_inputs(source, target):
    validate_fused_inputs({
        "target.imageL": target.imageL,
        "target.imageR": target.imageR,
        "source.imageL": source.imageL,
    }, target.imageL)
    return (
        torch.cat([target.imageL, source.imageL, target.imageL], dim=0),
        torch.cat([target.imageR, target.imageL, source.imageL], dim=0),
    )
```

Extract the current shape/device/dtype checks from `build_window_inputs` into `validate_fused_inputs()` and reuse them in both builders.

Override `CUDAGraph_FlowFormerCovFrontend.estimate_bidirectional()` to call `cuda_graph_estimate()` once and map slots zero, one, and two to target depth, forward match, and backward match. Batch shape remains three, so the same captured graph works after Pass A.

- [ ] **Step 4: Run frontend tests**

Run the command from Step 2. Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add Module/Frontend/Frontend.py Scripts/UnitTest/test_window_frontend.py
git commit -m "feat: add bidirectional batch3 frontend"
```

### Task 2: Add proximity factor records without changing archive schema

**Files:**
- Modify: `Module/Optimization/FactorArchive.py`
- Create: `Module/Optimization/ProximityFactorStore.py`
- Modify: `Scripts/UnitTest/test_factor_archive.py`
- Create: `Scripts/UnitTest/test_proximity_factor_store.py`

- [ ] **Step 1: Write failing compatibility and store tests**

Add a factor-archive test that round-trips `edge_kind="proximity"` while an existing gap archive still loads. Create store tests:

```python
def test_store_round_trip_preserves_aligned_records(self):
    store = PersistentFactorStore(source_archive())
    store.add(ProximityFactorRecord(
        edge=make_edge(0, 15), score=0.2, confidence=0.7,
        age=0, state="accepted",
        candidate_metrics={"mean_overlap": 0.6},
        validation_metrics={"mahalanobis_inlier_ratio": 0.8}))
    archive = store.to_archive()
    loaded = PersistentFactorStore.from_archive(archive)
    self.assertEqual(loaded.records[0].edge.b, 15)
    self.assertEqual(loaded.records[0].confidence, 0.7)

def test_store_rejects_record_count_mismatch(self):
    archive = proximity_archive()
    archive.metadata["edge_records"] = []
    with self.assertRaisesRegex(ValueError, "record count"):
        PersistentFactorStore.from_archive(archive)
```

Also reject confidence outside `[0,1]`, negative age, state other than `accepted`, non-proximity edge kinds, duplicate endpoints, and source identity mismatch.

- [ ] **Step 2: Run the focused tests and verify failure**

```bash
PYTHONPATH=. /home/zzh/MACVO/.venv/bin/python -m pytest -o addopts='' Scripts/UnitTest/test_factor_archive.py Scripts/UnitTest/test_proximity_factor_store.py -q
```

Expected: unknown edge kind and missing module failures.

- [ ] **Step 3: Implement the store**

Extend:

```python
KNOWN_EDGE_KINDS = frozenset({"adjacent", "skip2", "gap5", "gap10", "proximity"})
```

Create:

```python
@dataclass(frozen=True)
class ProximityFactorRecord:
    edge: Edge
    score: float
    confidence: float
    age: int
    state: str
    candidate_metrics: dict
    validation_metrics: dict

class PersistentFactorStore:
    def __init__(self, source_archive):
        self.source_archive = source_archive
        self.records = []
        self._keys = set()

    @property
    def keys(self):
        return frozenset(self._keys)

    def add(self, record):
        validate_record(record)
        key = (record.edge.a, record.edge.b)
        if key in self._keys:
            raise ValueError(f"duplicate proximity edge {key}")
        self.records.append(record)
        self._keys.add(key)

    def to_archive(self):
        metadata = dict(self.source_archive.metadata)
        metadata["generator"] = "dynamic_covisibility_proximity"
        metadata["edge_records"] = [record_to_json(record) for record in self.records]
        return FactorArchive(
            self.source_archive.initial_sensor_poses,
            self.source_archive.time_ns,
            self.source_archive.T_BS,
            tuple(record.edge for record in self.records),
            ("proximity",) * len(self.records),
            metadata)
```

`from_archive()` validates all kinds, aligned record count, endpoints, and source identity, then reconstructs records using archive edges plus JSON metadata.

- [ ] **Step 4: Run store and archive tests**

Run Step 2 command. Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add Module/Optimization/FactorArchive.py Module/Optimization/ProximityFactorStore.py Scripts/UnitTest/test_factor_archive.py Scripts/UnitTest/test_proximity_factor_store.py
git commit -m "feat: persist proximity factor records"
```

### Task 3: Build an atomic scheduled-keyframe depth cache

**Files:**
- Create: `Module/Optimization/CovisibilityCache.py`
- Create: `Scripts/UnitTest/test_covisibility_cache.py`

- [ ] **Step 1: Write failing schedule, proxy, round-trip, and atomicity tests**

Use a fake six-by-eight depth frontend and add:

```python
def test_keyframe_schedule_uses_stride_five(self):
    self.assertEqual(covisibility_keyframes(17, 5), [0, 5, 10, 15])

def test_cache_uses_one_bidirectional_call_per_keyframe(self):
    frontend = FakeFrontend()
    cache = build_covisibility_cache(fake_sequence(16), source_archive(16),
                                      temp_path(), frontend, stride=5,
                                      proxy_points=8)
    self.assertEqual(frontend.calls, [(0, 0), (5, 5), (10, 10), (15, 15)])

def test_proxy_sampling_is_deterministic_and_grid_distributed(self):
    first = deterministic_proxy_uv(8, 6, 8)
    second = deterministic_proxy_uv(8, 6, 8)
    self.assertTrue(np.array_equal(first, second))
    self.assertGreater(len(np.unique(first[:, 0])), 1)
    self.assertGreater(len(np.unique(first[:, 1])), 1)

def test_failed_publication_leaves_no_destination(self):
    with patch("Module.Optimization.CovisibilityCache.os.replace", side_effect=OSError("disk")):
        with self.assertRaisesRegex(OSError, "disk"):
            build_covisibility_cache(
                fake_sequence(6), source_archive(6), destination,
                FakeFrontend(), stride=5, proxy_points=8)
    self.assertFalse(destination.exists())
```

Also test exact float32 depth/covariance round-trip, timestamp/source-ID mismatch, missing metadata, partial array files, wrong shape, and reuse of a valid existing cache without frontend calls.

- [ ] **Step 2: Run the new tests and verify module absence**

```bash
PYTHONPATH=. /home/zzh/MACVO/.venv/bin/python -m pytest -o addopts='' Scripts/UnitTest/test_covisibility_cache.py -q
```

- [ ] **Step 3: Implement cache types and deterministic proxies**

Create:

```python
@dataclass(frozen=True)
class CovisibilityDepthCache:
    root: Path
    frame_ids: np.ndarray
    time_ns: np.ndarray
    depth: np.memmap
    depth_cov: np.memmap
    proxy_uv: np.ndarray
    proxy_depth: np.ndarray
    proxy_depth_cov: np.ndarray
    metadata: dict

    def depth_output(self, frame_id):
        index = int(np.where(self.frame_ids == frame_id)[0][0])
        return IStereoDepth.Output(
            depth=torch.from_numpy(np.asarray(self.depth[index])).float()[None, None],
            cov=torch.from_numpy(np.asarray(self.depth_cov[index])).float()[None, None])

def covisibility_keyframes(frame_count, stride=5):
    return list(range(0, frame_count, stride))

def deterministic_proxy_uv(width, height, count):
    columns = max(1, round((count * width / height) ** 0.5))
    rows = max(1, int(np.ceil(count / columns)))
    u = np.linspace(0, width - 1, columns, dtype=np.float32)
    v = np.linspace(0, height - 1, rows, dtype=np.float32)
    grid = np.stack(np.meshgrid(u, v), axis=-1).reshape(-1, 2)
    return grid[:count]
```

Build into `<destination>.tmp-<uuid>`, create `depth.npy` and `depth_cov.npy` with `np.lib.format.open_memmap`, write proxies and metadata, flush maps, then atomically rename the complete directory. Reject an existing invalid destination; load an existing valid destination directly.

For every scheduled frame call `frontend.estimate_bidirectional(frame.stereo, frame.stereo)` and discard both identity matches.

- [ ] **Step 4: Run cache tests**

Run Step 2 command. Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add Module/Optimization/CovisibilityCache.py Scripts/UnitTest/test_covisibility_cache.py
git commit -m "feat: cache covisibility keyframe depths"
```

### Task 4: Implement geometry-driven covisibility selection

**Files:**
- Create: `Module/Optimization/CovisibilitySelector.py`
- Create: `Scripts/UnitTest/test_covisibility_selector.py`

- [ ] **Step 1: Write failing NED projection and selection tests**

Add hand-computed synthetic tests:

```python
def test_identity_projection_has_full_overlap_and_zero_motion(self):
    metrics = score_covisibility_pair(identity_fixture())
    self.assertAlmostEqual(metrics.mean_overlap, 1.0)
    self.assertAlmostEqual(metrics.median_motion_px, 0.0)

def test_time_near_candidate_is_excluded(self):
    selector = CovisibilitySelector(
        cache=selector_cache(), poses=selector_poses(),
        intrinsics=selector_intrinsics(),
        config=CovisibilitySelectorConfig(min_temporal_gap=15))
    result, _ = selector.select(
        target=20, history=[0, 5, 10], accepted_pairs=[])
    self.assertNotEqual(result.source, 10)

def test_pair_space_nms_only_suppresses_near_both_endpoints(self):
    accepted = [(0, 20)]
    self.assertTrue(pair_suppressed(5, 25, accepted, radius=5))
    self.assertFalse(pair_suppressed(10, 25, accepted, radius=5))
```

Also test translated-camera pixels, behind-camera exclusion using NED depth component zero, out-of-range projections, depth consistency, every named gate reason, lower-score ordering, maximum one candidate, and absence of any ground-truth argument in the selector signature.

- [ ] **Step 2: Run and observe missing module failure**

```bash
PYTHONPATH=. /home/zzh/MACVO/.venv/bin/python -m pytest -o addopts='' Scripts/UnitTest/test_covisibility_selector.py -q
```

- [ ] **Step 3: Implement projection, metrics, gates, and NMS**

Define immutable records:

```python
@dataclass(frozen=True)
class CovisibilityMetrics:
    source: int
    target: int
    overlap_forward: float
    overlap_backward: float
    mean_overlap: float
    depth_consistency: float
    median_motion_px: float
    score: float
    status: str
    reason: str | None

class CovisibilitySelector:
    def __init__(self, *, cache, poses, intrinsics, config):
        self.cache = cache
        self.poses = poses
        self.intrinsics = intrinsics
        self.config = config

    def select(self, *, target, history, accepted_pairs):
        evaluated = [
            score_covisibility_pair(
                source, target, self.cache, self.poses,
                self.intrinsics, self.config)
            for source in sorted(history)
        ]
        eligible = [record for record in evaluated
                    if record.status == "eligible"
                    and not pair_suppressed(
                        record.source, record.target, accepted_pairs,
                        self.config.candidate_nms_frames)]
        eligible.sort(key=lambda record: (record.score, record.source))
        return (eligible[0] if eligible else None), evaluated
```

For each direction, convert proxy UV/depth with `pixel2point_NED`, transform with `pp.SE3(target_pose).Inv() @ pp.SE3(source_pose)`, require transformed NED depth `> 0`, project with `point2pixel_NED`, and sample target depth/covariance at rounded in-bounds pixels.

Use:

```python
normalized_depth_error = abs(projected_depth - sampled_depth) / torch.sqrt(
    source_depth_cov + sampled_depth_cov).clamp_min(1e-9)
depth_consistent = normalized_depth_error <= 3.0
score = ((1.0 - mean_overlap)
         + 0.5 * (1.0 - depth_consistency)
         + 0.1 * median_motion_px / image_diagonal)
```

Apply the approved thresholds in a `CovisibilitySelectorConfig` dataclass. Evaluate candidates in sorted source order, serialize every metric record, filter by gates and pair-space NMS, sort by `(score, source)`, and return the first candidate or `None`.

- [ ] **Step 4: Run selector tests**

Run Step 2 command. Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add Module/Optimization/CovisibilitySelector.py Scripts/UnitTest/test_covisibility_selector.py
git commit -m "feat: select geometric covisibility candidates"
```

### Task 5: Validate bidirectional matches and covariance-aware ICP inliers

**Files:**
- Modify: `Module/Optimization/MatchICP.py`
- Create: `Module/Optimization/ProximityICP.py`
- Modify: `Scripts/UnitTest/test_match_icp.py`
- Create: `Scripts/UnitTest/test_proximity_icp.py`

- [ ] **Step 1: Write failing correspondence-builder and validator tests**

Add:

```python
def test_correspondence_builder_matches_existing_builder(self):
    selected = select_fixture_uv()
    inputs = match_builder_inputs()
    legacy = build_match_edge(**inputs)
    direct = build_edge_from_correspondences(
        uv_a=selected, uv_b=selected + sampled_forward_flow(),
        **correspondence_builder_inputs(inputs))
    self.assertTrue(torch.equal(legacy.edge.points_a, direct.edge.points_a))

def test_inverse_flows_pass_forward_backward_validation(self):
    result = validate_proximity_match(forward_inverse_fixture())
    self.assertEqual(result.reason, None)
    self.assertEqual(result.forward_backward_inliers, 40)

def test_concentrated_matches_fail_grid_coverage(self):
    result = validate_proximity_match(single_cell_fixture())
    self.assertEqual(result.reason, "insufficient_grid_coverage")

def test_mahalanobis_filter_rejects_known_outlier(self):
    edge, uv = edge_with_one_world_outlier()
    filtered = filter_mahalanobis_inliers(edge, uv, poses, threshold=3.5)
    self.assertEqual(len(filtered.edge.points_a), len(edge.points_a) - 1)
```

Also test two-pixel forward/backward threshold, minimum count and ratio, invalid-depth ratio, six-of-24 grid occupancy, minimum Mahalanobis count/ratio, confidence formula, and only one forward factor emitted.

- [ ] **Step 2: Run focused tests and verify missing APIs**

```bash
PYTHONPATH=. /home/zzh/MACVO/.venv/bin/python -m pytest -o addopts='' Scripts/UnitTest/test_match_icp.py Scripts/UnitTest/test_proximity_icp.py -q
```

- [ ] **Step 3: Refactor factor construction around explicit correspondences**

Add to `MatchICP.py`:

```python
@dataclass(frozen=True)
class CorrespondenceEdgeBuildResult:
    edge: Edge | None
    uv_a: torch.Tensor
    uv_b: torch.Tensor
    reason: str | None

def build_edge_from_correspondences(*, a, b, uv_a, uv_b, stereo_a,
    stereo_b, depth_a, depth_b, match, frontend, covariance_model,
    min_num_point, match_cov_default, device):
    d_a = frontend.retrieve_pixels(uv_a, depth_a.depth).squeeze(0)
    d_b = frontend.retrieve_pixels(uv_b, depth_b.depth).squeeze(0)
    depth_cov_a = frontend.retrieve_pixels(uv_a, depth_a.cov).squeeze(0)
    depth_cov_b = frontend.retrieve_pixels(uv_b, depth_b.cov).squeeze(0)
    uv_cov_a = torch.full((len(uv_a), 3), match_cov_default, device=device)
    uv_cov_a[:, 2] = 0
    uv_cov_b = frontend.retrieve_pixels(uv_a, match.cov).T.clone()
    cov_a = covariance_model.estimate(
        stereo_a, uv_a, depth_a, depth_cov_a, uv_cov_a).cpu().double()
    cov_b = covariance_model.estimate(
        stereo_b, uv_b, depth_b, depth_cov_b, uv_cov_b).cpu().double()
    points_a = pixel2point_NED(
        uv_a.cpu(), d_a.cpu(), stereo_a.frame_K.cpu()).double()
    points_b = pixel2point_NED(
        uv_b.cpu(), d_b.cpu(), stereo_b.frame_K.cpu()).double()
    valid = (torch.isfinite(points_a).all(-1)
             & torch.isfinite(points_b).all(-1)
             & (d_a.cpu() > 0) & (d_b.cpu() > 0)
             & torch.isfinite(cov_a).all(dim=(-2, -1))
             & torch.isfinite(cov_b).all(dim=(-2, -1)))
    ids = torch.nonzero(valid).flatten()
    if len(ids):
        spd = ((torch.linalg.cholesky_ex(cov_a[ids]).info == 0)
               & (torch.linalg.cholesky_ex(cov_b[ids]).info == 0))
        ids = ids[spd]
    if len(ids) < min_num_point:
        return CorrespondenceEdgeBuildResult(
            None, uv_a[ids], uv_b[ids], "too_few_valid_observations")
    return CorrespondenceEdgeBuildResult(
        Edge(a, b, points_a[ids], points_b[ids], cov_a[ids], cov_b[ids]),
        uv_a[ids], uv_b[ids], None)
```

Move the existing post-selection construction into this function. Keep
`build_match_edge()` behavior unchanged by selecting/inbound-filtering UV and
then delegating to the new function.

- [ ] **Step 4: Implement proximity validation**

Create configuration and result dataclasses. Bilinearly sample backward flow at
forward target coordinates using `torch.nn.functional.grid_sample`, compute
closure error, depth-valid ratio, and source-grid occupancy. Delegate validated
correspondences to `build_edge_from_correspondences()`.

Compute Mahalanobis norms using the saved source poses and the same covariance
formula as `GlobalPoseICP._edge_linearization`. Retain only inliers and calculate:

```python
confidence = max(0.0, min(1.0, 0.25 * (
    candidate.mean_overlap
    + candidate.depth_consistency
    + forward_backward_ratio
    + mahalanobis_inlier_ratio)))
```

Return a `ProximityFactorRecord` only after every approved gate passes.

- [ ] **Step 5: Run match and proximity tests**

Run Step 2 command. Expected: all pass, including parity with the previous online skip-edge behavior.

- [ ] **Step 6: Commit**

```bash
git add Module/Optimization/MatchICP.py Module/Optimization/ProximityICP.py Scripts/UnitTest/test_match_icp.py Scripts/UnitTest/test_proximity_icp.py
git commit -m "feat: validate bidirectional proximity factors"
```

### Task 6: Generate dynamic proximity factors offline

**Files:**
- Modify: `Module/Optimization/ProximityICP.py`
- Create: `Scripts/Experiment/GenerateProximityICP.py`
- Create: `Scripts/UnitTest/test_generate_proximity_icp.py`

- [ ] **Step 1: Write failing sequential-generation and CLI tests**

Add a fake cache, selector, frontend, and validator:

```python
def test_generator_processes_targets_chronologically_and_adds_at_most_one_edge(self):
    result = generate_proximity_archive(**generation_fixture())
    self.assertEqual(result.targets, [15, 20, 25])
    self.assertLessEqual(result.accepted_edges, 3)

def test_failed_match_does_not_nms_suppress_future_candidate(self):
    validator.reject_pair(0, 15)
    result = generate_proximity_archive(
        **generation_fixture(validator=validator))
    self.assertIn((5, 20), result.attempted_pairs)

def test_cli_defaults_match_approved_thresholds(self):
    args = build_parser().parse_args(["--space", "/tmp/result"])
    self.assertEqual(args.min_temporal_gap, 15)
    self.assertEqual(args.max_candidates_per_target, 1)
```

Also test no-candidate, inference-failure, validation rejection, source timestamp mismatch, valid cache reuse, accepted-pair NMS, candidate JSON completeness, and no accepted edges status.

- [ ] **Step 2: Run and observe missing generator failure**

```bash
PYTHONPATH=. /home/zzh/MACVO/.venv/bin/python -m pytest -o addopts='' Scripts/UnitTest/test_generate_proximity_icp.py -q
```

- [ ] **Step 3: Implement chronological selection and validation orchestration**

For each scheduled target beginning at frame 15:

```python
candidate, evaluated = selector.select(
    target=target,
    history=[frame for frame in cache.frame_ids if frame <= target - 15],
    accepted_pairs=store.keys,
)
candidate_records.extend(evaluated)
if candidate is None:
    continue
depth_t, forward, backward = frontend.estimate_bidirectional(
    sequence[candidate.source].stereo,
    sequence[target].stereo,
)
validation = validate_proximity_match(
    candidate=candidate, forward=forward, backward=backward,
    source_frame=sequence[candidate.source], target_frame=sequence[target],
    source_depth=cache.depth_output(candidate.source), target_depth=depth_t,
    poses=source_archive.initial_sensor_poses,
    frontend=frontend, keypoint_selector=keypoint_selector,
    covariance_model=covariance_model, config=validation_config)
if validation.record is not None:
    store.add(validation.record)
```

Only `store.add()` changes NMS state. Serialize all evaluated candidate records and all validation outcomes, including explicit rejection reasons.

- [ ] **Step 4: Implement CLI and outputs**

`GenerateProximityICP.py` loads the saved sequence/configuration, source factor archive, and valid cache or builds it. It writes:

```text
covisibility_candidates.json
proximity_factors.npz
proximity_generation.json
```

Expose every approved threshold as a CLI argument with the approved defaults. Record source/cached/proximity SHA-256, frontend calls, Pass A/Pass B time, candidate counts, edge counts, observations, and rejection histograms. Return exit code 2 for `no_proximity_edges` without deleting diagnostics.

- [ ] **Step 5: Run generator tests**

Run Step 2 command. Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add Module/Optimization/ProximityICP.py Scripts/Experiment/GenerateProximityICP.py Scripts/UnitTest/test_generate_proximity_icp.py
git commit -m "feat: generate dynamic proximity ICP factors"
```

### Task 7: Compare dynamic covisibility against fixed gap-5

**Files:**
- Modify: `Scripts/Experiment/RefineGlobalPoseICP.py`
- Create: `Scripts/Experiment/CompareCovisibilityICP.py`
- Create: `Scripts/UnitTest/test_compare_covisibility_icp.py`
- Modify: `docs/WindowICP.md`

- [ ] **Step 1: Write failing parser, row, gate, and verification tests**

Add proximity to the refiner parser test. Create comparison tests:

```python
def test_basic_gate_requires_proximity_ate_below_short(self):
    gate = dynamic_covisibility_gate(rows(short=0.14, gap5=0.137, proximity=0.136,
                                          proximity_edges=20))
    self.assertTrue(gate["basic_passed"])

def test_selector_gate_requires_equal_or_better_gap5_ate_and_edge_budget(self):
    gate = dynamic_covisibility_gate(rows(short=0.14, gap5=0.137,
                                          proximity=0.136, proximity_edges=24))
    self.assertFalse(gate["selector_superiority"])
```

Also test exact mode order `before_global, short, gap5, proximity`, source SHA equality, proximity archive source identity, timestamps, fixed anchor, source `poses.npy` hash, missing fixed gap archive generation routing, and JSON/CSV consistency.

- [ ] **Step 2: Run tests and verify missing comparison behavior**

```bash
PYTHONPATH=. /home/zzh/MACVO/.venv/bin/python -m pytest -o addopts='' Scripts/UnitTest/test_offline_global_refinement.py Scripts/UnitTest/test_compare_covisibility_icp.py -q
```

- [ ] **Step 3: Allow proximity selection in the existing refiner**

Change `--edge-kinds` choices to include `proximity`. No solver code changes are required because factor archives already produce plain arbitrary-span `Edge` objects.

- [ ] **Step 4: Implement the strict comparison orchestrator**

Reuse or create one online source result. Reuse an existing valid
`long_factors_gap5_10.npz`; otherwise invoke `GenerateLongRangeICP.py` once.
Invoke `GenerateProximityICP.py`, then run offline refinements:

```text
short: source short archive only
gap5: source short + fixed archive filtered to gap5
proximity: source short + proximity archive filtered to proximity
```

Support `--segment-json PATH` as an alternative to explicit `--seq-from` and
`--seq-to`. Load integer `start` and `end` fields before launching online
odometry, reject conflicting explicit bounds, and copy the segment JSON SHA-256
into the comparison manifest.

Write `dynamic_covisibility_metrics.csv/json`,
`dynamic_covisibility_gate.json`, `comparison_manifest.json`, and
`verification.json`. `basic_passed` requires finite output, refined status,
exact anchor, objective decrease, and proximity ATE below short. The selector
gate also requires proximity ATE no greater than gap5 and accepted edge count no
greater than the actual number of fixed gap-5 edges in that comparison. For the
original segment that budget is 23.

- [ ] **Step 5: Document commands and interpretation**

Add the original-segment command:

```bash
PYTHONPATH=. /home/zzh/MACVO/.venv/bin/python Scripts/Experiment/CompareCovisibilityICP.py \
  --space /home/zzh/MACVO/Results/LongRangeICP_V203_short/20260914_175119_fa00a4/source/MACVO-Fast-WindowICP5-Global@V203/09_14_175121 \
  --result-root /home/zzh/MACVO/Results/CovisibilityICP_V203_original
```

Document that zero accepted proximity edges is a valid Stage 1 result and does not trigger automatic threshold relaxation.

- [ ] **Step 6: Run comparison tests**

Run Step 2 command. Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add Scripts/Experiment/RefineGlobalPoseICP.py Scripts/Experiment/CompareCovisibilityICP.py Scripts/UnitTest/test_offline_global_refinement.py Scripts/UnitTest/test_compare_covisibility_icp.py docs/WindowICP.md
git commit -m "feat: compare dynamic covisibility ICP"
```

### Task 8: Mine a difficult V203 revisit segment

**Files:**
- Create: `Scripts/Experiment/FindCovisibilitySegments.py`
- Create: `Scripts/UnitTest/test_find_covisibility_segments.py`

- [ ] **Step 1: Write failing revisit and ranking tests**

Use synthetic timed trajectories:

```python
def test_revisit_pair_requires_gap_position_and_view_angle(self):
    self.assertTrue(is_revisit_pair(poses, 0, 40, min_gap=30,
                                    max_distance=2.0, max_angle_deg=30.0))
    self.assertFalse(is_revisit_pair(poses, 0, 20, min_gap=30,
                                     max_distance=2.0, max_angle_deg=30.0))

def test_segment_requires_three_revisit_pairs_and_ranks_by_difficulty(self):
    segments = find_segments(
        gt, estimate, window=120, min_temporal_gap=30,
        max_distance=2.0, max_angle_deg=30.0, min_pairs=3)
    self.assertEqual(segments[0].start, expected_difficult_start)
```

Also test SO(3) geodesic viewing angle, windows without revisits, deterministic tie ordering, timestamp mismatch, exact 120-frame end index, and no use of GT in any selector module.

- [ ] **Step 2: Run and verify module absence**

```bash
PYTHONPATH=. /home/zzh/MACVO/.venv/bin/python -m pytest -o addopts='' Scripts/UnitTest/test_find_covisibility_segments.py -q
```

- [ ] **Step 3: Implement segment mining**

Load full `poses.npy` and `ref_poses.npy`, align timestamps using the existing
trajectory utilities, enumerate 120-frame windows, and count pairs satisfying:

```text
j - i >= 30
norm(gt_translation[j] - gt_translation[i]) <= 2.0
SO3_geodesic_angle(gt_rotation[i], gt_rotation[j]) <= 30 degrees
```

Compute each qualifying window's difficulty as normalized rolling estimate RTE
RMSE plus ROE RMSE, sort by `(-difficulty, start)`, and write all candidates to
CSV/JSON plus `selected_segment.json`.

- [ ] **Step 4: Run segment tests**

Run Step 2 command. Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add Scripts/Experiment/FindCovisibilitySegments.py Scripts/UnitTest/test_find_covisibility_segments.py
git commit -m "feat: find difficult covisibility segments"
```

### Task 9: Regression verification and original-segment experiment

**Files:**
- Create: `docs/validation/dynamic_covisibility_v203_1095_1215.csv`
- Modify: `docs/WindowICP.md`

- [ ] **Step 1: Run all focused tests**

```bash
PYTHONPATH=. /home/zzh/MACVO/.venv/bin/python -m pytest -o addopts='' \
  Scripts/UnitTest/test_window_frontend.py \
  Scripts/UnitTest/test_factor_archive.py \
  Scripts/UnitTest/test_proximity_factor_store.py \
  Scripts/UnitTest/test_covisibility_cache.py \
  Scripts/UnitTest/test_covisibility_selector.py \
  Scripts/UnitTest/test_match_icp.py \
  Scripts/UnitTest/test_proximity_icp.py \
  Scripts/UnitTest/test_generate_proximity_icp.py \
  Scripts/UnitTest/test_offline_global_refinement.py \
  Scripts/UnitTest/test_compare_covisibility_icp.py \
  Scripts/UnitTest/test_find_covisibility_segments.py -q
```

Expected: all focused tests pass.

- [ ] **Step 2: Run the complete non-local suite and CLI smoke tests**

```bash
PYTHONPATH=. /home/zzh/MACVO/.venv/bin/python -m pytest -o addopts='' Scripts/UnitTest -m 'not local' -q
PYTHONPATH=. /home/zzh/MACVO/.venv/bin/python Scripts/Experiment/GenerateProximityICP.py --help
PYTHONPATH=. /home/zzh/MACVO/.venv/bin/python Scripts/Experiment/CompareCovisibilityICP.py --help
PYTHONPATH=. /home/zzh/MACVO/.venv/bin/python Scripts/Experiment/FindCovisibilitySegments.py --help
git diff --check
```

Expected: no test failures, all CLIs exit zero, and no whitespace errors.

- [ ] **Step 3: Run Stage 1 without online odometry**

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=. \
/home/zzh/MACVO/.venv/bin/python Scripts/Experiment/CompareCovisibilityICP.py \
  --space /home/zzh/MACVO/Results/LongRangeICP_V203_short/20260914_175119_fa00a4/source/MACVO-Fast-WindowICP5-Global@V203/09_14_175121 \
  --result-root /home/zzh/MACVO/Results/CovisibilityICP_V203_original
```

Expected: reuse of the source short and gap-5 archives, 24 Pass A calls, at
most 23 Pass B calls, no modification of source outputs, and four strict metric
rows.

- [ ] **Step 4: Verify and document Stage 1**

Run the comparison verification command printed by the script. Copy the
generated CSV to `docs/validation/dynamic_covisibility_v203_1095_1215.csv` and
record candidate counts, rejection reasons, accepted edges, cache size, network
time, solver time, four metrics, and both gate values in `docs/WindowICP.md`.

- [ ] **Step 5: Commit Stage 1 evidence**

```bash
git add docs/WindowICP.md docs/validation/dynamic_covisibility_v203_1095_1215.csv
git commit -m "docs: validate dynamic covisibility on original V203 segment"
```

### Task 10: Find and run the second difficult revisit segment

**Files:**
- Create: `docs/validation/dynamic_covisibility_v203_revisit.csv`
- Modify: `docs/WindowICP.md`

- [ ] **Step 1: Mine V203 using the existing complete trajectory**

```bash
PYTHONPATH=. /home/zzh/MACVO/.venv/bin/python Scripts/Experiment/FindCovisibilitySegments.py \
  --space /home/zzh/MACVO/Results/WindowICP_Global_v03/20260914_155636_738ac1/V203/window_global/outputs/MACVO-Fast-WindowICP5-Global@V203/09_14_155638 \
  --window 120 --min-temporal-gap 30 --max-distance 2.0 \
  --max-angle-deg 30.0 --min-revisit-pairs 3 \
  --output-root /home/zzh/MACVO/Results/CovisibilitySegmentSearch_V203
```

Expected: ranked qualifying windows and one explicit selected `[start,end)` range.

- [ ] **Step 2: Run one online source and the same four-mode offline comparison**

Read the selected indices from `selected_segment.json`, then run:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=. \
/home/zzh/MACVO/.venv/bin/python Scripts/Experiment/CompareCovisibilityICP.py \
  --sequence V203 --segment-json /home/zzh/MACVO/Results/CovisibilitySegmentSearch_V203/selected_segment.json --seed 0 \
  --result-root /home/zzh/MACVO/Results/CovisibilityICP_V203_revisit
```

`CompareCovisibilityICP.py` reads `start` and `end` directly from the selected
segment JSON; do not select a different window after observing metrics.

- [ ] **Step 3: Verify and document the revisit experiment**

Run the printed verification command, copy its CSV to
`docs/validation/dynamic_covisibility_v203_revisit.csv`, and update
`docs/WindowICP.md` with the selected window's revisit-pair count, difficulty,
candidate/factor statistics, metrics, and gate outcomes.

- [ ] **Step 4: Run final verification**

```bash
PYTHONPATH=. /home/zzh/MACVO/.venv/bin/python -m pytest -o addopts='' Scripts/UnitTest -m 'not local' -q
git diff --check
git status --short --branch
```

Expected: all non-local tests pass and only intentional validation files are pending.

- [ ] **Step 5: Commit validation evidence**

```bash
git add docs/WindowICP.md docs/validation/dynamic_covisibility_v203_revisit.csv
git commit -m "docs: validate dynamic covisibility on V203 revisit"
```

Do not commit models, datasets, result directories, depth caches, or factor archives.
