# Offline Long-Range ICP Factors Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Save reproducible short-range ICP artifacts, generate real `t-5` and `t-10` factors offline with one fixed batch-three inference per scheduled target, and verify from identical initial poses whether those factors reduce V203 ATE.

**Architecture:** A typed factor archive becomes the boundary between inference and optimization. `WindowMACVO` snapshots and serializes its pre-global sensor poses plus short factors; an offline generator reuses the saved run configuration to create long factors; an offline refiner merges selected archives and writes tagged trajectories and metrics without modifying source outputs.

**Tech Stack:** Python 3.11, PyTorch/pypose, NumPy NPZ, SciPy sparse solve, existing FlowFormerCov CUDAGraph frontend, unittest/pytest, existing EuRoC and evo evaluation utilities.

---

## File structure

- Create `Module/Optimization/FactorArchive.py`: validated factor archive data model, atomic NPZ serialization, loading, filtering, merging, and sensor-to-body pose conversion.
- Create `Module/Optimization/MatchICP.py`: reusable deterministic match-to-ICP-factor construction shared by online skip edges and offline long edges.
- Create `Module/Optimization/LongRangeICP.py`: pure schedule construction and offline long-factor generation loop.
- Modify `Module/Optimization/WindowICP.py`: allow any valid forward edge while preserving short-edge diagnostics.
- Modify `Module/Optimization/GlobalPoseICP.py`: report edge-gap counts without changing the objective or sparse solver.
- Modify `Odometry/WindowMACVO.py`: use the shared match builder, snapshot pre-global state, write artifacts, and release the snapshot only after successful publication.
- Modify `MACVO.py`: persist the preprocessing configuration needed to reconstruct exact offline sequence inputs.
- Create `Scripts/Experiment/GenerateLongRangeICP.py`: command-line wrapper for offline inference and archive generation.
- Create `Scripts/Experiment/RefineGlobalPoseICP.py`: command-line wrapper for archive merging, sparse refinement, tagged trajectory output, and metrics.
- Create `Scripts/Experiment/CompareLongRangeICP.py`: reproducible short-sequence/full-sequence orchestration and comparison report.
- Create unit tests under `Scripts/UnitTest/` for each new boundary.
- Modify `docs/WindowICP.md`: artifact format, offline commands, and validation results.

### Task 1: Generalize ICP edges for long frame gaps

**Files:**
- Modify: `Module/Optimization/WindowICP.py`
- Modify: `Module/Optimization/GlobalPoseICP.py`
- Modify: `Scripts/UnitTest/test_window_icp.py`
- Modify: `Scripts/UnitTest/test_global_pose_icp.py`

- [ ] **Step 1: Write failing tests for forward long edges and gap diagnostics**

Add these tests, using the existing `fixture()` data:

```python
def test_edge_accepts_positive_long_gap(self):
    _, _, edges = fixture()
    base = edges[0]
    edge = Edge(0, 10, base.points_a, base.points_b, base.cov_a, base.cov_b)
    self.assertEqual((edge.a, edge.b), (0, 10))

def test_edge_rejects_non_forward_endpoint(self):
    _, _, edges = fixture()
    base = edges[0]
    with self.assertRaisesRegex(ValueError, "strictly forward"):
        Edge(4, 4, base.points_a, base.points_b, base.cov_a, base.cov_b)

def test_global_diagnostics_count_edge_gaps(self):
    truth, initial, edges = global_fixture()
    covariance = torch.eye(3, dtype=torch.float64)[None] * 0.001
    world = torch.tensor([[4.0, 0.2, -0.1]], dtype=torch.float64)
    edges.append(Edge(0, 10, truth[0].Inv().Act(world),
                      truth[10].Inv().Act(world), covariance, covariance))
    result = optimize_global_pose_graph(initial, edges, max_iters=15)
    self.assertEqual(result.diagnostics["edge_gaps"]["10"], 1)
```

- [ ] **Step 2: Run the focused tests and verify the current restriction fails**

Run:

```bash
python -m pytest Scripts/UnitTest/test_window_icp.py Scripts/UnitTest/test_global_pose_icp.py -q
```

Expected: the long-edge construction fails with `Only adjacent and two-step edges are supported`, and the diagnostics key is absent.

- [ ] **Step 3: Replace the hard-coded gap restriction and add deterministic counts**

Use this validation in `Edge.__post_init__`:

```python
if not isinstance(self.a, int) or not isinstance(self.b, int) or self.a < 0 or self.b <= self.a:
    raise ValueError("Edge endpoints must be non-negative and strictly forward")
```

Add a shared diagnostic helper in `WindowICP.py`:

```python
def edge_gap_counts(edges):
    counts = {}
    for edge in edges:
        key = str(edge.b - edge.a)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))
```

Use `"edge_gaps": edge_gap_counts(edges)` in both window and global diagnostics. Keep existing `adjacent_edges` and `skip_edges` fields for compatibility.

- [ ] **Step 4: Run the focused tests**

Run:

```bash
python -m pytest Scripts/UnitTest/test_window_icp.py Scripts/UnitTest/test_global_pose_icp.py -q
```

Expected: all focused tests pass.

- [ ] **Step 5: Commit**

```bash
git add Module/Optimization/WindowICP.py Module/Optimization/GlobalPoseICP.py Scripts/UnitTest/test_window_icp.py Scripts/UnitTest/test_global_pose_icp.py
git commit -m "feat: support long-range ICP edges"
```

### Task 2: Add the validated factor archive format

**Files:**
- Create: `Module/Optimization/FactorArchive.py`
- Create: `Scripts/UnitTest/test_factor_archive.py`

- [ ] **Step 1: Write failing round-trip, validation, filtering, merge, and atomic-publication tests**

Define `make_edge(a, b)` with one finite point and SPD identity covariance and
`make_archive(edges=(), kinds=())` with enough identity sensor poses for the
largest endpoint, integer timestamps, and identity `T_BS`. Then add:

```python
def test_archive_round_trip_is_exact(self):
    archive = make_archive([make_edge(0, 1), make_edge(0, 5)], ["adjacent", "gap5"])
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory, "factors.npz")
        save_factor_archive(path, archive)
        loaded = load_factor_archive(path)
    self.assertTrue(torch.equal(loaded.initial_sensor_poses, archive.initial_sensor_poses))
    self.assertTrue(torch.equal(loaded.edges[1].cov_b, archive.edges[1].cov_b))
    self.assertEqual(loaded.edge_kinds, ("adjacent", "gap5"))
    self.assertEqual(loaded.metadata, archive.metadata)

def test_merge_rejects_duplicate_edge_key(self):
    left = make_archive([make_edge(0, 5)], ["gap5"])
    right = make_archive([make_edge(0, 5)], ["gap5"])
    with self.assertRaisesRegex(ValueError, "duplicate"):
        merge_factor_archives(left, right)

def test_filter_keeps_only_requested_edge_kind(self):
    archive = make_archive([make_edge(0, 5), make_edge(0, 10)], ["gap5", "gap10"])
    filtered = filter_factor_archive(archive, {"gap5"})
    self.assertEqual(filtered.edge_kinds, ("gap5",))

def test_failed_publication_does_not_replace_destination(self):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory, "factors.npz")
        path.write_bytes(b"original")
        with patch("Module.Optimization.FactorArchive.os.replace", side_effect=OSError("disk")):
            with self.assertRaisesRegex(OSError, "disk"):
                save_factor_archive(path, make_archive())
        self.assertEqual(path.read_bytes(), b"original")
```

Also reject unsupported schema versions, malformed offsets, non-finite poses, timestamp/pose count mismatches, invalid `T_BS`, unknown edge kinds, endpoint overflow, and source-archive identity mismatches.

- [ ] **Step 2: Run the new test module and verify import failure**

Run:

```bash
python -m pytest Scripts/UnitTest/test_factor_archive.py -q
```

Expected: collection fails because `Module.Optimization.FactorArchive` does not exist.

- [ ] **Step 3: Implement the archive data model and conversion helper**

Create the core interface:

```python
SCHEMA_VERSION = 1
KNOWN_EDGE_KINDS = frozenset({"adjacent", "skip2", "gap5", "gap10"})

@dataclass(frozen=True)
class FactorArchive:
    initial_sensor_poses: torch.Tensor
    time_ns: np.ndarray
    T_BS: torch.Tensor
    edges: tuple[Edge, ...]
    edge_kinds: tuple[str, ...]
    metadata: dict

    def __post_init__(self):
        poses = self.initial_sensor_poses.detach().cpu().double().clone()
        extrinsic = self.T_BS.detach().cpu().double().clone()
        times = np.asarray(self.time_ns, dtype=np.int64).copy()
        if poses.ndim != 2 or poses.shape[1] != 7 or not torch.isfinite(poses).all():
            raise ValueError("initial_sensor_poses must be finite Nx7")
        if not torch.allclose(poses[:, 3:].norm(dim=-1),
                              torch.ones(len(poses), dtype=torch.float64), atol=1e-4):
            raise ValueError("initial_sensor_poses must contain unit quaternions")
        if times.shape != (len(poses),):
            raise ValueError("time_ns must match the pose count")
        if extrinsic.shape != (7,) or not torch.isfinite(extrinsic).all():
            raise ValueError("T_BS must be a finite SE3 vector")
        if not torch.allclose(extrinsic[3:].norm(), torch.tensor(1.0, dtype=torch.float64), atol=1e-4):
            raise ValueError("T_BS must contain a unit quaternion")
        if len(self.edges) != len(self.edge_kinds):
            raise ValueError("edge_kinds must match edges")
        if any(kind not in KNOWN_EDGE_KINDS for kind in self.edge_kinds):
            raise ValueError("unknown edge kind")
        keys = [(edge.a, edge.b) for edge in self.edges]
        if len(keys) != len(set(keys)):
            raise ValueError("factor archive contains duplicate edge keys")
        if any(edge.b >= len(poses) for edge in self.edges):
            raise ValueError("factor endpoint exceeds pose count")
        object.__setattr__(self, "initial_sensor_poses", poses)
        object.__setattr__(self, "time_ns", times)
        object.__setattr__(self, "T_BS", extrinsic)
        object.__setattr__(self, "edges", tuple(self.edges))
        object.__setattr__(self, "edge_kinds", tuple(self.edge_kinds))

def sensor_to_body_trajectory(poses, time_ns, T_BS):
    sensor = pp.SE3(poses)
    extrinsic = pp.SE3(T_BS)
    body = (extrinsic @ sensor @ extrinsic.Inv()).tensor().cpu().numpy()
    return np.concatenate([np.asarray(time_ns, dtype=np.int64)[:, None], body], axis=1)
```

Add `trajectory_identity(poses, time_ns, T_BS)` that hashes contiguous float64
pose bytes, int64 timestamp bytes, and float64 extrinsic bytes with SHA-256.
Store the digest as `metadata["source_id"]`; long archives copy it unchanged.

Serialize ragged observations using `edge_offsets`, write `metadata_json` with sorted JSON keys, open the temporary path as a binary file so NumPy does not append an extra suffix, call `os.replace` only after the file is closed, and remove a remaining temporary file in `finally`.

- [ ] **Step 4: Implement loading, filtering, and merging**

Loading must use `np.load(path, allow_pickle=False)`, require the exact schema fields, reconstruct every `Edge` from the offsets, and return `FactorArchive` so all semantic validation runs again.

Filtering keeps graph metadata and only edges whose kinds are selected:

```python
def filter_factor_archive(archive, kinds):
    selected = [(edge, kind) for edge, kind in zip(archive.edges, archive.edge_kinds) if kind in kinds]
    return replace(archive,
                   edges=tuple(edge for edge, _ in selected),
                   edge_kinds=tuple(kind for _, kind in selected))
```

Merging must require `torch.equal` for initial poses and `T_BS`, `np.array_equal` for timestamps, matching source identity metadata, and disjoint directed edge keys.

- [ ] **Step 5: Run archive tests**

Run:

```bash
python -m pytest Scripts/UnitTest/test_factor_archive.py -q
```

Expected: all archive tests pass.

- [ ] **Step 6: Commit**

```bash
git add Module/Optimization/FactorArchive.py Scripts/UnitTest/test_factor_archive.py
git commit -m "feat: serialize global ICP factor archives"
```

### Task 3: Save exact pre-global online artifacts

**Files:**
- Modify: `MACVO.py`
- Modify: `Odometry/WindowMACVO.py`
- Modify: `Scripts/UnitTest/test_window_global_termination.py`
- Modify: `Scripts/UnitTest/test_window_provenance.py`

- [ ] **Step 1: Write failing tests for snapshot timing and artifact publication**

Extend `make_system()` with timestamps and `T_BS`, then add:

```python
def test_termination_snapshots_interpolated_poses_before_global_writeback(self):
    system, _, _ = make_system()
    interpolated = system.graph.frames.data["pose"].tensor.double().clone()
    refined = interpolated.clone()
    refined[1, 0] += 0.2
    system.Optimizer = SimpleNamespace(optimize_res=None, terminate=Mock())
    system.MapRefiner = SimpleNamespace(elaborate_map=Mock())
    system.frame_cache = {}
    with patch.object(system, "_run_global_refinement") as refine:
        refine.side_effect = lambda before: system.graph.frames.data["pose"].tensor.copy_(refined.float())
        system.terminate()
    self.assertTrue(torch.equal(system._factor_snapshot.initial_sensor_poses, interpolated))

def test_save_diagnostics_writes_reusable_artifacts(self):
    system, _, _ = make_system()
    system._capture_factor_snapshot(system.graph.frames.data["pose"].tensor.double())
    with tempfile.TemporaryDirectory() as directory:
        Path(directory, "run_provenance.json").write_text('{"git_commit":"abc"}')
        system.save_window_diagnostics(directory)
        self.assertTrue(Path(directory, "poses_before_global.npy").is_file())
        self.assertTrue(Path(directory, "global_factors.npz").is_file())
        payload = json.loads(Path(directory, "global_refinement.json").read_text())
    self.assertEqual(payload["artifacts"]["status"], "saved")
```

Add a provenance test asserting the saved sandbox `config.yaml` contains the `Preprocess` section used by the run.

- [ ] **Step 2: Run the focused tests and verify missing snapshot behavior**

Run:

```bash
python -m pytest Scripts/UnitTest/test_window_global_termination.py Scripts/UnitTest/test_window_provenance.py -q
```

Expected: failures for missing `_factor_snapshot`, missing artifact files, and missing saved preprocessing configuration.

- [ ] **Step 3: Persist the exact preprocessing configuration**

Change the experiment sandbox configuration in `MACVO.py` to include:

```python
exp_space.config = {
    "Project": project_name,
    "Odometry": odomcfg_dict,
    "Data": {"args": datacfg_dict, "end_idx": args.seq_to, "start_idx": args.seq_from},
    "Preprocess": cfg_dict["Preprocess"],
}
```

This makes offline frame reconstruction independent of an external odometry YAML path.

- [ ] **Step 4: Capture the factor snapshot before global refinement**

Initialize `self._factor_snapshot = None` and `self.artifact_record = {"status": "pending"}` when global refinement is enabled. Add:

```python
def _capture_factor_snapshot(self, poses):
    if not self.global_refine:
        return
    graph = self.graph.frames.data
    edges = tuple(self._global_edges())
    kinds = tuple({1: "adjacent", 2: "skip2"}[edge.b - edge.a] for edge in edges)
    self._factor_snapshot = FactorArchive(
        initial_sensor_poses=poses,
        time_ns=graph["time_ns"].tensor.cpu().numpy(),
        T_BS=graph["T_BS"].tensor[0].double(),
        edges=edges,
        edge_kinds=kinds,
        metadata={"source": "WindowMACVO", "window_size": self.edge_window.size},
    )
```

Call `_capture_factor_snapshot(after)` immediately before `_run_global_refinement(after)` in `terminate()`. The factor snapshot owns the references required for later serialization, so active/inactive dictionaries may then be cleared without losing data.

- [ ] **Step 5: Publish the three artifacts and preserve failure diagnostics**

In `save_window_diagnostics(folder)`, write:

```python
try:
    archive = self._factor_snapshot
    if self.global_refine and archive is not None:
        np.save(Path(folder, "poses_before_global.npy"), sensor_to_body_trajectory(
            archive.initial_sensor_poses, archive.time_ns, archive.T_BS))
        save_factor_archive(Path(folder, "global_factors.npz"), archive)
        provenance_path = Path(folder, "run_provenance.json")
        provenance = json.loads(provenance_path.read_text()) if provenance_path.is_file() else {}
        self.artifact_record = {
            "status": "saved", "schema_version": SCHEMA_VERSION,
            "poses": "poses_before_global.npy", "factors": "global_factors.npz",
            "factor_bytes": Path(folder, "global_factors.npz").stat().st_size,
            "source_id": archive.metadata["source_id"],
            "git_commit": provenance.get("git_commit"),
            "pose_count": len(archive.initial_sensor_poses),
            "edge_count": len(archive.edges),
            "observation_count": sum(len(edge.points_a) for edge in archive.edges),
        }
except Exception as error:
    self.artifact_record = {"status": "failed", "reason": str(error)}
finally:
    Path(folder, "global_refinement.json").write_text(json.dumps(
        {"schema_version": SCHEMA_VERSION,
         "global_refinement": self.global_record,
         "artifacts": self.artifact_record},
        indent=2, allow_nan=False), encoding="utf-8")
```

Only set `_factor_snapshot = None` after both trajectory and archive publication succeed. Include `artifact_record` in `window_diagnostics.json` while retaining all existing fields.

- [ ] **Step 6: Run the focused tests**

Run:

```bash
python -m pytest Scripts/UnitTest/test_window_global_termination.py Scripts/UnitTest/test_window_provenance.py -q
```

Expected: all focused tests pass.

- [ ] **Step 7: Commit**

```bash
git add MACVO.py Odometry/WindowMACVO.py Scripts/UnitTest/test_window_global_termination.py Scripts/UnitTest/test_window_provenance.py
git commit -m "feat: save reproducible global ICP artifacts"
```

### Task 4: Extract deterministic match-to-factor construction

**Files:**
- Create: `Module/Optimization/MatchICP.py`
- Modify: `Odometry/WindowMACVO.py`
- Create: `Scripts/UnitTest/test_match_icp.py`

- [ ] **Step 1: Write failing parity and rejection-reason tests**

Create complete synthetic `StereoData`, depth-output, matcher-output, selector,
frontend, and covariance-model fixtures in the test module, then add:

```python
def test_match_builder_is_deterministic(self):
    first = build_match_edge(**self.inputs, a=0, b=5)
    second = build_match_edge(**self.inputs, a=0, b=5)
    self.assertEqual(first.reason, None)
    self.assertTrue(torch.equal(first.edge.points_a, second.edge.points_a))

def test_match_builder_reports_too_few_inbound_points(self):
    self.inputs["match"].flow.fill_(1e6)
    result = build_match_edge(**self.inputs, a=0, b=5)
    self.assertIsNone(result.edge)
    self.assertEqual(result.reason, "too_few_inbound_points")

def test_online_skip_wrapper_matches_shared_builder(self):
    expected = build_match_edge(**shared_inputs, a=0, b=2).edge
    actual = system._skip_edge_from_match(0, 2, shared_inputs["match"])
    self.assertTrue(torch.equal(actual.points_a, expected.points_a))
    self.assertTrue(torch.equal(actual.cov_b, expected.cov_b))
```

- [ ] **Step 2: Run the new test module and verify import failure**

Run:

```bash
python -m pytest Scripts/UnitTest/test_match_icp.py -q
```

Expected: collection fails because `Module.Optimization.MatchICP` does not exist.

- [ ] **Step 3: Implement the shared builder**

Create:

```python
@dataclass(frozen=True)
class MatchEdgeBuildResult:
    edge: Edge | None
    reason: str | None
    selected_points: int
    inbound_points: int

@torch.inference_mode()
def build_match_edge(*, a, b, stereo_a, stereo_b, depth_a, depth_b, match,
                     frontend, selector, covariance_model, num_point,
                     min_num_point, edge_width, match_cov_default, device):
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(1000003 + a * 1009 + b)
        uv_a = selector.select_point(stereo_a, num_point, depth_a, depth_b, match)
    selected_points = len(uv_a)
    uv_b = uv_a + frontend.retrieve_pixels(uv_a, match.flow).T
    inbound = torch.isfinite(uv_b).all(-1) & filterPointsInRange(
        uv_b, (edge_width, stereo_b.width - edge_width),
        (edge_width, stereo_b.height - edge_width))
    uv_a, uv_b = uv_a[inbound], uv_b[inbound]
    if len(uv_a) < min_num_point:
        return MatchEdgeBuildResult(None, "too_few_inbound_points",
                                    selected_points, int(inbound.sum()))
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
        return MatchEdgeBuildResult(None, "too_few_valid_observations",
                                    selected_points, int(inbound.sum()))
    edge = Edge(a, b, points_a[ids], points_b[ids], cov_a[ids], cov_b[ids])
    return MatchEdgeBuildResult(edge, None, selected_points, int(inbound.sum()))
```

- [ ] **Step 4: Replace the online implementation with the shared call**

Keep `_skip_edge_from_match` as a compatibility wrapper:

```python
def _skip_edge_from_match(self, a, b, match):
    frame_a, depth_a = self.frame_cache[a]
    frame_b, depth_b = self.frame_cache[b]
    return build_match_edge(
        a=a, b=b, stereo_a=frame_a.stereo, stereo_b=frame_b.stereo,
        depth_a=depth_a, depth_b=depth_b, match=match,
        frontend=self.Frontend, selector=self.KeypointSelector,
        covariance_model=self.ObsCovModel, num_point=self.num_point,
        min_num_point=self.min_num_point, edge_width=self.edge_width,
        match_cov_default=self.match_cov_default, device=self.device,
    ).edge
```

- [ ] **Step 5: Run parity and existing window tests**

Run:

```bash
python -m pytest Scripts/UnitTest/test_match_icp.py Scripts/UnitTest/test_window_frontend_routing.py Scripts/UnitTest/test_window_global_termination.py -q
```

Expected: all tests pass and the online path remains behaviorally identical.

- [ ] **Step 6: Commit**

```bash
git add Module/Optimization/MatchICP.py Odometry/WindowMACVO.py Scripts/UnitTest/test_match_icp.py
git commit -m "refactor: share ICP match factor construction"
```

### Task 5: Generate `t-5` and `t-10` factors offline

**Files:**
- Create: `Module/Optimization/LongRangeICP.py`
- Create: `Scripts/Experiment/GenerateLongRangeICP.py`
- Create: `Scripts/UnitTest/test_long_range_icp.py`

- [ ] **Step 1: Write failing scheduler, fixed-batch routing, cache-bound, and archive tests**

Define a fake indexable sequence and a fake frontend whose return objects encode
their source and target indices, then add:

```python
def test_schedule_for_twelve_frames(self):
    self.assertEqual(long_range_targets(12), [(5, (0,)), (10, (5, 0))])

def test_generator_uses_one_fixed_batch_call_per_target_plus_initialization(self):
    frontend = FakeFrontend()
    result = generate_long_range_archive(fake_sequence(16), fake_components(frontend))
    self.assertEqual(frontend.calls, [(None, 0, 0), (None, 0, 5), (0, 5, 10), (5, 10, 15)])
    self.assertEqual(result.diagnostics["frontend_calls"], 4)

def test_depth_cache_never_exceeds_three_scheduled_frames(self):
    result = generate_long_range_archive(fake_sequence(31), fake_components())
    self.assertLessEqual(result.diagnostics["max_cached_depths"], 3)

def test_t5_duplicate_slot_is_discarded(self):
    result = generate_long_range_archive(fake_sequence(6), fake_components())
    self.assertEqual(result.archive.edge_kinds, ("gap5",))
```

Also test timestamp mismatch, accepted/rejected counts by gap, local inference failure recording, and the no-accepted-edge status.

- [ ] **Step 2: Run the new tests and verify import failure**

Run:

```bash
python -m pytest Scripts/UnitTest/test_long_range_icp.py -q
```

Expected: collection fails because `Module.Optimization.LongRangeICP` does not exist.

- [ ] **Step 3: Implement the deterministic schedule and bounded caches**

Create:

```python
def long_range_targets(frame_count, stride=5, gaps=(5, 10)):
    return [
        (target, tuple(target - gap for gap in gaps if target - gap >= 0))
        for target in range(stride, frame_count, stride)
    ]
```

The generator processes scheduled indices only:

```python
frame0 = sequence[0]
frame_cache = {0: frame0}
depth_cache = {}
depth0, _, _ = frontend.estimate_window(None, frame0.stereo, frame0.stereo)
depth_cache[0] = depth0
for target, sources in long_range_targets(len(sequence)):
    current = sequence[target]
    if len(sources) == 1:
        depth_t, gap5_match, _ = frontend.estimate_window(
            None, frame_cache[sources[0]].stereo, current.stereo)
        matches = [(sources[0], "gap5", gap5_match)]
    else:
        depth_t, gap5_match, gap10_match = frontend.estimate_window(
            frame_cache[sources[1]].stereo,
            frame_cache[sources[0]].stereo,
            current.stereo,
        )
        matches = [(sources[0], "gap5", gap5_match),
                   (sources[1], "gap10", gap10_match)]
    depth_cache = {idx: value for idx, value in depth_cache.items()
                   if idx in sources}
    frame_cache = {idx: value for idx, value in frame_cache.items()
                   if idx in sources}
    depth_cache[target] = depth_t
    frame_cache[target] = current
    for source, kind, match in matches:
        built = build_match_edge(
            a=source, b=target,
            stereo_a=frame_cache[source].stereo,
            stereo_b=current.stereo,
            depth_a=depth_cache[source], depth_b=depth_t, match=match,
            frontend=frontend, selector=selector,
            covariance_model=covariance_model, num_point=num_point,
            min_num_point=min_num_point, edge_width=edge_width,
            match_cov_default=match_cov_default, device=device)
        attempted[kind] += 1
        if built.edge is None:
            rejected[kind][built.reason] += 1
        else:
            edges.append(built.edge)
            edge_kinds.append(kind)
            accepted[kind] += 1
    oldest_needed = target - 5
    depth_cache = {idx: value for idx, value in depth_cache.items()
                   if idx >= oldest_needed}
    frame_cache = {idx: value for idx, value in frame_cache.items()
                   if idx >= oldest_needed}
```

Use `build_match_edge` for every candidate. Store accepted edges and kinds in a `FactorArchive` whose poses/timestamps/extrinsic are copied from the source short archive. Diagnostics must include attempted, accepted, rejected-by-reason, observations, frontend calls, maximum cache sizes, and elapsed seconds.

- [ ] **Step 4: Implement the CLI using only the saved result configuration**

`GenerateLongRangeICP.py` must:

```python
space = Sandbox.load(args.space)
cfg = space.config
source = load_factor_archive(space.path("global_factors.npz"))
sequence = smart_transform(
    SequenceBase[StereoFrame].instantiate(cfg.Data.args.type, cfg.Data.args.args)
        .clip(cfg.Data.start_idx, cfg.Data.end_idx),
    cfg.Preprocess,
)
system = WindowMACVO.from_config(cfg)
result = generate_long_range_archive(sequence, source, frontend=system.Frontend,
    selector=system.KeypointSelector, covariance_model=system.ObsCovModel,
    num_point=system.num_point, min_num_point=system.min_num_point,
    edge_width=system.edge_width, match_cov_default=system.match_cov_default,
    device=system.device)
save_factor_archive(space.path(args.output), result.archive)
```

Default output is `long_factors_gap5_10.npz`; diagnostics are written to `long_factors_gap5_10.json`. Reject a source result lacking `Preprocess`, `global_factors.npz`, or matching pose timestamps before loading the network.

Set long-archive metadata to include `source_id`, SHA-256 of
`global_factors.npz`, source result path, frame range, gaps `[5, 10]`, stride
`5`, point thresholds, configuration identity, and accepted/rejected counts.

- [ ] **Step 5: Run generator tests**

Run:

```bash
python -m pytest Scripts/UnitTest/test_long_range_icp.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add Module/Optimization/LongRangeICP.py Scripts/Experiment/GenerateLongRangeICP.py Scripts/UnitTest/test_long_range_icp.py
git commit -m "feat: generate offline long-range ICP factors"
```

### Task 6: Add offline refinement and direct pose-file evaluation

**Files:**
- Create: `Scripts/Experiment/RefineGlobalPoseICP.py`
- Create: `Scripts/UnitTest/test_offline_global_refinement.py`

- [ ] **Step 1: Write failing short-only reproduction, tagged-output, and metric tests**

Define synthetic short and long archives plus a temporary sandbox containing
`poses.npy` and `ref_poses.npy`, then add:

```python
def test_short_only_refinement_uses_saved_initial_sensor_poses(self):
    source = make_archive(short_edges())
    result = refine_archives(source, None, edge_kinds=None, iterations=10, huber=3.0)
    expected = optimize_global_pose_graph(source.initial_sensor_poses, source.edges, max_iters=10)
    self.assertTrue(torch.allclose(result.poses, expected.poses, atol=1e-12, rtol=0))

def test_gap5_filter_excludes_gap10_edges(self):
    source, long = make_short_and_long_archives()
    result = refine_archives(source, long, edge_kinds={"gap5"}, iterations=5, huber=3.0)
    self.assertEqual(result.diagnostics["edge_gaps"].get("10", 0), 0)

def test_tagged_outputs_do_not_overwrite_source_poses(self):
    original = Path(space, "poses.npy").read_bytes()
    write_refinement_outputs(space, "gap5", result, source, metrics)
    self.assertEqual(Path(space, "poses.npy").read_bytes(), original)
    self.assertTrue(Path(space, "poses_global_gap5.npy").is_file())
```

Test metric evaluation against `ref_poses.npy`, missing-reference status, anchor preservation, non-finite solver output handling, and invalid output tags.

- [ ] **Step 2: Run the new tests and verify import failure**

Run:

```bash
python -m pytest Scripts/UnitTest/test_offline_global_refinement.py -q
```

Expected: collection fails because `Scripts.Experiment.RefineGlobalPoseICP` does not exist.

- [ ] **Step 3: Implement archive refinement and tagged output**

Use these interfaces:

```python
def refine_archives(short_archive, long_archive=None, edge_kinds=None,
                    iterations=5, huber=3.0):
    selected = long_archive
    if selected is not None and edge_kinds is not None:
        selected = filter_factor_archive(selected, edge_kinds)
    merged = short_archive if selected is None else merge_factor_archives(short_archive, selected)
    return optimize_global_pose_graph(
        merged.initial_sensor_poses, merged.edges,
        max_iters=iterations, huber_delta=huber)

def write_refinement_outputs(space, tag, result, archive, metrics):
    if not re.fullmatch(r"[a-z0-9_]+", tag):
        raise ValueError("output tag must contain lowercase letters, digits, or underscores")
    if result.diagnostics["status"] != "refined":
        return
    np.save(Path(space, f"poses_global_{tag}.npy"), sensor_to_body_trajectory(
        result.poses, archive.time_ns, archive.T_BS))
    Path(space, f"global_{tag}_diagnostics.json").write_text(
        json.dumps(result.diagnostics, indent=2, allow_nan=False), encoding="utf-8")
    Path(space, f"global_{tag}_metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
```

- [ ] **Step 4: Implement direct evaluation of a tagged pose file**

Load `ref_poses.npy` and the tagged pose file with `Trajectory.from_timed_SE3_numpy`, apply the same origin and timestamp alignment as `Trajectory.from_sandbox`, then call `evaluateATE`, `evaluateRTE`, `evaluateROE`, and `evaluateRPE`. Return JSON keys `RMSE_ATE`, `RMSE_RTE`, `RMSE_ROE`, and `RMSE_RPE`. If reference poses are absent, return `{"status": "unavailable", "reason": "ref_poses.npy is missing"}`.

- [ ] **Step 5: Implement CLI arguments and defaults**

Support:

```text
--space PATH
--long-factors PATH
--edge-kinds gap5 gap10
--iterations 5
--huber 3.0
--output-tag short|gap5|gap5_10
```

No long archive with tag `short` is valid. A long archive must share source identity, poses, timestamps, and extrinsic with the short archive.

- [ ] **Step 6: Run offline-refinement tests**

Run:

```bash
python -m pytest Scripts/UnitTest/test_offline_global_refinement.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add Scripts/Experiment/RefineGlobalPoseICP.py Scripts/UnitTest/test_offline_global_refinement.py
git commit -m "feat: rerun global ICP from saved factors"
```

### Task 7: Add a reproducible comparison orchestrator

**Files:**
- Create: `Scripts/Experiment/CompareLongRangeICP.py`
- Create: `Scripts/UnitTest/test_compare_long_range_icp.py`
- Modify: `docs/WindowICP.md`

- [ ] **Step 1: Write failing aggregation and ATE-gate tests**

Create temporary comparison JSON fixtures with explicit SHA-256 values and add:

```python
def test_comparison_rows_share_source_identity(self):
    rows = build_rows(source_space, fake_metrics)
    self.assertEqual({row["source_factor_sha256"] for row in rows}, {"same"})
    self.assertEqual([row["mode"] for row in rows],
                     ["before_global", "short", "gap5", "gap5_10"])

def test_stage_gate_requires_gap5_10_ate_below_short(self):
    rows = [{"mode": "short", "RMSE_ATE": 0.50},
            {"mode": "gap5_10", "RMSE_ATE": 0.49}]
    self.assertTrue(stage_one_passed(rows))
    rows[1]["RMSE_ATE"] = 0.51
    self.assertFalse(stage_one_passed(rows))
```

Also test incomplete source runs, more than one discovered result leaf, missing artifacts, and JSON/CSV consistency.

- [ ] **Step 2: Run the new test module and verify import failure**

Run:

```bash
python -m pytest Scripts/UnitTest/test_compare_long_range_icp.py -q
```

Expected: collection fails because `Scripts.Experiment.CompareLongRangeICP` does not exist.

- [ ] **Step 3: Implement online-source creation and offline reuse**

The orchestrator accepts either an existing `--space` or creates exactly one source run with:

```bash
python MACVO.py --odom Config/Experiment/MACVO/MACVO_Fast_WindowICP_Global.yaml \
  --data Config/Sequence/EuRoC_V203_local.yaml --resultRoot SOURCE_ROOT \
  --seed 0 --seq_from START --seq_to END --noeval --timing
```

After validating `run_provenance.json`, it runs long-factor generation once and calls offline refinement for `short`, `gap5`, and `gap5_10`. It directly evaluates `poses_before_global.npy` for the fourth baseline row.

Also implement `--verify-space PATH`. Verification discovers exactly one result
leaf beneath `PATH`, validates both archives, recomputes their SHA-256 values,
checks every tagged trajectory timestamp against `poses_before_global.npy`,
checks the first pose is exact, and confirms `poses.npy` still matches the hash
recorded before offline refinement.

- [ ] **Step 4: Write comparison artifacts and enforce the stage gate**

Write `long_range_metrics.json` and `long_range_metrics.csv` with:

```text
sequence, frame_from, frame_to, seed, mode, space,
source_factor_sha256, long_factor_sha256,
edges, observations, initial_cost, final_cost, solver_seconds,
RMSE_ATE, RMSE_RTE, RMSE_ROE, RMSE_RPE
```

Write `stage_gate.json` containing `passed`, short ATE, gap5+10 ATE, absolute change, and percentage change. Exit successfully even when the scientific gate fails, but clearly print `Stage 1 did not lower ATE; full V203 is not authorized by the design gate.`

- [ ] **Step 5: Document exact commands and artifact meanings**

Add to `docs/WindowICP.md`:

```bash
python Scripts/Experiment/CompareLongRangeICP.py \
  --sequence V203 --seq-from 1095 --seq-to 1215 --seed 0 \
  --result-root Results/LongRangeICP_V203_short
```

Document how `--space` skips the online run, how each NPZ/JSON/NPY file is used, and that tagged outputs never overwrite `poses.npy`.

- [ ] **Step 6: Run orchestrator unit tests**

Run:

```bash
python -m pytest Scripts/UnitTest/test_compare_long_range_icp.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add Scripts/Experiment/CompareLongRangeICP.py Scripts/UnitTest/test_compare_long_range_icp.py docs/WindowICP.md
git commit -m "feat: compare offline long-range ICP variants"
```

### Task 8: Run regression and integration verification

**Files:**
- Modify only if a regression exposes an implementation defect in files already listed above.

- [ ] **Step 1: Run all WindowICP and new offline tests**

Run:

```bash
python -m pytest \
  Scripts/UnitTest/test_window_icp.py \
  Scripts/UnitTest/test_global_pose_icp.py \
  Scripts/UnitTest/test_window_global_termination.py \
  Scripts/UnitTest/test_window_frontend.py \
  Scripts/UnitTest/test_window_frontend_routing.py \
  Scripts/UnitTest/test_factor_archive.py \
  Scripts/UnitTest/test_match_icp.py \
  Scripts/UnitTest/test_long_range_icp.py \
  Scripts/UnitTest/test_offline_global_refinement.py \
  Scripts/UnitTest/test_compare_long_range_icp.py -q
```

Expected: all focused tests pass.

- [ ] **Step 2: Run the complete non-local unit suite**

Run:

```bash
python -m pytest Scripts/UnitTest -q
```

Expected: the suite passes; dataset- or GPU-only tests may skip according to their existing markers, but no new failure is accepted.

- [ ] **Step 3: Check command-line entry points without loading the model**

Run:

```bash
python Scripts/Experiment/GenerateLongRangeICP.py --help
python Scripts/Experiment/RefineGlobalPoseICP.py --help
python Scripts/Experiment/CompareLongRangeICP.py --help
```

Expected: all commands exit zero and list the documented arguments.

- [ ] **Step 4: Check formatting and repository state**

Run:

```bash
git diff --check
git status --short
```

Expected: no whitespace errors and only intentional validation artifacts remain uncommitted.

- [ ] **Step 5: Commit any verified integration corrections**

If Step 1 or Step 2 required corrections, commit only those corrections:

```bash
git add Module Odometry Scripts MACVO.py docs/WindowICP.md
git commit -m "fix: complete offline long-range ICP integration"
```

If no corrections were needed, do not create an empty commit.

### Task 9: Validate the difficult V203 segment and conditionally full V203

**Files:**
- Create: `docs/validation/window_icp_long_range_v203_1095_1215.csv`
- Create conditionally: `docs/validation/window_icp_long_range_v203_full.csv`
- Modify: `docs/WindowICP.md`

- [ ] **Step 1: Confirm runtime prerequisites before the expensive run**

Run:

```bash
test -f Model/MACVO_FrontendCov.pth
test -f /media/zzh/data/EUROC_Dataset/EuRoC/V2_03_difficult/mav0/cam0/sensor.yaml
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
```

Expected: checkpoint and V203 data exist, and the CUDA device is visible with enough free memory for the existing batch-three graph.

- [ ] **Step 2: Run the one-online-run short comparison**

Run:

```bash
python Scripts/Experiment/CompareLongRangeICP.py \
  --sequence V203 --seq-from 1095 --seq-to 1215 --seed 0 \
  --result-root Results/LongRangeICP_V203_short
```

Expected: one online `WindowMACVO` source run, one offline long-factor generation pass, three offline global solves, and four metric rows. The command writes `stage_gate.json`.

- [ ] **Step 3: Verify strict comparability and archive integrity**

Run the comparison script's validation subcommand:

```bash
python Scripts/Experiment/CompareLongRangeICP.py \
  --verify-space Results/LongRangeICP_V203_short
```

Expected: identical timestamps and source-factor SHA256 across all four variants, valid archives, exact fixed anchor, finite trajectories, and no source output overwrite.

- [ ] **Step 4: Record short-sequence results**

Copy the generated CSV to `docs/validation/window_icp_long_range_v203_1095_1215.csv` and summarize edge counts, observations, generation time, solver time, objective changes, and all four trajectory metrics in `docs/WindowICP.md`.

- [ ] **Step 5: Apply the ATE gate**

Read `stage_gate.json`. Continue only when `passed` is true, meaning `gap5_10` ATE RMSE is strictly below short-only ATE RMSE with finite output, preserved anchor, and lower robust objective. If false, stop expensive execution and use the saved factors for offline weighting/outlier experiments.

- [ ] **Step 6: Run complete V203 only after the short gate passes**

Run:

```bash
python Scripts/Experiment/CompareLongRangeICP.py \
  --sequence V203 --seq-from 0 --seed 0 \
  --result-root Results/LongRangeICP_V203_full
```

Expected: one complete online artifact source, approximately one offline batch-three call per five frames plus initialization, and three offline solves from identical pre-global poses.

- [ ] **Step 7: Verify and document complete-sequence results**

Run:

```bash
python Scripts/Experiment/CompareLongRangeICP.py \
  --verify-space Results/LongRangeICP_V203_full
```

Copy the generated CSV to `docs/validation/window_icp_long_range_v203_full.csv`. Update `docs/WindowICP.md` with the strict short-only versus long-range ATE conclusion and note whether the missing-observability hypothesis was supported.

- [ ] **Step 8: Commit validation evidence**

```bash
git add docs/WindowICP.md docs/validation/window_icp_long_range_v203_1095_1215.csv
git commit -m "docs: validate offline long-range ICP on V203"
```

When the full-sequence CSV exists, add it with a separate `git add` command
before committing.

Do not commit model files, dataset files, result directories, or large factor archives.
