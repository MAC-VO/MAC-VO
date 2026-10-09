# MH01 Interactive Point Cloud Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Generate a standalone draggable 3D HTML viewer for the existing MH01 PLY point cloud.

**Architecture:** A focused CLI Python script reads PLY vertices, subsamples deterministically, and builds a Plotly `Scatter3d` figure with dark styling and camera controls. Output is self-contained HTML suitable for opening locally.

**Tech Stack:** Python, numpy, Plotly.

### Task 1: Viewer script
**Files:** Create `Scripts/Visualize/visualize_mh01_pointcloud.py`
- [ ] Parse ASCII and binary PLY using `plyfile`, with a clear dependency error.
- [ ] Add CLI arguments for input/output/max-points/seed.
- [ ] Build dark Plotly 3D scatter with equal aspect, hover coordinates, and title.
- [ ] Generate HTML and print summary.

### Task 2: Verify output
**Files:** Generated `Results/MH01_pointcloud_3d.html`
- [ ] Run the script against the checked-in MH01 PLY.
- [ ] Confirm output exists and contains Plotly `scatter3d` markup.
