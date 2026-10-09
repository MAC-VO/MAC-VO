#!/usr/bin/env python3
"""Create a standalone, mouse-driven 3D viewer for an MH01 PLY point cloud."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import plotly.graph_objects as go


def read_ply_vertices(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read x/y/z and optional RGB from an ASCII PLY vertex element."""
    with path.open("rb") as f:
        header: list[str] = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError("PLY header is missing end_header")
            decoded = line.decode("ascii").strip()
            header.append(decoded)
            if decoded == "end_header":
                break
        if not any(line.startswith("format ascii") for line in header):
            raise ValueError("Only ASCII PLY files are supported")
        vertex_line = next((line for line in header if line.startswith("element vertex ")), None)
        if vertex_line is None:
            raise ValueError("PLY has no vertex element")
        count = int(vertex_line.split()[-1])
        columns = [line.split()[-1] for line in header if line.startswith("property ")]
        rows = np.loadtxt(f, max_rows=count, dtype=np.float32)
    rows = np.atleast_2d(rows)
    xyz = rows[:, [columns.index(axis) for axis in ("x", "y", "z")]].astype(np.float32)
    rgb_names = ("red", "green", "blue")
    if all(name in columns for name in rgb_names):
        rgb = np.clip(rows[:, [columns.index(name) for name in rgb_names]], 0, 255).astype(np.uint8)
    else:
        rgb = np.full((len(xyz), 3), 180, dtype=np.uint8)
    return xyz, rgb


def build_viewer(xyz: np.ndarray, rgb: np.ndarray, title: str) -> go.Figure:
    fig = go.Figure(go.Scatter3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers",
        marker=dict(size=1.5, color="#39d9c5", opacity=0.88),
        hoverinfo="none", name="MH01 points"))
    fig.update_layout(title=dict(text=title, x=0.02, y=0.97, font=dict(size=20, color="#d8f3f0")),
        paper_bgcolor="#071316", plot_bgcolor="#071316", font=dict(color="#b9d7d4"),
        margin=dict(l=0, r=0, t=45, b=0), showlegend=False,
        scene=dict(aspectmode="data", bgcolor="#071316", xaxis=dict(title="X", gridcolor="#1d3b3d"),
                   yaxis=dict(title="Y", gridcolor="#1d3b3d"), zaxis=dict(title="Z", gridcolor="#1d3b3d")),
        uirevision="mh01")
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("Results/MACVO-Fast@MH01_MH01_pointcloud.ply"))
    parser.add_argument("--output", type=Path, default=Path("Results/MH01_pointcloud_3d.html"))
    parser.add_argument("--max-points", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    xyz, rgb = read_ply_vertices(args.input)
    if args.max_points > 0 and len(xyz) > args.max_points:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(xyz), args.max_points, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    build_viewer(xyz, rgb, f"MH01 Point Cloud · {len(xyz):,} points").write_html(args.output, include_plotlyjs=True, full_html=True)
    print(f"Wrote {args.output} ({len(xyz):,} points)")


if __name__ == "__main__":
    main()
