#!/usr/bin/env python3
"""View the complete MH01 point cloud in an interactive Open3D window."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("Results/MACVO-Fast@MH01_MH01_pointcloud.ply"))
    parser.add_argument("--point-size", type=float, default=2.0)
    args = parser.parse_args()

    try:
        import open3d as o3d
    except ImportError as exc:
        raise SystemExit("Open3D 未安装，请先运行: pip install open3d") from exc

    cloud = o3d.io.read_point_cloud(str(args.input))
    if cloud.is_empty():
        raise SystemExit(f"无法读取点云或点云为空: {args.input}")
    print(f"Loaded {len(cloud.points):,} points from {args.input}")
    viewer = o3d.visualization.Visualizer()
    viewer.create_window(window_name="MAC-VO · MH01 Full Point Cloud", width=1440, height=900)
    viewer.add_geometry(cloud)
    viewer.get_render_option().point_size = args.point_size
    viewer.get_render_option().background_color = [0.03, 0.07, 0.08]
    viewer.run()
    viewer.destroy_window()


if __name__ == "__main__":
    main()
