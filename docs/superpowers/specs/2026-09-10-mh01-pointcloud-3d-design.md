# MH01 点云可交互 3D 展示设计

## 目标
从 `Results/MACVO-Fast@MH01_MH01_pointcloud.ply` 读取点云，生成可双击打开的 `Results/MH01_pointcloud_3d.html`，支持鼠标拖动旋转、平移和滚轮缩放，视觉采用深色背景与青绿色点云，接近项目 GIF 的观感。

## 实现
新增 Python 脚本 `Scripts/Visualize/visualize_mh01_pointcloud.py`：使用 numpy 解析 ASCII/binary PLY（优先复用 plyfile，缺失时给出明确安装提示），按上限随机抽样点，使用 Plotly Scatter3d 输出独立 HTML。命令行参数支持输入 PLY、输出 HTML、最大点数和随机种子。

## 验证
运行脚本生成 HTML，并检查文件存在、点数统计和 HTML 中包含 Plotly 3D 图表配置。
