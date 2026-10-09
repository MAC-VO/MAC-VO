# MAC-VO 五帧滑动窗口 ICP v0.1

## 版本与范围

- 开发分支：experiment/sliding-window-icp-v1。
- 两帧实验基线：5542d91；标签 baseline/fast-icp-euroc-20260912。
- 求解器首次提交：c0f8d55；前端接入：553913d。
- 已验证代码提交：a0f49b4。
- 完成标签：window-icp-v0.1。
- 当前主目录位于开发分支，main未合并；未推送远程。
- 旧论文、译稿、可视化和用户已有报告不纳入本次提交。旧结果不覆盖。

本版本实现位姿级多帧ICP，不是联合优化地图点的BA，不依赖BAE。
网络仍使用原有FlowFormerCov权重、encoder FP16、decoder BF16、12轮解码、最多200个点、CovAwareSelector_NoDepth及MatchCovariance。
依赖环境不变：Python 3.12，torch 2.4.1+cu124，pypose 0.9.5。

## 配置和数据流

新配置：[MACVO_Fast_WindowICP.yaml](../Config/Experiment/MACVO/MACVO_Fast_WindowICP.yaml)。

~~~yaml
Odometry:
  type: WindowMACVO
  name: MACVO-Fast-WindowICP5
  args:
    window_size: 5
    skip_matching: true
    window_iterations: 10
    window_huber_delta: 3.0
  optimizer:
    type: TwoFrame_PGO
    args:
      graph_type: icp
      autodiff: false
      parallel: false
~~~

optimizer字段指定的是同步两帧ICP初值器。WindowMACVO先执行它，再执行自己的窗口联合求解；不是把旧优化器的frames参数扩成5就结束。

完整步骤：

1. 旧前端对 t-1→t 推理，生成当前双目深度和相邻匹配。
2. 旧两帧ICP根据完整三维协方差给当前帧一个位姿初值。
3. 立即写回并清空初值器结果，避免下次回写覆盖窗口结果。
4. 缓存最近5帧图像、深度和局部三维观测。
5. 用FlowFormerCov直接计算 t-2→t，独立选点、采样深度并生成三维协方差。
6. 窗口内建立相邻与间隔一帧的边，最多4+3=7条，每条最多200个观测。
7. 固定窗口最早位姿，在其局部坐标系联合优化其他最多4个位姿。
8. 将优化结果转换回世界坐标，并同时修正这些帧所拥有的世界点及其协方差。
9. 移除窗口外缓存和约束；窗口外轨迹保留，但不再参与本轮求解。

窗口启动阶段依次使用2、3、4、5帧；稳定阶段始终5帧。
间隔一帧匹配为网络直接推理，不是简单把两段光流相加。
跨帧匹配沿用既有batch=2 CUDA Graph，因此会同时重复计算一次当前立体匹配，只消费时间匹配半部分，并使用已有深度缓存。这是第一版为保持单个CUDA Graph内存池作出的性能取舍。

## 优化目标与求解

令 T_a、T_b 为两个相机到当前窗口锚定坐标系的位姿，p_a、p_b 为各自相机坐标中的固定三维观测：

$$
r_i=T_a p_{i,a}-T_b p_{i,b},
$$

$$
S_i=R_a\Sigma_{i,a}R_a^T+R_b\Sigma_{i,b}R_b^T.
$$

S保留全部非对角项。每个残差同时连接两个位姿，因此形成真实的多位姿耦合。用Cholesky分解进行白化：

$$
S_i=L_iL_i^T,\quad \bar r_i=L_i^{-1}r_i,\quad
\|\bar r_i\|^2=r_i^TS_i^{-1}r_i.
$$

目标为向量Huber损失之和，阈值为白化范数3：

$$
E=\sum_i \rho(\|\bar r_i\|),\qquad
\rho(n)=
\begin{cases}n^2&n\le3\\6n-9&n>3.\end{cases}
$$

左乘SE(3)扰动下，点位置 q_a=T_a p_a、q_b=T_b p_b 的Jacobian为：

$$
J_a=[I,-[q_a]_\times],\qquad
J_b=[-I,[q_b]_\times].
$$

锚定帧不产生变量列。将J用相同L白化，按Huber权重构造正规方程：

$$
(H+\lambda\operatorname{diag}H)\delta=-g.
$$

CPU float64直接求解最多24×24系统，不建立整个观测集合的巨大块对角权重矩阵。
每个外层步骤固定当前白化权重构造Jacobian，不对旋转相关协方差求导；试探位姿重新计算协方差和鲁棒目标，仅接受目标下降的步长。
因此这是显式记录目标的迭代重加权、阻尼ICP，并非包含协方差导数和logdet项的完整最大似然求解。

窗口输入、局部位姿复合和输出均进行四元数正规化，避免FP32历史回写误差被反复坐标变换放大。最早锚定帧的原始位姿行保持不变。

## 地图一致性和退化处理

历史位姿改变时，该帧作为源观测建立的世界点按

$$
\Delta T=T_{\rm new}T_{\rm old}^{-1},\qquad
P_{\rm new}=\Delta T P_{\rm old},\qquad
\Sigma_{\rm new}=R_{\Delta T}\Sigma_{\rm old}R_{\Delta T}^T
$$

更新。构造修正前先正规化旋转。其他帧拥有的点不修改。

- 无效深度、非有限观测和非正定协方差不能进入窗口约束。
- 每条边至少10个有效观测。
- 窗口不连通时保留旧两帧初值，记作not_refined，并记录原因。
- 如果跨帧约束成功恢复原本弱相邻匹配的非锚定帧，清除其插值标志，防止结束时覆盖恢复结果。
- 结束时的轨迹插值若改变位姿，也同步修正对应地图点。
- 当前仅支持AllKeyframe、mapping=false、CovarianceSanityFilter和同步ICP初值器。

## 命令

在当前开发分支运行完整MH01，结束后自动评估：

~~~bash
cd /home/zzh/MACVO
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python MACVO.py \
  --odom Config/Experiment/MACVO/MACVO_Fast_WindowICP.yaml \
  --data Config/Sequence/EuRoC_MH01_local.yaml \
  --seed 0 --resultRoot Results --timing
~~~

仍可使用两个旧配置运行两帧DISP/ICP；未指定Odometry.type时走旧MACVO路径。

三组重新运行并比较同一帧段：

~~~bash
.venv/bin/python Scripts/Experiment/CompareWindowICP.py \
  --sequences MH01 --seq_from 1500 --seq_to 1620 --seed 0
~~~

三组分别为 twoframe、window_adjacent、window_skip。
脚本串行运行，使用独立实验目录，不从旧目录选择“最新结果”，检查运行完整性和时间戳一致性，再输出CSV/JSON。

全部EuRoC完整三组比较：

~~~bash
.venv/bin/python Scripts/Experiment/CompareWindowICP.py \
  --sequences MH01 MH02 MH03 MH04 MH05 V101 V102 V103 V201 V202 V203 \
  --seed 0
~~~

仅运行MH02至V203的五帧“相邻＋间隔一帧”版本，并在每条序列结束后自动评估：

~~~bash
./Scripts/run_euroc_window_icp_mh02_v203.sh
~~~

可用环境变量指定随机种子和输出根目录：

~~~bash
SEED=0 RESULT_ROOT=/home/zzh/MACVO/Results/WindowICP_EuRoC \
./Scripts/run_euroc_window_icp_mh02_v203.sh
~~~

脚本默认顺序为MH02、MH03、MH04、MH05、V101、V102、V103、V201、V202、V203。
也可以在中断后显式传入尚未运行的序列，例如：

~~~bash
./Scripts/run_euroc_window_icp_mh02_v203.sh V103 V201 V202 V203
~~~

脚本使用文件锁防止同一输出根目录同时启动两个批次。MACVO的进度条、模块信息、Timer Report和异常会实时显示在终端，同时原样写入逐序列run.log和批次launcher日志；最终输出metrics.csv和metrics.json。设置DRY_RUN=1可只检查并打印命令。

仅跑所需窗口版本并评估（不重新跑基线）：

~~~bash
.venv/bin/python Scripts/Experiment/CompareWindowICP.py \
  --sequences MH01 --modes window_skip --seed 0
~~~

## 已完成验证

11项CPU测试：

~~~bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m unittest \
  Scripts.UnitTest.test_window_icp \
  Scripts.UnitTest.test_window_map \
  Scripts.UnitTest.test_window_provenance -v
~~~

覆盖：五帧位姿恢复、固定锚、双位姿Jacobian有限差分、完整协方差白化、窗口淘汰、断连拒绝、NaN拒绝、40帧FP32连续回写、地图点及协方差重锚定、恢复位姿不被插值覆盖、初始化失败状态记录。
仓库内置TartanAir双目10帧序列也完成实际GPU推理验证。

### MH01真实帧段

范围为序列索引1500至1619，共120帧，seed=0；不是完整MH01基准。
三组代码版本均为a0f49b4，使用同一时间戳序列。

| 模式 | ATE RMSE (m) | RTE RMSE (m/frame) | ROE RMSE (°/frame) | 平均Odom耗时 |
|---|---:|---:|---:|---:|
| 两帧ICP | 0.037271 | 0.001835 | 0.024682 | 595.8 ms |
| 五帧，仅相邻边 | 0.041320 | 0.001921 | 0.026319 | 657.7 ms |
| 五帧，相邻+间隔一帧 | 0.029949 | 0.001560 | 0.021205 | 1240.7 ms |

两种窗口模式均有119次有效优化，not_refined=0。
仅相邻模式最大缓存5帧/4边；跨帧模式最大5帧/7边。
跨帧模式窗口求解平均约63.7 ms，额外匹配平均约561.1 ms，显存日志约2.9GB（PyTorch保留显存口径）。

本段中跨帧版本更准确，但耗时约为两帧版本的2.08倍；仅扩大窗口并未改善精度。不能据此推断全EuRoC一定改善。
此外，两帧初值器和窗口求解器的鲁棒损失实现、阈值及同步方式不同；这个实验不是“仅改一个window_size、其他数值完全相同”的严格消融。

原始结果目录：
Results/WindowICP_comparison/20260912_214644_1452fb。
完整精度与耗时记录已版本化至 [CSV](validation/window_icp_mh01_1500_1620.csv)。
早期20260912_211312_d1c8a3运行用于定位四元数误差，含大量not_refined窗口，不能用作最终实验结论。

### batch=3融合前端验证

实现提交`ffd2fb4`将每个时间步的两次batch=2前端调用合并为一次固定batch=3 CUDA Graph：

1. 当前左图到当前右图，生成当前立体深度与协方差。
2. `t-1`左图到当前左图，生成相邻时间光流与协方差。
3. `t-2`左图到当前左图，生成跨帧时间光流与协方差。

首个时间步还没有`t-2`时，第三槽用相邻时间图像对填充但不消费输出，因此整条序列只捕获一个batch=3 CUDA Graph。`MACVO.run_pair`已拆分为前端推理和使用预计算结果构建因子两个阶段；普通MACVO仍使用原batch=2路径。WindowMACVO的关键点选择、协方差投影、直接`t-2 -> t`约束和窗口求解器均未改变。

在RTX 4060 Laptop GPU上使用MH01索引1500至1619、共120帧、seed=0验证。历史结果为提交`a0f49b4`的window_skip，新结果运行于`7e7f30a`（核心实现截至`fa4924c`）：

| 指标 | 两次batch=2 | 一次batch=3 | 变化 |
|---|---:|---:|---:|
| Frontend调用数 | 237 | 119 | -49.8% |
| 平均Odom耗时 | 1240.7 ms | 966.2 ms | -22.1% |
| 稳态Odom耗时（去掉前两项） | 1221.1 ms | 936.5 ms | -23.3% |
| 稳态Frontend耗时 | 553.9 ms/次 | 833.2 ms/次 | 单次处理三对输入 |
| ATE RMSE | 0.029949 m | 0.030170 m | +0.74% |
| RTE RMSE | 0.001560 | 0.001555 | -0.37% |
| ROE RMSE | 0.021205° | 0.021166° | -0.18% |
| RPE RMSE | 0.001646 | 0.001640 | -0.35% |
| 未优化窗口 | 0 | 0 | 不变 |

两次运行的120个时间戳完全一致。batch维度改变会引入轻微浮点顺序差异，最终四项误差变化均小于0.8%。新版本119个窗口全部refined，最大缓存仍为5帧/7边；CUDA保留显存日志约4.25GB，低于8GB设备容量。另用仓库内置10帧TartanAir双目序列验证得到9次前端调用、9个有效窗口和有限轨迹。

原定850 ms/帧目标没有达到。融合后的batch=3前端稳态仍需约833.2 ms，已经成为绝对瓶颈；若继续优化，应在不改变约束的前提下优先考虑图像特征缓存或更高效的三对联合前端，而不是优化约57 ms的CPU窗口求解。

### v0.3 active/inactive因子与全局pose-only ICP

v0.3新增独立配置[MACVO_Fast_WindowICP_Global.yaml](../Config/Experiment/MACVO/MACVO_Fast_WindowICP_Global.yaml)，现有v0.2配置保持默认关闭全局优化。在线阶段仍使用五帧、最多7条active边；离开窗口的相邻和`t-2 -> t`边在启用时转入inactive字典，复用原Edge对象，不再次复制观测张量。

序列结束后汇总inactive和当前active因子，固定第一帧，在第一帧局部坐标系执行CPU float64全局pose-only ICP。求解器逐边累积6x6位姿Hessian块，展开为SciPy COO/CSC稀疏矩阵并用稀疏线性求解；不会构造“全部观测×全部位姿”的dense Jacobian。目标函数、完整3x3协方差白化、向量Huber损失和SE(3)左乘更新与局部窗口求解保持一致。

全局结果仅在状态为refined且全部有限时写回。成功写回同步重锚定各源帧拥有的地图点和协方差；断连、奇异或非有限结果保留插值后的v0.2轨迹。早期v0.3运行的inactive观测只在运行期保留；后续离线长程版本新增了下述持久化归档。

#### V203困难片段验证

验证范围为V203索引1095至1214，共120帧、seed=0。该范围是现有完整V203结果中120帧滚动RTE和ROE组合误差最高的片段。两组均运行于提交`097085a`，时间戳完全一致。原始CSV见[window_icp_global_v203_1095_1215.csv](validation/window_icp_global_v203_1095_1215.csv)。

| 指标 | v0.2 window_skip | v0.3 window_global | 变化 |
|---|---:|---:|---:|
| ATE RMSE | 0.138230 m | 0.137488 m | -0.54% |
| RTE RMSE | 0.020222 m/frame | 0.019486 m/frame | -3.64% |
| ROE RMSE | 0.415006°/frame | 0.403546°/frame | -2.76% |
| RPE RMSE | 0.022668 | 0.021884 | -3.46% |
| 在线平均Odom耗时 | 1044.1 ms | 1020.4 ms | -2.28%（运行波动） |
| 未优化局部窗口 | 0 | 0 | 不变 |

v0.3共汇总237条因子，其中230条inactive、7条active，总计43885个三维观测；保留张量约8.43MB。全局鲁棒目标从110084.53降至109286.76，5轮中接受4步、拒绝13步，求解耗时4.59秒。第一帧逐位保持，输出120个位姿全部有限，最大四元数模长误差约`1.36e-7`。

该片段四项精度全部改善，在线阶段没有可测回退，因此v0.3可作为后续全序列实验的推荐候选配置。但这些因子仍只有相邻和间隔一帧约束，没有回环或新长程信息；当前结果不能证明完整V203或全部EuRoC都会改善。

#### 完整V203验证

提交`3a5f131`以`window_global`模式完成V203全部1865帧，seed=0。结果目录为`Results/WindowICP_Global_v03/20260914_155636_738ac1`，版本化指标见[window_icp_global_v203_full.csv](validation/window_icp_global_v203_full.csv)。

| 指标 | 完整V203 v0.3 |
|---|---:|
| ATE RMSE | 0.578908 m |
| RTE RMSE | 0.007755 m/frame |
| ROE RMSE | 0.168950°/frame |
| RPE RMSE | 0.008805 |
| 在线平均Odom耗时 | 971.9 ms/frame |
| 全局求解时间 | 36.07 s |

1865个位姿全部有限，1864个局部窗口全部refined，最大四元数模长误差为`1.52e-7`。全局后端汇总3727条边，其中3720条inactive、7条active，共720799个三维观测；保留张量为138393408字节，约131.98MiB。全局目标从615750.68降至601769.99，下降2.27%，5轮全部接受且第一帧逐位保持。

与旧提交`dc7c152`的完整V203结果相比，ATE高1.58%，RTE低7.55%，ROE低5.14%，RPE低7.03%。该比较同时包含v0.2的batch=3前端优化，不能把全部变化单独归因于全局后端；全局后端的严格单变量证据仍以上述同提交120帧对比为准。完整结果的ATE回退未超过预设2%阈值，其余相对指标均改善，因此v0.3仍可作为推荐候选，但在替代v0.2前应继续完成其他EuRoC序列对比。

### 离线长程因子实验

启用全局优化的新运行会额外保存：

- `poses_before_global.npy`：同一次运行中，插值完成但全局优化尚未写回的严格前轨迹；格式与`poses.npy`相同。
- `global_factors.npz`：全部`t-1`和`t-2`短程ICP因子、初始传感器位姿、时间戳及外参；观测保持CPU float64完整3×3协方差。
- `global_refinement.json`：求解结果、归档状态、边/观测数量、文件大小和Git提交。

`global_factors.npz`使用版本化、无pickle的NumPy格式并通过临时文件原子发布。离线工具会校验数组形状、ragged offsets、位姿四元数、协方差正定性、边端点、时间戳和源轨迹SHA-256。输入`poses.npy`及短程归档不会被覆盖。

从已有结果目录生成真实`t-5`和`t-10`边：

~~~bash
.venv/bin/python Scripts/Experiment/GenerateLongRangeICP.py \
  --space /absolute/path/to/result-leaf
~~~

生成器仅处理每5帧一个目标。初始化用一次固定batch=3得到第0帧深度；之后每个目标使用一次现有`estimate_window()`：槽0计算当前立体深度，槽1计算`t-5 -> t`，槽2计算`t-10 -> t`。在`t=5`时槽2复制`t-5`输入并丢弃输出。深度和帧缓存最多保留3个计划帧。

离线重复短程求解或加入长程边：

~~~bash
.venv/bin/python Scripts/Experiment/RefineGlobalPoseICP.py \
  --space /absolute/path/to/result-leaf --output-tag short

.venv/bin/python Scripts/Experiment/RefineGlobalPoseICP.py \
  --space /absolute/path/to/result-leaf \
  --long-factors long_factors_gap5_10.npz \
  --edge-kinds gap5 gap10 --output-tag gap5_10
~~~

输出使用`poses_global_<tag>.npy`、`global_<tag>_diagnostics.json`和`global_<tag>_metrics.json`。可以反复改变迭代次数和Huber阈值，不需要重新运行网络。

困难V203短序列的一键严格对比：

~~~bash
.venv/bin/python Scripts/Experiment/CompareLongRangeICP.py \
  --sequence V203 --seq-from 1095 --seq-to 1215 --seed 0 \
  --result-root Results/LongRangeICP_V203_short
~~~

脚本只运行一次在线源轨迹，然后离线比较`before_global`、`short`、`gap5`和`gap5_10`四组。它输出`long_range_metrics.csv/json`、`comparison_manifest.json`、`verification.json`和`stage_gate.json`。只有`gap5_10`输出有限、固定锚不变、目标下降且ATE RMSE严格低于short-only时，`stage_gate.json`中的`passed`才为true。

复用已生成的在线结果时传入`--space`，可跳过在线网络：

~~~bash
.venv/bin/python Scripts/Experiment/CompareLongRangeICP.py \
  --space /absolute/path/to/result-leaf \
  --sequence V203 --seq-from 1095 --seq-to 1215 --seed 0 \
  --result-root Results/LongRangeICP_V203_short_reuse
~~~

校验某次对比的严格同源性：

~~~bash
.venv/bin/python Scripts/Experiment/CompareLongRangeICP.py \
  --verify-space /absolute/path/to/comparison-run
~~~

#### V203困难片段离线长程验证

提交`085e1ab`在V203索引1095至1214、共120帧、seed=0上完成一次在线源运行。对比目录为`Results/LongRangeICP_V203_short/20260914_175119_fa00a4`，版本化结果见[window_icp_long_range_v203_1095_1215.csv](validation/window_icp_long_range_v203_1095_1215.csv)。四组轨迹共享同一`poses_before_global.npy`和短程因子SHA-256，验证脚本确认时间戳一致、`poses.npy`未被覆盖且第一帧锚保持。

| 模式 | ATE RMSE (m) | RTE RMSE | ROE RMSE (°) | RPE RMSE | 边数 |
|---|---:|---:|---:|---:|---:|
| 全局优化前 | 0.138230 | 0.020222 | 0.415005 | 0.022668 | — |
| short：`t-1,t-2` | 0.137488 | 0.019486 | 0.403545 | 0.021884 | 237 |
| short + `t-5` | 0.136785 | 0.019013 | 0.400499 | 0.021429 | 260 |
| short + `t-5,t-10` | 0.138097 | 0.017976 | 0.388844 | 0.020378 | 280 |

相对short-only，单独加入23条`t-5`边使ATE下降0.51%，并同步改善三项相对误差。默认Huber阈值3.0下，再加入20条有效`t-10`边后ATE反而升高0.44%，但RTE、ROE、RPE分别改善约7.75%、3.64%、6.88%。因此默认`gap5_10`严格ATE闸门为false，按设计没有启动完整V203网络运行。

离线长程生成共调用固定batch=3前端24次，其中1次初始化、23次目标推理；尝试23条gap-5和22条gap-10边，接受23和20条，2条gap-10因入界点不足被拒绝。43条长边共6651个观测，生成耗时26.66秒，帧与深度缓存峰值均为3。短程归档约8.1MiB，长程归档约1.3MiB。

同一归档上的附加敏感性检查表明结果高度依赖长边鲁棒权重：

| `t-5,t-10` Huber | ATE RMSE (m) | 相对short-only |
|---:|---:|---:|
| 1.0 | 0.140046 | +1.86% |
| 2.0 | 0.139362 | +1.36% |
| 3.0 | 0.138097 | +0.44% |
| 5.0 | 0.136319 | -0.85% |

Huber 5.0能够在该片段降低ATE，但这是使用同一困难片段选择参数的探索结果，不能替代预先设定的默认闸门，也不足以证明完整V203泛化。当前证据支持“真实长程边能够提供有效低频信息”，同时说明`t-10`边的协方差标定或鲁棒权重仍需单独校准；下一步应优先离线分析gap-10残差/协方差分布，而不是立即增加更复杂的Schur后端。

### 动态共视 proximity 实验

动态共视版本与固定gap实验严格区分。固定gap按时间选择`t-5/t-10`；动态版本对每个计划目标从全部合格历史关键帧中，根据当前`poses_before_global`、缓存深度、双向投影共视率和深度一致性选取最多一个候选，再使用真实正反向FlowFormerCov匹配验证。

离线生成命令：

~~~bash
PYTHONPATH=. .venv/bin/python Scripts/Experiment/GenerateProximityICP.py \
  --space /absolute/path/to/result-leaf
~~~

固定batch=3槽位为目标立体、历史到当前正向匹配、当前到历史反向匹配。候选必须通过正反向闭环误差、深度有效率、4×6网格覆盖和完整协方差Mahalanobis内点检查，才会写入`proximity_factors.npz`。候选评分、所有拒绝原因及匹配统计分别保存在`covisibility_candidates.json`和`proximity_generation.json`。

原V203困难片段的一键复用对比：

~~~bash
PYTHONPATH=. .venv/bin/python Scripts/Experiment/CompareCovisibilityICP.py \
  --space /home/zzh/MACVO/Results/LongRangeICP_V203_short/20260914_175119_fa00a4/source/MACVO-Fast-WindowICP5-Global@V203/09_14_175121 \
  --result-root /home/zzh/MACVO/Results/CovisibilityICP_V203_original
~~~

输出四组严格同源结果：`before_global`、`short`、`short+gap5`和`short+proximity`。基础闸门要求默认Huber=3时proximity ATE低于short-only；选择器优势还要求ATE不高于固定gap-5，且proximity边数不超过实际gap-5边预算。零条通过验证的proximity边也是有效实验结果，不自动放宽阈值。

#### 原V203困难片段结果

原片段V203 `[1095,1215)`复用同一`poses_before_global.npy`和短程归档完成动态共视验证。结果目录为`Results/CovisibilityICP_V203_original/20260914_195312_548f2c`，版本化指标见[dynamic_covisibility_v203_1095_1215.csv](validation/dynamic_covisibility_v203_1095_1215.csv)。源因子、源轨迹哈希、120个时间戳和固定锚均通过严格校验。

| 模式 | 新增边 | ATE RMSE (m) | RTE RMSE | ROE RMSE (°) | RPE RMSE |
|---|---:|---:|---:|---:|---:|
| 全局优化前 | 0 | 0.138230 | 0.020222 | 0.415005 | 0.022668 |
| short | 0 | 0.137488 | 0.019486 | 0.403545 | 0.021884 |
| short + gap-5 | 23 | 0.136785 | 0.019013 | 0.400499 | 0.021429 |
| short + proximity | 5 | 0.137186 | 0.019438 | 0.408617 | 0.021899 |

动态共视评估了231个几何候选，21个目标中12个进入真实双向匹配，最终接受5条边、574个观测；7条候选因正反向一致内点不足被拒绝，9个目标没有通过几何门槛的候选。接受边跨度为15或20帧。首次失败运行已完成24次Pass A并生成69,510,599字节缓存；修复CPU缓存深度与GPU关键点的设备边界后，正式运行复用该缓存，仅执行12次Pass B，耗时16.32秒。

proximity ATE相对short下降0.22%，因此基础动态共视闸门通过；但比固定gap-5高0.29%，选择器优势闸门未通过。当前原片段证据说明少量几何筛选边能够改善ATE，但候选仍集中在15至20帧跨度，尚未验证真正的远距离重访优势。

#### 独立V203重访片段结果

使用完整V203的GT只做评测窗口筛选，并显式排除原`[1095,1215)`区间。筛选条件为120帧窗口、时间间隔至少30帧、位置距离不超过2米、SO(3)视角差不超过30°、至少3对重访。最终固定选择`[1733,1853)`：包含1123对满足条件的GT帧对，现有完整轨迹的窗口RTE RMSE为0.012779、ROE RMSE为0.193926°，难度分数4.57897。GT没有进入动态选择器或匹配器。

该片段运行于提交`1e9e1c2`，结果目录为`Results/CovisibilityICP_V203_revisit/20260914_195852_96444b`，版本化指标见[dynamic_covisibility_v203_revisit.csv](validation/dynamic_covisibility_v203_revisit.csv)。所有模式共享同一前全局轨迹、短程因子和时间戳，源文件哈希及固定锚校验通过。

| 模式 | 新增边 | ATE RMSE (m) | RTE RMSE | ROE RMSE (°) | RPE RMSE |
|---|---:|---:|---:|---:|---:|
| 全局优化前 | 0 | 0.032591 | 0.017574 | 0.252719 | 0.018649 |
| short | 0 | 0.042799 | 0.011431 | 0.191720 | 0.012372 |
| short + gap-5 | 23 | 0.039412 | 0.011438 | 0.187130 | 0.012335 |
| short + proximity | 13 | 0.037601 | 0.011343 | 0.191239 | 0.012286 |

动态选择评估231个几何候选，21个目标中15个进入真实匹配，接受13条、拒绝2条。接受边包含7条gap-15、1条gap-20、1条gap-25、3条gap-30和1条gap-40，共1882个验证后观测。Pass A执行24次batch=3、耗时23.87秒；Pass B执行15次、耗时15.06秒；缓存约69.5MB。

proximity ATE相对short下降12.14%，相对使用23条边的固定gap-5仍下降4.60%，而只使用13条边，因此基础闸门与选择器优势闸门均通过。这是动态几何共视选择优于固定时间gap的首个严格同源证据。

同时必须注意：该片段全局优化前ATE为0.032591，仍低于三种全局优化后的轨迹。short/global优化虽然大幅改善RTE、ROE和RPE，却重新分配了误差并增加ATE；动态proximity减少了这种ATE回退，但没有完全消除。该结果支持继续研究动态共视边，不支持直接声称当前全局pose-only后端总能优于在线前轨迹。

#### 完整V203动态共视验证

提交`2d03875`完成V203全部1865帧、seed=0的严格同源对比。结果目录为`Results/CovisibilityICP_V203_full/20260914_201422_ce9c65`，版本化指标见[dynamic_covisibility_v203_full.csv](validation/dynamic_covisibility_v203_full.csv)。在线阶段1864个局部窗口全部refined，平均Odom耗时968.75 ms/帧，前端平均840.44 ms/次，终止short全局求解约36.17秒。

| 模式 | 新增边 | ATE RMSE (m) | RTE RMSE | ROE RMSE (°) | RPE RMSE |
|---|---:|---:|---:|---:|---:|
| 全局优化前 | 0 | 0.578985 | 0.008433 | 0.178319 | 0.009513 |
| short | 0 | 0.578908 | 0.007755 | 0.168951 | 0.008805 |
| short + gap-5 | 371 | 0.578774 | 0.007313 | 0.164465 | 0.008364 |
| short + proximity | 161 | 0.577528 | 0.007650 | 0.167708 | 0.008699 |

动态共视评估68,635个几何候选，370个目标中324个进入真实双向匹配，接受161条边、24,232个观测。接受边跨度从15至165帧，其中73条gap-15、17条gap-30，并包含110、115、125、145、160和165帧的真正长程边。验证拒绝包括76条正反向内点不足、11条网格覆盖不足和76条Mahalanobis内点不足。

Pass A执行373次固定batch=3，生成约1.08GB深度缓存，耗时323.27秒；候选搜索及Pass B执行324次前端调用，耗时390.61秒。固定gap生成执行373次前端调用，接受371条gap-5和362条gap-10边，耗时327.22秒。

完整V203上proximity ATE相对short下降0.238%，相对固定gap-5下降0.215%，同时只使用其43.4%的新增边，因此基础闸门和选择器优势闸门均通过。相对误差方面固定gap-5仍最好：动态proximity主要优化ATE与边效率，而未在RTE、ROE、RPE上超过密集gap-5。

与同一次运行的全局优化前轨迹相比，动态proximity ATE下降约0.252%，RTE、ROE和RPE分别明显低于优化前结果。这说明动态共视在完整V203上能够带来可复现但幅度较小的整体改善；它尚未超过历史旧提交约0.5699米的完整V203 ATE，因此当前证据仍不足以替代`window-icp-v0.2`作为默认运行版本。

### 在线ORB-BoW回环与后台位姿图

在线版本`MACVO_Fast_WindowICP_ORBLoop.yaml`保留原WindowICP的相邻边和`t-2` skip边，并增加异步ORB-BoW候选、共享FlowFormerCov固定batch=3双向验证、串行GPU访问和单线程后台稀疏位姿图。回环验证继续复用同一模型实例和同一CUDA Graph，不允许跟踪与回环GPU推理并发。关键帧图像使用无损PNG落盘，GPU侧只保留有界待验证目标。

完整V203运行命令：

~~~bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. \
  .venv/bin/python Scripts/Experiment/CompareOnlineORBLoop.py \
  --sequence V203 --seq-from 0 --seed 0 \
  --modes window_orb_loop \
  --result-root /home/zzh/MACVO/Results/OnlineORBLoop_V203_full
~~~

提交`ca88f45`完成全部1865帧，结果目录为`Results/OnlineORBLoop_V203_full/20260914_232539_309810`，版本化汇总见[online_orb_loop_v203_full.csv](validation/online_orb_loop_v203_full.csv)。

| 指标 | 在线ORB版本 |
|---|---:|
| ATE RMSE | 0.465289 m |
| RTE RMSE | 0.006868 m/frame |
| ROE RMSE | 0.160229°/frame |
| RPE RMSE | 0.007925 |
| 在线平均Odom耗时 | 1127.0 ms/frame |
| 中位Odom耗时 | 1045.6 ms/frame |
| P95 Odom耗时 | 1911.4 ms/frame |
| 峰值CUDA保留显存 | 4.049 GiB |

运行只创建1个前端模型和1个batch=3 CUDA Graph，最大并发GPU调用为1。正常跟踪调用1865次，额外回环前端调用82次，回环推理累计73.49秒。ORB初筛/几何验证写入51条loop因子，其中29条跨度大于100帧、22条跨度大于300帧，最大跨度1455帧。后台提交52次、完成49次、合并3次，写回48次；最终3778条因子包括1864条相邻边、1863条skip边和51条loop边。最终1865位姿求解耗时17.90秒。关键帧PNG约161MiB，压缩位姿因子约1.8MiB。

但“通过几何初筛”不等于“对优化有效”。按最终信息矩阵计算，51条loop因子的switch中位数仅`7.35e-6`，最大值0.02725；switch大于0.01的3条边都是`0→30/45/50`近程冗余边，没有一条跨度大于100帧的有效loop。使用同一次运行的内部相机位姿和序列化因子，移除全部51条loop后重新求解，ATE为0.465328米；保留loop为0.465289米，只改善0.0000389米、约0.0084%。因此当前长程ORB回环没有带来实质ATE改善。

相对完整V203同源旧实验的优化前WindowICP轨迹0.578985米，在线版本ATE下降约19.64%。证据表明该收益几乎全部来自相邻/skip压缩因子的后台全局位姿图及48次周期写回，而不是长程回环。代价是平均在线耗时相对旧完整运行约增加16.34%，末段大图后台求解使P95达到1.91秒。当前版本是已验证完整V203中ATE最好的本分支候选，但不能宣称ORB回环策略已成功。

下一步应把后台位姿图改为独立固定周期触发，使“仅短程位姿图”与“短程位姿图+回环”拥有相同写回时序；同时对ORB候选增加更严格的独立ICP质量/位姿一致性分层，并优先分析FlowFormer在真正长跨度重访上的错误对应。对比脚本现默认运行`window_skip`、`window_pose_graph`和`window_orb_loop`三组，并在新运行中直接输出最终最大switch、switch>0.01的有效loop数和有效长程loop数。

#### 完全移除`t-2 -> t`消融

no-skip版本保持同一模型、同一固定batch=3 CUDA Graph和同一在线ORB/后台写回策略。跟踪阶段第三槽复制相邻输入作为占位并丢弃输出，因此网络批量不变；被移除的是skip匹配观测、5帧WindowICP中的skip残差、持久化`skip2`位姿因子及其稀疏图块。独立配置为`MACVO_Fast_WindowICP_ORBLoop_NoSkip.yaml`。

困难片段V203 `[1095,1215)`首先完成稳定性验证。no-skip的120个位姿全部有限，0个未优化窗口，119条adjacent、0条skip2和1条loop因子。与保留skip的在线ORB版本相比，ATE从0.104312升至0.109183米（+4.67%），RTE、ROE和RPE分别升高10.94%、10.07%和10.75%；平均耗时从1116.1降至998.1毫秒/帧（-10.57%）。该结果满足完整运行闸门，但已经显示skip对困难运动有明显稳定作用。

提交`527b6e0`随后完成V203全部1865帧、seed=0。结果目录为`Results/OnlineORBLoop_NoSkip_V203_full/20260915_155752_24a8ff`，版本化汇总见[online_orb_loop_no_skip_v203.csv](validation/online_orb_loop_no_skip_v203.csv)。时间戳与保留skip基准逐项相同，1864个窗口全部refined，位姿全部有限，最大四元数模长误差为`1.93e-7`。

| 指标 | 保留`t-2 -> t` | 完全移除 | 变化 |
|---|---:|---:|---:|
| ATE RMSE | 0.465289 m | 0.558976 m | +20.14% |
| RTE RMSE | 0.006868 | 0.010689 | +55.65% |
| ROE RMSE | 0.160229° | 0.201635° | +25.84% |
| RPE RMSE | 0.007925 | 0.011791 | +48.78% |
| 平均Odom耗时 | 1127.0 ms | 980.8 ms | -12.98% |
| 中位Odom耗时 | 1045.6 ms | 914.3 ms | -12.55% |
| P95 Odom耗时 | 1911.4 ms | 1703.0 ms | -10.90% |
| 峰值CUDA保留显存 | 4.049 GiB | 4.049 GiB | 不变 |
| 位姿因子 | 3778 | 1915 | -49.31% |
| 最终全图求解 | 17.90 s | 9.96 s | -44.37% |

no-skip全局图包含1864条adjacent和51条loop，没有`skip2`。后台写回仍为48次，额外回环前端调用仍为82次，因此调度频率和网络工作量与基准一致。有效长程loop仍为0，说明差异主要来自skip约束本身，而不是回环集合变化。

从no-skip到保留skip，ATE相对改善16.76%，RTE、ROE和RPE分别改善35.75%、20.54%和32.79%。代价是平均耗时增加约14.91%，最终位姿图求解时间增加约79.75%。即使没有skip，adjacent-only后台图的ATE 0.558976仍比旧WindowICP优化前0.578985低3.46%，但明显弱于保留skip的0.465289。完整V203证据支持保留`t-2 -> t`；若继续优化效率，应降低skip构边/压缩开销或降低全局求解频率，而不是删除该约束。

## 每次运行的追溯信息

- config.yaml：本次Odometry配置。
- run_provenance.json：Git提交、分支、工作区状态、配置SHA256、种子、torch/PyPose版本、预期帧数、完成/失败状态。
- window_diagnostics.json：逐窗口位姿ID、锚、边数、匹配数、求解目标、步长接受次数、耗时及失败原因。
- poses_before_global.npy / global_factors.npz / global_refinement.json：严格前后对比和离线后端实验输入。
- long_factors_gap5_10.npz：离线生成的真实`t-5`及`t-10`观测。
- poses.npy / ref_poses.npy / tensor_map.npz：兼容已有评估与地图导出。
- elapsed_time.json：前端、协方差及里程计计时。

v0.2跨帧约束只保存在运行期有界侧缓存。当前实验分支在全局模式下会保存独立因子归档；`tensor_map.npz`仍只包含原邻接地图结构，不重复嵌入inactive观测。

## 版本操作

~~~bash
git log --oneline baseline/fast-icp-euroc-20260912..HEAD
git diff --stat baseline/fast-icp-euroc-20260912..window-icp-v0.1
~~~

回到两帧算法进行实验不必切换Git分支，使用MACVO_Fast_ICP_local.yaml即可。
若要基于旧代码继续开发，可从baseline标签新建分支；切换前保留自己的未提交修改，不使用强制重置。

## 尚未实现

没有边缘化先验、共享地图点BA或BAE后端；没有估计重复图像观测带来的跨因子相关性。在线ORB-BoW回环已经能够提出并验证任意跨度候选，但完整V203中所有真正长程边都被switch抑制，尚未形成有效的低频漂移约束。最早位姿固定不等于全局漂移已消除。
完整EuRoC评测仍需运行上面的批量命令。
