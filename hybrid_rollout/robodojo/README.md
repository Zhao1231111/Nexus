# robodojo

See the [public project guide](../../README.md) for setup, scope, dependencies, tests and licensing.

## 本地 Isaac Sim 6 启动

当前兼容入口使用外部 RoboDojo 的 Isaac Sim 6.0.1 / IsaacLab 3.0 环境。
RPC server 默认不创建窗口，但显式加载 `isaaclab.python.rendering.kit`，保留完整
相机渲染组件；不会再由 AppLauncher 自动选择 `headless.rendering.kit`。
`run_local.sh` 调用同一 server 入口，演示重录也共用这一配置。
无窗口模式仍显式注册 Kit visualizer：IsaacLab 3.0 通过它调用 `app.update()`，
没有这个循环时，仅加载完整 experience 也不能完成相机取图。

```bash
"$ROBODOJO_PYTHON" -m hybrid_rollout.robodojo.robodojo_server.server \
  --task build_tower --output /path/to/new_smoke_output --port 19113
```

另一个终端使用同一 Python 环境与 PYTHONPATH 运行无策略客户端：

```bash
"$ROBODOJO_PYTHON" -m hybrid_rollout.robodojo.rpc_smoke --port 19113 --seed 0
```

客户端执行 metadata → reset → teacher_observation → 单步 hold → teacher_observation
→ finish_pilot，检查一次控制 ACK、三路 RGB 和两帧视频；不导入或启动 Astra／Pi05。
默认只等待服务连接，不重试会修改仿真状态的 RPC。该命令会实际重置与推进模拟器。

交互诊断时加 `--visual`，入口会同时选择 Kit visualizer；远程桌面显示仍需要正确的
DISPLAY 与 X11 授权。无人值守模式无需桌面配置。显式 `--experience` 可以覆盖默认值，
用于复现兼容问题；实际加载的绝对路径、headless、visualizer 和相机开关保存在输出目录
`launch_config.json`，并通过 `simulator_launch` 日志事件报告。

相机 experience 与窗口开关是独立设置。此前 `--visual --viz kit` 通过的结果，不能单独
作为显式 headless 模式通过的证据；后者必须完成 reset、相机观测、真实动作和结果落盘。

本地 `build_tower` 已验证默认配置：未设置 DISPLAY / XAUTHORITY，实际 headless=true、
visualizers=[kit]、offscreen_render=true；完成 800 步 reset、三路 480×640 RGB 观测、
一次真实 hold 控制与两帧视频落盘，Astra 调用为 0。只更换 experience、未注册 Kit
visualizer 的对照仍停在 reset 的 600 步，因此不能把此前问题完全归因于 experience 文件。
`headless.rendering.kit` 配合正确渲染循环的组合尚未单独验证。

2026-10-04，`organize_table` 也完成同一默认无窗口配置的完整 RPC 冒烟：
metadata → reset → 三路观测 → 一次 hold 控制 → 三路观测 → finish_pilot。
返回 step_id=1、video_frames=2、reason=rpc_smoke_complete，Astra 调用为 0；
视频解码确认两帧，原生日志没有 Python traceback 或 CUDA 错误，实验进程已清理。
本地产物在 `/data/nexus-runtime/datasets/rpc_smoke_20261004_rpc_organize_table_headless_kit_nativecompat_03/`，
包含 `launch_config.json`、`summary.json`、`evaluation_outcome.json` 和 `sensors.mp4`。
单步 smoke 的 complete=false / valid_for_success_rate=false 是预期结果。

`organize_table` 还触发了上游静态几何 pose（CPU）与 env_origin（CUDA）的设备
不匹配。共享入口只适配相对坐标计算，将原点转换到 pose 的 device/dtype 后执行
原来的减法，保留世界 pose、姿态与原生奖励条件；适配记录到 `launch_config.json`。

以上是模拟器 RPC 冒烟，不代表已完成 `run_local.sh` 的模型／策略评测、完整任务或布料验证。
