# 演示协议实现

本模块实现 [演示协议](../../docs/demostration_protocol.md)，默认任务为 `organize_table`。
它独立于现有 `gpt_only` / `pi05_plus_gpt` 控制器，不调用付费模型，也不使用物体 pose 或分割真值。

## 原始数据

两种来源明确区分：

- `robodojo.py` 重放官方 HDF5 中的完整绝对关节动作，重新记录三路 RGB-D、实际相机内外参、实际关节与指关节、实测 link6 pose。
- `official.py import` 导入官方已有的深度演示。这不是重录；原文件未提供实际指关节读数，不能把归一化 state 值重新命名为实测指关节。来源与限制保存到 `recording.json`。

每条归档有 `recording.json`、`source_trajectory.npz` 和 `frames/000000.npz` 等文件。新录制第 0 帧在动作之前，帧 i 对应源动作 i−1 的执行后观测；导入遵循官方 `state[t]` 当前观测、`action[t]` 下一帧绝对目标的约定，最后一个 action 是 padding。完整源机器人轨迹单独保存，原轨迹与重放实测状态不会混在一起。

统一坐标系为 `environment_origin`，位置单位为米，关节单位为弧度，四元数为 wxyz。
`T_world_eef` 和 `T_world_camera` 分别将 EEF／光学相机坐标转换到统一坐标。
相机使用 +x 向右、+y 向下、+z 向前，深度是像素对齐的 optical z。
所有图像标注使用原分辨率的 `[u,v]`；查询时取最近整数像素，不进行隐式深度补全。
无效深度保留为 NaN；官方 uint16 毫米深度中的 0 和 65535 按无效处理。

### 下载一条官方深度演示

安装 `requirements-tools.txt` 后，在 Nexus 根目录执行：

```bash
python -m hybrid_rollout.demonstrations.official download \
  --task organize_table --episode 0 \
  --output /data/nexus-runtime/datasets/organize_table_depth_new
```

只下载一条文件，解析并记录不可变 revision，校验官方 LFS 大小及 SHA256。不会下载整个约 4.5 TB 的深度集合。
官方数据说明见 <https://robodojo-benchmark.com/doc/usage/install-and-download/>。

```bash
python -m hybrid_rollout.demonstrations.official import \
  --file /data/nexus-runtime/datasets/organize_table_depth/episode_0000000.hdf5 \
  --task organize_table --extrinsics-axes usd \
  --output /data/nexus-runtime/datasets/organize_table_depth/nexus_recording_v1
```

外参输入的轴约定必须显式指定。`usd` 适用于标准 pinhole 相机 prim 与安装参考系重合的原生归档；导入时右乘 `diag(1,-1,-1,1)` 转成光学参考系。其他相机存在额外安装偏移时，需要先标定，不能仅靠这项轴转换。导入保留原文件 K，不擅自替换为重新估计的内参。

### 在配套布局上重录

重录需要与源演示明确配套的场景布局。公开资产中的 `Eval_Layout` 是评测布局，不应仅凭同编号就假定它与训练 episode 对应。布局文件仅在模拟器建立场景时使用，不进入演示标注工作区或 Astra 交付上下文。

```bash
ROBODOJO_SOURCE=/data/nexus-runtime/src/RoboDojo \
ROBODOJO_ENV=/data/nexus-runtime/envs/robodojo-isaac6 \
DEMO_LOG=/data/nexus-runtime/logs/robodojo-isaac6/organize-table-record.log \
bash hybrid_rollout/demonstrations/record_robodojo.sh \
  --task organize_table \
  --trajectory /path/to/episode_0000000.hdf5 \
  --layout /path/to/paired_source_layout.json \
  --layout-provenance '具体文件来源与 episode 对应证据' \
  --output /path/to/new_recording \
  --headless --enable_cameras --device cuda:0
```

`--max-actions 2` 可执行有界采集冒烟，但归档始终标为 incomplete，不能导出成完整演示。
采集继续保留任务完成后的退回动作，原生 success 只记录在结果中，不提前截断源动作。
录制使用原生 RoboDojo RGB-D 采集器，不另外增加相机渲染或物理步。
默认显式加载 `isaaclab.python.rendering.kit`；`--headless` 关闭窗口，但保留完整的
渲染扩展。RPC 入口使用相同配置，避免回到此前卡在首次相机更新的默认
`isaaclab.python.headless.rendering.kit` 启动组合。可显式传入 `--experience` 复现其他配置。
无窗口运行仍注册 Kit visualizer 以驱动 `app.update()`；这里的 visualizer
承担相机渲染循环，不要求桌面窗口。实际解析后的 experience、headless、
visualizer 和相机开关写入 `launch_config.json`。
本地 `organize_table` 已完成两动作 RGB-D 采集验证：初始帧与两个执行后帧的
时间戳为 0、0.04、0.08 秒，三路 RGB/depth 均为 480×640，并保留实测双臂 EEF
与指关节。该验证使用明确标记的评测布局，仅证明采集链路可运行，不是源演示的重录。

## 人工选关键帧与阶段

```bash
python -m hybrid_rollout.demonstrations.cli browse \
  --recording /path/to/recording --stride 25 --output /path/to/previews
python -m hybrid_rollout.demonstrations.cli frame \
  --recording /path/to/recording --frame 100 --output /path/to/frame_100
```

缩略图页仅帮助浏览，不自动选择关键帧或阶段。精确查看时使用 `frame` 命令导出三路原分辨率 RGB-D 与传感器 JSON。
人工编写 selection JSON，包含任务、阶段目的、固定 active/target、有序关键帧及人工描述。
以下帧号仅演示格式，不能当作已经确定的 `organize_table` 阶段边界：

```json
{
  "task": "organize_table",
  "stages": [{
    "id": "clock_to_drawer",
    "purpose": "将闹钟搬到抽屉顶面",
    "active_object": "alarm_clock",
    "target_object": "drawer_top",
    "keyframes": [
      {"frame_id": 0, "description": "人工填写该帧的实际情形"},
      {"frame_id": 100, "description": "人工填写该帧的实际情形"}
    ]
  }]
}
```

同一阶段对象不变；换对象时划分新阶段。阶段按时间排序，可共享边界帧。

## Agent 语义标注与数学工具

```bash
python -m hybrid_rollout.demonstrations.cli prepare \
  --recording /path/to/recording --selection /path/to/selection.json \
  --output /path/to/agent_workspace
```

每个阶段生成 `context.json`，列出全部关键帧的三路图像、人工描述与传感器文件。
Agent 查看全部帧后，编写一个阶段标注文件。构造定义属于整个阶段，不能逐帧更改；
每帧保留全部关键点名称，无法可靠观察时使用 `unavailable` 和原因。

```json
{
  "stage_id": "clock_to_drawer",
  "reviewed_frame_ids": [0, 100],
  "keypoint_semantics": {
    "clock_center": "可跨关键帧跟踪的闹钟功能中心",
    "drawer_center": "抽屉顶面目标区域中心"
  },
  "normal_semantics": {},
  "definitions": {
    "active": {
      "kind": "point", "description": "使用闹钟功能中心",
      "origin": {"op": "point", "name": "clock_center"}
    },
    "target": {
      "kind": "point", "description": "使用抽屉目标区域中心",
      "origin": {"op": "point", "name": "drawer_center"}
    }
  },
  "keyframes": [
    {"frame_id": 0, "keypoints": {
      "clock_center": {"status": "observed", "camera": "cam_head", "pixel": [320, 240]},
      "drawer_center": {"status": "unavailable", "reason": "该帧无法可靠观察"}
    }},
    {"frame_id": 100, "keypoints": {
      "clock_center": {"status": "unavailable", "reason": "夹爪遮挡"},
      "drawer_center": {"status": "observed", "camera": "cam_head", "pixel": [250, 200]}
    }}
  ]
}
```

这些像素也是格式示例，实际语义、像素与构造必须由 agent 看图决定。
对于 pose，定义还包含 `axes` 中的两条方向表达、`primary_axis`、`secondary_axis` 和
`axis_sources`（明确记录来自观测还是约定）；工具保留主轴，正交化副轴，再以右手规则补全第三轴。
两轴近乎平行或依赖点缺失时，不输出确定 pose。

夹爪／EEF 参考可以直接引用归档机器人状态，定义示例：

```json
{"kind": "pose", "description": "左夹爪以归档 link6 为功能参考，非指尖或 TCP",
 "source": {"kind": "robot_eef", "arm": "left"}}
```

此时不从图像背景深度重新估计 link6。工具保存本体状态来源及三路二维投影；
标注图以青色十字和 `L6/R6 (proprio)` 区别于黄色图像语义点，投影不证明该点未被遮挡。
两个可独立运动的夹爪可以用 `kind: collection`、`description` 和 `members` 表达，
每个命名成员各自采用同样的 point／pose 定义，不生成平均刚体位姿。
成员独立保存 available／unavailable；target 为 pose 时额外保存各 pose 成员的
`T_target_active_members`，原单主体 `T_target_active` 保持为空。

支持的组合表达为 `point`、`normal`、`vector`（统一参考系内的明确向量）、`mean`、
`add`、`subtract`、`scale`、`cross`、`normalize`。它们是数学运算，不预设杯口、工具端或托盘等语义。
`normal_regions` 按法向量语义名称保存每帧的 `camera`、`polygon`、可选 `toward_world` 和
`max_rms_m`，以深度平面拟合得到法向量；默认法向朝向观察相机，区域不足或非平面时返回不可用。

查询请求保存为 JSON，例如：

```json
{"op": "backproject", "camera": "cam_head", "pixel": [320, 240]}
```

```bash
python -m hybrid_rollout.demonstrations.cli query \
  --recording /path/to/recording --frame 100 --request /path/to/query.json
python -m hybrid_rollout.demonstrations.cli resolve \
  --recording /path/to/recording --selection /path/to/selection.json \
  --annotation /path/to/annotations/clock_to_drawer.json --output /path/to/resolved.json \
  --render-output /path/to/stage_review
```

`query` 的 `op` 可为 `depth`、`backproject`、`normal`；后者使用图像多边形。
`resolve` 计算保存的全部构造，保存二维输入、定义、计算证据和最终 point／pose。
两对象都形成 pose 时，额外计算 `T_target_active`；point 不会被补成虚假的完整 pose。
`--render-output` 可选，生成该阶段的三路标注预览，用于整包导出之前复核像素位置。
二维点／法向区域可选 `label_offset: [dx,dy]`，只移动图上名称以避免重叠，不改变查询像素或几何。

可复用 agent 工作流见 [nexus-demonstration-annotator/SKILL.md](nexus-demonstration-annotator/SKILL.md)。
该 skill 仅显式调用；阶段语义、像素与构造由执行 agent 自主选择，工具与最终交付由主 agent 复核。

## 导出给 Astra

```bash
python -m hybrid_rollout.demonstrations.cli export \
  --recording /path/to/recording --selection /path/to/selection.json \
  --annotations /path/to/annotations --samples 8 --output /path/to/astra_demo
```

`annotations` 目录每阶段一个 `<stage_id>.json`。输出包括 `demonstration.json`、
有序 `astra_context.md`、关键帧原 RGB-D、标注图、传感器参数、机器人状态及完整标注审计。
交付同时保留 agent 的观察备注及具体不可用原因，模型上下文不会只收到内部依赖缺失消息。
每对相邻关键帧生成一张 head camera 概览图，均匀采样 RGB-D 重投影到区间首帧光学相机后叠加，
绘制整段实测左右 link6 的二维轨迹，色带由浅到深表达时间。投影可见不代表未被其他物体遮挡。

`astra_context.md` 是后续控制器装配演示输入的入口；当前尚未自动注入现有 rollout controller。
它不包含完整数值 action chunk，也不需要将全部原始轨迹放进模型上下文。

## 验证

```bash
python -m pytest hybrid_rollout/demonstrations/test_protocol.py -q
```

离线测试覆盖光学坐标变换、深度有效性、平面法向、pose 构造、关键帧顺序、缺失观察与交付。
仿真采集另做有界冒烟；它只验证采集链路，不证明完整源演示的重放成功率。

此前 2026-10-04 的本地验证：演示协议与受影响的原生评测／GPT-only 回归共 40 项通过。
`organize_table` 两动作采集入口退出码为 0，三帧归档位于
`/data/nexus-runtime/datasets/organize_table_capture_smoke_20261004_robodojo_nativecompat_organize_smoke_03/`。
该归档使用评测布局，source 明确记录未与原演示配对，不能导出为完整演示。

官方深度演示的已导入归档为
`/data/nexus-runtime/datasets/organize_table_depth/nexus_recording_v1/`（838 帧）。
人工选帧预览位于同级 `preview_v1/preview_000.png` 与 `preview_001.png`。
`overview_geometry_check_0_200.png` 是对帧 0–200 的工具验证图，未将该区间选定为任务阶段。
原演示的配套布局仍未找到。用户已确认首版直接使用这条官方 RGB-D 与归一化夹爪 state，
并给出 7 个阶段、17 个关键帧；原始时间到观测帧的换算保存在
`/data/nexus-runtime/datasets/organize_table_depth/selection_user_v1.json`。
独立标注工作区为同级 `agent_workspace_independent_v1/`，不改变人工阶段或描述。

本轮补充机器人本体参考、独立双夹爪表示、阶段标注预览和缺失原因交付后，演示离线测试共 29 项通过。
skill 的结构校验通过；独立 agent 的真实标注用于检查该工作流是否可执行。

本条已完成的交付包位于
`/data/nexus-runtime/datasets/organize_table_depth/astra_demo_independent_v1/`。
独立执行 agent 使用 `gpt-6.1-sol` / `xhigh`，主 agent 已复核全部关键帧三路视图及 10 张概览。
入口为 `processing_report.md`、`astra_context.md` 和 `demonstration.json`；
`verification.json`、`review.json`、`tool_verification.json` 记录实际交付验证与复核。
最终 target 在帧 311、361、811 因遮挡不可用，具体原因进入模型上下文；其余相关参考保留。
