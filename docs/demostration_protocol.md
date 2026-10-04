# 演示数据处理与交付规范

## 原始演示来源与采集

初版采用官方已有的完整 RGB-D 演示，保留原始动作与机器人状态轨迹，以及三路 RGB-D、相机内外参、帧号或时间戳和归档双臂 EEF pose。允许使用官方归一化夹爪 state，并记录其来源；原文件缺失的实测指关节保持缺失。导入遵循官方归档的同步与标定约定，不声称通过导入独立验证了原始同步。

需要重新录制时，在与源演示明确配套的布局上重放完整原始动作，同步采集上述传感器数据及实际指关节。RGB 与深度需要像素对齐，各项数据需要对应同一观测时刻。不使用仿真物体 pose、分割真值等特权数据，使处理流程可以迁移到真机。

## 阶段与关键帧

每个关键帧包含该时刻的：
- 像素对齐的 RGB 图与深度图
- 人工编写的描述性文字
- Active obj 与 Target obj 的语义 point 或 pose，以及生成它们的二维关键点标注与定义（用于隐式表示稀疏的 target object-centric 轨迹）
- 机械臂 proprioception
- 双臂 EEF pose
- 夹爪开合程度

阶段内部的 active object 和 target object 保持不变。

每个阶段包含：
- 阶段目的与相关描述（文字说明）；
- active object；
- target object；
- 有序关键帧；

## 关键帧之间的轨迹展示：

相邻关键帧之间生成一张 head cam 的轨迹概览图：
- 从 head camera 区间内均匀抽取若干 RGB 帧，对齐后叠加到同一张图，使用由浅到深或固定色带表示时间；
- 绘制左右 EEF 的二维投影轨迹。

这张图主要让 Astra 理解运动形状、绕行方式、双臂配合和动作顺序。

## 处理原则

- 关键帧由人工挑选确定，阶段是关键帧的集合，由人工划分
- 每个阶段内关键帧的 Active obj 与 Target obj 语义 point / pose 标注由 agent 完成。agent 查看该阶段的全部关键帧，自主选定并命名能够跨关键帧稳定跟踪的二维语义关键点。
- agent 自主决定如何由语义关键点拓展成 point 或 pose，包括原点、坐标轴和所需法向量的定义，并调用几何工具完成计算。我们不预先穷举这些具体定义。
- 同一阶段必须沿用相同的关键点语义和 point / pose 构造定义。二维标注、构造定义和最终结果都需要保存，以便复核和重新计算；无法可靠观察时不生成确定的 point / pose。
- 所有 point / pose 使用统一且明确记录的参考系、单位和姿态约定。
- 夹爪／EEF 的功能参考可引用归档机器人本体状态中的 link6 pose，并保存来源与二维投影。
  该引用不等于 TCP、指尖或接触点，也不通过投影处的背景深度反推位姿。
  多个可独立运动的主体分别保存 point／pose，不以平均位姿表示单一刚体。
- 其余确定性数据则由自动化脚本完成。

## 数学工具

为了方便 agent 标注语义 point / pose，为 agent 制作可调用的数学工具，包括获取某个像素的深度、将二维关键点反投影并转换到统一坐标系、计算图像区域的法向量，以及根据 agent 给出的定义构造 pose。工具负责几何计算，不替 agent 选择语义关键点或定义 pose。

## 交付给 Astra

按阶段交付有序关键帧及相邻关键帧之间的轨迹概览图。每个关键帧提供标注图、文字描述、Active / Target 的语义 point 或 pose，以及机器人状态。完整原始轨迹保留为数据来源，不默认作为大段数字直接放入上下文。

## 实现入口与当前数据

实现位于 [`hybrid_rollout/demonstrations`](../hybrid_rollout/demonstrations/README.md)，默认任务为 `organize_table`。
提供官方动作重放采集、单条官方深度演示下载与导入、人工选帧预览、阶段标注工作区、几何查询与构造重算、交付包导出。
具体命令、文件格式和标注示例见该模块 README。

原始数据、阶段选择、二维标注与构造定义分别保存；同阶段所有关键帧必须共用语义名称与定义。
缺失深度、无法观察或几何退化时，相关 point / pose 标为 `unavailable`，不填入猜测结果。
交付包保存 RGB-D、标注图、机器人状态与计算审计，并生成供后续控制器装配输入的 `astra_context.md`。
当前尚未自动将该文件注入既有 rollout controller。

本地 Isaac Sim 6.0.1 的无窗口采集已通过 `organize_table` 两动作验证：
默认使用完整 rendering experience 与 Kit 渲染循环，保存初始帧及两个动作后的
三路 RGB-D、实测 EEF 和指关节，时间戳为 0、0.04、0.08 秒。
该测试使用明确标记的评测布局，归档为 `bounded_smoke`；不作为完整演示或成功率结果。
实际启动配置与局部兼容适配保存在 `launch_config.json`。

官方同时公开了 `RoboDojo_depth` 深度演示。本地已下载并导入 `organize_table` 的
`episode_0000000.hdf5`（838 帧），归档位置为
`/data/nexus-runtime/datasets/organize_table_depth/nexus_recording_v1`。
其机器人轨迹与已有 RGB 版一致，导入保存官方标定和毫米深度的来源与约定。
官方文件不提供实际指关节读数，因此导入数据明确保留该缺失，并区分其 reported 夹爪 state 与新录制的实测指关节。

官方 RGB-D 导入不等于重新录制。完整重录需要与源 episode 明确对应的布局文件；
目前公开资产目录中只找到 `Eval_Layout`，尚未找到本条训练演示的配套布局。
因此不能将任意评测布局重放称为该演示的忠实重录。
本条演示采用用户确定的 7 个阶段、17 个关键帧，选择与原始时间换算保存在
`/data/nexus-runtime/datasets/organize_table_depth/selection_user_v1.json`。
用户已确认首版采用官方 RGB-D 和归一化夹爪 state；缺失的实测指关节不作补造。
语义标注使用显式调用的 [nexus-demonstration-annotator](../hybrid_rollout/demonstrations/nexus-demonstration-annotator/SKILL.md)
流程，由独立执行 agent 看图选择参考并调用几何工具，主 agent 验证工具并复核交付。

本条演示交付已完成，位于
`/data/nexus-runtime/datasets/organize_table_depth/astra_demo_independent_v1/`。
包含 7 阶段、17 关键帧、51 张标注图、51 个深度数组和 10 张阶段内轨迹概览。
独立标注采用 `gpt-6.1-sol` / `xhigh`；主 agent 完成逐阶段视觉复核、坐标链及原始数据一致性验证。
帧 311、361 的手办放置标记和帧 811 的键盘标记因遮挡不可用，原因保留在上下文与审计中。
处理说明、验证结果及使用入口见包内 `processing_report.md`。
