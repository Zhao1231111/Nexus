# Project Nexus — Handoff

本文保留早期开发交接内容。当前项目思路及模块职责以
[project_concept.md](project_concept.md) 为准。

## 背景

我们关注 GPT-6 Astra 在 RoboDojo 中作为 embodied policy 的能力。现有工作表明，Astra 已具备较强的任务理解、语义推理和空间判断能力，但在精确操作中仍大量依赖 RGB 反馈进行 trial-and-error，不断修正 EEF。

我们的判断是：其中一部分试错并不是高层 reasoning 不足，而是 Astra 缺乏可靠的 metric geometry。RoboICL 通过 demonstration context 提供了更好的几何参考，但仍然主要依赖视觉历史，而不是显式的几何查询。

因此项目的第一步是研究：

\[
\text{RGB reasoning} + \text{callable geometry tools}
\]

即让 Astra 继续负责语义理解和决策，但在需要时主动查询深度、3D 坐标等几何信息，以减少通过真实机器人动作“探测几何”的必要。

## 基础代码选择

以 `anonymous-report-421/GPT-as-Policy` 为基础仓库。

主要原因是它已经搭好了我们需要的 agent infrastructure：RoboDojo、Codex app-server、persistent Codex thread、图像查看、工具调用、EEF 控制和 rollout 记录都已经打通，而且可以直接使用 Codex 订阅运行 Astra。

当前应从其 `gpt_only` 路径发展，而不是从 `pi05_plus_gpt` 开始。

RoboICL 暂时不作为基础仓库。它的优势在于精确控制 demonstration、history 和 anchored memory，是另一条很有价值的研究方向；后续可以吸收它的 context/memory 思路，但第一阶段不要把 geometry、ICL 和 hierarchical control 同时混在一起。


## 开发原则

先解决当前研究问题，不提前为未来功能设计复杂框架。

现在需要做的是：理解 `GPT-as-Policy` 的 `gpt_only` 执行链，在不改变原 baseline 的前提下增加一个 geometry-enabled 方法，让 Astra 能在同一个 persistent Codex agent 中查询几何信息并继续控制 RoboDojo。

做到这一点并完成初步对比后，再决定下一步。
