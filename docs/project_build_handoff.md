# Project Nexus — Handoff

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

## 当前方法

第一阶段只增加一种能力：

\[
\boxed{\text{Astra can actively query metric geometry}}
\]

概念流程是：

\[
RGB
\rightarrow
\text{Astra semantic reasoning}
\rightarrow
\text{geometry query}
\rightarrow
\text{metric 3D information}
\rightarrow
\text{Astra planning}
\rightarrow
EEF action
\]

重点不是把完整 point cloud 或 depth map 无条件塞给模型，而是把 geometry 设计成 agent tool，让 Astra 自己决定什么时候需要查询、查询哪里、如何利用结果。

这样可以直接比较：

\[
\text{RGB-only Astra}
\quad\text{vs}\quad
\text{RGB + geometry-tool Astra}
\]

看显式 geometry 是否减少 EEF correction、失败接触和 Astra decision 次数。

## 后续方向

如果 geometry tool 有效，再逐步扩展。

一条路线是增强几何能力，例如 region geometry、surface normal、clearance、height map 等。另一条更重要的路线是走向真正的 hierarchical policy：

\[
\text{Astra}
\rightarrow
\text{structured spatial / motor goal}
\rightarrow
\text{low-level visuomotor policy}
\rightarrow
\text{actions}
\]

长期希望高层利用 frozen MLLM 的推理和泛化能力，而低层不再承担语言理解，只执行非语言的空间或运动目标，例如 SE(3) subgoal、3D waypoint、object motion 或 contact target。

RoboICL 的 demonstration context 和 anchored memory 可以在 geometry 本身验证有效之后再加入，从而分别研究 geometry、experience 和 hierarchy 的贡献。

## 开发原则

先解决当前研究问题，不提前为未来功能设计复杂框架。

现在需要做的是：理解 `GPT-as-Policy` 的 `gpt_only` 执行链，在不改变原 baseline 的前提下增加一个 geometry-enabled 方法，让 Astra 能在同一个 persistent Codex agent 中查询几何信息并继续控制 RoboDojo。

做到这一点并完成初步对比后，再决定下一步。