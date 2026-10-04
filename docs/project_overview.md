# Nexus 项目介绍与代码导航

当前项目思路见 [project_concept.md](project_concept.md)。本文保留既有实现的代码导航，
其中研究方向和开发顺序以当前项目思路文档为准。

本文面向两类读者：第一次接手仓库的开发人员，以及需要继续开发本项目的 Agent。
目标是用最短路径说明项目现状、运行架构、关键代码与开发边界。

## 内容导航

- [1. 项目定位](#1-项目定位)
- [2. 当前状态](#2-当前状态)
- [3. 总体架构](#3-总体架构)
- [4. 两条策略路径](#4-两条策略路径)
- [5. 目录导航](#5-目录导航)
- [6. 核心代码导航](#6-核心代码导航)
- [7. 运行产物与数据契约](#7-运行产物与数据契约)
- [8. 关键工程约束](#8-关键工程约束)
- [9. 推荐阅读顺序](#9-推荐阅读顺序)
- [10. Nexus 后续开发入口](#10-nexus-后续开发入口)
- [11. 验证与测试](#11-验证与测试)

## 1. 项目定位

Nexus 基于公开项目 **GPT 6 Astra as an Embodied Policy**。基础仓库已经打通：

- RoboDojo / RoboLab 仿真环境；
- Codex app-server 和持久化模型线程；
- RGB 图像、本体状态和任务文本输入；
- 双臂 EEF 控制；
- π₀.₅ 策略服务；
- rollout 记录、评测冻结、结果审计和报告发布。

Nexus 的研究目标：详细研究背景见 [project_build_handoff.md](project_build_handoff.md)。

目标方法可以概括为：

```text
RGB 观察
  -> Astra 语义推理
  -> 主动调用 geometry tool
  -> 获得深度、3D 坐标或局部几何信息
  -> Astra 规划双臂 EEF 目标
  -> 仿真器执行并返回新观察
```



## 2. 当前状态

仓库当前包含一套已完成的评测发布和一份 Nexus 开发交接说明。

已经实现：

- `pi05_plus_gpt`：π₀.₅ 生成动作候选，Astra 审核并按需修正；
- `gpt_only`：Astra 根据 RGB 和本体状态直接输出双臂 EEF 目标；
- 10 个 RoboDojo 任务的冻结评测、case 配对和公开结果；
- 独立 RoboLab 集成；
- 实验归档、人工审核和双语报告工具链。

## 3. 总体架构

项目按“任务调度、单回合运行、模型控制、仿真执行、结果发布”分层。

```text
集群或本地任务入口
  cluster_entrypoint.sh
        |
        v
case 选择与批次隔离
  batch.py
        |
        v
单回合私有环境
  episode_entrypoint.sh
        |
        v
进程监督
  run_local.sh
        |
        +-- π₀.₅ Policy Server      仅 Hybrid 使用
        +-- RoboDojo Simulator RPC  原生执行与成功判定
        +-- Codex Controller        持久化 Astra 线程与工具分发
                    |
                    v
             文件化控制状态机
                    |
                    v
          rollout artifacts + logs
                    |
                    v
        audit / paired evaluation / report
```



### 3.1 外层调度

`cluster_entrypoint.sh` 校验运行环境、认证配置、模型版本和评测 manifest，然后进入
`batch.py`。`batch.py` 原子占用 case，为每个 episode 启动独立进程，并在结束后检查：

- 子进程是否全部退出；
- RPC 端口是否释放；
- controller 和 simulator 是否产生完整结果；
- artifact manifest 是否通过验证。



### 3.2 单回合进程

`run_local.sh` 为一次 episode 创建全新的结果目录和源码快照，然后启动三个相互隔离的组件：

- Policy Server：加载 OpenPI/JAX checkpoint，仅为 Hybrid 提供 π₀.₅ 推理；
- Simulator Server：持有唯一 RoboDojo 环境和物理状态；
- Codex Controller：运行持久化 Astra thread，并把模型工具调用映射到 rollout API。

三个组件只通过 loopback RPC、WebSocket 和文件产物交换数据。

### 3.3 权威边界

- Astra 负责语义判断、规划和工具选择；
- π₀.₅ 只负责 Hybrid 的候选动作生成；
- Simulator 是动作是否执行、episode 是否结束、任务是否成功的唯一权威；
- 调度器退出码不能代替 simulator 原生结果；
- 报告只消费已冻结和已审计的产物，不参与在线决策。



## 4. 两条策略路径



### 4.1 GPT-only：Nexus 的主开发基线

实现入口是
[gpt_only_client.py](../hybrid_rollout/robodojo/robodojo_server/gpt_only_client.py)。

```text
robodojo_start
  -> 三路 RGB + 14D 本体状态 + 双臂 EEF pose
  -> robodojo_act
  -> 左右臂绝对 EEF target
  -> 参数和边界校验
  -> 局部 DLS 转换为 14D 关节动作
  -> 执行 1～5 步
  -> 新观察和原生终止状态
  -> 循环
```

GPT-only 不启动 π₀.₅，也不允许模型直接输出关节动作。每个 target 必须明确包含左右臂的
位置、wxyz 四元数和夹爪状态。当前每次决策限制为：

- EEF 平移不超过 5 cm；
- EEF 旋转不超过 0.35 rad；
- 执行 1～5 个控制步；
- 每个真实控制 ACK 后重新计算局部 IK。



### 4.2 π₀.₅ + GPT：已完成的参考方法

实现入口是
[client.py](../hybrid_rollout/robodojo/robodojo_server/client.py)。

```text
当前观察
  -> π₀.₅ 生成 50 x 14 候选动作
  -> 机器人 FK 预览候选 EEF 轨迹
  -> Astra 检查上次执行结果和下次动作意图
  -> student：执行候选前 1～15 步
     或 edit/eef：执行 1～5 步修正
  -> 新观察
```

只有观察到执行失败，或判断下一段候选意图明确错误时，Astra 才能接管。单纯不确定不能
触发修正。该逻辑由 `gate_assessment.py` 和 `validation.py` 强制执行。

## 5. 目录导航



### 仓库根目录

- `[README.md](../README.md)`：公开发布说明、结果摘要和运行方式；
- `[docs/](./)`：Nexus 交接文档与本项目介绍；
- `[hybrid_rollout/](../hybrid_rollout/)`：在线控制、评测和报告源码；
- `[public_results/](../public_results/)`：公开 case、分数和 provenance；
- `[report_web/](../report_web/)`：已经编译的静态报告，不是主要开发源码；
- `[release-manifest.json](../release-manifest.json)`：公开发布文件清单；
- `[requirements-tools.txt](../requirements-tools.txt)`：轻量离线工具依赖。



### `hybrid_rollout/robodojo/`

- `skill/`：Hybrid 的 Codex prompt、动态工具 schema 和 app-server 控制器；
- `robodojo-gpt-only-rollout/`：GPT-only 专用 skill 和上下文；
- `robodojo_server/`：模拟器 RPC、控制状态机、校验、FK/IK 和录像；
- `pi05_server/`：π₀.₅ checkpoint 身份、服务端和客户端；
- `codex_backend/`：隔离认证、provider 配置和配置校验；
- `eval_panels/`：冻结评测 panel；
- `evaluation.py`：case、seed、layout 和资源哈希契约；
- `batch.py`：case 占用、episode 隔离和结果收集；
- `artifact_manifest.py`：归档完整性检查；
- `case_ledger.py`、`paired_evaluation.py`：case 状态和配对结果；
- `campaign.py`、`overnight.py`、`*_recovery.py`、`*_audit.py`：历史运维和恢复工具。

初次阅读不需要深入 campaign、迁移和账户恢复脚本。这些文件不定义在线动作语义。

### `hybrid_rollout/robolab/`

这是独立 RoboLab 适配，不是 RoboDojo 的薄封装。它使用不同相机、机器人、动作空间和
Isaac Sim 环境。Nexus 当前研究应以 RoboDojo `gpt_only` 为主线。

### `hybrid_rollout/report_site/`

- `build.py`：从冻结归档提取 case、分数、token、视频和人工标注；
- `app/src/content/report/`：项目专用双语报告内容；
- `app/src/content/dashboard/`：数据视图；
- `app/src/` 其余大部分文件：通用 Data App runtime；
- `preview.py`：支持 HTTP Range 的只读视频预览；
- `package.py`、`package_article.py`：静态交付物打包。



## 6. 核心代码导航



### 理解进程和启动顺序

- `[cluster_entrypoint.sh](../hybrid_rollout/robodojo/cluster_entrypoint.sh)`
- `[batch.py](../hybrid_rollout/robodojo/batch.py)`
- `[episode_entrypoint.sh](../hybrid_rollout/robodojo/episode_entrypoint.sh)`
- `[run_local.sh](../hybrid_rollout/robodojo/run_local.sh)`



### 理解 Codex/Astra 如何控制 episode

- `[skill/run.py](../hybrid_rollout/robodojo/skill/run.py)`：Codex thread、工具注册和事件循环；
- `[skill/schema.py](../hybrid_rollout/robodojo/skill/schema.py)`：工具参数 schema；
- `[skill/transport.py](../hybrid_rollout/robodojo/skill/transport.py)`：app-server JSON-RPC；
- `[robodojo-gpt-only-rollout/SKILL.md](../hybrid_rollout/robodojo/robodojo-gpt-only-rollout/SKILL.md)`：Direct 方法提示和动作约束。



### 理解在线控制

- `[gpt_only_client.py](../hybrid_rollout/robodojo/robodojo_server/gpt_only_client.py)`：Nexus 主基线；
- `[client.py](../hybrid_rollout/robodojo/robodojo_server/client.py)`：Hybrid 状态机；
- `[session.py](../hybrid_rollout/robodojo/robodojo_server/session.py)`：仿真权威状态；
- `[kinematics.py](../hybrid_rollout/robodojo/robodojo_server/kinematics.py)`：双臂 FK 和局部 DLS；
- `[validation.py](../hybrid_rollout/robodojo/robodojo_server/validation.py)`：动作边界；
- `[gate_assessment.py](../hybrid_rollout/robodojo/robodojo_server/gate_assessment.py)`：Hybrid 接管条件；
- `[protocol.py](../hybrid_rollout/robodojo/robodojo_server/protocol.py)`：无自动重试的版本化 RPC。



### 理解 π₀.₅

- `[pi05_server/checkpoint.py](../hybrid_rollout/robodojo/pi05_server/checkpoint.py)`
- `[pi05_server/server.py](../hybrid_rollout/robodojo/pi05_server/server.py)`
- `[pi05_server/client.py](../hybrid_rollout/robodojo/pi05_server/client.py)`



### 理解评测可信度

- `[evaluation.py](../hybrid_rollout/robodojo/evaluation.py)`
- `[artifact_manifest.py](../hybrid_rollout/robodojo/artifact_manifest.py)`
- `[case_ledger.py](../hybrid_rollout/robodojo/case_ledger.py)`
- `[paired_evaluation.py](../hybrid_rollout/robodojo/paired_evaluation.py)`
- `[test_contract.py](../hybrid_rollout/robodojo/test_contract.py)`
- `[test_native_evaluation.py](../hybrid_rollout/robodojo/test_native_evaluation.py)`



## 7. 运行产物与数据契约

一次 episode 通常生成以下结构：

```text
run-root/
  controller/
    run.json                 方法、模型、checkpoint 和运行参数
    history.json             已执行决策历史
    request_*.json           每次模型决策输入
    response_*.json          每次模型公开决策
    proposal_*.npz           π₀.₅ 候选，仅 Hybrid 存在
    execution_*.npz          实际执行动作和状态
    observations/            RGB、NPZ 和 observation JSON
    result.json              controller 汇总
    token_usage.json         模型用量
    codex_workspace/         Codex 工具审计和公开事件

  sim/
    resolved_config.json     解析后的仿真配置
    reset.json               episode 身份和初始状态摘要
    native_evaluation.json   原生成功条件注册信息
    evaluation_outcome.json  原生成功、分数和有效性
    summary.json             仿真汇总
    sensors.mp4              原始传感器录像
    <episode-id>/action_*.json

  logs/
    policy.log
    sim.log
    controller.log

  launch_manifest.json
  policy_identity.json
  artifact_manifest.json
  source_snapshot/
```

新增 geometry tool 时，应为查询输入、查询结果、标定信息和 schema/version 建立明确产物，
使一次决策能够从 observation、geometry query 一直追溯到最终 EEF 动作。

## 8. 关键工程约束



### 无回滚

一个 simulator process 只运行一个 fresh episode。不能在同一 episode 中 reset、恢复旧状态
或重放不确定动作。

### 不重试不确定的物理调用

只有执行前参数校验失败可以修正后重试。RPC 断开、ACK 不确定或物理异常必须终止当前
episode，不能假设动作未执行。

### 原生评测权威

模型文本、视觉判断、调度器退出码和 partial score 都不能代替 RoboDojo 原生 success、
terminated 和 truncated 状态。

### 评测输入冻结

生产评测必须固定 panel、case、layout、seed、native source 和 checkpoint identity。比较两个
方法时应复用同一 case 定义和原生评测过程。

### 新鲜输出目录

运行目录和每次工具输出目录必须是新的。现有结果不能被静默覆盖。

### 只公开必要信息

公开决策日志保存模型的简短证据和动作目的，不保存私有推理流。凭据、认证文件和原始私有
归档不能进入公开产物。

## 9. 推荐阅读顺序



### 第一轮：15～30 分钟建立全局概念

1. 本文；
2. [project_build_handoff.md](project_build_handoff.md)；
3. [README.md](../README.md)；
4. [run_local.sh](../hybrid_rollout/robodojo/run_local.sh)；
5. [GPT-only SKILL.md](../hybrid_rollout/robodojo/robodojo-gpt-only-rollout/SKILL.md)。



### 第二轮：读透 GPT-only 基线

1. `skill/run.py` 中的 `tool_specs`、`CodexPolicy` 和 `main`；
2. `gpt_only_client.py` 的 `start`、`_observation`、`act`；
3. `session.py` 的 `reset`、`chunk_step`、`dispatch`；
4. `kinematics.py` 的 `DualKinematics.target`；
5. `validation.py` 和 `protocol.py`。

完成这一轮后，应能回答：一张 RGB 图如何进入 Astra，一次 EEF target 如何变成真实动作，
以及成功标签从哪里产生。

### 第三轮：理解 Hybrid 作为参考方法

1. `robodojo_server/client.py`；
2. `gate_assessment.py`；
3. `pi05_server/client.py`；
4. `pi05_server/checkpoint.py`。



### 第四轮：理解实验和结果

1. `evaluation.py`；
2. `batch.py`；
3. `artifact_manifest.py`；
4. `paired_evaluation.py`；
5. `report_site/build.py`；
6. `public_results/evaluation_cases.json` 和 `scores.csv`。



