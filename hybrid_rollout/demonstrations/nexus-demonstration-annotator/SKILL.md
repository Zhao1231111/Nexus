---
name: nexus-demonstration-annotator
description: "标注 Nexus 机器人演示中人工确定阶段和关键帧的 RGB-D 语义 point／pose，调用几何工具重算并保存可复核结果。显式调用时使用，不自动选阶段或关键帧。"
---

# Nexus 演示语义标注

把人工选定的阶段与关键帧处理成可复核的语义几何标注。使用前读取仓库中的
`docs/demostration_protocol.md` 和本模块的 [工具与 JSON 契约](../README.md)。
调用方应提供仓库根目录、Python 解释器、recording、selection、prepared workspace、
annotations 和 review 输出目录。路径由本次任务提供，不绑定特定 episode。
若从技能目录的符号链接进入，先解析其真实路径，再按仓库文件位置解析相对文档链接。

## 输入与判断边界

- 保留人工阶段、对象、关键帧顺序和描述。发现图像与人工描述不符时另写观察备注，不改写原描述。
- 以整个阶段为标注单位：先查看全部关键帧的三路原分辨率 RGB，再确定该阶段共用的语义关键点和构造定义。
  可以制作联系图、裁剪或放大辅助判断；最终 `[u,v]` 必须对应原图分辨率。
- 关键点与功能参考由你选择，不套用固定物体模板。选择能在本阶段稳定辨认的功能部位；
  point 足够时使用 point，不为了填满字段而构造 pose。同一语义名称不能逐帧代表不同部位。
- 只使用 RGB-D、相机标定和机器人本体状态；不读物体 pose、分割真值或场景布局来补标注。
  不把画在遮挡物上的像素深度当成被遮挡对象深度，不把上一帧结果无声明地当成本帧测量。
- 无法可靠观察的语义仍保留名称，写 `unavailable` 和具体原因；无效深度和退化构造也保留缺失。
  不通过悄悄换点、提高平面拟合阈值或猜测遮挡位置来获得表面完整的结果。

## 几何与来源

几何工具只做计算，不替你选择语义。所有三维结果位于 `environment_origin`，单位米；
四元数为 wxyz，光学相机 +x 向右、+y 向下、+z 向前，深度为 optical z。

定义 pose 时，写清原点、两条有序轴的构造、法向区域和每条轴的来源。
人为约定的轴可以补全 frame，但须明确它是约定，不是物体可辨识的真实朝向。
点与法向来自不同表面时须说明它们为何共同定义该功能参考。

夹爪／EEF 可以引用归档的 `source: {kind: robot_eef, arm: left|right}` pose。
该 pose 是 link6，不是 TCP、指尖或接触点；二维投影仅用于展示本体来源，不能用投影像素的背景深度重算。
多个独立主体用 `collection.members` 分别保存 point／pose，不平均成一个刚体。
夹持物体仍需物体观测，不能直接用 EEF pose 冒充物体 pose。

归一化夹爪 state 与实际指关节测量按归档 source 区分；缺失指关节保持缺失。
二维人工估计与量化深度是近似观测，不承诺精确标定或毫米级语义定位。

## 工具调用与复核

在仓库根目录用调用方提供的解释器运行：

```bash
<python> -m hybrid_rollout.demonstrations.cli query \
  --recording <recording> --frame <frame_id> --request <query.json>
<python> -m hybrid_rollout.demonstrations.cli resolve \
  --recording <recording> --selection <selection.json> \
  --annotation <annotations/stage_id.json> --output <review/stage_id/resolved.json> \
  --render-output <review/stage_id/frames>
```

`query` 支持像素深度、三维反投影和平面法向。表达式、pose 和机器人引用由 `resolve` 重算。
调用几何工具实际验证选择，不手写最终三维数值。CLI `resolve` 的标准输出包含完整审计；
保存到日志后只读取简短状态摘要，详细证据从 `resolved.json` 按需读取。

每阶段写 `<stage_id>.json`，包含 `reviewed_frame_ids`、语义名称、阶段共用 `definitions` 和
每帧二维点／法向区域。用 `resolve` 校验后查看生成的标注图，确认像素落在正确部位、
法向区域没有混入其他物体、参考没有跳到不同功能部位。每次修正都重新计算。
切换相机后，若静止对象的同名参考出现明显三维跳变，先重投影到另一视图并核对实际表面；
外观相似的正面、背面或顶面可用局部法向辅助区分，不直接把跳变解释为物体运动或标定误差。

提交全部阶段标注和简短中文说明：看过哪些帧、各对象采用什么参考、哪些帧不可用及原因。
可以按阶段向调用方报告进展。不要启动仿真、调用 Astra、改写工具实现或擅自选择新关键帧。
整包导出和独立复核由调用方完成；此次真实标注同时用于验证本 skill 的可执行性。
