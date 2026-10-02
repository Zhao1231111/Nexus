"""Observation-to-action GPT baseline, without a pi05 client or proposals."""

# 本模块实现 GPT-only 的 RGB/本体状态到 EEF 动作基线，不连接 π₀.₅，也不存在候选
# proposal。GPTOnlyTools 复用 RoboDojoTools 的观测保存、RPC、EEF 局部 IK 和结束逻辑，
# 但把 Hybrid 的 start -> infer -> execute 状态机改为 start -> act。
# 模型只能输出左右臂受限的绝对 EEF target；host 每执行一个真实控制步后重新计算一次局部
# IK，再读取新的模拟器状态。成功、失败和 episode 终止仍由 RoboDojo 原生环境判定，模型
# 不能自行声明成功或提前 stop。
import uuid

import numpy as np

from ..io import InputError, write_json
from ..prompt_context import CONTEXT_VERSION
from ..settings import MODEL, EFFORT, PROVIDER
from .client import RoboDojoTools, ARMS
from .decision_log import decision_event, emit
from .validation import validate_response, validate_public_language


def validate_action(response, request):
    """在触发任何 RPC/物理动作前校验 GPT-only 响应。

    这里先检查 Direct 方法特有的顶层契约，再逐臂复用 Hybrid 的 ``validate_response``，
    从而共享 5 cm、0.35 rad、单位四元数、步数等 EEF 安全边界。
    """
    # request_id 把动作绑定到当前观测；旧观测生成的迟到响应必须拒绝。
    if not isinstance(response, dict) or response.get('request_id') != request['request_id']:
        raise ValueError('Response must match the current observation request_id')
    # reason 是可公开审计的简短视觉证据和动作目的，不是私有思维链。
    if not isinstance(response.get('reason'), str) or not response['reason'].strip():
        raise ValueError('Brief visible evidence and action purpose are required')
    # 公开决策统一使用英文，保证报告和审计格式与已冻结 baseline 一致。
    validate_public_language(response)
    mode = response.get('mode')
    if mode == 'eef':
        # Direct baseline 不开放 14D 关节数组入口，避免绕过 EEF 边界和局部 IK。
        if response.get('actions') not in (None, []):
            raise ValueError('Direct joint actions are not available in EEF-only evaluation')
        # 双臂任务要求显式给出两只手；不能省略“保持不动”的那一臂。
        if set(response.get('target', {})) != set(ARMS):
            raise ValueError('EEF action requires explicit left and right targets')
        for arm in ARMS:
            # 通用 validator 面向单臂结构，因此把双臂 response/request 投影成逐臂视图。
            validate_response(dict(response, target=response['target'][arm]),
                dict(request, current_eef=request['current_eef'][arm]))
    else:
        # GPT-only 没有 student/edit/stop 分支；冻结评测必须等待原生终止。
        raise ValueError('GPT-only supports bounded EEF actions only; no joint/student/edit/stop')


class GPTOnlyTools(RoboDojoTools):
    """复用公共 RoboDojo 工具基础设施的 Direct EEF 控制状态机。"""

    def __init__(self, output, task, **kwargs):
        # student=None 明确表示本方法不会创建或访问 π₀.₅ client。
        super().__init__(output, task, None, **kwargs)
        # simulator 记录中的控制源从 episode 开始就是 GPT EEF。
        self.source = 'gpt_eef'
        # 保留公共计数器字段，并增加 Direct 专用字段；gpt_joint_steps 持续为 0，明确证明
        # 没有直接关节动作路径。
        self.counters.update(gpt_joint_steps=0, gpt_eef_steps=0)

    def next_call(self):
        """根据当前 phase 返回模型唯一合法的下一次服务调用。"""
        if self.phase == 'start':
            # start 的路径和 fresh output_dir 规则与 Hybrid 完全复用。
            return super().next_call()
        if self.phase == 'act':
            # observation_path 必须精确匹配当前观测，output_dir 必须是新的序号目录。
            return dict(tool='robodojo_act', observation_path=str(self.observation_path),
                output_dir=str(self.output/'observations'/f'{len(self.history)+1:03d}'),
                response='Choose bounded dual-arm EEF action from current observation')
        return None

    def infer(self, **arguments):
        """显式封死从继承关系误调用 π₀.₅ infer 的路径。"""
        raise InputError('pi05 is unavailable in the GPT-only evaluation')

    def execute(self, **arguments):
        """显式封死 Hybrid ``robodojo_execute`` 路径。"""
        raise InputError('Use robodojo_act; no pi05 proposal exists')

    def _observation(self, directory, *, result=None):
        """保存一帧权威观测，并为下一次 Direct 动作生成新的 request_id。"""
        # 父类从 simulator 的缓存 post-ACK 状态读取三路 RGB、14D state 和双臂 EEF，
        # 同时写 observation.npz、PNG 和 observation.json。
        packet = super()._observation(directory, result=result)
        packet['evaluation_method'] = 'gpt_only'
        if self.phase != 'done':
            # 每个观测只对应一个新 request_id。prediction_id=None 表明没有学生策略 proposal。
            self.request = dict(packet, request_id=uuid.uuid4().hex,
                decision=len(self.history), prediction_id=None, gate_policy='gpt_only')
            packet.update(request_id=self.request['request_id'],
                request_path=str(self.output/f'request_{len(self.history):03d}.json'))
            # request_NNN.json 是本次模型动作的不可变输入快照，便于把 response 与观测配对。
            write_json(self.output/f'request_{len(self.history):03d}.json', self.request)
        # 父类落盘发生在加入 Direct 字段之前，因此最后覆盖写一次完整 packet。
        write_json(self.observation_path, packet)
        return packet

    def start(self, **arguments):
        """连接模拟器、执行唯一一次 reset，并写出本局 Direct 方法身份。"""
        # _check 同时验证工具顺序、task/output_dir 精确值，以及输出目录尚不存在。
        directory = self._check('robodojo_start', arguments)
        # 一个 GPTOnlyTools 实例只持有一个 simulator RPC 连接，避免多 owner 并发控制。
        self.sim = self.rpc_factory('127.0.0.1', self.sim_port, timeout=600)
        self.meta = self.sim.request('metadata')
        if self.meta['task'] != self.task:
            raise RuntimeError('Launched task differs from requested task')
        # reset 只能发生一次。source/policy_version 会写入 simulator 侧归档，证明本局从未使用
        # π₀.₅ 学生策略。
        reset = self.sim.request('reset', seed=self.seed, source='gpt_eef', policy_version='gpt_only')
        if reset.get('instruction'):
            # 原生环境返回的实际 instruction 优先于启动参数中的占位 metadata。
            self.meta['instruction'] = reset['instruction']
        self.episode, self.tick = reset['episode_id'], reset['step_id']
        # 冻结 benchmark case 禁止模型 stop，必须运行到原生 success/timeout；非 benchmark
        # 调试局仍由公共 finish 逻辑处理预算或控制器退出。
        self.require_native_termination = bool(reset.get('metadata', {}).get('evaluation_case'))

        # combination.json 绑定模型、prompt/context 和“无 student”事实，不推进物理时间。
        self._rpc('begin_combination', teacher_model=MODEL, teacher_model_provider=PROVIDER,
            context_version=CONTEXT_VERSION, evaluation_method='gpt_only',
            prompt_sha256=self.prompt_sha256, student_policy_version=None, student_policy_sha256=None)
        # simulator 默认控制源为 student；Direct 启动后立即显式切换到 gpt_eef，仍不执行动作。
        self._rpc('switch_control_source', source='gpt_eef', reason='GPT-only EEF observation-to-action baseline')

        # run.json 是 controller 侧的本局方法契约。所有 student/checkpoint 字段显式置空，
        # 防止下游仅凭字段缺失误判是否使用过 π₀.₅。
        self.run = dict(schema='robodojo_rollout.run.v1', evaluation_method='gpt_only',
            teacher='codex_tools', context_version=CONTEXT_VERSION, teacher_model=MODEL,
            teacher_model_provider=PROVIDER, teacher_reasoning_effort=EFFORT,
            task=self.task, instruction=self.meta['instruction'], seed=self.seed,
            evaluation_case=reset.get('metadata', {}).get('evaluation_case'), robot_profile=self.profile,
            checkpoint=None, student_backend=None, student_identity_sha256=None,
            student_server_metadata=None, pi05_enabled=False, pi05_inference_calls=0,
            action_horizon=5, action_dim=14, action_space='eef_only', gate_policy='gpt_only', no_rollback=True,
            teacher_prompt_sha256=self.prompt_sha256, initial_state_hash=reset.get('initial_state_hash'),
            reset_metadata=reset.get('metadata'), max_episode_steps=self.meta['max_episode_steps'],
            control_dt=self.meta['control_dt'], max_decisions=self.max_decisions,
            require_native_termination=self.require_native_termination, training_steps=0,
            training_bundle_ready=False)
        write_json(self.output/'run.json', self.run)
        # 历史在 episode 开始时必须为空，之后每个 act 完成后整体覆盖写入。
        write_json(self.output/'history.json', self.history)
        # reset 完成后唯一合法的下一步变为 robodojo_act。
        self.phase = 'act'
        # 返回 step_id=0 的首帧观测，并生成 request_000.json。
        return self._observation(directory)

    def act(self, **arguments):
        """校验并执行一次 GPT 编写的双臂 EEF 决策。"""
        # 在访问 response 前先校验 observation_path/output_dir 和状态机 phase。
        directory = self._check('robodojo_act', arguments)
        response = arguments.get('response')
        try:
            # 校验发生在任何 simulator mutation 之前；失败会转成 InputError 反馈给模型改参。
            validate_action(response, self.request)
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            raise InputError(str(error)) from error

        decision = len(self.history)
        # 给公开 response 打上方法标签，并在执行前落盘模型原始决策。
        response = dict(response, evaluation_method='gpt_only')
        write_json(self.output/f'response_{decision:03d}.json', response)

        # 先让 simulator 持久化决策收据，再执行物理动作。收据失败意味着尚未进入动作循环，
        # 调用方会终止而不是猜测记录状态。
        receipt = self._rpc('record_codex_decision', decision=decision, prediction_id=None, response=response)
        emit(dict(decision_event(episode_id=self.episode, step_id=self.tick, decision=decision,
            prediction_id=None, response=response), simulator_receipt=receipt))

        wanted = 'gpt_eef'
        if wanted != self.source:
            # 正常 Direct 路径在 start 时已切换；该分支保留公共状态一致性并记录 control_epoch。
            self._rpc('switch_control_source', source=wanted, reason=response['reason'])
            self.source = wanted

        start = self.tick
        trace = []
        # 一个模型 target 可执行 1～5 步，但每一步都基于最新测量关节状态重新求一次局部 IK。
        # 这样不会预先排队多个依赖旧状态的关节动作。
        for _ in range(response['steps']):
            # target_proposal 只做 robot-only DLS，返回一条 14D 绝对关节/夹爪动作和诊断；
            # 它不推进物理时间，也不读取物体真值。
            proposal = self.target_proposal(response['target'])
            # 每次 chunk_step 只发送一行动作，并用当前 request_id 标记其 teacher 来源。
            result = self._rpc('chunk_step', actions=np.asarray([proposal['action']], np.float32),
                               teacher_request_id=self.request['request_id'])
            for row in result['steps']:
                # 把名义 EEF target 和本步 IK 诊断附到真实 ACK，便于离线追溯。
                row['edited_target'], row['ik_diagnostics'] = response['target'], proposal['diagnostics']
            trace.extend(result['steps'])
            # tick 只采用 simulator 返回的权威 step_id，不能在 controller 本地自行加一。
            self.tick = result['step_id']
            if any(r['terminated'] or r['truncated'] for r in result['steps']):
                # 原生环境已经结束时立即停止余下重复 target，不再消耗物理步。
                break

        # edit_NNN.json 这个历史文件名沿用公共下游格式；在 Direct 中它实际保存逐步 EEF/IK
        # 记录，而不是 Hybrid 的 proposal edit。
        write_json(self.output/f'edit_{decision:03d}.json', [dict(target=r['edited_target'],
            diagnostics=r['ik_diagnostics'], valid=r['valid'],
            executed_action=np.asarray(r['executed_action']).tolist()) for r in trace])
        # 只有 simulator 标为 valid 的 ACK 才进入计数和 execution 数组。
        valid = [r for r in trace if r['valid']]
        self.counters['gpt_eef_steps'] += len(valid)
        # 即使本次 0 个有效步，也固定 reshape 为 (0, 14)，保持序列化 tensor layout 稳定。
        executed = np.asarray([r['executed_action'] for r in valid], np.float32).reshape(-1, 14)
        np.savez_compressed(self.output/f'execution_{decision:03d}.npz', actions=executed,
            states=np.asarray([r['obs']['states'] for r in valid], np.float32).reshape(-1, 14))
        # terminal 只依据最后一个有效 simulator ACK；模型自己的 reason 不参与终止判定。
        done = bool(valid and (valid[-1]['terminated'] or valid[-1]['truncated']))

        # history 的一行对应一次模型决策，可同时关联 request、response、动作区间和原生结果。
        self.history.append(dict(decision=decision, start_tick=start, end_tick=self.tick,
            response=response, evaluation_method='gpt_only', simulator_decision_receipt=receipt,
            executed_steps=len(valid), source=self.source, source_detail=response['mode'], prediction_id=None,
            executed_gripper_closed=self.gripper_history(executed), terminal=done,
            native_success=bool(valid and valid[-1]['success'])))
        # history/progress 在返回下一张观测前落盘，崩溃后仍能定位最后确认的动作边界。
        write_json(self.output/'history.json', self.history)
        write_json(self.output/'progress.json', dict(episode_id=self.episode, step_id=self.tick,
            evaluation_method='gpt_only', **self.counters))

        # max_decisions=0 表示不设 controller 决策数上限；原生 step horizon 仍然生效。
        exhausted = self.max_decisions > 0 and len(self.history) >= self.max_decisions
        self.phase = 'act'
        # 原生终止或 controller 决策预算耗尽时完成归档；预算耗尽不等价于原生成功。
        final = self.finish('terminal' if done else 'decision_budget') if done or exhausted else None
        # 无论是否刚结束，都保存最后一次 post-ACK 观测；done 时不会再生成新 request_id。
        return self._observation(directory, result=final)

    def finish(self, reason):
        """复用公共结束逻辑，并在最终结果中再次声明 Direct 方法身份。"""
        result = super().finish(reason)
        # 显式的 false/0 比缺字段更适合下游审计，证明本局没有 π₀.₅ inference。
        result.update(evaluation_method='gpt_only', pi05_enabled=False, pi05_inference_calls=0)
        write_json(self.output/'result.json', result)
        return result
