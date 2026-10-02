"""Add three rollout services to a full, persistent Codex policy agent."""

# 本模块是“模型线程”和“物理 rollout 状态机”之间的适配层，主要负责：
# 1. 启动隔离的 Codex app-server，并创建一个贯穿整局 episode 的持久化 thread；
# 2. 按评测方法注册动态工具，把模型工具调用分发给 RoboDojoTools 或 GPTOnlyTools；
# 3. 把观测 JSON 和 RGB 图像作为工具结果返回给模型；
# 4. 记录公开事件、token 用量和所有工具调用，处理有限的同线程网络恢复；
# 5. 严格区分“执行前参数错误”和“执行结果不确定”：前者允许模型改参重试，后者必须
#    终止，绝不能重复发送可能已经执行过的物理动作。
# 本模块只编排控制流程，不直接持有模拟器状态，也不自行生成机器人动作。
import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

from ..io import InputError, write_json
from ..prompt_context import CONTEXT_VERSION
from ..settings import MODEL, EFFORT, PROVIDER, DEFAULT_CODEX_IMAGE_MAX_EDGE
from ..robodojo_server.client import RoboDojoTools
from ..method import evaluation_method
from .schema import _obj, response_schema, direct_response_schema
from .transport import StdioAppServer
from .workspace import prepare_workspace
from .image_preview import image_max_edge, prepare_image
from .network_recovery import NETWORK_CONTINUE_DELAYS, closed_network_turn, is_network_error

# Hybrid skill 的根目录；GPT-only 会在运行时切换到同级的专用 skill 目录。
SKILL_ROOT = Path(__file__).parent

# 写入归档的控制器协议版本。版本名强调：只有“工具输入在执行前被拒绝”才可恢复。
CONTROLLER_VERSION = 'recoverable_tool_input_v1'

# 这些是 Codex app-server 原生工作项，不由下面的动态 rollout handler 执行。
# 收到它们时只记录公开事件，并把它们视为 Agent 仍在正常分析的证据。
NATIVE_WORK_ITEMS = frozenset(('commandExecution', 'fileChange', 'imageView',
    'webSearch', 'mcpToolCall', 'collabAgentToolCall'))


def rejected_input(error, rollout, arguments, count):
    """把执行前的 ``InputError`` 转成可返回给模型的结构化纠错信息。

    这个函数只解释错误，不能替模型修改参数、裁剪 target、再次调用 handler 或 reset
    模拟器。其他异常（尤其是 RPC 返回不确定）不会进入这里，而是作为致命错误终止 episode。
    """
    # no_execution=True 是关键审计字段：它声明该错误发生在任何物理动作之前。
    packet = dict(error=str(error), error_type='recoverable_tool_input',
        no_execution=True, retryable=True, controller_version=CONTROLLER_VERSION,
        rejected_calls=count, step_id=rollout.tick, next_call=rollout.next_call(),
        correction='Correct the arguments and retry next_call in this same episode. '
                   'Do not reset or replay already executed actions.')
    if getattr(rollout, 'observation_path', None):
        packet['observation_path'] = str(rollout.observation_path)
    request = getattr(rollout, 'request', None)
    if isinstance(request, dict) and rollout.phase in ('act', 'execute'):
        # 把当前 request_id 和测量到的 EEF 位姿原样返回，方便模型修正同一轮请求。
        packet['request_id'] = request.get('request_id')
        packet['current_eef'] = request.get('current_eef')
        # 5 cm 约束是三维欧氏距离而不是逐轴约束。这里只附加诊断，不改变原 validator。
        response = arguments.get('response') if isinstance(arguments, dict) else None
        if ('exceeds 5 cm' in str(error) and isinstance(response, dict)
                and response.get('mode') == 'eef'):
            distances = {}
            for arm in ('left', 'right'):
                try:
                    target = response['target'][arm]['position']
                    current = request['current_eef'][arm]['position']
                    if len(target) != 3 or len(current) != 3:
                        continue
                    distance = math.dist(target, current)
                    if math.isfinite(distance):
                        distances[arm] = distance
                except (KeyError, TypeError, ValueError, OverflowError):
                    # 诊断计算失败不能遮蔽最初的权威校验错误。
                    continue
            packet['eef_translation'] = dict(distance_m=distances, max_distance_m=0.05,
                metric='Euclidean distance from the current EEF, not a per-axis limit')
    return packet


def toml_value(value):
    """把 Python 配置值编码为 Codex ``-c`` 参数要求的 TOML 字面量。

    字典必须编码为 TOML inline table；直接使用 JSON 对象或错误引用 dotted key 会改变
    Codex 对配置层级的解释。
    """
    if isinstance(value, dict):
        return '{' + ', '.join(json.dumps(k) + ' = ' + toml_value(v) for k, v in value.items()) + '}'
    return json.dumps(value)


def agent_config(audit, agent):
    """生成 rollout 专用的 Codex 配置，并把写权限限制在 Agent 工作目录。"""
    # 保留 Codex 常规工具/skill 配置，同时显式启用 shell 与图片查看；即使继承的 profile
    # 曾关闭它们，在线策略仍需要这两个能力分析观测和本地文件。
    return dict(model=MODEL, model_provider=PROVIDER, model_reasoning_effort=EFFORT,
        # 数据库和日志放进隔离的 rollout state 目录，避免污染开发者日常 Codex home。
        sqlite_home=str(Path(os.environ.get('ROLLOUT_CODEX_STATE_DIR', os.environ.get('CODEX_HOME', str(audit))))/'runtime_db'),
        log_dir=str(Path(os.environ.get('ROLLOUT_CODEX_STATE_DIR', os.environ.get('CODEX_HOME', str(audit))))/'runtime_logs'),
        default_permissions='rollout_agent', **{
            'features.shell_tool': True, 'features.view_image': True,
            'shell_environment_policy.inherit': 'all',
            'permissions.rollout_agent.extends': ':workspace',
            # 父目录保存 host 侧 RPC 日志和实验产物，只能读；只有 agent 工作区允许写。
            'permissions.rollout_agent.filesystem': {
                str(audit.parent): 'read', str(agent): 'write'},
        })


def tool_specs(method='pi05_plus_gpt'):
    """按评测方法构造 app-server 动态工具的 JSON Schema。

    GPT-only 只有 ``start -> act``；Hybrid 则是 ``start -> infer -> execute``。工具描述使用
    英文是因为在线策略的公开输出契约要求使用英文，但这里的注释说明其实际控制语义。
    """
    string = {'type': 'string'}

    def spec(name, description, **properties):
        # _obj 会关闭 additionalProperties，并把列出的字段全部标为 required。
        return dict(type='function', name=name, description=description,
                    inputSchema=_obj(properties))
    if method == 'gpt_only':
        # Direct 方法没有 π₀.₅ 服务，也没有 joint/student/edit/stop 分支。
        return [spec('robodojo_start', 'Blocking: start the requested episode and return RGB/proprio.',
                     task=string, output_dir=string),
                spec('robodojo_act', 'Blocking: execute GPT-authored bounded dual-arm EEF targets '
                     '(1-5 steps), then return RGB/proprio and native outcome. '
                     'No pi05 inference, proposals or direct joint-action commands exist.',
                     observation_path=string, response=direct_response_schema(), output_dir=string)]
    # Hybrid 方法必须为每次 execute 先取得一份新的 π₀.₅ proposal。
    return [
        spec('robodojo_start', 'Blocking: start the requested RoboDojo episode; save and return current RGB/proprio.',
             task=string, output_dir=string),
        spec('pi05_infer', 'Blocking: infer pi05 once from the latest observation; save chunk and return full robot-only FK. '
             'action_diagnostics provides shape, finiteness, opening ranges and maximum successive joint step. '
             'Use these for routine arithmetic; native tools remain available for independent or additional checks. '
             'Diagnostics are not a safety or gate verdict: perform the unchanged visual and intent assessment.',
             observation_path=string, output_dir=string),
        spec('robodojo_execute', 'Blocking: validate and execute a reviewed fresh chunk or short EEF correction; save and return new observation and terminal result.',
             proposal_path=string, response=response_schema(), output_dir=string),
    ]


def content_items(packet, *, images, max_image_edge=DEFAULT_CODEX_IMAGE_MAX_EDGE):
    """把 host 返回包转换成 Codex 工具结果中的文本和图片附件。

    图片缩放只服务于 teacher 预览。原始 PNG、NPZ、观测字典和 π₀.₅ 输入数组均保持不变，
    因而不会让模型预览设置影响策略输入或评测记录。
    """
    # 浅复制 packet，并替换副本中的 image descriptor；绝不原地修改权威记录。
    visible_packet = packet
    attachments = []
    if images:
        visible_packet = dict(packet, images=[])
        for item in packet.get('images', []):
            # descriptor 记录预览尺寸/路径，payload 是实际发送给 Codex 的 PNG 字节。
            descriptor, payload = prepare_image(item, max_image_edge)
            visible_packet['images'].append(descriptor)
            data = base64.b64encode(payload).decode('ascii')
            attachments.append(dict(type='inputImage', imageUrl='data:image/png;base64,' + data))
    # JSON 文本始终排在前面，随后追加 0～3 张 RGB 图片。
    return [dict(type='inputText', text=json.dumps(visible_packet, separators=(',', ':'))), *attachments]


class CodexPolicy:
    """一个 episode 对应的持久化 Codex 策略线程及其事件循环。"""

    def __init__(self, workspace, codex, *, timeout=900, transport_factory=StdioAppServer,
                 method='pi05_plus_gpt'):
        # 先规范化 method，禁止通过 provider、checkpoint 等旁路信息猜测评测方法。
        self.method = evaluation_method(method)
        direct = self.method == 'gpt_only'
        # GPT-only 和 Hybrid 使用不同的策略说明，但共用当前 app-server 编排实现。
        skill_root = SKILL_ROOT.parent/'robodojo-gpt-only-rollout' if direct else SKILL_ROOT
        # 该尺寸只影响发送给模型的预览图，不影响原始传感器帧或 π₀.₅ 输入。
        self.image_max_edge = image_max_edge(os.environ.get(
            'CODEX_IMAGE_MAX_EDGE', str(DEFAULT_CODEX_IMAGE_MAX_EDGE)))
        self.workspace = Path(workspace)
        # exist_ok=False 保证每局使用新目录，避免旧线程、旧工具结果混入本次评测。
        self.workspace.mkdir(parents=True, exist_ok=False)
        # 提前固化恢复语义，后续审计可以确认没有因输入错误 reset 或 replay 动作。
        write_json(self.workspace.parent/'controller_protocol.json', dict(
            controller_version=CONTROLLER_VERSION, context_version=CONTEXT_VERSION,
            recoverable_input_error_limit=None, validation_unchanged=True,
            same_thread=True, simulator_reset_on_input_error=False,
            action_replay_on_input_error=False))
        # Agent 可写工作区与 host 侧审计目录分离；workspace.json 会告诉 Agent 各路径用途。
        self.agent_workspace = prepare_workspace(self.workspace, skill_root, method=self.method)
        self.timeout = timeout

        # developerInstructions 由固定 skill、可选 Hybrid gate 和版本化 teacher context 拼接。
        # Direct 不需要 outcome/intent gate，所以不读取 gate_prompt.md。
        skill = (skill_root/'SKILL.md').read_text()
        gate = '' if direct else (skill_root/'gate_prompt.md').read_text()
        context = (skill_root/'context/teacher_context.md').read_text()
        self.prompt = (skill + ('' if direct else '\n\n# Unchanged baseline gate prompt\n\n' + gate)
                       + '\n\n' + context + '\n\nAgent working directory: '
                       + str(self.agent_workspace)
                       + '\nResolve context/ and workspace.json relative to that directory.\n')
        # prompt 哈希写入 run 记录，用于证明线上决策使用了哪一版上下文。
        self.prompt_sha256 = hashlib.sha256(self.prompt.encode()).hexdigest()

        # 保存本局实际使用的文本和工具 schema，而不是依赖运行后可能变化的源码文件。
        (self.workspace/'SKILL.md').write_text(skill)
        if not direct:
            (self.workspace/'gate_prompt.md').write_text(gate)
        (self.workspace/'PROMPT.md').write_text(self.prompt)
        specs = tool_specs(self.method)
        write_json(self.workspace/'tools.json', specs)

        # 记录真实 CLI 版本和完整启动参数，确保 app-server 行为可追溯。
        version = subprocess.run([codex, '--version'], capture_output=True, text=True, check=True).stdout.strip()
        config = agent_config(self.workspace, self.agent_workspace)
        argv = [codex, 'app-server', '--stdio', '--strict-config']
        for key, value in config.items():
            argv += ['-c', key + '=' + toml_value(value)]
        write_json(self.workspace/'launch.json', dict(argv=argv, config=config))
        self.transport = transport_factory(argv, self.workspace)
        try:
            # 先完成 JSON-RPC initialize 握手，再创建非临时 thread。整个 episode 复用此 thread，
            # 因而历史观察、模型公开消息和工作笔记可以连续积累。
            self.transport.request('initialize', dict(
                clientInfo=dict(name='robodojo-hybrid-policy', version='1'),
                capabilities=dict(experimentalApi=True)), timeout)
            self.transport.notify('initialized', {})
            response = self.transport.request('thread/start', dict(
                cwd=str(self.agent_workspace), model=MODEL, modelProvider=PROVIDER,
                config=dict(model_reasoning_effort=EFFORT), developerInstructions=self.prompt,
                dynamicTools=specs, ephemeral=False,
                allowProviderModelFallback=False, approvalPolicy='never',
                approvalsReviewer='auto_review', permissions='rollout_agent',
                runtimeWorkspaceRoots=[str(self.agent_workspace)]), timeout)
            # 不允许 provider 静默回退到其他模型或 reasoning effort，否则不同 case 不可比。
            if response.get('model') != MODEL or response.get('reasoningEffort') != EFFORT:
                raise RuntimeError(f'Codex model/effort differs from the required {MODEL}/{EFFORT}: '
                                   f'{response.get("model")}/{response.get("reasoningEffort")}')
            self.thread_id = response['thread']['id']
            # worker.json 是本局模型侧身份清单：方法、版本、thread、prompt 和恢复策略都在此记录。
            write_json(self.workspace/'worker.json', dict(model=MODEL, model_provider=PROVIDER,
                controller_version=CONTROLLER_VERSION,
                evaluation_method=self.method,
                reasoning_effort=EFFORT, context_version=CONTEXT_VERSION,
                codex_version=version, thread_id=self.thread_id, pid=self.transport.process.pid,
                prompt_sha256=self.prompt_sha256, no_rollback=True, tools=[s['name'] for s in specs],
                service_tools_are_additive=True, agent_workspace=str(self.agent_workspace),
                native_tools='Codex defaults; shell and image viewing explicitly enabled',
                network_recovery=dict(revision='same_thread_continue_v1',
                    backoff_seconds=list(NETWORK_CONTINUE_DELAYS), restart_simulator=False,
                    new_thread=False, replay_actions=False),
                codex_image_max_edge=self.image_max_edge, policy_images_resized=False,
                baseline_full_conversation_available=False))
        except BaseException:
            # 初始化任一步失败都关闭子进程，不能留下一个身份不完整的后台 app-server。
            self.close()
            raise

    def _turn(self, text):
        """在同一个持久化 thread 中启动新 turn，并返回 app-server 分配的 turn ID。"""
        result = self.transport.request('turn/start', dict(threadId=self.thread_id,
            model=MODEL, effort=EFFORT, approvalPolicy='never', approvalsReviewer='auto_review',
            permissions='rollout_agent', cwd=str(self.agent_workspace),
            runtimeWorkspaceRoots=[str(self.agent_workspace)], input=[dict(type='text', text=text)]), self.timeout)
        return result['turn']['id']

    def _network_continue(self, rollout, old_turn, error, consecutive, serial):
        """在确认旧 turn 已关闭后，同线程发送一次网络恢复提示。

        恢复只唤醒模型，不 reset 模拟器、不创建新 thread，也不重放任何服务调用或物理动作。
        ``consecutive`` 用来选择有上限的退避时间，超过预设次数立即失败。
        """
        if not is_network_error(error) or consecutive >= len(NETWORK_CONTINUE_DELAYS):
            raise RuntimeError(f'Codex network continue exhausted or ineligible: {error}')
        delay = NETWORK_CONTINUE_DELAYS[consecutive]
        record = dict(revision='same_thread_continue_v1', thread_id=self.thread_id,
            previous_turn_id=old_turn, error=error, step_id=rollout.tick,
            phase=rollout.phase, next_call=rollout.next_call(), delay_seconds=delay,
            consecutive_attempt=consecutive+1, status='waiting', physical_actions_replayed=0)
        path = self.workspace/f'network_continue_{serial:04d}.json'
        # 在等待前落盘，进程若被中断，审计仍能看到恢复停在哪个阶段。
        write_json(path, record)
        print(json.dumps(dict(event='codex_network_continue_wait', **record)), flush=True)
        # 退避期间保持 Python、模拟器和当前观测存活；SIGTERM 仍会正常打断 sleep。
        for _ in range(delay):
            time.sleep(1)
        text = ('Continue the same rollout. The previous turn ended because of a network error. '
                'The simulator and recorded actions are unchanged; do not reset or repeat executed actions. '
                f'Current step_id={rollout.tick}. Use the latest recorded observation and next_call: '
                + json.dumps(rollout.next_call()))
        if getattr(rollout, 'observation_path', None):
            text += '\nLatest observation: ' + str(rollout.observation_path)
        record['status'] = 'turn_start_requested'
        # 在发送 turn/start 之前先记录状态；若 RPC 响应丢失，调用方不能盲目再发第二个 turn。
        write_json(path, record)
        new_turn = self._turn(text)
        record.update(status='continue_sent', new_turn_id=new_turn)
        write_json(path, record)
        print(json.dumps(dict(event='codex_network_continue_sent', thread_id=self.thread_id,
            old_turn_id=old_turn, turn_id=new_turn, step_id=rollout.tick)), flush=True)
        return new_turn

    def run(self, rollout):
        """驱动 Codex 事件循环，直到 rollout 进入 ``done`` 或出现不可恢复错误。"""
        # 初始 turn 明确授权的数据范围，并把状态机给出的 next_call 作为唯一首个服务调用。
        turn_id = self._turn('Act as the autonomous policy agent for this single simulation rollout. '
            'Use your normal file, image, shell/code and planning tools as useful, alongside '
            'the rollout service tools. Read workspace.json for paths and interpreter; '
            'keep working notes and analysis in your agent workspace. Use English for every '
            'public explanation, assessment, progress update, note, and final report. '
            'The user authorizes sending this episode\'s three RGB images, proprio, robot-only FK, '
            'task text and same-episode history to OpenAI Codex for online decisions. '
            'First call: ' + json.dumps(rollout.next_call()))

        # call_index：所有动态工具调用（成功和拒绝都计数）的审计序号。
        call_index = 0
        # errors：执行前 InputError 的累计数，只用于记录，不因成功调用而清零。
        errors = 0
        # continuations：模型正常结束 turn、但 episode 未完成时的上下文续跑次数。
        continuations = 0
        # network_*：只追踪网络故障唤醒，不与工具参数拒绝混用。
        network_error = None
        network_consecutive = network_serial = 0
        completed_turns = 0
        # 0 表示不设置额外 token 上限；原生 episode horizon 仍然有效。
        token_limit = int(os.environ.get('CODEX_MAX_TOTAL_TOKENS', '0'))
        # deadline 是“无有效 Agent 活动”的超时，每次真实工具/分析活动后都会向后延长。
        deadline = time.monotonic() + self.timeout
        while True:
            remaining = deadline - time.monotonic()
            try:
                if remaining <= 0:
                    raise TimeoutError('Codex made no completed service or native tool call within the configured timeout')
                event = self.transport.next_message(remaining)
            except TimeoutError:
                if not network_error:
                    raise
                # 已收到网络错误但缺少终态通知时，先只读查询 thread。不能据此直接再启动一个
                # turn，否则可能让两个 turn 同时控制同一模拟器。
                response = self.transport.request('thread/read', dict(
                    threadId=self.thread_id, includeTurns=True), min(30, self.timeout))
                turn = closed_network_turn(response, self.thread_id, turn_id)
                if turn is None:
                    raise
                event = dict(method='turn/completed', params=dict(threadId=self.thread_id, turn=turn))
            method, params = event.get('method'), event.get('params', {})

            # tokenUsage 是 thread 级累计值。每次更新都覆盖写入，便于中途终止后仍保留用量。
            if method == 'thread/tokenUsage/updated' and params.get('threadId') == self.thread_id:
                usage = params.get('tokenUsage', {})
                write_json(self.workspace.parent/'token_usage.json', dict(
                    model=MODEL, effort=EFFORT, provider=PROVIDER, usage=usage,
                    total_tokens_include_cached_input=True, configured_limit=token_limit))
                print(json.dumps(dict(event='token_usage', total=usage.get('total', {}))), flush=True)
                if token_limit and usage.get('total', {}).get('totalTokens', 0) >= token_limit:
                    raise RuntimeError('Configured total-token budget reached; stopping this episode')

            # item/tool/call 只表示本模块注册的动态工具。Codex 原生 shell/image 等工具由
            # app-server 自己执行，并通过后面的 item/started、item/completed 事件报告。
            elif method == 'item/tool/call':
                # threadId/turnId 必须同时匹配，防止迟到事件或其他线程越权操作当前 episode。
                if params.get('threadId') != self.thread_id or params.get('turnId') != turn_id:
                    raise RuntimeError('Tool call belongs to another thread or turn')
                name, arguments = params['tool'], params['arguments']
                print(json.dumps(dict(event='tool_start', tool=name, call=call_index,
                                      phase=rollout.phase, step_id=rollout.tick)), flush=True)
                audit = dict(call_id=params['callId'], tool=name, arguments=arguments)
                # 在执行 handler 前先写请求。即使进程随后崩溃，也能知道最后尝试了什么。
                write_json(self.workspace/f'call_{call_index:04d}_request.json', audit)
                # 两种方法暴露的状态机不同；不要给 GPT-only 偷偷接入 π₀.₅ handler。
                handlers = (dict(robodojo_start=rollout.start, robodojo_act=rollout.act)
                    if self.method == 'gpt_only' else
                    dict(robodojo_start=rollout.start, pi05_infer=rollout.infer, robodojo_execute=rollout.execute))
                try:
                    # 部分 app-server 版本可能把 arguments 作为 JSON 字符串返回；统一解码后再
                    # 做类型检查。解析失败仍属于执行前错误，没有物理副作用。
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError as error:
                            raise InputError(f'Tool arguments must be a JSON object: {error}') from error
                    if name not in handlers:
                        raise InputError('Unknown host service; native Codex tools are executed by app-server')
                    if not isinstance(arguments, dict):
                        raise InputError('Tool arguments must be a JSON object')
                    # handler 内部会再次验证 phase、路径、request_id 和动作边界。
                    packet = handlers[name](**arguments)
                except InputError as error:
                    # 只有明确的 InputError 才进入可恢复分支；RPC/物理/文件异常会向外抛出。
                    errors += 1
                    packet = rejected_input(error, rollout, arguments, errors)
                    success = False
                else:
                    success = True
                    # 成功服务调用说明网络链路已恢复，只重置网络连续失败计数；参数拒绝次数
                    # 是审计事实，不能清零或覆盖。
                    network_consecutive = 0
                    network_error = None
                # 无论成功还是拒绝，都必须先落盘结果再回复模型。若 reply 传输失败，host 也
                # 不能再次调用 handler，因为第一次调用可能已经执行了物理动作。
                write_json(self.workspace/f'call_{call_index:04d}_result.json', packet)
                # π₀.₅ infer 重用刚才 observation 的图片，不重复附图；start/act/execute 返回
                # 新 observation 时才附加 RGB 预览。
                self.transport.reply(event['id'], dict(success=success,
                    contentItems=content_items(packet, images=success and name != 'pi05_infer',
                                               max_image_edge=self.image_max_edge)))
                call_index += 1
                print(json.dumps(dict(event='tool_done', tool=name, step_id=rollout.tick,
                    phase=rollout.phase, counters=rollout.counters)), flush=True)
                # 一次完整服务调用已经取得进展，从当前时刻重新计算 Agent 空闲超时。
                deadline = time.monotonic() + self.timeout

            # 原生工作项和公开 Agent 消息只用于记录活动，不参与 rollout 状态转换。
            elif method in ('item/started', 'item/completed'):
                if params.get('threadId') != self.thread_id or params.get('turnId') != turn_id:
                    continue
                item = params.get('item', {})
                kind = item.get('type')
                if kind in NATIVE_WORK_ITEMS or (kind == 'agentMessage' and method == 'item/completed'):
                    # 只保留公开消息和工具活动，绝不复制模型私有 reasoning item。
                    with (self.workspace/'agent_events.jsonl').open('a') as stream:
                        stream.write(json.dumps(dict(method=method, **params), ensure_ascii=False)+'\n')
                    event_record = dict(event='agent_activity', phase=method.split('/')[-1],
                        item_type=kind, item_id=item.get('id'), step_id=rollout.tick,
                        status=item.get('status'))
                    if kind == 'agentMessage':
                        event_record['text'] = item.get('text', '')
                    elif kind == 'commandExecution':
                        event_record['command'] = item.get('command')
                        event_record['exit_code'] = item.get('exitCode')
                    print(json.dumps(event_record, ensure_ascii=False), flush=True)
                    if kind in NATIVE_WORK_ITEMS:
                        # 查看图片、运行脚本等是真实分析进展，不能误判为 Agent 卡住。
                        deadline = time.monotonic() + self.timeout

            # 一个模型 turn 结束并不一定代表物理 episode 结束；以 rollout.phase 为准。
            elif method == 'turn/completed' and params.get('threadId') == self.thread_id:
                turn = params.get('turn', {})
                if turn.get('id') != turn_id:
                    continue
                write_json(self.workspace/f'turn_{completed_turns:02d}_completed.json', turn)
                completed_turns += 1
                failure = turn.get('error') or network_error
                if turn.get('status') == 'failed' and is_network_error(failure):
                    if rollout.phase == 'done':
                        # 模拟器已经原生结束时，不再为一条总结消息消耗新的模型调用。
                        return
                    # 只有确认旧 turn 已失败关闭后，才在同一 thread 中发送恢复 turn。
                    turn_id = self._network_continue(rollout, turn_id, failure,
                        network_consecutive, network_serial)
                    network_serial += 1
                    network_consecutive += 1
                    network_error = None
                    deadline = time.monotonic() + self.timeout
                    continue
                if turn.get('status') != 'completed':
                    raise RuntimeError(f'Codex turn ended: {turn.get("status")}')
                if rollout.phase == 'done':
                    return
                if continuations >= 2:
                    raise RuntimeError('Codex ended repeatedly before completing the episode')
                # 这是同 episode 的上下文续跑，不包含人工动作判断，也不替模型选择动作。
                continuations += 1
                turn_id = self._turn('The same episode is unfinished. Continue the skill; next call: ' +
                                     json.dumps(rollout.next_call()))
                deadline = time.monotonic() + self.timeout

            # error/turn/failed 可能先于 turn/completed 到达。网络错误先记下并等待权威终态，
            # 非网络错误则立即失败。
            elif method in ('error', 'turn/failed'):
                if (params.get('threadId') not in (None, self.thread_id)
                        or params.get('turnId') not in (None, turn_id)):
                    continue
                if is_network_error(params.get('error', params)):
                    network_error = params.get('error', params)
                if method == 'error' and params.get('willRetry') is True:
                    # 这是 app-server 内部的模型传输重试，不会重新调用 infer、host handler
                    # 或任何物理动作，因此可以继续等待。
                    print(json.dumps(dict(event='codex_transport_retry',
                        error=params.get('error'), step_id=rollout.tick)), flush=True)
                    continue
                if network_error and is_network_error(params.get('error', params)):
                    if rollout.phase == 'done':
                        return
                    # 等待 turn/completed；若迟迟不到，再由上方 timeout 分支只读核验。
                    # 当前 turn 可能仍活跃时绝不能发送 continue。
                    deadline = min(deadline, time.monotonic() + 30)
                    print(json.dumps(dict(event='codex_network_wait_terminal',
                        error=network_error, step_id=rollout.tick)), flush=True)
                    continue
                raise RuntimeError(f'Codex error: {params}')

            # 带 id 的未知 method 是 app-server 向客户端发来的能力请求。这里没有实现对应
            # 能力，必须 fail closed，不能悄悄忽略后让协议双方状态分叉。
            elif 'id' in event and 'method' in event:
                raise RuntimeError(f'Unexpected Codex capability request: {method}')

    def close(self):
        """关闭 app-server transport；实现应同时回收其拥有的 Codex 子进程。"""
        self.transport.close()


def main():
    """命令行入口：组装学生策略、Codex worker 和对应方法的 rollout 状态机。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--evaluation-method', default=evaluation_method(), choices=('pi05_plus_gpt','gpt_only'))
    parser.add_argument('--codex', required=True)
    parser.add_argument('--sim-port', type=int, default=19113)
    parser.add_argument('--student-port', type=int, default=18830)
    parser.add_argument('--max-decisions', type=int, default=180, help='0 disables the decision-count limit')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    args.output = args.output.resolve()
    # controller 输出目录必须全新，避免混入旧 request/response 或覆盖审计证据。
    args.output.mkdir(parents=True, exist_ok=False)
    student = worker = rollout = None

    def terminate(signum, frame):
        # SIGTERM 转成正常 Python 清理路径，只停止本进程拥有的连接和子进程。
        raise KeyboardInterrupt('Stopping only this owned rollout and Codex process')

    signal.signal(signal.SIGTERM, terminate)
    try:
        if args.evaluation_method == 'pi05_plus_gpt':
            # Hybrid 必须显式绑定 checkpoint，并在连接时核对服务端 checkpoint 身份。
            if args.checkpoint is None:
                raise ValueError('Hybrid evaluation requires an explicit checkpoint')
            from ..pi05_server.client import Pi05Client
            student = Pi05Client(args.student_port, args.checkpoint)

        # 先创建模型 worker，取得 prompt_sha256；run.json 会把它与本局控制记录绑定。
        worker = CodexPolicy(args.output/'codex_workspace', args.codex, method=args.evaluation_method)
        if args.evaluation_method == 'gpt_only':
            # 延迟导入避免 Hybrid 路径无意加载 Direct 专用实现。
            from ..robodojo_server.gpt_only_client import GPTOnlyTools
            rollout = GPTOnlyTools(args.output, args.task, sim_port=args.sim_port,
                seed=args.seed, max_decisions=args.max_decisions, prompt_sha256=worker.prompt_sha256)
        else:
            rollout = RoboDojoTools(args.output, args.task, student, sim_port=args.sim_port,
                seed=args.seed, max_decisions=args.max_decisions, prompt_sha256=worker.prompt_sha256)
        # run() 阻塞到原生结束、预算结束或不可恢复错误。
        worker.run(rollout)
    except BaseException:
        # failure.json 先保存原始 traceback 和最后确认的 episode/tick，方便区分控制错误与
        # simulator 原生失败。这里不会把异常吞掉，写完后仍向上抛出。
        write_json(args.output/'failure.json', dict(error=traceback.format_exc(),
            controller_version=CONTROLLER_VERSION,
            episode_id=rollout.episode if rollout else None, step_id=rollout.tick if rollout else 0,
            counters=rollout.counters if rollout else {}, completed=False))
        if rollout is not None and rollout.episode is not None and rollout.phase != 'done':
            try:
                # finish 只把当前已确认状态标为 controller_error，不 reset、不补动作。
                rollout.finish('controller_error')
            except Exception:
                # finish 本身失败时也不能重连或重试不确定动作；保留上面的原始 failure。
                pass
        raise
    finally:
        # 按依赖方向关闭：先断开 rollout/simulator，再关闭学生服务连接和 Codex transport。
        if rollout is not None:
            rollout.close()
        if student is not None:
            student.close()
        if worker is not None:
            worker.close()


if __name__ == '__main__':
    main()
