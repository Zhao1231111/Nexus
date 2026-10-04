"""无策略 RPC 冒烟：一次 reset、一次实际 hold 控制、显式 finish_pilot。"""

import argparse
import json
import time

import numpy as np

from .robodojo_server.protocol import RPCClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=19113)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--connect-timeout', type=float, default=120.)
    parser.add_argument('--rpc-timeout', type=float, default=210.)
    args = parser.parse_args()
    if args.connect_timeout <= 0 or args.rpc_timeout <= 0:
        parser.error('Timeouts must be positive')

    deadline = time.monotonic() + args.connect_timeout
    while True:
        try:
            client = RPCClient('127.0.0.1', args.port, timeout=args.rpc_timeout)
            break
        except OSError:
            # 只等待服务开始监听。下面可能推进仿真的 RPC 各发送一次，失败时
            # 直接报错，不通过重试重复执行 reset 或控制动作。
            if time.monotonic() >= deadline:
                raise
            time.sleep(1.)
    try:
        metadata = client.request('metadata')
        print(f'RPC metadata OK: {metadata["task"]}', flush=True)
        print('Starting RPC reset...', flush=True)
        reset = client.request('reset', seed=args.seed, source='teacher', policy_version='rpc_smoke')
        if reset['step_id'] != 0:
            raise RuntimeError('Fresh reset must begin at step zero')
        print('RPC reset OK', flush=True)
        identity = dict(episode_id=reset['episode_id'], step_id=0)
        before = client.request('teacher_observation', **identity)
        states = np.asarray(before['states'], dtype=np.float32)
        if states.shape != (14,) or not np.isfinite(states).all():
            raise RuntimeError('Invalid measured dual-arm state')

        # 当前状态 hold 仍进入真实原生控制器并推进一个控制周期；不启动模型，
        # 不要求任务成功，也不把这一动作宣称为成功率实验。
        ack = client.request('chunk_step', **identity, actions=[states.copy()])
        if ack['step_id'] != 1 or len(ack['steps']) != 1 or not ack['steps'][0]['valid']:
            raise RuntimeError('Expected one valid native control ACK')
        identity['step_id'] = 1
        after = client.request('teacher_observation', **identity)
        shapes = {name: list(np.asarray(after[name]).shape) for name in metadata['cameras']}
        if len(shapes) != 3 or any(len(shape) != 3 or shape[2] != 3 or
                                   shape != list(np.asarray(before[name]).shape)
                                   for name, shape in shapes.items()):
            raise RuntimeError(f'Expected three consistent RGB observations: {shapes}')
        # 显式完成 smoke，验证结果与视频 flush 路径；返回的 complete/success
        # 仍由原生任务条件决定，单步冒烟通常不完成任务。
        result = client.request('finish_pilot', **identity, reason='rpc_smoke_complete')
        if result['video_frames'] != 2 or result['reason'] != 'rpc_smoke_complete':
            raise RuntimeError('Smoke evidence was not finalized with two video frames')
        print('RPC_SMOKE_PASS ' + json.dumps(dict(task=metadata['task'], step_id=1,
              camera_shapes=shapes, video_frames=result['video_frames'],
              reason=result['reason'], astra_calls=0)), flush=True)
    finally:
        client.close()


if __name__ == '__main__':
    main()
