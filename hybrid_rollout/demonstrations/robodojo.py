"""官方 HDF5 动作重放与 RGB-D 录制；Isaac 模块只在 main 内导入。"""

import argparse
import hashlib
from pathlib import Path
import subprocess

import numpy as np

from .data import CAMERAS, Recorder, finite, pose_matrix, read_json, write_json


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def load_official_trajectory(path):
    """只读取官方机器人字段；不把 HDF5 内其他字段透传给模型。"""
    import h5py
    arrays = {}
    with h5py.File(path, 'r') as file:
        for group in ('action', 'state'):
            for arm in ('left', 'right'):
                for field, dimension in (('arm_joint_states', 6), ('ee_joint_states', 1), ('ee_poses', 7)):
                    key = f'{group}/{arm}_{field}'
                    array = np.asarray(file[key], dtype=float)
                    if array.ndim != 2 or array.shape[1] != dimension or not np.isfinite(array).all():
                        raise ValueError(f'Invalid official trajectory field: {key}')
                    arrays[key] = array
        frequency = float(file['additional_info/frequency'][()])
        instruction = file['instruction'][()]
        instruction = instruction.decode('utf-8') if isinstance(instruction, bytes) else str(instruction)
    lengths = {len(array) for array in arrays.values()}
    if len(lengths) != 1 or not next(iter(lengths)) or not np.isfinite(frequency) or frequency <= 0:
        raise ValueError('Official robot trajectories must be finite, synchronized and nonempty')
    actions = np.concatenate([arrays[f'action/{arm}_{field}'] for arm in ('left', 'right')
                              for field in ('arm_joint_states', 'ee_joint_states')], axis=1)
    if np.any(actions[:, [6, 13]] < 0) or np.any(actions[:, [6, 13]] > 1):
        raise ValueError('Official gripper commands must be normalized to [0,1]')
    return dict(arrays=arrays, actions=actions, frequency=frequency, instruction=instruction)


def configure_sensors(cfg, frequency):
    """配置在原生任务配置处理之后应用，避免被任务专用相机覆盖。"""
    cfg.eval_cfg.observation.collect_freq = frequency
    cfg.eval_cfg.observation.vision.update(dict(depth=True, approximate_depth=False,
                                               intrinsic_matrix=False, extrinsic_matrix=False))
    cfg.camera.default_frequency = frequency
    for name in ('common', *CAMERAS):
        cfg.camera.annotator[name] = dict(enabled=True,
            rgb_capture=dict(type='rgb', device='cpu'),
            distance_to_image_plane_capture=dict(type='distance_to_image_plane', device='cpu'))


def to_numpy(value):
    if hasattr(value, 'detach'):
        value = value.detach()
    if hasattr(value, 'cpu'):
        value = value.cpu()
    return np.asarray(value)


def capture_frame(env, index, action, physics_start):
    """一次原生 render 后读取所有传感器；读取期间不推进物理世界。"""
    before = int(env.sim._sim_step_counter)
    raw = env.get_obs()
    manager = env.robot_manager
    origin = to_numpy(manager.scene.env_origins[0])
    robots = {robot.arm_name.split('_')[0]: robot for robot in manager.robot_list if robot.type == 'target'}
    if set(robots) != {'left', 'right'}:
        raise ValueError('Recorder requires the two target X5 arms')
    states, poses, grippers, commands = [], [], [], []
    for arm in ('left', 'right'):
        robot = robots[arm]
        joints = manager.get_joint(robot, env_idx_list=[0])[0]
        fingers = finite(manager.get_end_effector_real_val(robot, env_idx_list=[0])[0], (2,), 'measured gripper joints')
        low, high = robot.gripper_scale
        opening = (fingers[0] - low) / (high - low) if robot.gripper_move['sign'] == 1 else (high - fingers[0]) / (high - low)
        states.extend([*joints, opening])
        grippers.append(fingers)
        commands.append(float(raw['state'][f'{arm}_ee_joint_state'][0]))
        end = manager.get_real_endpose(robot, env_idx_list=[0], is_relative=True)[0]
        poses.append(pose_matrix(end[:3], end[3:]))
    frame = dict(frame_id=index, timestamp_s=(before - physics_start) * env.dt,
                 source_action_index=index - 1, action=np.asarray(action, dtype=float),
                 states=np.asarray(states), T_world_eef=np.asarray(poses),
                 gripper_joints=np.asarray(grippers), gripper_commanded=np.asarray(commands))
    names = env.camera_manager.camera_names[0]
    for name in CAMERAS:
        camera = env.camera_manager.cameras[0][names.index(name)]
        sensor = raw['vision'][name]
        frame[f'{name}_rgb'] = np.asarray(sensor['color'], dtype=np.uint8)[..., :3].copy()
        frame[f'{name}_depth_m'] = np.asarray(sensor['depth'], dtype=np.float32).copy()
        # 原生 CameraManager 外参指向安装 Xform。读取实际 optical prim 的世界
        # 位姿后，把 USD(+y up,-z forward) 转为光学(+y down,+z forward)。
        position, quaternion = camera.get_world_pose(camera_axes='usd')
        T = pose_matrix(to_numpy(position) - origin, to_numpy(quaternion))
        frame[f'{name}_T_world_camera'] = T @ np.diag([1., -1., -1., 1.])
        frame[f'{name}_K'] = to_numpy(camera.get_intrinsics_matrix(device='cpu')).copy()
    if int(env.sim._sim_step_counter) != before:
        raise RuntimeError('Physics advanced while collecting a synchronized observation')
    return frame


def main():
    # cv2 必须先于 Kit 的依赖路径加载，保持与现有模拟器入口一致。
    import cv2  # noqa: F401
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', default='organize_table')
    parser.add_argument('--trajectory', type=Path, required=True)
    parser.add_argument('--layout', type=Path, required=True, help='Explicit source demonstration layout JSON')
    parser.add_argument('--layout-provenance', required=True, help='Evidence connecting this layout to the source episode')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-actions', type=int, help='Bounded smoke only; archive remains incomplete')
    parser.add_argument('--startup-stack-after', type=int,
                        help='Diagnostic only: dump a Python stack if native construction/reset stalls')
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    from hybrid_rollout.robodojo.robodojo_server.launcher import configure_camera_app, resolved_camera_app
    configure_camera_app(args)
    if args.max_actions is not None and args.max_actions < 1:
        parser.error('--max-actions must be positive')
    if args.startup_stack_after is not None and args.startup_stack_after < 1:
        parser.error('--startup-stack-after must be positive')
    if args.output.exists():
        parser.error('Recording output must be a new directory')
    trajectory = load_official_trajectory(args.trajectory)
    layout = read_json(args.layout)
    from env.isaac_sim_compat import add_isaac_sim_compat_search_path, enable_isaac_sim_compat_extensions
    add_isaac_sim_compat_search_path(args)
    launcher = AppLauncher(args)
    app = launcher.app
    env = recorder = None
    try:
        enable_isaac_sim_compat_extensions(app)
        print('DEMO_STAGE app_ready', flush=True)
        from hybrid_rollout.robodojo.robodojo_server.native_compat import install_geometry_device_compat
        install_geometry_device_compat()
        launch_config = resolved_camera_app(launcher)
        launch_config['native_compat'] = ['geometry_relative_pose_device']
        print(f'DEMO_LAUNCH {launch_config}', flush=True)
        import importlib
        from omegaconf import OmegaConf
        from env.global_configs import BENCHMARK, ENV_CONFIG_PATH, ROOT_DIR
        from src.eval_client import eval_env
        from utils.load_file import load_yaml
        from utils.pipeline_utils import process_config, process_randomization
        from hybrid_rollout.robodojo.robodojo_server.server import NoPolicyConnection
        from hybrid_rollout.robodojo.robodojo_server.session import register_native_evaluation
        registry = importlib.import_module(f'task.{BENCHMARK}.task_registry')
        root = Path(ENV_CONFIG_PATH)
        evaluation = load_yaml(str(root / 'arx_x5.yml'))
        evaluation.update(task_name=args.task, num_envs=1, device_id=0, seed=0,
                          policy_name='official_demo_replay', additional_info='nexus_rgbd', eval_batch=False)
        values = {key: load_yaml(str(root / key / (evaluation['config'][key] + '.yml')))
                  for key in ('sim', 'scene', 'camera', 'robot')}
        values.update(eval_cfg=evaluation, deploy_cfg=dict(port=1, policy_name='official_demo_replay'),
                      task_env=load_yaml(registry.task_config_path(str(Path(ROOT_DIR) / 'task' / BENCHMARK / 'config'), args.task)))
        cfg = OmegaConf.create(values)
        cfg.sim.scene.num_envs, cfg.sim.seed = 1, [args.seed]
        cfg.sim.device = args.device
        cfg = process_randomization(cfg)
        cfg, _ = process_config(cfg, task_name=args.task)
        configure_sensors(cfg, trajectory['frequency'])
        cfg.eval_cfg.eval_num = 1
        for robot in cfg.robot.robots:
            robot.need_planner = False  # 官方绝对关节动作重放不需要 IK 或场景规划。
        print('DEMO_STAGE config_ready', flush=True)
        original = eval_env.WsModelClient
        import faulthandler
        if args.startup_stack_after:
            # 实际冒烟曾停在环境初始化且长期无日志。只在显式诊断时保存调用栈，
            # 方便定位加载／编译阶段；不增加模拟步或改变数据采集。
            faulthandler.dump_traceback_later(args.startup_stack_after)
        try:
            eval_env.WsModelClient = NoPolicyConnection
            print('DEMO_STAGE env_construct_begin', flush=True)
            env = eval_env.create_eval_env(cfg, app)
        finally:
            eval_env.WsModelClient = original
            if args.startup_stack_after:
                faulthandler.cancel_dump_traceback_later()
        print('DEMO_STAGE env_construct_done', flush=True)
        # seed 只控制环境随机数；布局必须来自显式配套文件，不能默认拿评测布局。
        env.seed_manager.get_seed_scene_info = lambda seed: layout
        env._stream_vision = lambda *args, **kwargs: None
        env.step_lim = max(env.step_lim, len(trajectory['actions']) + 1)
        print('DEMO_STAGE reset_begin', flush=True)
        native_sim_step = env.sim_step
        reset_steps = 0

        def monitored_sim_step(*values, **options):
            nonlocal reset_steps
            result = native_sim_step(*values, **options)
            reset_steps += 1
            if reset_steps % 100 == 0:
                print(f'DEMO_RESET_PROGRESS physics_steps={reset_steps}', flush=True)
            return result

        # 与现有 session.reset 的诊断方式一致，只包装既有步进，原样调用一次，
        # 不额外推进物理步；无论 reset 成败，都恢复原方法和诊断计时器。
        env.sim_step = monitored_sim_step
        if args.startup_stack_after:
            faulthandler.dump_traceback_later(args.startup_stack_after)
        try:
            env.reset(seed=[args.seed])
            print('DEMO_STAGE reset_done', flush=True)
            register_native_evaluation(env)
        finally:
            env.sim_step = native_sim_step
            if args.startup_stack_after:
                faulthandler.cancel_dump_traceback_later()
        print('DEMO_STAGE scene_ready', flush=True)
        revision = subprocess.run(['git', '-C', ROOT_DIR, 'rev-parse', 'HEAD'], check=True,
                                  capture_output=True, text=True).stdout.strip()
        source = dict(trajectory=str(args.trajectory.resolve()), trajectory_sha256=digest(args.trajectory),
                      layout=str(args.layout.resolve()), layout_sha256=digest(args.layout),
                      layout_provenance=args.layout_provenance, source_frequency_hz=trajectory['frequency'],
                      source_actions=len(trajectory['actions']), simulator_revision=revision,
                      seed=args.seed, replay='absolute joint commands, original frequency',
                      kind='native_replay', gripper_state_provenance='measured finger joint positions',
                      camera_capture='native RoboDojo tiled RGB-D capture',
                      native_termination='recorded but does not truncate source replay')
        recorder = Recorder(args.output, args.task, trajectory['instruction'], source)
        # 保留原始机器人轨迹，重放实测轨迹另存逐帧文件，二者不会混为一谈。
        np.savez_compressed(args.output / 'source_trajectory.npz', **trajectory['arrays'])
        write_json(args.output / 'launch_config.json', launch_config)
        write_json(args.output / 'resolved_config.json', OmegaConf.to_container(cfg, resolve=True))
        start = int(env.sim._sim_step_counter)
        recorder.append(capture_frame(env, 0, [], start))
        count = min(args.max_actions or len(trajectory['actions']), len(trajectory['actions']))
        native_success_seen = False
        for index, action in enumerate(trajectory['actions'][:count]):
            if not app.is_running():
                raise RuntimeError('Simulator closed before source replay completed')
            # 演示可能包含任务成功后的退回动作，录制必须保留整条源轨迹；这里只解除
            # EvalEnv 的提前停止，不跳过奖励判断，也不改变现有 baseline 的执行入口。
            env.end_flag[0] = False
            command = {key: value for arm, offset in (('left', 0), ('right', 7))
                       for key, value in ((f'{arm}_arm_joint_state', action[offset:offset + 6]),
                                          (f'{arm}_ee_joint_state', action[offset + 6:offset + 7]))}
            env.take_action(command)
            native_success_seen |= bool(env.end_flag[0] and env.success[0])
            recorder.append(capture_frame(env, index + 1, action, start))
            if (index + 1) % 50 == 0:
                print(f'DEMO_PROGRESS actions={index + 1}/{count}', flush=True)
        complete = count == len(trajectory['actions']) and args.max_actions is None
        recorder.finish(complete=complete, native_success_seen=native_success_seen,
                        recorded_actions=count, status='recorded' if complete else 'bounded_smoke')
        # 原生进度条使用 \r 且不保证在结束时换行。显式另起一行，使启动脚本
        # 能可靠识别完成标记；否则真实录制成功也会被误判为缺失 DEMO_PASS。
        print(f'\nDEMO_PASS complete={complete} actions={count} output={args.output}', flush=True)
    except BaseException as error:
        if recorder:
            recorder.finish(complete=False, status='error', error=f'{type(error).__name__}: {error}')
        import traceback
        traceback.print_exc()
        raise
    finally:
        if env:
            env.close()
        app.close()


if __name__ == '__main__':
    main()
