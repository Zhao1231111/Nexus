"""Launch RoboDojo's native evaluation environment behind the existing RPC."""
import argparse
import importlib
import json
import os
import shutil
import traceback
from pathlib import Path


class NoPolicyConnection:
    """Native reset callback only; inference is owned by the Codex tool host."""
    def __init__(self, **kwargs):
        pass

    def call(self, func_name, **kwargs):
        if func_name != 'reset':
            raise RuntimeError('Native autonomous policy loop is disabled')

    def close(self):
        pass


def main():
    import cv2  # Load before Isaac's extension dependency paths.
    from isaaclab.app import AppLauncher
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', default='build_tower')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--port', type=int, default=19113)
    p.add_argument('--eval-seed', type=int, default=0, help='Published layout group, NOT the reset layout ID')
    p.add_argument('--eval-manifest', type=Path)
    p.add_argument('--case-file', type=Path)
    p.add_argument(
        '--visual',
        action='store_true',
        help='Show the Isaac Sim window for interactive smoke-test diagnosis.',
    )
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()
    # IsaacLab 3.0 不再仅凭 headless=False 打开窗口；未显式选择 visualizer 时，
    # AppLauncher 会重新强制无头运行。因此 --visual 必须同时选择 Kit visualizer。
    # 正式评测仍默认无头运行，无论是否显示窗口都保留相机 RPC 观测路径。
    args.headless = not args.visual
    if args.visual:
        args.visualizer = ['kit']
        args.visualizer_explicit = True
    args.enable_cameras = True
    # Isaac Sim 6 删除了一小部分 RoboDojo 仍会导入的旧模块。这里必须在创建
    # SimulationApp 之前注入兼容搜索路径，保证 Kit 启动阶段也能解析这些模块。
    from env.isaac_sim_compat import (
        add_isaac_sim_compat_search_path,
        enable_isaac_sim_compat_extensions,
    )
    add_isaac_sim_compat_search_path(args)
    from ..evaluation import read_panel, verify_assets, case_identity
    from ..io import require
    from env.global_configs import ROOT_DIR, BENCHMARK
    registry = importlib.import_module(f'task.{BENCHMARK}.task_registry')
    evaluation_case = None
    if args.eval_manifest:
        panel = read_panel(args.eval_manifest, os.environ.get('ROLLOUT_EVAL_MANIFEST_SHA256'))
        selected = json.loads(args.case_file.read_text())
        evaluation_case = selected['case']
        require(evaluation_case in panel['cases'] and selected['identity'] == case_identity(panel, evaluation_case),
                'Selected case differs from the frozen panel')
        require((args.task, args.eval_seed) == (evaluation_case['runtime_task'], evaluation_case['eval_seed']),
                'CLI task/eval_seed differs from selected case')
        verify_assets(panel, ROOT_DIR, [evaluation_case])
        args.output.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(ROOT_DIR)/evaluation_case['layout']['path'], args.output/'scene_layout.json')
    # Match native main.py: monitor Articulation tasks before AppLauncher.
    import yaml
    task_config_path = registry.task_config_path(str(Path(ROOT_DIR)/'task'/BENCHMARK/'config'), args.task)
    with open(task_config_path) as stream:
        task_values = yaml.safe_load(stream) or {}
    enable_monitor = bool(task_values.get('Articulation'))
    if enable_monitor:
        from src.eval_client.physx_warning_monitor import get_monitor
        get_monitor().start(enabled=True)
    app = AppLauncher(args).app
    # 部分旧扩展只能在 SimulationApp 创建后启用；该调用与 RoboDojo 的 Isaac 6
    # 冒烟入口保持一致，不改变 Nexus RPC 协议或上层控制逻辑。
    enable_isaac_sim_compat_extensions(app)
    env = session = None
    try:
        from omegaconf import OmegaConf
        from env.global_configs import ENV_CONFIG_PATH, ROOT_DIR, BENCHMARK
        from utils.load_file import load_yaml
        from utils.pipeline_utils import process_config, process_randomization
        from src.eval_client import eval_env
        from .rpc import serve
        from .session import RoboDojoSession
        root = Path(ENV_CONFIG_PATH)
        evaluation = load_yaml(str(root/'arx_x5.yml'))
        evaluation.update(task_name=args.task, num_envs=1, device_id=0, eval_batch=False,
            policy_name='Pi_05', additional_info='codex_hybrid', seed=args.eval_seed,
            physx_monitor_enabled=enable_monitor)
        values = {key: load_yaml(str(root/key/(evaluation['config'][key]+'.yml')))
                  for key in ('sim', 'scene', 'camera', 'robot')}
        values.update(eval_cfg=evaluation, deploy_cfg=dict(port=1, policy_name='Pi_05'),
            task_env=load_yaml(registry.task_config_path(str(Path(ROOT_DIR)/'task'/BENCHMARK/'config'), args.task)))
        cfg = OmegaConf.create(values)
        cfg.sim.scene.num_envs = 1
        cfg = process_randomization(cfg)
        cfg, _ = process_config(cfg, task_name=args.task)
        cfg.eval_cfg.eval_num = 1
        cfg.camera.default_frequency = cfg.eval_cfg.observation.collect_freq
        # Isaac Sim 6 的布料对象必须在第一次 PhysX 初始化前生成，不能再依赖 RPC
        # reset 后动态替换。因此冻结评测用例存在时，启动阶段就使用其 layout_id；
        # 普通本地运行仍从布局 0 启动，并可按原协议执行后续 reset。
        cfg.sim.seed = [int(evaluation_case['layout_id']) if evaluation_case else 0]
        # EEF corrections use bounded robot-only DLS, never object-aware cuRobo.
        for robot in cfg.robot.robots:
            robot.need_planner = False
        original = eval_env.WsModelClient
        try:
            eval_env.WsModelClient = NoPolicyConnection
            env = eval_env.create_eval_env(cfg, app)
        finally:
            eval_env.WsModelClient = original
        if evaluation_case:
            resolved = env.seed_manager.seed_info[evaluation_case['layout_id']]['scene_layout']
            require(Path(resolved).resolve() == (Path(ROOT_DIR)/evaluation_case['layout']['path']).resolve(),
                    'Native SeedManager resolved a different layout')
        from ..io import write_json
        write_json(args.output/'resolved_config.json', OmegaConf.to_container(cfg, resolve=True))
        session = RoboDojoSession(env, args.output, args.task,
            evaluation_identity=selected['identity'] if evaluation_case else None)
        serve(session, args.port)
    except BaseException:
        # SimulationApp.close may exit before Python prints an uncaught error.
        # Persist the original failure before native cleanup can obscure it.
        traceback.print_exc()
        raise
    finally:
        if session:
            session._write_summary('server_close')
        if env:
            env.close()
        app.close()


if __name__ == '__main__':
    main()
