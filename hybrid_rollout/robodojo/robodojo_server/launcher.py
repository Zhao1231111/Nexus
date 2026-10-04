"""RPC 与演示采集共用的 Isaac Sim 相机启动配置。"""


CAMERA_EXPERIENCE = 'isaaclab.python.rendering.kit'


def configure_camera_app(args):
    """选择完整相机 experience；窗口开关仍由调用入口和 AppLauncher 决定。"""
    # 当前 Isaac Sim 6.0.1 / IsaacLab 3.0 组合的默认 headless.rendering.kit
    # 曾在首次相机更新时停住。experience 决定加载的扩展集合，headless 决定
    # 是否创建窗口；两者独立。无窗口运行也可以加载完整 rendering experience。
    # 保留用户显式传入的 --experience，以便后续复现和诊断上游兼容问题。
    if not args.experience:
        args.experience = CAMERA_EXPERIENCE
    args.enable_cameras = True  # RPC 与录制始终需要相机，不能仅因无窗口而禁用。
    # IsaacLab 3.0 的 SimulationContext.render() 只更新已注册的 visualizer；
    # 没有 KitVisualizer 时，不会调用 app.update()，原生相机也无法产出新帧。
    # 默认保留 Kit 渲染循环，并设置 headless=True。KitVisualizer 会读取
    # SimulationApp 的 headless 配置，从而泵送渲染而不创建 viewport 窗口。
    if args.visualizer is None:
        args.visualizer = ['kit']
        args.visualizer_explicit = True
        args.headless = True
    if args.headless and 'kit' in args.visualizer:
        # 废弃的 --headless CLI 标记在此版本还会禁用所有 visualizer。保留
        # headless=True（关闭窗口），仅清除该禁用意图，避免连相机循环也被关闭。
        args.headless_explicit = False


def resolved_camera_app(launcher):
    """保存实际启动结果，便于区分请求配置与 AppLauncher 最终解析的配置。"""
    # 这些字段来自项目固定使用的 IsaacLab 3.0 AppLauncher；该版本没有公开
    # 的配置 getter。集中在这里读取，避免各入口重复依赖其内部字段。
    return dict(experience=launcher._sim_experience_file,
                headless=launcher._headless, enable_cameras=launcher._enable_cameras,
                visualizers=list(launcher._cli_visualizer_types),
                offscreen_render=launcher._offscreen_render)
