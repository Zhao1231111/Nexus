"""当前外部 RoboDojo 兼容分支所需的局部适配，不改写其任务判定规则。"""


def install_geometry_device_compat():
    """修正静态几何世界 pose 与环境原点在不同设备上时的相对坐标计算。"""
    import torch
    from env.scene_manager.objects.geometry import GeometryObject

    if getattr(GeometryObject, '_nexus_relative_pose_device_compat', False):
        return  # 同一进程重复初始化入口时，不叠加 wrapper。
    native_get_state = GeometryObject.get_state

    def get_state(self, is_relative=False):
        # 上游从 USD 读取静态几何 pose 后在 CPU 上建立张量，而 env_origin
        # 可能来自 CUDA scene。先用原生方法获取世界状态，只适配相同的减原点
        # 运算；不改变旋转、对象位置来源、奖励条件或场景状态。
        state = native_get_state(self, is_relative=False)
        if is_relative and hasattr(self, 'env_origin'):
            root_pose = state['root_pose'].clone()
            origin = torch.as_tensor(self.env_origin, dtype=root_pose.dtype,
                                     device=root_pose.device).reshape(-1)
            if origin.numel() != 3:
                raise ValueError('Geometry environment origin must have three coordinates')
            root_pose[:3] -= origin
            state = dict(state, root_pose=root_pose)
        return state

    GeometryObject.get_state = get_state
    GeometryObject._nexus_relative_pose_device_compat = True
