"""回归 organize_table 暴露的 CPU 静态 pose / CUDA 环境原点组合。"""

import sys
from types import ModuleType

import pytest

from .robodojo_server.native_compat import install_geometry_device_compat


def test_geometry_relative_pose_keeps_world_state_and_handles_cuda_origin(monkeypatch):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('Cross-device regression requires CUDA; no Isaac Sim app is needed')

    class GeometryObject:
        def get_state(self, is_relative=False):
            # 模拟上游真实组合：USD pose 在 CPU，scene 的 env_origin 在 CUDA。
            pose = torch.tensor([2., 4., 6., 1., 0., 0., 0.])
            if is_relative:
                pose[:3] -= self.env_origin
            return dict(root_pose=pose, extra='native_state_field')

    geometry = GeometryObject()
    geometry.env_origin = torch.tensor([1., 2., 3.], device='cuda')
    with pytest.raises(RuntimeError, match='same device'):
        geometry.get_state(is_relative=True)  # 先证明触发的是设备不匹配这个实际缺陷。
    module = ModuleType('env.scene_manager.objects.geometry')
    module.GeometryObject = GeometryObject
    monkeypatch.setitem(sys.modules, module.__name__, module)
    install_geometry_device_compat()
    wrapper = GeometryObject.get_state
    install_geometry_device_compat()
    assert GeometryObject.get_state is wrapper  # 重复安装不能不断叠加转换。

    world = geometry.get_state(is_relative=False)
    relative = geometry.get_state(is_relative=True)
    torch.testing.assert_close(world['root_pose'], torch.tensor([2., 4., 6., 1., 0., 0., 0.]))
    torch.testing.assert_close(relative['root_pose'], torch.tensor([1., 2., 3., 1., 0., 0., 0.]))
    assert relative['extra'] == world['extra'] == 'native_state_field'
    assert geometry.env_origin.device.type == 'cuda'  # 不迁移或修改原始 scene 状态。
