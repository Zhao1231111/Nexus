"""演示数据契约：只保存传感器、机器人状态和动作，不保存物体真值。"""

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


CAMERAS = ('cam_head', 'cam_left_wrist', 'cam_right_wrist')
SCHEMA = 'nexus_demonstration_v1'
CONVENTIONS = dict(
    frame='environment_origin', length_unit='m', joint_unit='rad',
    quaternion_order='wxyz', transform='T_X_Y maps coordinates in Y to X',
    camera_axes='optical: +x right, +y down, +z forward',
    depth='aligned optical z in metres; nonfinite/nonpositive values are invalid',
    pixel='[u,v], original RGB resolution, integer coordinates index pixel centres',
    eef='measured link6; not TCP or gripper functional frame',
    state_order='left 6 joints, left opening, right 6 joints, right opening',
    gripper='0 closed, 1 open; not clipped; measurement provenance is recorded in source',
    synchronization='one observation after rendering, without intervening physics steps',
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    """原子替换索引；中断时不会把半截 JSON 当成完整演示。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def finite(value, shape, name):
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f'{name}: expected finite array of shape {shape}')
    return array


def rigid(value):
    """校验真实 SE(3)，避免把视图矩阵或含缩放的安装矩阵用于几何计算。"""
    matrix = finite(value, (4, 4), 'transform')
    rotation = matrix[:3, :3]
    if (not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5)):
        raise ValueError('Expected a rigid homogeneous transform')
    return matrix


def intrinsic(value):
    matrix = finite(value, (3, 3), 'intrinsic')
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0 or not np.allclose(matrix[2], [0, 0, 1]):
        raise ValueError('Invalid pinhole intrinsic matrix')
    return matrix


def pose_matrix(position, quaternion_wxyz):
    position = finite(position, (3,), 'position')
    quaternion = finite(quaternion_wxyz, (4,), 'quaternion_wxyz')
    if not np.isclose(np.linalg.norm(quaternion), 1, atol=1e-3):
        raise ValueError('Expected a unit wxyz quaternion')
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
    matrix[:3, 3] = position
    return matrix


def validate_frame(frame):
    """逐帧白名单限制；未经声明的物体 pose／分割数据不会混入归档。"""
    expected = {'frame_id', 'timestamp_s', 'source_action_index', 'action', 'states',
                'T_world_eef', 'gripper_joints', 'gripper_commanded'}
    expected |= {f'{camera}_{field}' for camera in CAMERAS
                 for field in ('rgb', 'depth_m', 'K', 'T_world_camera')}
    if set(frame) != expected:
        raise ValueError(f'Frame fields differ from sensor whitelist: {set(frame) ^ expected}')
    finite(frame['states'], (14,), 'states')
    # 官方归档可能只有归一化开合字段；未知的实测指关节／历史指令使用空数组，
    # 不由状态值反推或伪造。新录制必须由采集适配器提供真实读数。
    for field, shape in (('gripper_joints', (2, 2)), ('gripper_commanded', (2,))):
        value = np.asarray(frame[field])
        if value.shape != (0,):
            finite(value, shape, field)
    poses = finite(frame['T_world_eef'], (2, 4, 4), 'T_world_eef')
    for matrix in poses:
        rigid(matrix)
    # 第 0 帧尚未执行动作，使用空数组而不是伪造零动作。
    action = np.asarray(frame['action'])
    if action.shape not in ((0,), (14,)) or not np.isfinite(action).all():
        raise ValueError('Expected empty initial action or finite 14D applied action')
    for camera in CAMERAS:
        rgb, depth = np.asarray(frame[f'{camera}_rgb']), np.asarray(frame[f'{camera}_depth_m'])
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError(f'{camera}: expected uint8 RGB')
        if depth.shape != rgb.shape[:2] or depth.dtype.kind != 'f':
            raise ValueError(f'{camera}: RGB/depth must be pixel aligned, with floating metric depth')
        intrinsic(frame[f'{camera}_K'])
        rigid(frame[f'{camera}_T_world_camera'])


class Recording:
    """一个原始演示；每个 NPZ 对应一个同步观测时刻。"""

    def __init__(self, root):
        self.root = Path(root)
        self.manifest = read_json(self.root / 'recording.json')
        if self.manifest['schema'] != SCHEMA or self.manifest['conventions'] != CONVENTIONS:
            raise ValueError('Unsupported demonstration schema or conventions')
        self.frames = self.manifest['frames']

    def load(self, frame_id):
        if isinstance(frame_id, bool) or not isinstance(frame_id, int) or not 0 <= frame_id < len(self.frames):
            raise ValueError(f'Frame index out of range: {frame_id}')
        with np.load(self.root / self.frames[frame_id]['path'], allow_pickle=False) as data:
            frame = {key: data[key].copy() for key in data.files}
        validate_frame(frame)
        if int(frame['frame_id']) != frame_id:
            raise ValueError('Frame ID differs from recording index')
        return frame


class Recorder:
    def __init__(self, root, task, instruction, source):
        self.root = Path(root)
        # 每次录制必须使用新目录，避免将两次重放的帧拼成同一条轨迹。
        self.root.mkdir(parents=True, exist_ok=False)
        (self.root / 'frames').mkdir()
        self.manifest = dict(schema=SCHEMA, task=task, instruction=instruction,
                             conventions=CONVENTIONS, source=source, complete=False, frames=[])
        write_json(self.root / 'recording.json', self.manifest)

    def append(self, frame):
        validate_frame(frame)
        index = len(self.manifest['frames'])
        if int(frame['frame_id']) != index:
            raise ValueError('Frames must be consecutive from zero')
        timestamp = float(frame['timestamp_s'])
        if not np.isfinite(timestamp) or (index and timestamp <= self.manifest['frames'][-1]['timestamp_s']):
            raise ValueError('Observation timestamps must increase')
        source_index = int(frame['source_action_index'])
        if source_index != index - 1 or np.asarray(frame['action']).size != (14 if index else 0):
            raise ValueError('Frame zero precedes actions; frame i records source action i-1')
        relative = f'frames/{index:06d}.npz'
        np.savez_compressed(self.root / relative, **frame)
        self.manifest['frames'].append(dict(frame_id=index, timestamp_s=timestamp,
                                            source_action_index=source_index, path=relative))
        write_json(self.root / 'recording.json', self.manifest)

    def finish(self, **outcome):
        self.manifest.update(outcome)
        write_json(self.root / 'recording.json', self.manifest)
