"""可被 agent 独立调用的几何工具；不选择物体语义或猜测不可见部位。"""

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from .data import finite, intrinsic, rigid


class GeometryError(ValueError):
    """观测无效或构造退化；调用方应保存 unavailable，而不是填入猜测 pose。"""


def unit(vector):
    vector = finite(vector, (3,), 'direction')
    norm = np.linalg.norm(vector)
    if norm < 1e-8:
        raise GeometryError('Direction is zero or construction is degenerate')
    return vector / norm


def depth_at(depth, pixel):
    """精确读取指定像素，不隐式跨物体边缘取邻域深度。"""
    depth = np.asarray(depth)
    pixel = finite(pixel, (2,), 'pixel')
    if depth.ndim != 2 or not (0 <= pixel[0] <= depth.shape[1] - 1 and 0 <= pixel[1] <= depth.shape[0] - 1):
        raise GeometryError('Pixel outside original image')
    u, v = np.rint(pixel).astype(int)
    value = float(depth[v, u])
    if not np.isfinite(value) or value <= 0:
        raise GeometryError('Pixel has no valid metric depth')
    return dict(pixel=[int(u), int(v)], depth_m=value)


def backproject(depth, K, T_world_camera, pixel):
    sample = depth_at(depth, pixel)
    ray = np.linalg.solve(intrinsic(K), [*sample['pixel'], 1.0])
    # 深度是 optical z，不是沿射线的欧氏距离。
    camera_point = ray * (sample['depth_m'] / ray[2])
    point = rigid(T_world_camera) @ np.r_[camera_point, 1.0]
    return dict(**sample, point_world_m=point[:3].tolist())


def project(points_world, K, T_world_camera):
    """返回像素与前向可见标记；后方点不能通过齐次除法伪装成可见点。"""
    points = np.asarray(points_world, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('Expected finite Nx3 world points')
    camera = (np.linalg.inv(rigid(T_world_camera)) @ np.c_[points, np.ones(len(points))].T).T[:, :3]
    valid = camera[:, 2] > 1e-6
    pixels = np.full((len(points), 2), np.nan)
    mapped = (intrinsic(K) @ camera[valid].T).T
    pixels[valid] = mapped[:, :2] / mapped[:, 2:3]
    return pixels, valid


def surface_normal(depth, K, T_world_camera, polygon, toward=None, max_rms_m=0.003, min_points=12):
    """对 agent 指定的图像多边形拟合平面，并报告拟合质量。"""
    depth = np.asarray(depth)
    polygon = np.asarray(polygon, dtype=float)
    if (polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3
            or not np.isfinite(polygon).all() or not np.isfinite(max_rms_m) or max_rms_m <= 0):
        raise ValueError('Expected polygon and positive plane RMS threshold')
    height, width = depth.shape
    if np.any(polygon < 0) or np.any(polygon[:, 0] >= width) or np.any(polygon[:, 1] >= height):
        raise GeometryError('Normal region outside original image')
    mask_image = Image.new('1', (width, height))
    ImageDraw.Draw(mask_image).polygon([tuple(p) for p in polygon], fill=1)
    mask = np.asarray(mask_image) & np.isfinite(depth) & (depth > 0)
    v, u = np.where(mask)
    if len(u) < max(3, min_points):
        raise GeometryError('Too few valid depth pixels in normal region')
    rays = np.linalg.solve(intrinsic(K), np.stack([u, v, np.ones(len(u))]))
    points_camera = (rays * (depth[v, u] / rays[2])).T
    matrix = rigid(T_world_camera)
    points = points_camera @ matrix[:3, :3].T + matrix[:3, 3]
    center = points.mean(axis=0)
    # 用 3x3 协方差求法向，避免对大区域执行大型 SVD。
    centered = points - center
    eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered / len(points))
    if eigenvalues[1] < 1e-10:
        raise GeometryError('Normal region is geometrically degenerate')
    rms = float(np.sqrt(max(0, eigenvalues[0])))
    if rms > max_rms_m:
        raise GeometryError(f'Region is not planar enough: RMS={rms:.6f}m')
    normal = eigenvectors[:, 0]
    # 默认朝向观察相机；agent 可以明确指定世界坐标方向消除符号歧义。
    reference = matrix[:3, 3] - center if toward is None else unit(toward)
    if abs(normal @ unit(reference)) < 1e-6:
        raise GeometryError('Normal sign is ambiguous for the supplied reference')
    if normal @ reference < 0:
        normal = -normal
    return dict(normal_world=normal.tolist(), centroid_world_m=center.tolist(),
                rms_m=rms, valid_pixels=int(len(points)),
                sign_reference='toward_camera' if toward is None else list(toward))


def evaluate(expression, points, normals):
    """组合式向量表达：具体语义与构造树由 agent 给出，不执行任意 Python。"""
    op = expression['op']
    if op == 'point':
        return np.asarray(points[expression['name']], dtype=float)
    if op == 'normal':
        return np.asarray(normals[expression['name']], dtype=float)
    if op == 'vector':
        return finite(expression['value'], (3,), 'explicit world vector')
    if op == 'mean':
        values = [evaluate(item, points, normals) for item in expression['items']]
        if not values:
            raise ValueError('mean requires at least one item')
        return np.mean(values, axis=0)
    if op == 'normalize':
        return unit(evaluate(expression['value'], points, normals))
    if op == 'scale':
        factor = float(expression['factor'])
        if not np.isfinite(factor):
            raise ValueError('scale factor must be finite')
        return factor * evaluate(expression['value'], points, normals)
    if op in ('add', 'subtract', 'cross'):
        left = evaluate(expression['left'], points, normals)
        right = evaluate(expression['right'], points, normals)
        return {'add': lambda: left + right, 'subtract': lambda: left - right,
                'cross': lambda: np.cross(left, right)}[op]()
    raise ValueError(f'Unknown geometry expression: {op}')


def construct(definition, points, normals, robot_poses=None):
    # 双臂是两个可独立运动的主体，不能通过平均位置／姿态伪装成一个刚体。
    # collection 的每个成员仍是协议中的 point 或 pose，并独立保留缺失原因。
    if definition['kind'] == 'collection':
        members = {}
        for name, member in definition['members'].items():
            try:
                members[name] = dict(status='available', **construct(member, points, normals, robot_poses))
            except (GeometryError, KeyError) as error:
                members[name] = dict(status='unavailable', reason=str(error))
        return dict(kind='collection', members=members)
    source = definition.get('source')
    if source:
        # 已记录的机器人本体状态比从遮挡图像重新估计安装 link 更直接。
        # 此入口只允许实际归档的左右 link6，不推断 TCP、指尖或夹持物体位姿。
        if definition['kind'] != 'pose' or source.get('kind') != 'robot_eef' or source.get('arm') not in ('left', 'right'):
            raise ValueError('Robot geometry source must identify a left/right link6 pose')
        if robot_poses is None:
            raise GeometryError('Recorded robot EEF pose is unavailable')
        index = ('left', 'right').index(source['arm'])
        matrix = rigid(robot_poses[index])
        quaternion = Rotation.from_matrix(matrix[:3, :3]).as_quat()[[3, 0, 1, 2]]
        return dict(kind='pose', T_world=matrix.tolist(), position_world_m=matrix[:3, 3].tolist(),
                    quaternion_wxyz=quaternion.tolist(), source='recorded_robot_eef',
                    arm=source['arm'], control_frame='link6')
    origin = finite(evaluate(definition['origin'], points, normals), (3,), 'origin')
    if definition['kind'] == 'point':
        return dict(kind='point', point_world_m=origin.tolist())
    if definition['kind'] != 'pose':
        raise ValueError('Geometry kind must be point or pose')
    axes = definition['axes']
    primary, secondary = definition['primary_axis'], definition['secondary_axis']
    if primary == secondary or set(axes) != {primary, secondary} or not set(axes) <= {'x', 'y', 'z'}:
        raise ValueError('Pose requires two explicitly ordered, distinct axes')
    first = unit(evaluate(axes[primary], points, normals))
    second = unit(evaluate(axes[secondary], points, normals))
    # 保留主轴，正交化副轴；平行或近乎平行时不输出确定姿态。
    orthogonal = second - first * (first @ second)
    if np.linalg.norm(orthogonal) < 1e-4:
        raise GeometryError('Pose axes are nearly parallel')
    values = {primary: first, secondary: unit(orthogonal)}
    missing = ({'x', 'y', 'z'} - set(values)).pop()
    cyclic = {'x': ('y', 'z'), 'y': ('z', 'x'), 'z': ('x', 'y')}
    left, right = cyclic[missing]
    values[missing] = np.cross(values[left], values[right])
    matrix = np.eye(4)
    matrix[:3, :3] = np.column_stack([values[axis] for axis in 'xyz'])
    matrix[:3, 3] = origin
    quaternion_xyzw = Rotation.from_matrix(rigid(matrix)[:3, :3]).as_quat()
    return dict(kind='pose', T_world=matrix.tolist(), position_world_m=origin.tolist(),
                quaternion_wxyz=quaternion_xyzw[[3, 0, 1, 2]].tolist())
