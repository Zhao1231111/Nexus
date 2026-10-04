"""下载单条官方深度演示，或将其传感器字段导入 Nexus（不是重录）。"""

import argparse
import io
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

import numpy as np
from PIL import Image

from .data import CAMERAS, Recorder, pose_matrix, read_json, write_json
from .robodojo import digest, load_official_trajectory


REPOSITORY = 'RoboDojo-Benchmark/RoboDojo'


def download(output, task='organize_table', episode=0, revision='main'):
    """先解析不可变 revision，再按 LFS SHA256 校验单个文件，避免下载整个数据集。"""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with urlopen(f'https://huggingface.co/api/datasets/{REPOSITORY}/revision/{quote(revision, safe="")}', timeout=60) as response:
        revision = json.load(response)['sha']
    relative = f'data/RoboDojo_depth/{task}/arx_x5/data/episode_{episode:07d}.hdf5'
    url = f'https://huggingface.co/api/datasets/{REPOSITORY}/tree/{revision}/{relative.rsplit("/", 1)[0]}'
    entry = None
    while url:
        with urlopen(url, timeout=60) as response:
            entries = json.load(response)
            link = response.headers.get('Link', '')
        entry = next((item for item in entries if item['path'] == relative), None)
        if entry:
            break
        # 官方 tree API 使用 Link 分页，较大 episode 编号不能只查第一页。
        next_links = [part for part in link.split(',') if 'rel="next"' in part]
        url = next_links[0].split('<', 1)[1].split('>', 1)[0] if next_links else None
    if entry is None or 'lfs' not in entry:
        raise ValueError(f'Official depth episode not found: {relative}')
    temporary = output / f'episode_{episode:07d}.hdf5.partial'
    resolved = f'https://huggingface.co/datasets/{REPOSITORY}/resolve/{revision}/{relative}'
    with urlopen(resolved, timeout=120) as response, temporary.open('wb') as stream:
        while block := response.read(8 * 1024 * 1024):
            stream.write(block)
    if temporary.stat().st_size != entry['lfs']['size'] or digest(temporary) != entry['lfs']['oid']:
        raise ValueError('Downloaded file differs from official LFS size/SHA256')
    target = temporary.with_suffix('')
    temporary.rename(target)
    source = dict(repo_id=REPOSITORY, revision=revision, repository_path=relative,
                  sha256=entry['lfs']['oid'], size_bytes=entry['lfs']['size'], resolved_url=resolved)
    write_json(output / 'source_official_hf.json', source)
    return dict(file=str(target.resolve()), **source)


def import_episode(path, output, extrinsics_axes, task='organize_table'):
    """保持官方 state[t]/action[t] 的对齐；缺失的物理读数明确为空。"""
    import h5py
    trajectory = load_official_trajectory(path)
    count = len(trajectory['actions'])
    if extrinsics_axes not in ('usd', 'optical'):
        raise ValueError('Explicit extrinsics_axes must be usd or optical')
    # 原文件没有完整记录轴约定；必须由调用方确认。usd 适用于 standard pinhole
    # 相机 prim 与安装节点重合的 RoboDojo 归档，不支持未经标定的其他相机偏移。
    conversion = np.diag([1., -1., -1., 1.]) if extrinsics_axes == 'usd' else np.eye(4)
    with h5py.File(path, 'r') as file:
        for name in CAMERAS:
            camera = file[f'vision/{name}']
            if camera['approximate_depths'].shape[0] != count or len(camera['colors']) != count:
                raise ValueError('Official RGB-D and robot observations have different lengths')
            if camera['approximate_depths'].dtype != np.uint16:
                raise ValueError('Expected official uint16 millimetre depth')
            if camera['extrinsic_matrix'].shape != (count, 4, 4):
                raise ValueError('Expected per-frame official camera-to-environment extrinsics')
        trajectory_hash = digest(path)
        source = dict(kind='official_rgbd_import', trajectory=str(Path(path).resolve()),
                      trajectory_sha256=trajectory_hash, source_actions=count,
                      source_frequency_hz=trajectory['frequency'], extrinsics_input_axes=extrinsics_axes,
                      depth_quantization_m=.001, depth_invalid_values=[0, 65535],
                      gripper_state_provenance='official reported normalized state; physical finger measurement is not supplied',
                      alignment='state[t] observation; action[t] next-frame target; final action is padding',
                      synchronization_evidence='upstream export metadata; not independently proven by import',
                      upstream_rgb_alignment=str(file.attrs.get('fix_obs_shift_rgb', 'not supplied')),
                      calibration_assumption='recorded camera-to-environment; standard pinhole has zero local optical prim offset',
                      raw_finger_joints_available=False)
        receipt_path = Path(path).parent / 'source_official_hf.json'
        if receipt_path.exists():
            receipt = read_json(receipt_path)
            if receipt['sha256'] != trajectory_hash:
                raise ValueError('Official source receipt SHA256 differs from this episode')
            source['official_release'] = receipt
        recorder = Recorder(output, task, trajectory['instruction'], source)
        np.savez_compressed(recorder.root / 'source_trajectory.npz', **trajectory['arrays'])
        try:
            for index in range(count):
                poses = []
                states = []
                for arm in ('left', 'right'):
                    pose = trajectory['arrays'][f'state/{arm}_ee_poses'][index]
                    poses.append(pose_matrix(pose[:3], pose[3:]))
                    states.extend(trajectory['arrays'][f'state/{arm}_arm_joint_states'][index])
                    states.extend(trajectory['arrays'][f'state/{arm}_ee_joint_states'][index])
                frame = dict(frame_id=index, timestamp_s=index / trajectory['frequency'], source_action_index=index - 1,
                             action=trajectory['actions'][index - 1] if index else np.array([]),
                             states=np.asarray(states), T_world_eef=np.asarray(poses),
                             gripper_joints=np.array([]), gripper_commanded=np.array([]))
                for name in CAMERAS:
                    camera = file[f'vision/{name}']
                    encoded = np.asarray(camera['colors'][index]).tobytes()
                    with Image.open(io.BytesIO(encoded)) as image:
                        frame[f'{name}_rgb'] = np.asarray(image.convert('RGB')).copy()
                    raw_depth = camera['approximate_depths'][index]
                    depth = raw_depth.astype(np.float32) * .001
                    depth[(raw_depth == 0) | (raw_depth == 65535)] = np.nan
                    frame[f'{name}_depth_m'] = depth
                    frame[f'{name}_K'] = camera['intrinsic_matrix'][()]
                    frame[f'{name}_T_world_camera'] = camera['extrinsic_matrix'][index] @ conversion
                recorder.append(frame)
                if (index + 1) % 100 == 0:
                    print(f'DEMO_IMPORT_PROGRESS frames={index + 1}/{count}', flush=True)
            recorder.finish(complete=True, status='official_import', recorded_observations=count)
        except BaseException as error:
            recorder.finish(complete=False, status='error', error=f'{type(error).__name__}: {error}')
            raise
    return dict(recording=str(recorder.root.resolve()), observations=count, re_recorded=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    fetch = commands.add_parser('download')
    fetch.add_argument('--task', default='organize_table')
    fetch.add_argument('--episode', type=int, default=0)
    fetch.add_argument('--revision', default='main')
    fetch.add_argument('--output', type=Path, required=True)
    convert = commands.add_parser('import')
    convert.add_argument('--task', default='organize_table')
    convert.add_argument('--file', type=Path, required=True)
    convert.add_argument('--output', type=Path, required=True)
    convert.add_argument('--extrinsics-axes', required=True, choices=('usd', 'optical'))
    args = parser.parse_args()
    result = (download(args.output, args.task, args.episode, args.revision) if args.command == 'download'
              else import_episode(args.file, args.output, args.extrinsics_axes, args.task))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
