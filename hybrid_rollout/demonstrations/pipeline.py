"""离线工作区与交付包：人工选择阶段，agent 补充语义标注，脚本重算。"""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .annotations import load_selection, resolve_stage
from .data import CAMERAS, CONVENTIONS, Recording, read_json, write_json
from .render import annotate, overview


def robot_state(frame):
    return dict(states=frame['states'].tolist(), T_world_eef=frame['T_world_eef'].tolist(),
                gripper_joints=frame['gripper_joints'].tolist(),
                gripper_commanded=frame['gripper_commanded'].tolist(),
                timestamp_s=float(frame['timestamp_s']), source_action_index=int(frame['source_action_index']))


def save_keyframe(frame, directory, annotation=None, robot_references=None):
    directory.mkdir(parents=True, exist_ok=True)
    cameras = {}
    for camera in CAMERAS:
        Image.fromarray(frame[f'{camera}_rgb']).save(directory / f'{camera}_rgb.png')
        np.save(directory / f'{camera}_depth_m.npy', frame[f'{camera}_depth_m'])
        camera_data = dict(rgb=f'{camera}_rgb.png', depth=f'{camera}_depth_m.npy',
                           K=frame[f'{camera}_K'].tolist(),
                           T_world_camera=frame[f'{camera}_T_world_camera'].tolist())
        if annotation:
            annotate(frame[f'{camera}_rgb'], annotation['keypoints'], annotation.get('normal_regions', {}),
                     camera, robot_references).save(directory / f'{camera}_annotated.png')
            camera_data['annotated_rgb'] = f'{camera}_annotated.png'
        cameras[camera] = camera_data
    write_json(directory / 'sensors.json', dict(conventions=CONVENTIONS, cameras=cameras,
                                               robot=robot_state(frame)))
    return cameras


def browse(recording_root, output, stride=25):
    """生成带真实观测编号的缩略图页，仅供人工选帧，不自动决定关键帧。"""
    if stride < 1:
        raise ValueError('Preview stride must be positive')
    recording = Recording(recording_root)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    indices = sorted(set(range(0, len(recording.frames), stride)) | {len(recording.frames) - 1})
    pages = []
    for offset in range(0, len(indices), 24):
        group = indices[offset:offset + 24]
        page = Image.new('RGB', (960, 202 * ((len(group) + 2) // 3)), '#20252d')
        draw = ImageDraw.Draw(page)
        for slot, frame_id in enumerate(group):
            frame = recording.load(frame_id)
            x, y = (slot % 3) * 320, (slot // 3) * 202
            page.paste(Image.fromarray(frame['cam_head_rgb']).resize((320, 180)), (x, y))
            draw.text((x + 5, y + 184), f'frame {frame_id} | t={float(frame["timestamp_s"]):.3f}s', fill='white')
        path = output / f'preview_{offset // 24:03d}.png'
        page.save(path)
        pages.append(str(path.resolve()))
    return dict(task=recording.manifest['task'], frame_count=len(recording.frames),
                complete=recording.manifest['complete'], pages=pages,
                note='Preview sampling does not select keyframes or stages')


def prepare(recording_root, selection_path, output):
    recording = Recording(recording_root)
    selection = load_selection(recording, selection_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'selection.json', selection)
    contexts = []
    for stage in selection['stages']:
        directory = output / stage['id']
        frames = []
        for item in stage['keyframes']:
            frame = recording.load(item['frame_id'])
            path = directory / f'{item["frame_id"]:06d}'
            save_keyframe(frame, path)
            frames.append(dict(**item, sensors=str((path / 'sensors.json').resolve()),
                               images={camera: str((path / f'{camera}_rgb.png').resolve()) for camera in CAMERAS}))
        context = dict(stage=stage, keyframes=frames, conventions=CONVENTIONS,
                       annotation_instructions=(
                           '查看本阶段全部关键帧；自主命名稳定的二维语义关键点，并为 active/target '
                           '给出阶段共用的 point/pose 构造定义。每帧保留全部名称；遮挡或无法可靠观察'
                           '的点用 unavailable 和原因表达。几何工具只计算，不替你决定语义。'))
        write_json(directory / 'context.json', context)
        contexts.append(str((directory / 'context.json').resolve()))
    return dict(stage_contexts=contexts, selection=str((output / 'selection.json').resolve()))


def export(recording_root, selection_path, annotations_root, output, samples=8):
    recording = Recording(recording_root)
    if not recording.manifest['complete']:
        raise ValueError('Cannot publish a partial replay as a complete demonstration')
    selection = load_selection(recording, selection_path)
    # 输出前先校验整个阶段集合，避免输入不完整却产生貌似可用的交付包。
    resolved = [(stage, resolve_stage(recording, stage, read_json(Path(annotations_root) / f'{stage["id"]}.json')))
                for stage in selection['stages']]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    package = dict(schema='nexus_astra_demonstration_v1', task=selection['task'],
                   instruction=recording.manifest['instruction'], conventions=CONVENTIONS,
                   source_recording=str(recording.root.resolve()),
                   source=recording.manifest['source'], stages=[])
    context = ['# 演示参考', '', recording.manifest['instruction'], '',
               '所有三维信息位于 environment_origin，长度为米，姿态四元数为 wxyz。',
               '深度为光学 z；EEF 是实测 link6。Unavailable 表示无法可靠计算，不是零位姿。', '']
    context += ['夹爪状态来源：' + recording.manifest['source'].get('gripper_state_provenance', '未声明'), '']
    for stage, result in resolved:
        directory = output / stage['id']
        directory.mkdir()
        write_json(directory / 'annotation_audit.json', result)
        delivered = dict(id=stage['id'], purpose=stage['purpose'], active_object=stage['active_object'],
                         target_object=stage['target_object'], definitions=result['annotation']['definitions'],
                         keypoint_semantics=result['annotation']['keypoint_semantics'],
                         normal_semantics=result['annotation'].get('normal_semantics', {}),
                         observation_notes=result['annotation'].get('observation_notes', []), keyframes=[], overviews=[])
        context += [f'## {stage["id"]}: {stage["purpose"]}', '',
                    f'Active: {stage["active_object"]}; Target: {stage["target_object"]}', '',
                    '功能参考系定义：', json.dumps(delivered['definitions'], ensure_ascii=False), '',
                    '语义名称：', json.dumps(dict(keypoints=delivered['keypoint_semantics'],
                        normals=delivered['normal_semantics']), ensure_ascii=False), '']
        if delivered['observation_notes']:
            context += ['标注观察备注：', *delivered['observation_notes'], '']
        for item, geometry, annotation in zip(stage['keyframes'], result['keyframes'], result['annotation']['keyframes']):
            frame_id = item['frame_id']
            frame = recording.load(frame_id)
            assets = save_keyframe(frame, directory / f'{frame_id:06d}', annotation,
                                  geometry['evidence'].get('robot_references'))
            relative = f'{stage["id"]}/{frame_id:06d}'
            # 构造失败的内部消息可能只显示依赖名称；同时交付观察原因，让模型知道是遮挡还是无效深度。
            missing = {kind: {name: evidence['reason'] for name, evidence in geometry['evidence'][kind].items()
                             if evidence['status'] == 'unavailable'} for kind in ('keypoints', 'normals')}
            missing = {kind: entries for kind, entries in missing.items() if entries}
            payload = dict(**item, assets=relative, cameras=assets, robot=robot_state(frame),
                           geometry=geometry['geometry'], T_target_active=geometry['T_target_active'],
                           T_target_active_members=geometry['T_target_active_members'],
                           unavailable_observations=missing)
            write_json(output / relative / 'keyframe.json', payload)
            delivered['keyframes'].append(payload)
            if len(delivered['keyframes']) > 1:
                start = delivered['keyframes'][-2]['frame_id']
                filename = f'overview_{start:06d}_{frame_id:06d}.png'
                interval = overview(recording, start, frame_id, directory / filename, samples)
                interval['path'] = f'{stage["id"]}/{filename}'
                delivered['overviews'].append(interval)
                # 在上下文中按“前一关键帧 → 区间运动 → 当前关键帧”的顺序展示。
                context += [f'![Frames {start} to {frame_id}]({interval["path"]})', '']
            context += [f'### Frame {frame_id}', '', item['description'], '']
            for camera in CAMERAS:
                context += [f'![{camera} frame {frame_id}]({relative}/{camera}_annotated.png)', '']
            context += ['```json', json.dumps(dict(robot=payload['robot'], geometry=payload['geometry'],
                         T_target_active=payload['T_target_active'],
                         T_target_active_members=payload['T_target_active_members'],
                         unavailable_observations=payload['unavailable_observations']), ensure_ascii=False), '```', '']
        package['stages'].append(delivered)
    write_json(output / 'demonstration.json', package)
    (output / 'astra_context.md').write_text('\n'.join(context), encoding='utf-8')
    return dict(manifest=str((output / 'demonstration.json').resolve()),
                context=str((output / 'astra_context.md').resolve()), stages=len(package['stages']))
