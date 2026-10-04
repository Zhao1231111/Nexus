"""演示离线入口；无模拟器和付费模型调用。"""

import argparse
import json
from pathlib import Path

from .annotations import load_selection, resolve_stage
from .data import Recording, read_json, write_json
from .geometry import backproject, depth_at, surface_normal
from .pipeline import browse, export, prepare, save_keyframe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest='command', required=True)
    for name in ('browse', 'frame', 'prepare', 'export', 'resolve', 'query'):
        command = subcommands.add_parser(name)
        command.add_argument('--recording', type=Path, required=True)
        if name in ('prepare', 'export', 'resolve'):
            command.add_argument('--selection', type=Path, required=True)
        if name != 'query':
            command.add_argument('--output', type=Path, required=True)
        if name == 'browse':
            command.add_argument('--stride', type=int, default=25)
        if name == 'export':
            command.add_argument('--annotations', type=Path, required=True)
            command.add_argument('--samples', type=int, default=8)
        if name == 'resolve':
            command.add_argument('--annotation', type=Path, required=True)
            command.add_argument('--render-output', type=Path,
                                 help='Optional stage preview directory for visual annotation review')
        if name in ('frame', 'query'):
            command.add_argument('--frame', type=int, required=True)
        if name == 'query':
            command.add_argument('--request', type=Path, required=True, help='Geometry request JSON')
    args = parser.parse_args()
    if args.command == 'browse':
        result = browse(args.recording, args.output, args.stride)
    elif args.command == 'frame':
        frame = Recording(args.recording).load(args.frame)
        save_keyframe(frame, args.output)
        result = dict(frame_id=args.frame, sensors=str((args.output / 'sensors.json').resolve()),
                      images=[str((args.output / f'{name}_rgb.png').resolve()) for name in ('cam_head', 'cam_left_wrist', 'cam_right_wrist')])
    elif args.command == 'prepare':
        result = prepare(args.recording, args.selection, args.output)
    elif args.command == 'export':
        result = export(args.recording, args.selection, args.annotations, args.output, args.samples)
    elif args.command == 'resolve':
        recording = Recording(args.recording)
        selection = load_selection(recording, args.selection)
        annotation = read_json(args.annotation)
        stage = next(stage for stage in selection['stages'] if stage['id'] == annotation['stage_id'])
        result = resolve_stage(recording, stage, annotation)
        write_json(args.output, result)
        if args.render_output:
            # agent 可在整包导出之前复核本阶段标注；使用同一渲染入口，避免预览与交付不一致。
            for item, resolved in zip(annotation['keyframes'], result['keyframes']):
                directory = args.render_output / f'{item["frame_id"]:06d}'
                save_keyframe(recording.load(item['frame_id']), directory, item,
                              resolved['evidence']['robot_references'])
    else:
        frame = Recording(args.recording).load(args.frame)
        request = read_json(args.request)
        camera = request['camera']
        values = (frame[f'{camera}_depth_m'], frame[f'{camera}_K'], frame[f'{camera}_T_world_camera'])
        op = request['op']
        if op == 'depth':
            result = depth_at(values[0], request['pixel'])
        elif op == 'backproject':
            result = backproject(*values, request['pixel'])
        elif op == 'normal':
            result = surface_normal(*values, request['polygon'], request.get('toward_world'),
                                    request.get('max_rms_m', 0.003))
        else:
            raise ValueError(f'Unknown geometry tool: {op}')
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
