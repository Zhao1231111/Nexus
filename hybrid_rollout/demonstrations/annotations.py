"""人工阶段选择与 agent 标注契约；重算只依赖已保存的传感器数据。"""

from .data import CAMERAS, Recording, read_json
from .geometry import GeometryError, backproject, construct, evaluate, project, surface_normal


def text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must contain descriptive text')
    return value


def leaf_definitions(definition):
    """检查独立主体集合，并遍历其 point／pose 定义。"""
    text(definition['description'], 'construction description')
    if definition['kind'] == 'collection':
        if not definition['members']:
            raise ValueError('Geometry collection requires named members')
        for name, member in definition['members'].items():
            text(name, 'geometry member name')
            yield from leaf_definitions(member)
    else:
        yield definition


def load_selection(recording, path):
    selection = read_json(path)
    if selection['task'] != recording.manifest['task'] or not selection['stages']:
        raise ValueError('Selection must contain stages for the recorded task')
    ids = set()
    previous_end = -1
    for stage in selection['stages']:
        identity = text(stage['id'], 'stage id')
        if identity in ids or not identity.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Stage IDs must be unique file-safe names')
        ids.add(identity)
        for field in ('purpose', 'active_object', 'target_object'):
            text(stage[field], field)
        indices = [item['frame_id'] for item in stage['keyframes']]
        if not indices or any(isinstance(i, bool) or not isinstance(i, int) for i in indices):
            raise ValueError('A stage requires manually selected integer frame IDs')
        if indices != sorted(set(indices)) or indices[0] < previous_end:
            raise ValueError('Keyframes/stages must follow recorded order; boundaries may be shared')
        for keyframe in stage['keyframes']:
            recording.load(keyframe['frame_id'])  # 校验实际帧存在，防止视频编号与观测编号混用。
            text(keyframe['description'], 'manual keyframe description')
        previous_end = indices[-1]
    return selection


def resolve_stage(recording, stage, annotation):
    """一次读取整个阶段；所有关键帧沿用同一语义及构造定义。"""
    indices = [item['frame_id'] for item in stage['keyframes']]
    if annotation['stage_id'] != stage['id'] or annotation['reviewed_frame_ids'] != indices:
        raise ValueError('Agent must review every selected keyframe in this stage')
    semantics = annotation['keypoint_semantics']
    normal_semantics = annotation.get('normal_semantics', {})
    for name, meaning in {**semantics, **normal_semantics}.items():
        text(name, 'semantic name')
        text(meaning, 'semantic meaning')
    definitions = annotation['definitions']
    if set(definitions) != {'active', 'target'}:
        raise ValueError('Stage requires shared active and target definitions')
    leaves = [leaf for definition in definitions.values() for leaf in leaf_definitions(definition)]
    for definition in leaves:
        if definition.get('source'):
            source = definition['source']
            if (definition['kind'] != 'pose' or source.get('kind') != 'robot_eef'
                    or source.get('arm') not in ('left', 'right')):
                raise ValueError('Robot reference must be a named left/right link6 pose')
            continue
        expressions = [definition['origin'], *definition.get('axes', {}).values()]
        for expression in expressions:
            try:
                # 用占位向量检查构造树及语义引用。占位点退化不代表真实观察退化，
                # 但引用未声明的关键点是定义错误，不能被误报成该帧的遮挡。
                evaluate(expression, {name: [1., 2., 3.] for name in semantics},
                         {name: [0., 0., 1.] for name in normal_semantics})
            except GeometryError:
                pass
            except KeyError as error:
                raise ValueError(f'Invalid or undeclared geometry definition reference: {error}') from error
        if definition['kind'] == 'pose':
            # 明确记录每条轴来自观测还是人为约定，不把补全轴伪装成实测朝向。
            if set(definition['axis_sources']) != set(definition['axes']):
                raise ValueError('Every supplied pose axis requires a documented source')
            for source in definition['axis_sources'].values():
                text(source, 'axis source')
    annotated = annotation['keyframes']
    if [item['frame_id'] for item in annotated] != indices:
        raise ValueError('Annotations must match the complete ordered stage keyframes')
    results = []
    for item in annotated:
        frame = recording.load(item['frame_id'])
        if set(item['keypoints']) != set(semantics) or set(item.get('normal_regions', {})) != set(normal_semantics):
            raise ValueError('Each frame must retain all semantic names, including unavailable observations')
        points, normals, evidence = {}, {}, {'keypoints': {}, 'normals': {}}
        for kind, entries in (('keypoints', item['keypoints']), ('normals', item.get('normal_regions', {}))):
            for name, observation in entries.items():
                status = observation['status']
                if status == 'unavailable':
                    evidence[kind][name] = dict(status=status, reason=text(observation['reason'], 'unavailable reason'))
                    continue
                if status != 'observed':
                    raise ValueError('Observation status must be observed or unavailable')
                camera = observation['camera']
                prefix = f'{camera}_'
                if prefix + 'depth_m' not in frame:
                    raise ValueError(f'Unknown camera: {camera}')
                args = (frame[prefix + 'depth_m'], frame[prefix + 'K'], frame[prefix + 'T_world_camera'])
                try:
                    if kind == 'keypoints':
                        result = backproject(*args, observation['pixel'])
                        points[name] = result['point_world_m']
                    else:
                        result = surface_normal(*args, observation['polygon'],
                                                toward=observation.get('toward_world'),
                                                max_rms_m=observation.get('max_rms_m', 0.003))
                        normals[name] = result['normal_world']
                    evidence[kind][name] = dict(status='available', camera=camera, **result)
                except GeometryError as error:
                    evidence[kind][name] = dict(status='unavailable', camera=camera, reason=str(error))
        geometry = {}
        for role, definition in definitions.items():
            try:
                geometry[role] = dict(status='available', **construct(definition, points, normals, frame['T_world_eef']))
            except (GeometryError, KeyError) as error:
                # 缺少依赖只使相关对象不可用，不阻止其他可观察对象的几何结果。
                geometry[role] = dict(status='unavailable', reason=f'Missing/invalid geometric evidence: {error}')
        relative = None
        active, target = geometry['active'], geometry['target']
        if active['status'] == target['status'] == 'available' and active['kind'] == target['kind'] == 'pose':
            import numpy as np
            relative = (np.linalg.inv(target['T_world']) @ active['T_world']).tolist()
        # 双夹爪必须分别变换到 target；集合没有一个可替代成员的平均刚体位姿。
        member_transforms = {}
        if active.get('kind') == 'collection' and target.get('kind') == 'pose' and target['status'] == 'available':
            import numpy as np
            inverse = np.linalg.inv(target['T_world'])
            for name, member in active['members'].items():
                if member['status'] == 'available' and member['kind'] == 'pose':
                    member_transforms[name] = (inverse @ member['T_world']).tolist()
        robot_references = {}
        for definition in leaves:
            if definition.get('source', {}).get('kind') != 'robot_eef':
                continue
            arm = definition['source']['arm']
            point = frame['T_world_eef'][('left', 'right').index(arm), :3, 3]
            projections = {}
            for camera in CAMERAS:
                pixels, forward = project([point], frame[f'{camera}_K'], frame[f'{camera}_T_world_camera'])
                height, width = frame[f'{camera}_rgb'].shape[:2]
                visible = bool(forward[0] and 0 <= pixels[0, 0] < width and 0 <= pixels[0, 1] < height)
                projections[camera] = dict(status='projected' if visible else 'outside_image',
                                           pixel=pixels[0].tolist() if visible else None)
            # 二维投影与图像上观察到的语义点分开保存，不能将背景深度当作 link6 深度。
            robot_references[f'{arm}_link6_origin'] = dict(source='recorded_robot_proprioception',
                point_world_m=point.tolist(), projections=projections, occlusion_tested=False)
        evidence['robot_references'] = robot_references
        results.append(dict(frame_id=item['frame_id'], evidence=evidence, geometry=geometry,
                            T_target_active=relative, T_target_active_members=member_transforms))
    return dict(stage_id=stage['id'], annotation=annotation, keyframes=results)
