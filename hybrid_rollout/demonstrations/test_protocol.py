"""协议的离线回归：相机几何、观测缺失、人工边界与交付内容。"""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from .annotations import load_selection, resolve_stage
from .data import CAMERAS, Recorder, Recording, pose_matrix, write_json
from .geometry import GeometryError, backproject, construct, depth_at, project, surface_normal
from .pipeline import export, prepare
from .render import align_head
from .robodojo import capture_frame, load_official_trajectory
from .official import import_episode


def make_frame(index):
    matrix = np.eye(4)
    # 两臂位于相机前方，轨迹投影的可见性不依赖默认世界方向。
    poses = np.stack([np.eye(4), np.eye(4)])
    poses[:, 2, 3] = 1
    poses[:, 0, 3] = [index * 0.01, -index * 0.01]
    frame = dict(frame_id=index, timestamp_s=index * 0.04, source_action_index=index - 1,
                 action=np.zeros(14) if index else np.array([]), states=np.zeros(14),
                 T_world_eef=poses, gripper_joints=np.zeros((2, 2)), gripper_commanded=np.ones(2))
    for camera in CAMERAS:
        frame.update({f'{camera}_rgb': np.full((48, 64, 3), 120, np.uint8),
                      f'{camera}_depth_m': np.ones((48, 64), np.float32),
                      f'{camera}_K': np.array([[40., 0, 32], [0, 40, 24], [0, 0, 1]]),
                      f'{camera}_T_world_camera': matrix.copy()})
    return frame


@pytest.fixture
def recording(tmp_path):
    root = tmp_path / 'recording'
    recorder = Recorder(root, 'organize_table', 'Place objects on the table.', {'fixture': True})
    for index in range(3):
        recorder.append(make_frame(index))
    recorder.finish(complete=True)
    return Recording(root)


def selection():
    return dict(task='organize_table', stages=[dict(id='carry', purpose='搬运物体', active_object='cup',
        target_object='tray', keyframes=[dict(frame_id=0, description='搬运开始'), dict(frame_id=2, description='搬运结束')])])


def annotation():
    # 同阶段所有帧共用的语义和构造，只允许观测像素随帧变化。
    return dict(stage_id='carry', reviewed_frame_ids=[0, 2],
        keypoint_semantics=dict(cup_center='杯口中心', tray_center='托盘表面中心'),
        definitions={role: dict(kind='point', description='使用对应表面中心点',
                                origin=dict(op='point', name=name))
                     for role, name in [('active', 'cup_center'), ('target', 'tray_center')]},
        keyframes=[dict(frame_id=index, keypoints={name: dict(status='observed', camera='cam_head', pixel=pixel)
                                                  for name, pixel in [('cup_center', [32, 24]), ('tray_center', [40, 24])]})
                   for index in (0, 2)])


def test_backproject_and_projection_in_rotated_translated_frame():
    K = np.array([[100., 0, 20], [0, 200, 10], [0, 0, 1]])
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler('z', 90, degrees=True).as_matrix()
    T[:3, 3] = [1, 2, 3]
    result = backproject(np.full((40, 60), 2.), K, T, [30, 30])
    # 光学点 [0.2,0.2,2] 经旋转和平移，应为 [0.8,2.2,5]。
    assert np.allclose(result['point_world_m'], [0.8, 2.2, 5])
    pixels, valid = project([result['point_world_m'], [1, 2, 2]], K, T)
    assert valid.tolist() == [True, False]
    assert np.allclose(pixels[0], [30, 30])
    assert np.isnan(pixels[1]).all()


@pytest.mark.parametrize('invalid', [0., -1., np.inf, np.nan])
def test_invalid_depth_is_never_guessed(invalid):
    depth = np.ones((3, 3))
    depth[1, 1] = invalid
    with pytest.raises(GeometryError, match='valid metric depth'):
        depth_at(depth, [1, 1])


def test_plane_normal_sign_and_nonplanarity():
    K = np.array([[40., 0, 16], [0, 40, 16], [0, 0, 1]])
    depth = np.ones((32, 32))
    polygon = [[5, 5], [26, 5], [26, 26], [5, 26]]
    result = surface_normal(depth, K, np.eye(4), polygon, toward=[0, 0, 1])
    assert np.allclose(result['normal_world'], [0, 0, 1])
    assert result['rms_m'] < 1e-8
    depth[:, 16:] += .3
    with pytest.raises(GeometryError, match='not planar'):
        surface_normal(depth, K, np.eye(4), polygon)


def test_construct_right_handed_pose_and_reject_parallel_axes():
    definition = dict(kind='pose', origin=dict(op='point', name='tip'), primary_axis='z', secondary_axis='x',
                      axes=dict(z=dict(op='vector', value=[0, 0, 1]), x=dict(op='vector', value=[1, 0, 1])))
    result = construct(definition, {'tip': [1, 2, 3]}, {})
    assert np.allclose(np.asarray(result['T_world'])[:3, :3], np.eye(3))
    assert np.allclose(result['quaternion_wxyz'], [1, 0, 0, 0])
    definition['axes']['x']['value'] = [0, 0, 2]
    with pytest.raises(GeometryError, match='parallel'):
        construct(definition, {'tip': [1, 2, 3]}, {})


def test_missing_evidence_is_local_to_dependent_object(recording):
    data = annotation()
    data['keyframes'][1]['keypoints']['cup_center'] = dict(status='unavailable', reason='夹爪遮挡')
    result = resolve_stage(recording, selection()['stages'][0], data)
    last = result['keyframes'][1]
    assert last['geometry']['active']['status'] == 'unavailable'
    assert last['geometry']['target']['status'] == 'available'
    assert last['T_target_active'] is None
    assert result['annotation']['keyframes'][1]['keypoints']['cup_center']['reason'] == '夹爪遮挡'


def test_agent_cannot_skip_stage_keyframes(recording):
    data = annotation()
    data['reviewed_frame_ids'] = [0]
    with pytest.raises(ValueError, match='every selected'):
        resolve_stage(recording, selection()['stages'][0], data)


def test_keypoint_names_must_persist_even_when_occluded(recording):
    data = annotation()
    del data['keyframes'][1]['keypoints']['cup_center']
    with pytest.raises(ValueError, match='retain all semantic'):
        resolve_stage(recording, selection()['stages'][0], data)


def test_manual_keyframe_order_is_required(recording, tmp_path):
    data = selection()
    data['stages'][0]['keyframes'].reverse()
    path = tmp_path / 'selection.json'
    write_json(path, data)
    with pytest.raises(ValueError, match='recorded order'):
        load_selection(recording, path)


def test_recorder_rejects_truth_fields_and_inconsistent_rgbd(tmp_path):
    recorder = Recorder(tmp_path / 'new', 'organize_table', 'task', {})
    frame = make_frame(0)
    frame['object_pose'] = np.eye(4)
    with pytest.raises(ValueError, match='whitelist'):
        recorder.append(frame)
    del frame['object_pose']
    frame['cam_head_depth_m'] = np.ones((1, 1), np.float32)
    with pytest.raises(ValueError, match='pixel aligned'):
        recorder.append(frame)


def test_export_builds_ordered_context_and_preserves_audit(recording, tmp_path):
    selected = tmp_path / 'selection.json'
    annotations = tmp_path / 'annotations'
    annotations.mkdir()
    write_json(selected, selection())
    write_json(annotations / 'carry.json', annotation())
    workspace = prepare(recording.root, selected, tmp_path / 'workspace')
    context = json.loads(open(workspace['stage_contexts'][0]).read())
    assert [item['frame_id'] for item in context['keyframes']] == [0, 2]
    result = export(recording.root, selected, annotations, tmp_path / 'package')
    package = json.loads(open(result['manifest']).read())
    stage = package['stages'][0]
    assert [item['frame_id'] for item in stage['keyframes']] == [0, 2]
    assert len(stage['overviews']) == 1
    assert (tmp_path / 'package' / stage['overviews'][0]['path']).exists()
    assert np.load(tmp_path / 'package/carry/000000/cam_head_depth_m.npy').shape == (48, 64)
    audit = json.loads((tmp_path / 'package/carry/annotation_audit.json').read_text())
    assert audit['annotation'] == annotation()
    # 交付包含关键帧状态，不把完整原始动作轨迹塞入模型上下文。
    text = open(result['context']).read()
    assert 'source_trajectory' not in text and 'action/' not in text
    assert text.index('Frame 0') < text.index('![Frames 0 to 2]') < text.index('Frame 2')


def test_partial_replay_cannot_be_exported(recording, tmp_path):
    recording.manifest['complete'] = False
    write_json(recording.root / 'recording.json', recording.manifest)
    with pytest.raises(ValueError, match='partial replay'):
        export(recording.root, tmp_path / 'unused', tmp_path, tmp_path / 'package')


def test_rgbd_alignment_uses_optical_camera_translation():
    source, reference = make_frame(0), make_frame(1)
    source['cam_head_rgb'][:] = 0
    source['cam_head_rgb'][24, 32] = [255, 0, 0]
    source['cam_head_T_world_camera'][0, 3] = 0.1
    aligned, valid = align_head(source, reference)
    # fx=40,z=1，相机向右平移 0.1m 后，该世界点在参考图像向右移动 4px。
    assert valid[24, 36] and aligned[24, 36, 0] == 255


def test_capture_measures_fingers_and_uses_sensor_optical_frame():
    robots = [SimpleNamespace(arm_name=f'{arm}_arm', type='target', gripper_scale=[0., .04],
                              gripper_move={'sign': 1}) for arm in ('left', 'right')]
    camera = SimpleNamespace(get_world_pose=lambda camera_axes: (np.array([1., 2., 3.]), np.array([1., 0, 0, 0])),
                             get_intrinsics_matrix=lambda device: np.eye(3))
    manager = SimpleNamespace(robot_list=robots, scene=SimpleNamespace(env_origins=np.array([[.5, 0, 0]])),
        get_joint=lambda robot, env_idx_list: {0: np.zeros(6)},
        get_end_effector_real_val=lambda robot, env_idx_list: {0: np.array([.01, .01])},
        get_real_endpose=lambda robot, env_idx_list, is_relative: {0: np.array([0., 0, 1, 1, 0, 0, 0])})
    raw = dict(state={f'{arm}_ee_joint_state': [1.] for arm in ('left', 'right')},
               vision={name: dict(color=np.zeros((4, 4, 3), np.uint8), depth=np.ones((4, 4))) for name in CAMERAS})
    env = SimpleNamespace(sim=SimpleNamespace(_sim_step_counter=30), dt=.004, robot_manager=manager,
        camera_manager=SimpleNamespace(camera_names=[list(CAMERAS)], cameras=[[camera] * 3]), get_obs=lambda: raw)
    result = capture_frame(env, 0, [], 30)
    assert result['states'][6] == .25 and result['gripper_commanded'][0] == 1
    assert np.allclose(result['cam_head_T_world_camera'], [[1, 0, 0, .5], [0, -1, 0, 2], [0, 0, -1, 3], [0, 0, 0, 1]])


def test_official_loader_preserves_original_robot_trajectory(tmp_path):
    h5py = pytest.importorskip('h5py')
    path = tmp_path / 'official.hdf5'
    with h5py.File(path, 'w') as file:
        for group in ('action', 'state'):
            for arm in ('left', 'right'):
                for field, dimension in (('arm_joint_states', 6), ('ee_joint_states', 1), ('ee_poses', 7)):
                    file[f'{group}/{arm}_{field}'] = np.zeros((4, dimension))
        file['additional_info/frequency'] = 25
        file['instruction'] = 'organize table'
        file['privileged/object_pose'] = np.eye(4)
    result = load_official_trajectory(path)
    assert result['actions'].shape == (4, 14)
    assert result['frequency'] == 25
    assert set(key.split('/')[0] for key in result['arrays']) == {'action', 'state'}


def test_official_rgbd_import_preserves_alignment_and_missing_measurements(tmp_path):
    h5py = pytest.importorskip('h5py')
    path = tmp_path / 'depth.hdf5'
    import io
    from PIL import Image
    buffer = io.BytesIO()
    Image.fromarray(np.zeros((48, 64, 3), np.uint8)).save(buffer, format='JPEG')
    with h5py.File(path, 'w') as file:
        for group in ('action', 'state'):
            for arm in ('left', 'right'):
                for field, dimension in (('arm_joint_states', 6), ('ee_joint_states', 1), ('ee_poses', 7)):
                    values = np.zeros((3, dimension))
                    if field == 'ee_poses':
                        values[:, 3] = 1
                    file[f'{group}/{arm}_{field}'] = values
        file['action/left_arm_joint_states'][0, 0] = .2
        file['additional_info/frequency'] = 25
        file['instruction'] = 'organize_table'
        for camera in CAMERAS:
            file[f'vision/{camera}/colors'] = np.array([buffer.getvalue()] * 3)
            depths = np.full((3, 48, 64), 1250, dtype=np.uint16)
            depths[:, 0, 0], depths[:, 0, 1] = 0, 65535
            file[f'vision/{camera}/approximate_depths'] = depths
            file[f'vision/{camera}/intrinsic_matrix'] = make_frame(0)[f'{camera}_K']
            file[f'vision/{camera}/extrinsic_matrix'] = np.stack([np.eye(4)] * 3)
    result = import_episode(path, tmp_path / 'imported', 'usd')
    recording = Recording(result['recording'])
    frame = recording.load(1)
    assert frame['source_action_index'] == 0 and frame['action'][0] == .2
    assert frame['cam_head_depth_m'][2, 2] == 1.25
    assert np.isnan(frame['cam_head_depth_m'][0, :2]).all()
    assert frame['gripper_joints'].size == frame['gripper_commanded'].size == 0
    assert np.allclose(frame['cam_head_T_world_camera'], np.diag([1, -1, -1, 1]))
    assert recording.manifest['source']['kind'] == 'official_rgbd_import'
    assert not result['re_recorded']


def test_undeclared_semantic_reference_is_definition_error(recording):
    data = annotation()
    data['definitions']['active']['origin']['name'] = 'not_declared'
    with pytest.raises(ValueError, match='undeclared'):
        resolve_stage(recording, selection()['stages'][0], data)


def robot_definition(arm):
    """测试只引用已归档的 link6；不将指尖或背景深度作为机器人位姿。"""
    return dict(kind='pose', description=f'{arm} link6 的归档本体位姿',
                source=dict(kind='robot_eef', arm=arm))


def test_robot_reference_uses_proprioception_not_image_depth(recording):
    data = annotation()
    data['definitions'] = {role: robot_definition('left') for role in ('active', 'target')}
    frame = recording.load(2)
    # 图像上该投影像素的深度故意与 link6 的真实 z 不同；不能反投影背景替代机器人状态。
    frame['cam_head_depth_m'][:] = .25
    np.savez_compressed(recording.root / 'frames/000002.npz', **frame)
    last = resolve_stage(recording, selection()['stages'][0], data)['keyframes'][-1]
    assert np.allclose(last['geometry']['active']['T_world'], frame['T_world_eef'][0])
    assert np.allclose(last['T_target_active'], np.eye(4))
    reference = last['evidence']['robot_references']['left_link6_origin']
    assert reference['source'] == 'recorded_robot_proprioception'
    assert reference['occlusion_tested'] is False
    assert reference['projections']['cam_head']['status'] == 'projected'
    assert np.allclose(reference['projections']['cam_head']['pixel'], [32.8, 24])


def test_independent_grippers_keep_member_transforms_and_local_missing_evidence(recording):
    data = annotation()
    data['definitions']['active'] = dict(kind='collection', description='左右夹爪独立运动', members={
        'left': robot_definition('left'), 'right': robot_definition('right'),
        'hidden': dict(kind='point', description='被遮挡的辅助点', origin=dict(op='point', name='cup_center'))})
    data['definitions']['target'] = robot_definition('left')
    data['keyframes'][-1]['keypoints']['cup_center'] = dict(status='unavailable', reason='遮挡')
    last = resolve_stage(recording, selection()['stages'][0], data)['keyframes'][-1]
    members = last['geometry']['active']['members']
    assert members['left']['status'] == members['right']['status'] == 'available'
    assert members['hidden']['status'] == 'unavailable'
    assert last['T_target_active'] is None  # 集合不存在可替代两个夹爪的单一刚体变换。
    assert np.allclose(last['T_target_active_members']['left'], np.eye(4))
    assert np.allclose(np.asarray(last['T_target_active_members']['right'])[:3, 3], [-.04, 0, 0])
    assert set(last['evidence']['robot_references']) == {'left_link6_origin', 'right_link6_origin'}


def test_export_retains_independent_robot_references(recording, tmp_path):
    data = annotation()
    data['definitions']['active'] = dict(kind='collection', description='双臂 link6', members={
        'left': robot_definition('left'), 'right': robot_definition('right')})
    data['definitions']['target'] = robot_definition('left')
    selected, annotations = tmp_path / 'selection.json', tmp_path / 'annotations'
    write_json(selected, selection())
    write_json(annotations / 'carry.json', data)
    result = export(recording.root, selected, annotations, tmp_path / 'package')
    package = json.loads(open(result['manifest']).read())
    last = package['stages'][0]['keyframes'][-1]
    assert set(last['T_target_active_members']) == {'left', 'right'}
    audit = json.loads((tmp_path / 'package/carry/annotation_audit.json').read_text())
    assert audit['annotation'] == data
    assert audit['keyframes'][-1]['evidence']['robot_references']['right_link6_origin']['occlusion_tested'] is False
    assert 'T_target_active_members' in open(result['context']).read()


def test_resolve_cli_renders_stage_before_full_export(recording, tmp_path, monkeypatch, capsys):
    """agent 的阶段复核入口应真实生成与最终交付同分辨率的标注图。"""
    from PIL import Image
    from .cli import main
    selected, annotated = tmp_path / 'selection.json', tmp_path / 'carry.json'
    write_json(selected, selection())
    write_json(annotated, annotation())
    monkeypatch.setattr('sys.argv', ['demonstrations', 'resolve', '--recording', str(recording.root),
        '--selection', str(selected), '--annotation', str(annotated), '--output', str(tmp_path / 'resolved.json'),
        '--render-output', str(tmp_path / 'review')])
    main()
    result = json.loads(capsys.readouterr().out)
    assert [item['frame_id'] for item in result['keyframes']] == [0, 2]
    for frame_id in (0, 2):
        for camera in CAMERAS:
            image = Image.open(tmp_path / 'review' / f'{frame_id:06d}' / f'{camera}_annotated.png')
            assert image.size == (64, 48)
        image = Image.open(tmp_path / 'review' / f'{frame_id:06d}' / 'cam_head_annotated.png')
        assert image.getpixel((32, 24)) == (255, 207, 64)  # 黄色标记确实落在原分辨率的指定像素。


def test_export_exposes_missing_observation_reason_to_astra(recording, tmp_path):
    """不能把 agent 的遮挡原因只留在审计文件，让模型仅收到内部 KeyError 名称。"""
    data = annotation()
    data['keyframes'][-1]['keypoints']['cup_center'] = dict(status='unavailable', reason='中心被夹爪遮挡')
    data['observation_notes'] = ['当前参考是可见表面点，不是物体质量中心。']
    selected, annotations = tmp_path / 'selection.json', tmp_path / 'annotations'
    write_json(selected, selection())
    write_json(annotations / 'carry.json', data)
    result = export(recording.root, selected, annotations, tmp_path / 'package')
    package = json.loads(open(result['manifest']).read())
    stage = package['stages'][0]
    assert stage['observation_notes'] == data['observation_notes']
    assert stage['keyframes'][-1]['unavailable_observations'] == {'keypoints': {'cup_center': '中心被夹爪遮挡'}}
    context = open(result['context']).read()
    assert '中心被夹爪遮挡' in context and data['observation_notes'][0] in context


def test_recording_launcher_rejects_silent_native_failure(tmp_path):
    """Kit 可能吞掉异常退出码；不能仅凭零退出码认定采集成功。"""
    import os
    from pathlib import Path
    import subprocess
    environment = tmp_path / 'env'
    (environment / 'bin').mkdir(parents=True)
    executable = environment / 'bin/python'
    executable.write_text('#!/bin/sh\nif [ "$1" = "-c" ]; then exit 0; fi\necho "simulator failure"\nexit 0\n')
    executable.chmod(0o755)
    script = Path(__file__).with_name('record_robodojo.sh')
    variables = dict(os.environ, ROBODOJO_SOURCE=str(tmp_path), ROBODOJO_ENV=str(environment),
                     DEMO_LOG=str(tmp_path / 'failure.log'))
    result = subprocess.run(['bash', str(script)], env=variables, capture_output=True, text=True, timeout=15)
    assert result.returncode == 1
    assert 'status=0' in result.stderr


@pytest.mark.parametrize('pixel', [[-0.1, 1], [2.1, 1], [1, -1]])
def test_outside_pixels_do_not_round_into_valid_observations(pixel):
    with pytest.raises(GeometryError, match='outside'):
        depth_at(np.ones((3, 3)), pixel)


def test_recording_launcher_cleans_owned_children_on_interrupt(tmp_path):
    """针对实际冒烟中的残留进程回归：停止入口必须同时停止其子进程。"""
    import os
    from pathlib import Path
    import signal
    import subprocess
    import sys
    import time
    environment = tmp_path / 'env'
    (environment / 'bin').mkdir(parents=True)
    executable = environment / 'bin/python'
    pid_file = tmp_path / 'child.pid'
    executable.write_text(f'#!{sys.executable}\n'
        'import pathlib,subprocess,sys,time,os\n'
        'if sys.argv[1] == "-c": sys.exit(0)\n'
        'child=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"])\n'
        'pathlib.Path(os.environ["TEST_DEMO_CHILD_PID"]).write_text(str(child.pid))\n'
        'while True: time.sleep(1)\n')
    executable.chmod(0o755)
    variables = dict(os.environ, ROBODOJO_SOURCE=str(tmp_path), ROBODOJO_ENV=str(environment),
                     DEMO_LOG=str(tmp_path / 'interrupt.log'), TEST_DEMO_CHILD_PID=str(pid_file))
    script = Path(__file__).with_name('record_robodojo.sh')
    process = subprocess.Popen(['bash', str(script)], env=variables, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 5
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(.05)
        assert pid_file.exists(), 'Fake simulator did not create its child'
        child_pid = int(pid_file.read_text())
        process.send_signal(signal.SIGTERM)
        process.communicate(timeout=15)
        assert process.returncode == 143
        # 被 init 接管的已退出 zombie 可能短暂保留 /proc，但不再是活动进程。
        status = Path(f'/proc/{child_pid}/stat')
        assert not status.exists() or status.read_text().split()[2] == 'Z'
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            process.communicate(timeout=15)
