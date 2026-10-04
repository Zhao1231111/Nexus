"""关键帧标注与相邻关键帧的 head camera 运动概览。"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .geometry import project


def annotate(rgb, keypoints, regions, camera, robot_references=None):
    """在原分辨率图像标注名称，保留二维标注与像素查询的对应关系。"""
    image = Image.fromarray(rgb).copy()
    draw = ImageDraw.Draw(image)
    font_path = Path(__file__).resolve().parents[1] / 'assets/fonts/NotoSansCJKsc-Regular.otf'
    font = ImageFont.truetype(str(font_path), 14)

    def label(anchor, name, color, offset):
        # 显示偏移只调整文字，不改变用于深度查询的像素；靠近边缘时将名称移回图内。
        # offset 可保存到二维标注中，便于复核图上密集名称的重叠问题。
        box = draw.textbbox((0, 0), name, font=font, stroke_width=1)
        x = min(max(anchor[0] + offset[0], -box[0]), max(-box[0], image.width - box[2]))
        y = min(max(anchor[1] + offset[1], -box[1]), max(-box[1], image.height - box[3]))
        # 文字移开后用引线保持与原像素的对应，避免远处名称看起来属于旁边的对象。
        endpoint = (min(max(anchor[0], x + box[0]), x + box[2]),
                    min(max(anchor[1], y + box[1]), y + box[3]))
        if np.linalg.norm(np.asarray(endpoint) - anchor) > 12:
            draw.line([tuple(anchor), endpoint], fill='black', width=3)
            draw.line([tuple(anchor), endpoint], fill=color, width=1)
        draw.text((x, y), name, font=font, fill=color, stroke_width=1, stroke_fill='black')

    for name, item in keypoints.items():
        if item['status'] != 'observed' or item['camera'] != camera:
            continue
        u, v = item['pixel']
        draw.ellipse((u - 4, v - 4, u + 4, v + 4), fill='#ffcf40', outline='black')
        label((u, v), name, '#ffcf40', item.get('label_offset', [6, 2]))
    for name, item in regions.items():
        if item['status'] == 'observed' and item['camera'] == camera:
            polygon = [tuple(point) for point in item['polygon']]
            draw.line(polygon + polygon[:1], fill='#40e0ff', width=2)
            label(polygon[0], name, '#40e0ff', item.get('label_offset', [0, 0]))
    for name, reference in (robot_references or {}).items():
        projection = reference['projections'][camera]
        if projection['status'] != 'projected':
            continue
        u, v = projection['pixel']
        # 空心十字表示本体状态投影，黄色圆点表示图像语义标注，防止两种来源混淆。
        draw.line([(u - 5, v), (u + 5, v)], fill='#66ddff', width=2)
        draw.line([(u, v - 5), (u, v + 5)], fill='#66ddff', width=2)
        caption = ('L6' if name.startswith('left') else 'R6') + ' (proprio)'
        label((u, v), caption, '#66ddff', [7, 0])
    return image


def align_head(source, reference):
    """使用 RGB-D 将区间帧重投影到首帧光学相机，近点覆盖远点。"""
    rgb = source['cam_head_rgb']
    reference_rgb = reference['cam_head_rgb']
    if (rgb.shape == reference_rgb.shape
            and np.allclose(source['cam_head_K'], reference['cam_head_K'], atol=1e-8)
            and np.allclose(source['cam_head_T_world_camera'], reference['cam_head_T_world_camera'], atol=1e-8)):
        return rgb.astype(float), np.ones(rgb.shape[:2], dtype=bool)
    depth = source['cam_head_depth_m']
    v, u = np.where(np.isfinite(depth) & (depth > 0))
    rays = np.linalg.solve(source['cam_head_K'], np.stack([u, v, np.ones(len(u))]))
    local = rays * (depth[v, u] / rays[2])
    world = source['cam_head_T_world_camera'] @ np.vstack([local, np.ones(len(u))])
    target = np.linalg.inv(reference['cam_head_T_world_camera']) @ world
    forward = target[2] > 1e-6
    projected = reference['cam_head_K'] @ target[:3, forward]
    pixels = np.rint(projected[:2] / projected[2]).astype(int)
    height, width = reference_rgb.shape[:2]
    inside = (pixels[0] >= 0) & (pixels[0] < width) & (pixels[1] >= 0) & (pixels[1] < height)
    pixels = pixels[:, inside]
    colors = rgb[v[forward][inside], u[forward][inside]]
    distances = target[2, forward][inside]
    # 对每个目标像素选择最近点，避免多次写入的顺序影响对齐结果。
    linear = pixels[1] * width + pixels[0]
    order = np.lexsort((distances, linear))
    _, first = np.unique(linear[order], return_index=True)
    chosen = order[first]
    aligned = np.zeros_like(reference_rgb, dtype=float)
    mask = np.zeros((height, width), dtype=bool)
    aligned[pixels[1, chosen], pixels[0, chosen]] = colors[chosen]
    mask[pixels[1, chosen], pixels[0, chosen]] = True
    return aligned, mask


def overview(recording, start, end, output, samples=8):
    if start >= end or samples < 2:
        raise ValueError('Overview requires increasing keyframes and >=2 samples')
    indices = np.unique(np.rint(np.linspace(start, end, min(samples, end - start + 1))).astype(int))
    reference = recording.load(start)
    canvas = reference['cam_head_rgb'].astype(float)
    # 时间色带固定为从浅到深；背景保留 RGB，移动物体以时间色调叠加。
    for index, frame_id in enumerate(indices):
        frame = recording.load(int(frame_id))
        aligned, mask = align_head(frame, reference)
        fraction = index / max(1, len(indices) - 1)
        tint = np.array([210 - 130 * fraction, 225 - 150 * fraction, 255 - 110 * fraction])
        overlay = aligned * 0.8 + tint * 0.2
        canvas[mask] = canvas[mask] * 0.65 + overlay[mask] * 0.35
    image = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    width, height = image.size
    previous = [None, None]
    for frame_id in range(start, end + 1):
        frame = recording.load(frame_id)
        pixels, front = project(frame['T_world_eef'][:, :3, 3], reference['cam_head_K'],
                                reference['cam_head_T_world_camera'])
        fraction = (frame_id - start) / (end - start)
        for arm, base_color in enumerate(((255, 70, 40), (30, 160, 255))):
            point = pixels[arm]
            visible = front[arm] and 0 <= point[0] < width and 0 <= point[1] < height
            if not visible:
                previous[arm] = None  # 不跨不可见区间连线，避免虚构轨迹。
                continue
            current = tuple(point)
            color = tuple(int(240 * (1 - fraction) + component * fraction) for component in base_color)
            if previous[arm] is not None:
                draw.line([previous[arm], current], fill=color, width=3)
            previous[arm] = current
            if frame_id in (start, end):
                draw.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), fill=color)
                label = f'{"L" if arm == 0 else "R"}:{frame_id}'
                draw.text((point[0] + 5, point[1]), label, fill=color, stroke_width=1, stroke_fill='black')
    draw.rectangle((0, 0, width, 22), fill='black')
    draw.text((5, 5), f'Head RGB-D aligned | frames {start} -> {end} | L red, R blue | light -> dark', fill='white')
    image.save(output)
    return dict(start_frame=start, end_frame=end, sampled_frames=indices.tolist(),
                alignment='RGB-D reprojection to start-frame optical camera',
                eef='measured link6 projections; image visibility is not an occlusion test',
                path=str(output))
