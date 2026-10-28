"""为候选轨迹 CSV 添加候选终点、专家轨迹和专家终点，保留其他列。"""
import argparse
import csv
import json
import math
import os
import tempfile
from pathlib import Path

DEFAULT_ROOT = Path('/mnt/sfs_turbo/gaoxuan_gac/WoTE/exp/eval/WoTE/collision_analysis_190')
CANDIDATE_ENDPOINT = 'trajectory_endpoint_xy_heading'
EXPERT_TRAJECTORY = 'expert_trajectory_xy_heading'
EXPERT_ENDPOINT = 'expert_endpoint_xy_heading'


def format_point(point):
    """把轨迹点格式化成保留两位小数的 [x, y, heading]。"""
    values = [float(value) for value in point]
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError('轨迹点必须包含三个有限数值')
    return '[' + ', '.join(f'{value:.2f}' for value in values) + ']'


def prepare_rows(path, poses=None, fields='both'):
    """准备新列，保留其他字段值和原有顺序；重复运行不会增加重复列。"""
    with path.open('r', encoding='utf-8', newline='') as source:
        reader = csv.DictReader(source)
        columns = reader.fieldnames
        if not columns or len(columns) != len(set(columns)):
            raise ValueError(f'{path}: 缺少表头或存在重复列名')
        if fields in ('both', 'candidate') and 'trajectory_xy_heading_json' not in columns:
            raise ValueError(f'{path}: 缺少轨迹列')
        rows = list(reader)
    if len(rows) != 256:
        raise ValueError(f'{path}: 预期 256 条轨迹，实际 {len(rows)} 条')
    added = []
    if fields in ('both', 'candidate'):
        added.append(CANDIDATE_ENDPOINT)
        for row in rows:
            trajectory = json.loads(row['trajectory_xy_heading_json'])
            if len(trajectory) != 8 or len(trajectory[-1]) != 3:
                raise ValueError(f'{path}: 轨迹 {row.get("index")} 的终点格式错误')
            row[CANDIDATE_ENDPOINT] = format_point(trajectory[-1])
    if fields in ('both', 'expert'):
        if poses is None or poses.shape != (8, 3):
            raise ValueError(f'{path}: 专家轨迹形状应为 (8, 3)')
        points = [format_point(point) for point in poses]
        trajectory = '[' + ', '.join(points) + ']'
        added.extend((EXPERT_TRAJECTORY, EXPERT_ENDPOINT))
        for row in rows:
            row[EXPERT_TRAJECTORY] = trajectory
            row[EXPERT_ENDPOINT] = points[-1]
    return columns + [name for name in added if name not in columns], rows


def atomic_write(path, columns, rows):
    """同目录创建临时文件再替换，保留原文件权限。"""
    mode = path.stat().st_mode & 0o777
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', newline='', dir=path.parent, delete=False) as target:
            temp_path = Path(target.name)
            writer = csv.DictWriter(target, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        os.chmod(temp_path, mode)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT, help='包含场景文件夹的目录')
    parser.add_argument('--tokens', nargs='+', help='只处理指定的场景 token')
    parser.add_argument('--expected-scenes', type=int, default=190, help='预期场景数，默认 190')
    parser.add_argument('--fields', choices=('both', 'candidate', 'expert'), default='both',
                        help='选择更新全部列、候选终点或专家轨迹列，默认全部')
    args = parser.parse_args()
    paths = sorted(args.root.glob('*/candidates.csv'))
    if args.tokens:
        wanted = set(args.tokens)
        paths = [path for path in paths if path.parent.name in wanted]
        missing = wanted - {path.parent.name for path in paths}
        if missing:
            raise ValueError(f'未找到场景 CSV: {sorted(missing)}')
    if len(paths) != args.expected_scenes:
        raise ValueError(f'预期 {args.expected_scenes} 个 CSV，实际找到 {len(paths)} 个')
    trajectories = {}
    if args.fields in ('both', 'expert'):
        # 只加载自车位姿，不读取相机图片或激光雷达。
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        from navsim.common.dataloader import SceneLoader
        from navsim.common.dataclasses import SensorConfig
        data = Path(os.environ['OPENSCENE_DATA_ROOT'])
        repo = Path(os.environ['NAVSIM_DEVKIT_ROOT'])
        config = OmegaConf.load(repo / 'navsim/planning/script/config/common/scene_filter/navtest.yaml')
        config.tokens = [path.parent.name for path in paths]
        config.max_scenes = None
        loader = SceneLoader(data / 'navsim_logs/test', None, instantiate(config), SensorConfig.build_no_sensors())
        missing = set(config.tokens) - set(loader.tokens)
        if missing:
            raise ValueError(f'数据集中未找到场景: {sorted(missing)}')
        for path in paths:
            trajectories[path.parent.name] = loader.get_scene_from_token(path.parent.name).get_future_trajectory(num_trajectory_frames=8).poses
    # 全部场景先校验，再开始写入；避免输入格式错误造成部分更新。
    for path in paths:
        prepare_rows(path, trajectories.get(path.parent.name), args.fields)
    for index, path in enumerate(paths, 1):
        columns, rows = prepare_rows(path, trajectories.get(path.parent.name), args.fields)
        atomic_write(path, columns, rows)
        print(f'[{index}/{len(paths)}] {path.parent.name}', flush=True)
    print(f'完成：{len(paths)} 个场景，每个场景 256 条候选轨迹。', flush=True)


if __name__ == '__main__':
    main()
