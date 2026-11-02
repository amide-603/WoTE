"""按场景 token 和候选轨迹排名运行 NAVSIM 评分，并为低分轨迹生成 BEV 动画。"""
import argparse
import json
import lzma
import pickle
import re
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

from navsim.common.dataloader import MetricCacheLoader, SceneLoader
from navsim.common.dataclasses import SensorConfig, Trajectory
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import EgoAreaIndex
from navsim.visualization.analyze_wote_collisions import (
    CACHE, CONFIG, DATA, ROOT, collision_events, render_gif,
)
from navsim.visualization.score_top_candidates import METRICS, write_csv
from navsim.visualization.visualize_wote_scene import DEFAULT_RESULTS, load_results
from navsim.visualization.plots import plot_bev_frame

ROOT_OUTPUT = Path('/mnt/sfs_turbo/gaoxuan_gac/WoTE/exp/eval/WoTE/selected_candidates')


def plot_selected_bev(scene, human, trajectories, ranks, indices, output):
    """在同一张地图上展示专家轨迹和本次选择的所有候选轨迹。"""
    fig, ax = plot_bev_frame(scene, scene.scene_metadata.num_history_frames - 1)
    xy = np.vstack([np.zeros((1, 2)), human[:, :2]])
    ax.plot(xy[:, 1], xy[:, 0], color='green', linewidth=2.5, label='Expert', zorder=10)
    colors = plt.get_cmap('tab20')
    for position, (rank, index) in enumerate(zip(ranks, indices)):
        xy = np.vstack([np.zeros((1, 2)), trajectories[index, :, :2]])
        ax.plot(xy[:, 1], xy[:, 0], color=colors(position % 20), linewidth=1.8,
                label=f'Rank {rank} (index {index})', zorder=9)
    ax.legend(loc='upper right', fontsize='small')
    ax.set_title('Selected candidate trajectories')
    fig.savefig(output, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--token', required=True, help='场景 token')
    parser.add_argument('--ranks', required=True, nargs='+', type=int, help='模型原始综合奖励排名（1～256）')
    parser.add_argument('--output', type=Path, help='可选：自定义结果目录；默认按场景 token 创建')
    parser.add_argument('--results-dir', type=Path, default=DEFAULT_RESULTS, help='保存场景 NPZ 的目录')
    parser.add_argument('--cache-dir', type=Path, default=CACHE, help='NAVSIM metric cache 目录')
    parser.add_argument('--config', type=Path, default=CONFIG, help='原评测的 Hydra 配置')
    args = parser.parse_args()
    if not re.fullmatch(r'[0-9a-fA-F]{16}', args.token):
        parser.error('token 应为 16 位十六进制字符串')
    if len(set(args.ranks)) != len(args.ranks) or any(r < 1 or r > 256 for r in args.ranks):
        parser.error('轨迹排名必须互不重复，范围为 1～256')
    if args.output is None:
        args.output = ROOT_OUTPUT / args.token

    results = load_results(args.results_dir / f'{args.token}.npz')
    trajectories = np.asarray(results['all_trajectory'])
    if trajectories.shape == (1, 256, 8, 3):
        trajectories = trajectories[0]
    rewards = np.asarray(results['final_rewards']).reshape(-1)
    if rewards.shape != (256,) or not np.isfinite(rewards).all():
        raise ValueError('模型综合奖励必须包含 256 个有限数值')
    # 与原 candidates.csv 的排序方式保持一致；排名从 1 开始，编号从 0 开始。
    order = np.argsort(-rewards)
    indices = [int(order[rank - 1]) for rank in args.ranks]
    if trajectories.shape != (256, 8, 3) or not np.isfinite(trajectories[indices]).all():
        raise ValueError(f'候选轨迹格式错误或包含非有限值: {trajectories.shape}')
    caches = MetricCacheLoader(args.cache_dir)
    if args.token not in caches.metric_cache_paths:
        raise ValueError(f'找不到场景 {args.token} 的 metric cache')
    with lzma.open(caches.metric_cache_paths[args.token], 'rb') as file:
        cache = pickle.load(file)
    cfg = OmegaConf.load(args.config)
    simulator, scorer = instantiate(cfg.simulator), instantiate(cfg.scorer)
    args.output.mkdir(parents=True, exist_ok=True)
    fields = ['token', 'rank', 'candidate_index', 'model_final_reward', 'valid', 'error', *METRICS,
              'first_at_fault_collision_s', 'collision_objects_json', 'first_offroad_s', 'gif', 'gif_error']
    rows = []
    filt = OmegaConf.load(ROOT / 'navsim/planning/script/config/common/scene_filter/navtest.yaml')
    filt.tokens, filt.max_scenes = [args.token], None
    loader = SceneLoader(DATA / 'navsim_logs/test', DATA / 'sensor_blobs/test', instantiate(filt), SensorConfig.build_no_sensors())
    scene = loader.get_scene_from_token(args.token)
    human = scene.get_future_trajectory(num_trajectory_frames=8).poses
    plot_selected_bev(scene, human, trajectories, args.ranks, indices, args.output / 'selected_trajectories_bev.png')
    had_failure = False
    for rank, index in zip(args.ranks, indices):
        # 移除本次排名的旧动画，避免本次高分或失败时误读旧文件。
        gif = args.output / f'rank_{rank}_bev.gif'
        gif.unlink(missing_ok=True)
        row = dict(token=args.token, rank=rank, candidate_index=index, model_final_reward=float(rewards[index]),
                   valid=False, error='', gif='', gif_error='')
        try:
            scores = asdict(pdm_score(cache, Trajectory(trajectories[index]), simulator.proposal_sampling, simulator, scorer))
            row.update(scores, valid=True)
            events = collision_events(scorer)
            collision_time = scorer.time_to_at_fault_collision(1)
            if events and not np.isclose(events[0]['time_s'], collision_time):
                raise ValueError('碰撞事件时间与评分器不一致')
            if not events and np.isfinite(collision_time):
                raise ValueError('评分器记录了碰撞，但未能还原碰撞对象')
            row['first_at_fault_collision_s'] = events[0]['time_s'] if events else ''
            row['collision_objects_json'] = json.dumps(events, ensure_ascii=False)
            offroad = np.flatnonzero(scorer._ego_areas[1, :, EgoAreaIndex.NON_DRIVABLE_AREA])
            row['first_offroad_s'] = round(float(offroad[0] * scorer.proposal_sampling.interval_length), 3) if offroad.size else ''
        except Exception as exc:
            row.update(valid=False, error=f'{type(exc).__name__}: {exc}')
            had_failure = True
        if row['valid'] and row['score'] < 0.8:
            try:
                render_gif(gif, scene, scorer, trajectories[index], human, events[0] if events else None)
                row['gif'] = gif.name
            except Exception as exc:
                row['gif_error'] = f'{type(exc).__name__}: {exc}'
                had_failure = True
        rows.append(row)
        # 每完成一条就写入，避免中断后丢失已完成的结果。
        write_csv(args.output / 'scores.csv', fields, rows)
        if row['valid']:
            print(f"rank={rank} index={index} score={row['score']:.6f} NC={row['no_at_fault_collisions']} GIF={row['gif'] or '无'}", flush=True)
        else:
            print(f"index={index} 评分失败: {row['error']}", file=sys.stderr, flush=True)
        if row['gif_error']:
            print(f"index={index} GIF 生成失败: {row['gif_error']}", file=sys.stderr, flush=True)
    print(f'结果：{args.output / "scores.csv"}', flush=True)
    print(f'轨迹图：{args.output / "selected_trajectories_bev.png"}', flush=True)
    if had_failure:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
