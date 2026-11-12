"""对已有场景的前 K 条候选轨迹逐条运行 NAVSIM 评分，并汇总无责任碰撞情况。"""
import argparse
import csv
import json
import lzma
import os
import pickle
import tempfile
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

from navsim.common.dataloader import MetricCacheLoader
from navsim.common.dataclasses import Trajectory
from navsim.evaluate.pdm_score import pdm_score

BASE = Path('/mnt/sfs_turbo/gaoxuan_gac/WoTE/exp/eval/WoTE')
ROOT = BASE / 'collision_analysis_190'
CACHE = Path('/mnt/sfs_turbo/gaoxuan_gac/NAVSIMv1/metric_cache/wote_298957c/navtest')
CONFIG = BASE / 'navtest_resume/code/hydra/config.yaml'
ORIGINAL = BASE / 'navtest_resume/2026.10.28.10.10.03.csv'
METRICS = ['no_at_fault_collisions', 'drivable_area_compliance', 'driving_direction_compliance',
           'ego_progress', 'time_to_collision_within_bound', 'comfort', 'score']
DETAIL_FIELDS = ['token', 'rank', 'candidate_index', 'model_final_reward', 'valid', 'error'] + METRICS
SUMMARY_FIELDS = ['token', 'valid_candidates', 'failed_candidates', 'rank1_nc',
                  'non_fault_collision_free_count_rank2_to_k', 'non_fault_collision_free_ranks',
                  'non_fault_collision_free_candidate_indices']


def write_csv(path, fields, rows):
    """使用同目录临时文件和原子替换，避免中断时留下不完整文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix='.tmp_', suffix='.csv')
    try:
        with os.fdopen(fd, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def read_rows(path):
    with path.open(newline='', encoding='utf-8') as file:
        return list(csv.DictReader(file))


def summarize(rows, k):
    grouped = {}
    for row in rows:
        grouped.setdefault(row['token'], []).append(row)
    result = []
    for token, group in sorted(grouped.items()):
        valid = [r for r in group if str(r['valid']).lower() == 'true']
        free = sorted((r for r in valid if 2 <= int(r['rank']) <= k and
                       np.isclose(float(r['no_at_fault_collisions']), 1.0)), key=lambda r: int(r['rank']))
        first = next((r for r in valid if int(r['rank']) == 1), None)
        result.append({'token': token, 'valid_candidates': len(valid), 'failed_candidates': len(group) - len(valid),
                       'rank1_nc': first['no_at_fault_collisions'] if first else '',
                       'non_fault_collision_free_count_rank2_to_k': len(free),
                       'non_fault_collision_free_ranks': json.dumps([int(r['rank']) for r in free]),
                       'non_fault_collision_free_candidate_indices': json.dumps([int(r['candidate_index']) for r in free])})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--tokens', nargs='+')
    args = parser.parse_args()
    if not 2 <= args.k <= 256:
        parser.error('k 必须在 2 到 256 之间')
    paths = sorted(args.root.glob('*/candidates.csv'))
    if args.tokens:
        requested = set(args.tokens)
        paths = [p for p in paths if p.parent.name in requested]
        if {p.parent.name for p in paths} != requested:
            raise ValueError('指定的场景缺少 candidates.csv')
    elif len(paths) != 190:
        raise ValueError(f'预期 190 个场景，实际 {len(paths)} 个')
    originals = {r['token']: r for r in read_rows(ORIGINAL) if r.get('token') and r.get('valid', '').lower() == 'true'}
    config = OmegaConf.load(CONFIG)
    simulator, scorer = instantiate(config.simulator), instantiate(config.scorer)
    caches = MetricCacheLoader(CACHE)
    details = []
    args.output.mkdir(parents=True, exist_ok=True)
    for n, path in enumerate(paths, 1):
        token = path.parent.name
        scene_output = args.output / 'scenes' / f'{token}.csv'
        try:
            candidates = sorted(read_rows(path), key=lambda r: int(r['rank']))[:args.k]
            if [int(r['rank']) for r in candidates] != list(range(1, args.k + 1)):
                raise ValueError('候选轨迹排名不完整')
            # 已完成场景可恢复；异常场景下次重试，避免将失败误认为无碰撞。
            if scene_output.exists():
                saved = read_rows(scene_output)
                if (len(saved) == args.k and all(r['valid'].lower() == 'true' for r in saved)
                        and [int(r['rank']) for r in saved] == list(range(1, args.k + 1))
                        and [int(r['candidate_index']) for r in saved] == [int(r['index']) for r in candidates]):
                    details.extend(saved)
                    print(f'[{n}/{len(paths)}] {token} 已完成，跳过', flush=True)
                    continue
            with lzma.open(caches.metric_cache_paths[token], 'rb') as file:
                cache = pickle.load(file)
            scene_rows = []
            for candidate in candidates:
                row = dict(token=token, rank=int(candidate['rank']), candidate_index=int(candidate['index']),
                           model_final_reward=candidate['final_reward'], valid=False, error='')
                try:
                    poses = np.asarray(json.loads(candidate['trajectory_xy_heading_json']), dtype=np.float64)
                    if poses.shape != (8, 3) or not np.isfinite(poses).all():
                        raise ValueError('轨迹应为有限的 8×3 数组')
                    scores = asdict(pdm_score(cache, Trajectory(poses), simulator.proposal_sampling, simulator, scorer))
                    if row['rank'] == 1:
                        original = originals[token]
                        for name in METRICS:
                            if not np.isclose(scores[name], float(original[name]), rtol=1e-5, atol=1e-6):
                                raise ValueError(f'第一名的 {name} 与原评分不一致: {scores[name]} vs {original[name]}')
                    row.update(scores, valid=True)
                except Exception as exc:
                    row['error'] = str(exc)
                scene_rows.append(row)
            write_csv(scene_output, DETAIL_FIELDS, scene_rows)
            details.extend(scene_rows)
            failed = sum(not r['valid'] for r in scene_rows)
            print(f'[{n}/{len(paths)}] {token}: {len(scene_rows)-failed}/{args.k} 成功', flush=True)
        except Exception as exc:
            print(f'[{n}/{len(paths)}] {token}: 场景失败 {exc}', flush=True)
            traceback.print_exc()
            failed_rows = [dict(token=token, rank=rank, candidate_index='', model_final_reward='', valid=False,
                                error=f'场景失败: {exc}') for rank in range(1, args.k + 1)]
            write_csv(scene_output, DETAIL_FIELDS, failed_rows)
            details.extend(failed_rows)
        # 汇总随场景更新，即使运行中断也可以检查已有结果。
        write_csv(args.output / 'candidate_scores.csv', DETAIL_FIELDS, details)
        write_csv(args.output / 'scene_summary.csv', SUMMARY_FIELDS, summarize(details, args.k))
    # 恢复运行时所有场景可能均被跳过，也要重新生成汇总。
    write_csv(args.output / 'candidate_scores.csv', DETAIL_FIELDS, details)
    write_csv(args.output / 'scene_summary.csv', SUMMARY_FIELDS, summarize(details, args.k))
    failures = sum(str(row['valid']).lower() != 'true' for row in details)
    print(f'完成：{len(paths)} 个场景，{len(details)} 条轨迹，失败 {failures} 条', flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
