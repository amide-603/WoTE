"""对已保存的全部候选轨迹运行 NAVSIM 评分，并计算逐场景 Oracle。"""
import argparse
import csv
import json
import lzma
import multiprocessing as mp
import os
import pickle
import tempfile
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

from navsim.common.dataloader import MetricCacheLoader
from navsim.common.dataclasses import Trajectory
from navsim.evaluate.pdm_score import pdm_score
from navsim.visualization.score_top_candidates import CACHE, CONFIG, ORIGINAL, METRICS
from navsim.visualization.visualize_wote_scene import DEFAULT_RESULTS

FIELDS = ['token', 'rank', 'candidate_index', 'model_final_reward', 'model_im_reward'] + METRICS
GROUPS = ['selected', 'oracle_32', 'oracle_128', 'oracle_256']
SUMMARY_FIELDS = ['token'] + [f'{g}_{k}' for g in GROUPS for k in ['rank', 'candidate_index'] + METRICS]
WORKER = {}


def atomic_csv(path, fields, rows):
    """同目录临时文件加原子替换；中断不会留下半个 CSV。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix='.tmp_', suffix='.csv')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def make_summary(token, rows):
    """Oracle 在模型排名前 K 条中选择实际总分最高者；同分时优先模型排名靠前。"""
    if len(rows) != 256 or [int(r['rank']) for r in rows] != list(range(1, 257)):
        raise ValueError('结果不完整，无法计算 Oracle')
    result = {'token': token}
    for group, subset in [('selected', rows[:1]), ('oracle_32', rows[:32]),
                          ('oracle_128', rows[:128]), ('oracle_256', rows)]:
        best = max(subset, key=lambda row: float(row['score']))
        for key in ['rank', 'candidate_index'] + METRICS:
            result[f'{group}_{key}'] = best[key]
    return result


def init_worker(config, cache_paths, originals, output, checkpoint_every):
    cfg = OmegaConf.load(config)
    WORKER.update(sim=instantiate(cfg.simulator), scorer=instantiate(cfg.scorer),
                  caches=cache_paths, originals=originals, output=Path(output), checkpoint=checkpoint_every)


def score_scene(job):
    token, file_name = job
    path = WORKER['output'] / 'scenes' / f'{token}.csv'
    partial = WORKER['output'] / 'partial' / f'{token}.csv'
    try:
        with np.load(file_name, allow_pickle=False) as d:
            poses = np.asarray(d['all_trajectory'])
            rewards = np.asarray(d['final_rewards']).reshape(-1)
            im = np.asarray(d['im_rewards']).reshape(-1)
            selected = np.asarray(d['trajectory']).reshape(8, 3)
        if poses.shape == (1, 256, 8, 3): poses = poses[0]
        if poses.shape != (256, 8, 3) or rewards.shape != (256,) or im.shape != (256,):
            raise ValueError('NPZ 轨迹或奖励形状错误')
        if not all(np.isfinite(x).all() for x in (poses, rewards, im)):
            raise ValueError('NPZ 包含无效数值')
        # 同分按原始编号顺序；第 1 名与模型 argmax 保持一致。
        order = np.argsort(-rewards, kind='stable')
        if not np.allclose(selected, poses[order[0]], rtol=1e-5, atol=1e-6):
            raise ValueError('模型选中轨迹与综合奖励第 1 名不一致')
        rows = read_csv(path) if path.exists() else read_csv(partial) if partial.exists() else []
        if len(rows) > 256:
            raise ValueError('已有结果行数异常')
        for rank, row in enumerate(rows, 1):
            index = int(order[rank - 1])
            if (row['token'] != token or int(row['rank']) != rank or int(row['candidate_index']) != index
                or not np.isclose(float(row['model_final_reward']), rewards[index], rtol=1e-6, atol=1e-7)
                or not all(np.isfinite(float(row[key])) for key in METRICS)):
                raise ValueError(f'已有结果第 {rank} 行与当前输入不一致')
        if len(rows) < 256:
            with lzma.open(WORKER['caches'][token], 'rb') as f: cache = pickle.load(f)
            for rank in range(len(rows) + 1, 257):
                index = int(order[rank - 1])
                score = asdict(pdm_score(cache, Trajectory(poses[index]), WORKER['sim'].proposal_sampling,
                                         WORKER['sim'], WORKER['scorer']))
                if not all(np.isfinite(score[k]) for k in METRICS):
                    raise ValueError(f'第 {rank} 名评分包含非有限值')
                if rank == 1:
                    original = WORKER['originals'][token]
                    for key in METRICS:
                        if not np.isclose(score[key], float(original[key]), rtol=1e-5, atol=1e-6):
                            raise ValueError(f'第 1 名 {key} 与原评分不同: {score[key]} vs {original[key]}')
                row = dict(token=token, rank=rank, candidate_index=index,
                           model_final_reward=float(rewards[index]), model_im_reward=float(im[index]), **score)
                rows.append(row)
                if rank % WORKER['checkpoint'] == 0 and rank < 256:
                    atomic_csv(partial, FIELDS, rows)
            atomic_csv(path, FIELDS, rows)
            partial.unlink(missing_ok=True)
        return token, make_summary(token, rows), ''
    except Exception:
        return token, None, traceback.format_exc()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--results-dir', type=Path, default=DEFAULT_RESULTS)
    p.add_argument('--cache-dir', type=Path, default=CACHE)
    p.add_argument('--config', type=Path, default=CONFIG)
    p.add_argument('--original', type=Path, default=ORIGINAL)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--checkpoint-every', type=int, default=16)
    p.add_argument('--tokens', nargs='+', help='仅处理指定场景，供小批量验证')
    args = p.parse_args()
    if args.workers < 1 or args.checkpoint_every < 1: p.error('workers 和 checkpoint-every 必须大于 0')
    inputs = {f.stem: f for f in args.results_dir.glob('*.npz')}
    originals = {r['token']: r for r in read_csv(args.original) if r.get('valid', '').lower() == 'true'}
    caches = MetricCacheLoader(args.cache_dir).metric_cache_paths
    tokens = sorted(set(args.tokens) if args.tokens else set(inputs))
    missing = [t for t in tokens if t not in inputs or t not in caches or t not in originals]
    if not tokens or missing: raise ValueError(f'场景缺少模型结果、cache 或原始有效评分：{missing[:10]}，共 {len(missing)}')
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {'version': 1, 'results_dir': str(args.results_dir.resolve()), 'cache_dir': str(args.cache_dir.resolve()),
                'config': str(args.config.resolve()), 'original': str(args.original.resolve()),
                'tokens': tokens, 'ranking': 'final_rewards 降序，同分按候选编号升序',
                'oracle': '模型排名前 K 条中实际 score 最高，同分选择排名靠前的轨迹',
                'metrics': METRICS}
    meta = args.output / 'manifest.json'
    if meta.exists() and json.loads(meta.read_text(encoding='utf-8')) != metadata:
        raise ValueError('输出目录已有不同的实验配置，不能混用')
    if not meta.exists(): meta.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    start = time.monotonic(); summaries = {}; failures = []
    jobs = [(t, str(inputs[t])) for t in tokens]
    print(f'开始：{len(tokens)} 场景，{len(tokens)*256} 条轨迹，CPU worker={args.workers}', flush=True)
    with mp.get_context('spawn').Pool(args.workers, initializer=init_worker,
             initargs=(str(args.config), caches, originals, str(args.output), args.checkpoint_every)) as pool:
        for n, (token, summary, error) in enumerate(pool.imap_unordered(score_scene, jobs, chunksize=1), 1):
            if error:
                failures.append({'token': token, 'error': error})
                print(f'[{n}/{len(tokens)}] {token} 失败 {error.splitlines()[-1]}', flush=True)
            else:
                summaries[token] = summary
                print(f'[{n}/{len(tokens)}] {token} 完成 score={float(summary["selected_score"]):.4f} oracle256={float(summary["oracle_256_score"]):.4f}', flush=True)
            if n % 25 == 0 or n == len(tokens):
                atomic_csv(args.output / 'scene_summary.csv', SUMMARY_FIELDS, [summaries[t] for t in sorted(summaries)])
                overall = []
                for group in GROUPS:
                    item = {'rule': group, 'scenes': len(summaries)}
                    item.update({key: float(np.mean([float(row[f'{group}_{key}']) for row in summaries.values()]))
                                 if summaries else '' for key in METRICS})
                    overall.append(item)
                atomic_csv(args.output / 'overall_summary.csv', ['rule', 'scenes'] + METRICS, overall)
                (args.output / 'failures.json').write_text(json.dumps(failures, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(f'结束：成功 {len(summaries)}，失败 {len(failures)}，耗时 {(time.monotonic()-start)/60:.1f} 分钟', flush=True)
    if failures: raise SystemExit(1)


if __name__ == '__main__': main()
