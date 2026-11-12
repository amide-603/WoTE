"""比较正常与同场景错排未来 BEV 时的 WoTE 选轨和 NAVSIM 评分。

固定原推理 Top 32，保持当前 BEV 与当前/未来自车特征，将送入评分网络
的未来 BEV 进行多次可复现的一对一错排。未来 BEV 本身含自车信息。
"""
import argparse
import csv
import json
import lzma
import os
import pickle
import traceback
import types
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from navsim.common.dataloader import MetricCacheLoader, SceneLoader
from navsim.common.dataclasses import Trajectory
from navsim.evaluate.pdm_score import pdm_score
from navsim.visualization.compare_ranking_rules import save_csv, sample_tokens
from navsim.visualization.score_top_candidates import CACHE, CONFIG, METRICS
from navsim.visualization.visualize_wote_scene import DEFAULT_RESULTS

SIM = ['nc', 'dac', 'ep', 'ttc', 'comfort']
K = 32


def scores(model, feature):
    im = torch.softmax(model.reward_head(feature).squeeze(-1), dim=-1)
    sim = [head(feature).sigmoid() for head in model.sim_reward_heads]
    return im, torch.cat(sim, dim=-1).permute(0, 2, 1), model.weighted_reward_calculation(im, sim)


def load_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def validate_rows(rows, token, repeats):
    if len(rows) != K * repeats or any(r['token'] != token for r in rows):
        raise ValueError('现有场景结果不完整或 token 不一致')
    for repeat in range(1, repeats + 1):
        group = [r for r in rows if int(r['repeat']) == repeat]
        if len(group) != K or sum(int(r['normal_selected']) for r in group) != 1 or sum(int(r['shuffled_selected']) for r in group) != 1:
            raise ValueError(f'第 {repeat} 次错排结果不完整')
        indices = [int(r['candidate_index']) for r in group]
        donors = [int(r['donor_candidate_index']) for r in group]
        if len(set(indices)) != K or set(indices) != set(donors) or any(x == y for x, y in zip(indices, donors)):
            raise ValueError('错排不是一对一无自配对')
        for r in group:
            if int(r['normal_selected']) or int(r['shuffled_selected']):
                if not all(r[f'actual_{metric}'] != '' and np.isfinite(float(r[f'actual_{metric}'])) for metric in METRICS):
                    raise ValueError('选中轨迹缺少真实评分')


def write_summaries(output, completed, repeats):
    """按每个场景和每次错排汇总；总体同时报告每次及全部错排均值。"""
    if not completed:
        return
    scene_rows = []
    for group in completed:
        for repeat in range(1, repeats + 1):
            g = [r for r in group if int(r['repeat']) == repeat]
            n = next(r for r in g if int(r['normal_selected']))
            s = next(r for r in g if int(r['shuffled_selected']))
            row = {'token': n['token'], 'repeat': repeat,
                   'normal_index': int(n['candidate_index']), 'normal_original_rank': int(n['original_rank']),
                   'shuffled_index': int(s['candidate_index']), 'shuffled_original_rank': int(s['original_rank']),
                   'top1_changed': int(n['candidate_index'] != s['candidate_index']),
                   'saved_top1_index': int(n['saved_top1_index']),
                   'normal_differs_from_saved_top1': int(n['candidate_index'] != n['saved_top1_index']),
                   'max_pose_diff_vs_saved': n['max_pose_diff_vs_saved']}
            for name in METRICS:
                row[f'normal_{name}'] = float(n[f'actual_{name}'])
                row[f'shuffled_{name}'] = float(s[f'actual_{name}'])
            row['score_difference_shuffled_minus_normal'] = row['shuffled_score'] - row['normal_score']
            scene_rows.append(row)
    all_summary = []
    for repeat in [*range(1, repeats + 1), 0]:
        group = [r for r in scene_rows if r['repeat'] == repeat] if repeat else scene_rows
        normal = np.array([r['normal_score'] for r in group])
        shuffled = np.array([r['shuffled_score'] for r in group])
        row = {'repeat': repeat if repeat else 'all_mean', 'completed_scenes': len(completed),
               'comparison_count': len(group), 'normal_mean_score': normal.mean(),
               'shuffled_mean_score': shuffled.mean(), 'mean_score_change': (shuffled-normal).mean(),
               'top1_changed_count': sum(r['top1_changed'] for r in group),
               'saved_top1_changed_count': sum(r['normal_differs_from_saved_top1'] for r in group),
               'improved_count': int(np.sum(shuffled > normal + 1e-9)),
               'worsened_count': int(np.sum(shuffled < normal - 1e-9)),
               'unchanged_score_count': int(np.sum(np.abs(shuffled-normal) <= 1e-9))}
        for name in METRICS:
            for prefix in ('normal', 'shuffled'):
                row[f'{prefix}_mean_{name}'] = np.mean([r[f'{prefix}_{name}'] for r in group])
        for prefix in ('normal', 'shuffled'):
            row[f'{prefix}_nc_equal_1_count'] = sum(r[f'{prefix}_no_at_fault_collisions'] == 1 for r in group)
        all_summary.append(row)
    scene_averages = []
    by_token = {}
    for row in scene_rows:
        by_token.setdefault(row['token'], []).append(row)
    for token, entries in by_token.items():
        scene_averages.append({'token': token, 'normal_score': entries[0]['normal_score'],
                               'shuffled_mean_score': np.mean([r['shuffled_score'] for r in entries]),
                               'mean_score_change': np.mean([r['score_difference_shuffled_minus_normal'] for r in entries]),
                               'top1_changed_count': sum(r['top1_changed'] for r in entries)})
    save_csv(output / 'scene_summary.csv', list(scene_rows[0]), scene_rows)
    save_csv(output / 'scene_averages.csv', list(scene_averages[0]), scene_averages)
    save_csv(output / 'overall_summary.csv', list(all_summary[0]), all_summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=3000)
    parser.add_argument('--sample-size', type=int, default=3000,
                        help='固定抽样总数；count 取前几个，便于小批量验证')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='npu:0')
    args = parser.parse_args()
    if not 1 <= args.count <= args.sample_size or args.repeats < 1:
        parser.error('count、sample-size 或 repeats 不合法')
    cache_loader = MetricCacheLoader(CACHE)
    tokens = sample_tokens(args.sample_size, args.seed, cache_loader)[:args.count]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'scenes').mkdir(exist_ok=True)
    metadata = {'tokens': tokens, 'sample_size': args.sample_size, 'seed': args.seed,
                'device': args.device, 'k': K, 'shuffle_repeats': args.repeats,
                'permutation': 'independent random cycles, same scene, no fixed points',
                'candidate_set': 'saved original R_final top 32',
                'ranking': 'rerun R_final, rank only the 32 saved candidates',
                'actual': 'NAVSIM evaluates selected rerun trajectory, cached between groups/repeats',
                'note': 'future BEV includes injected ego; current BEV and ego tensors unchanged'}
    metadata_path = args.output / 'metadata.json'
    if metadata_path.exists() and json.loads(metadata_path.read_text()) != metadata:
        raise ValueError('输出目录属于不同实验配置')
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    torch.set_num_threads(1)
    if args.device.startswith('npu'):
        import torch_npu  # noqa: F401
        torch.backends.mha.set_fastpath_enabled(False)
    from navsim.agents.WoTE.configs.default import WoTEConfig
    from navsim.agents.WoTE.WoTE_agent import WoTEAgent
    cfg = WoTEConfig()
    agent = WoTEAgent(cfg, cfg.trajectory_sampling, lr=1e-4, checkpoint_path=os.environ['WOTE_CHECKPOINT_PATH'])
    agent.initialize(); agent.is_eval = True; agent.eval(); agent.to(args.device)
    model = agent.WoTE_model
    captured = {}
    original_method = model._compute_reward_feature
    def capture(self, ego, bev, *method_args, **kwargs):
        captured['ego'] = [x.detach().clone() for x in ego]
        captured['bev'] = [x.detach().clone() for x in bev]
        return original_method(ego, bev, *method_args, **kwargs)
    model._compute_reward_feature = types.MethodType(capture, model)
    root, data = Path(os.environ['NAVSIM_DEVKIT_ROOT']), Path(os.environ['OPENSCENE_DATA_ROOT'])
    filt = OmegaConf.load(root / 'navsim/planning/script/config/common/scene_filter/navtest.yaml')
    filt.tokens = tokens; filt.max_scenes = None
    loader = SceneLoader(data / 'navsim_logs/test', data / 'sensor_blobs/test', instantiate(filt), agent.get_sensor_config())
    if set(tokens) - set(loader.tokens):
        raise ValueError(f'缺少场景: {set(tokens) - set(loader.tokens)}')
    config = OmegaConf.load(CONFIG)
    simulator, scorer = instantiate(config.simulator), instantiate(config.scorer)
    completed, failures = [], []
    for n, token in enumerate(tokens, 1):
        path = args.output / 'scenes' / f'{token}.csv'
        try:
            if path.exists():
                rows = load_rows(path)
                validate_rows(rows, token, args.repeats)
            else:
                with np.load(DEFAULT_RESULTS / f'{token}.npz', allow_pickle=False) as old:
                    saved_poses = np.asarray(old['all_trajectory']).reshape(256, 8, 3)
                    saved_final = np.asarray(old['final_rewards']).reshape(256)
                    indices = np.argsort(-saved_final, kind='stable')[:K]
                captured.clear()
                _, result = agent.compute_trajectory_with_results(loader.get_agent_input_from_token(token))
                max_pose_diff = float(np.max(np.abs(saved_poses - result['all_trajectory'].reshape(256, 8, 3))))
                if max_pose_diff > 0.05:
                    raise ValueError(f'重推理与保存轨迹最大差异 {max_pose_diff:.6f} 超过 0.05')
                with torch.no_grad():
                    ego, bev = captured['ego'], captured['bev']
                    if len(bev) < 2 or len(ego) != len(bev):
                        raise ValueError('缺少未来特征')
                    normal = scores(model, original_method(ego, bev, 1, 256))
                    for key, tensor in zip(('im_rewards', 'sim_rewards', 'final_rewards'), normal):
                        if not np.allclose(tensor.cpu().numpy(), result[key], rtol=1e-5, atol=1e-6):
                            raise ValueError(f'重算与原模型不一致: {key}')
                    normal = [x.cpu().numpy() for x in normal]
                    ni = int(indices[np.argmax(normal[2][0, indices])])
                    rng = np.random.default_rng((int(token, 16) + args.seed) % 2**64)
                    idx = torch.as_tensor(indices, device=bev[0].device)
                    variants = []
                    for repeat in range(1, args.repeats + 1):
                        cycle = rng.permutation(K)
                        donors = np.empty(K, dtype=int)
                        donors[cycle] = np.roll(cycle, 1)
                        if np.any(donors == np.arange(K)) or len(set(donors)) != K:
                            raise ValueError('置换不满足一对一无自配对')
                        donor_idx = torch.as_tensor(indices[donors], device=bev[0].device)
                        modified = [bev[0]]
                        for future in bev[1:]:
                            changed = future.clone()
                            changed[idx] = future[donor_idx]
                            modified.append(changed)
                        shuffled = scores(model, original_method(ego, modified, 1, 256))
                        cos = torch.nn.functional.cosine_similarity(bev[-1][idx].flatten(1), bev[-1][donor_idx].flatten(1), dim=1).cpu().numpy()
                        shuffled = [x.cpu().numpy() for x in shuffled]
                        si = int(indices[np.argmax(shuffled[2][0, indices])])
                        variants.append((repeat, donors, shuffled, cos, si))
                with lzma.open(cache_loader.metric_cache_paths[token], 'rb') as f:
                    cache = pickle.load(f)
                selected = {ni} | {v[4] for v in variants}
                actual = {i: asdict(pdm_score(cache, Trajectory(result['all_trajectory'].reshape(256, 8, 3)[i]),
                                              simulator.proposal_sampling, simulator, scorer)) for i in selected}
                rows = []
                for repeat, donors, shuffled, cos, si in variants:
                    for rank, (index, donor) in enumerate(zip(indices, indices[donors]), 1):
                        index = int(index)
                        row = {'token': token, 'repeat': repeat, 'original_rank': rank, 'candidate_index': index,
                               'donor_candidate_index': int(donor), 'future_bev_cosine': float(cos[rank-1]),
                               'normal_selected': int(index == ni), 'shuffled_selected': int(index == si),
                               'saved_top1_index': int(indices[0]), 'max_pose_diff_vs_saved': max_pose_diff}
                        for prefix, (ims, sims, finals) in [('normal', normal), ('shuffled', shuffled)]:
                            row.update({f'{prefix}_{name}': float(sims[0, i, index]) for i, name in enumerate(SIM)})
                            row[f'{prefix}_im'] = float(ims[0, index])
                            row[f'{prefix}_final_reward'] = float(finals[0, index])
                        row.update({f'actual_{key}': actual[index][key] if index in actual else '' for key in METRICS})
                        rows.append(row)
                validate_rows(rows, token, args.repeats)
                save_csv(path, list(rows[0]), rows)
            # 比较编号时统一类型；磁盘读取的 CSV 值是字符串。
            for row in rows:
                for key in ('candidate_index', 'saved_top1_index', 'original_rank'):
                    row[key] = int(row[key])
            completed.append(rows)
            print(f'[{n}/{len(tokens)}] {token} 完成，累计 {len(completed)}', flush=True)
        except Exception as exc:
            failures.append({'token': token, 'error': str(exc), 'traceback': traceback.format_exc()})
            print(f'[{n}/{len(tokens)}] {token} 失败: {exc}', flush=True)
        if n % 25 == 0 or n == len(tokens):
            write_summaries(args.output, completed, args.repeats)
            (args.output / 'failures.json').write_text(json.dumps(failures, indent=2, ensure_ascii=False) + '\n')
    print(f'完成 {len(completed)}/{len(tokens)} 场景，失败 {len(failures)}', flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
