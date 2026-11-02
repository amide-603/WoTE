"""抽样 navtest 场景，比较原始奖励、仅 SIM 和仅 IM 三种轨迹排序。"""
import argparse
import csv
import json
import lzma
import math
import os
import pickle
import random
import tempfile
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

from navsim.common.dataloader import MetricCacheLoader, SceneLoader
from navsim.common.dataclasses import Trajectory
from navsim.evaluate.pdm_score import pdm_score
from navsim.visualization.visualize_wote_scene import DEFAULT_RESULTS
from navsim.visualization.score_top_candidates import CACHE, CONFIG, ORIGINAL, METRICS

RULES = ('original', 'sim_only', 'im_only')
FIELDS = ['token', 'rule', 'rank', 'candidate_index', 'rule_reward', 'model_final_reward',
          'model_im_reward', 'model_sim_nc', 'model_sim_dac', 'model_sim_ep', 'model_sim_ttc',
          'model_sim_comfort', 'valid', 'error'] + METRICS


def save_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix='.tmp_', suffix='.csv')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp): os.unlink(temp)


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))


def sample_tokens(count, seed, cache):
    valid = [r['token'] for r in read_csv(ORIGINAL) if r.get('valid', '').lower() == 'true'
             and r['token'] in cache.metric_cache_paths and (DEFAULT_RESULTS / (r['token'] + '.npz')).is_file()]
    valid = sorted(set(valid))
    if len(valid) < count: raise ValueError(f'可用场景仅 {len(valid)} 个')
    return random.Random(seed).sample(valid, count)


def get_prediction(agent, loader, token, device):
    _, result = agent.compute_trajectory_with_results(loader.get_agent_input_from_token(token))
    with np.load(DEFAULT_RESULTS / (token + '.npz'), allow_pickle=False) as old:
        differences = {}
        for key in ('trajectory', 'all_trajectory', 'final_rewards', 'im_rewards'):
            a, b = old[key], result[key]
            if a.shape != b.shape or not np.allclose(a, b, rtol=1e-3, atol=1e-3):
                raise ValueError(f'{key} 重推理与旧结果不一致')
            differences[key] = float(np.max(np.abs(a-b)))
        if int(np.argmax(old['final_rewards'])) != int(np.argmax(result['final_rewards'])):
            raise ValueError('重推理与原推理的第一名不一致')
    sim = np.asarray(result['sim_rewards']).reshape(5,256)
    im = np.asarray(result['im_rewards']).reshape(256)
    final = np.asarray(result['final_rewards']).reshape(256)
    poses = np.asarray(result['all_trajectory']).reshape(256,8,3)
    if not all(np.isfinite(x).all() for x in (sim, im, final, poses)) or (sim <= 0).any() or (im <= 0).any():
        raise ValueError('模型输出异常，不能计算对数')
    simscore = .5*np.log(sim[0].astype('float64')) + .5*np.log(sim[1].astype('float64')) + np.log(5*sim[3].astype('float64')+2*sim[4].astype('float64')+5*sim[2].astype('float64'))
    reconstructed = .1*np.log(im.astype('float64')) + simscore
    if not np.allclose(reconstructed, final, rtol=1e-4, atol=1e-5):
        raise ValueError('综合奖励与默认公式不一致')
    return sim, im, final, poses, simscore, differences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--device', default='npu:0')
    args = parser.parse_args()
    if args.count < 1 or not 1 <= args.k <= 256: parser.error('count 或 k 不合法')
    cache_loader = MetricCacheLoader(CACHE)
    tokens = sample_tokens(args.count, args.seed, cache_loader)
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {'count':args.count, 'seed':args.seed, 'k':args.k, 'device':args.device,
                'eligible_population':'valid NAVSIM score, saved NPZ and metric cache', 'tokens':tokens,
                'ranking':'original=0.1*log(IM)+0.5*log(NC)+0.5*log(DAC)+log(5*TTC+2*Comfort+5*EP); sim_only removes IM; im_only uses IM',
                'collision_free_definition':'no_at_fault_collisions == 1; does not exclude non-fault collisions'}
    meta = args.output/'sample.json'
    if meta.exists() and json.loads(meta.read_text()) != metadata: raise ValueError('现有输出目录对应不同实验配置')
    meta.write_text(json.dumps(metadata, indent=2, ensure_ascii=False)+'\n')
    if args.device.startswith('npu'):
        import torch_npu
        import torch
        torch.backends.mha.set_fastpath_enabled(False)
    from navsim.agents.WoTE.configs.default import WoTEConfig
    from navsim.agents.WoTE.WoTE_agent import WoTEAgent
    cfg = WoTEConfig()
    agent = WoTEAgent(cfg, cfg.trajectory_sampling, lr=1e-4, checkpoint_path=os.environ['WOTE_CHECKPOINT_PATH'])
    agent.initialize(); agent.is_eval = True; agent.to(args.device)
    root, data = Path(os.environ['NAVSIM_DEVKIT_ROOT']), Path(os.environ['OPENSCENE_DATA_ROOT'])
    filt = OmegaConf.load(root/'navsim/planning/script/config/common/scene_filter/navtest.yaml')
    filt.tokens=tokens; filt.max_scenes=None
    loader = SceneLoader(data/'navsim_logs/test', data/'sensor_blobs/test', instantiate(filt), agent.get_sensor_config())
    missing=set(tokens)-set(loader.tokens)
    if missing: raise ValueError(f'数据缺失 {len(missing)} 场景')
    config=OmegaConf.load(CONFIG); simulator=instantiate(config.simulator); scorer=instantiate(config.scorer)
    originals={r['token']:r for r in read_csv(ORIGINAL) if r.get('valid','').lower()=='true'}
    all_rows=[]; failures=[]
    for n,token in enumerate(tokens,1):
        output=args.output/'scenes'/f'{token}.csv'
        try:
            if output.exists():
                prev=read_csv(output)
                if len(prev)==len(RULES)*args.k and all(x['valid'].lower()=='true' for x in prev):
                    all_rows.extend(prev); print(f'[{n}/{len(tokens)}] {token} 已完成',flush=True); continue
            sim,im,final,poses,simscore,differences=get_prediction(agent,loader,token,args.device)
            scores={'original':final, 'sim_only':simscore, 'im_only':im}
            ranking={rule:np.argsort(-scores[rule],kind='stable')[:args.k] for rule in RULES}
            with lzma.open(cache_loader.metric_cache_paths[token],'rb') as f: cache=pickle.load(f)
            index_scores={}
            for idx in sorted({int(i) for rule in RULES for i in ranking[rule]}):
                index_scores[idx]=asdict(pdm_score(cache,Trajectory(poses[idx]),simulator.proposal_sampling,simulator,scorer))
            first=index_scores[int(ranking['original'][0])]
            for key in METRICS:
                if not np.isclose(first[key],float(originals[token][key]),rtol=1e-5,atol=1e-6):
                    raise ValueError(f'第一名 {key} 与旧评分不同: {first[key]} vs {originals[token][key]}')
            scene_rows=[]
            for rule in RULES:
                for rank,idx in enumerate(ranking[rule],1):
                    idx=int(idx)
                    row=dict(token=token,rule=rule,rank=rank,candidate_index=idx,rule_reward=float(scores[rule][idx]),
                             model_final_reward=float(final[idx]),model_im_reward=float(im[idx]),
                             model_sim_nc=float(sim[0,idx]),model_sim_dac=float(sim[1,idx]),
                             model_sim_ep=float(sim[2,idx]),model_sim_ttc=float(sim[3,idx]),
                             model_sim_comfort=float(sim[4,idx]),valid=True,error='')
                    row.update(index_scores[idx]);scene_rows.append(row)
            save_csv(output,FIELDS,scene_rows);all_rows.extend(scene_rows)
            print(f'[{n}/{len(tokens)}] {token} 成功，去重后评分 {len(index_scores)} 条，推理最大误差 {max(differences.values()):.6g}',flush=True)
        except Exception as e:
            failures.append(dict(token=token,error=str(e),traceback=traceback.format_exc()))
            print(f'[{n}/{len(tokens)}] {token} 失败: {e}',flush=True)
        # 每完成一个场景就保存完整明细及各规则逐场景汇总，便于中断后恢复。
        save_csv(args.output/'candidate_scores.csv',FIELDS,all_rows)
        groups={}
        for row in all_rows: groups.setdefault((row['token'],row['rule']),[]).append(row)
        summary=[]
        for (t,rule),group in groups.items():
            group.sort(key=lambda r:int(r['rank']))
            summary.append(dict(token=t,rule=rule,top1_score=group[0]['score'],
                                top10_score_mean=float(np.mean([float(x['score']) for x in group])),
                                no_at_fault_collision_count=sum(np.isclose(float(x['no_at_fault_collisions']),1) for x in group),
                                scores_by_rank=json.dumps([float(x['score']) for x in group]),
                                collision_free_ranks=json.dumps([int(x['rank']) for x in group if np.isclose(float(x['no_at_fault_collisions']),1)])))
        save_csv(args.output/'scene_summary.csv',['token','rule','top1_score','top10_score_mean',
                 'no_at_fault_collision_count','scores_by_rank','collision_free_ranks'],summary)
        (args.output/'failures.json').write_text(json.dumps(failures,indent=2,ensure_ascii=False)+'\n')
    print(f'完成 {len(tokens)-len(failures)}/{len(tokens)} 场景，{len(all_rows)} 条排序记录，失败 {len(failures)}',flush=True)
    if failures: raise SystemExit(1)


if __name__=='__main__': main()
