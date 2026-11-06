"""在已保存原始 Top K 轨迹中按 0.5*预测 NC+0.5*预测 DAC 选轨。

复用已保存的预测或重新推理，严格校验候选轨迹和原始奖励；使用全部
候选已计算好的 NAVSIM 评分，不重复运行 NAVSIM。每个场景原子保存。
"""
import argparse
import csv
import json
import os
import traceback
from pathlib import Path

import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf
from navsim.common.dataloader import SceneLoader
from navsim.visualization.compare_ranking_rules import save_csv
from navsim.visualization.score_top_candidates import METRICS
from navsim.visualization.visualize_wote_scene import DEFAULT_RESULTS

BASE = Path('/mnt/sfs_turbo/gaoxuan_gac/WoTE/exp/eval/WoTE')
ORACLE = BASE / 'oracle_all_256'
SHUFFLE = BASE / 'future_bev_shuffle_3000_top32_repeat5_seed42/scenes'
COLLISIONS = BASE / 'collision_analysis_190'


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def cached_predictions(token, order, saved_final, saved_im, k):
    """优先复用 3000 场景或 190 碰撞场景的预测，验证轨迹编号及奖励。"""
    path = SHUFFLE / f'{token}.csv'
    if path.exists():
        rows = [r for r in read_csv(path) if int(r['repeat']) == 1 and int(r['original_rank']) <= k]
        rows.sort(key=lambda r: int(r['original_rank']))
        indices = [int(r['candidate_index']) for r in rows]
        if len(rows) != k or indices != order[:k].tolist() or any(float(r['max_pose_diff_vs_saved']) != 0 for r in rows):
            raise ValueError('已缓存预测与保存轨迹不匹配')
        pred = np.array([[float(r['normal_'+key]) for r in rows] for key in ('nc','dac','ep','ttc','comfort')])
        final = np.array([float(r['normal_final_reward']) for r in rows])
        im = np.array([float(r['normal_im']) for r in rows])
        source = 'shuffle_3000'
    elif (COLLISIONS / token / 'candidates.csv').exists():
        rows = sorted(read_csv(COLLISIONS / token / 'candidates.csv'), key=lambda r:int(r['rank']))[:k]
        indices = [int(r['index']) for r in rows]
        if len(rows) != k or indices != order[:k].tolist():
            raise ValueError('碰撞场景缓存预测与保存轨迹不匹配')
        pred = np.array([[float(r['sim_'+key]) for r in rows] for key in ('nc','dac','ep','ttc','comfort')])
        final = np.array([float(r['final_reward']) for r in rows])
        im = np.array([float(r['im_reward']) for r in rows])
        source = 'collision_190'
    else:
        return None
    if not np.isfinite(pred).all() or not np.allclose(final, saved_final[indices], rtol=1e-4, atol=1e-5) or not np.allclose(im, saved_im[indices], rtol=1e-4, atol=1e-6):
        raise ValueError('缓存预测奖励与原始模型输出不一致')
    if np.any(pred <= 0) or np.any(pred > 1):
        raise ValueError('预测分数超出范围')
    rec = .1*np.log(im) + .5*np.log(pred[0]) + .5*np.log(pred[1]) + np.log(5*pred[3]+2*pred[4]+5*pred[2])
    if not np.allclose(rec, final, rtol=1e-4, atol=1e-5):
        raise ValueError('缓存预测分数不能重现原始综合奖励')
    return pred, source


def summary(rows, total):
    out=[]
    for name in ('original','reranked'):
        row={'rule':name,'completed_scenes':len(rows),'target_scenes':total}
        for metric in METRICS:
            row[metric]=float(np.mean([float(r[name+'_'+metric]) for r in rows]))
        nc=[float(r[name+'_no_at_fault_collisions']) for r in rows]
        row['nc_equal_1_count']=sum(x==1 for x in nc)
        row['nc_less_1_count']=sum(x<1 for x in nc)
        out.append(row)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--device',default='npu:0')
    p.add_argument('--k',default=16,type=int)
    p.add_argument('--tokens',nargs='+',help='仅用于小批量验证')
    args=p.parse_args()
    if not 1 <= args.k <= 32: p.error('k 必须在 1 到 32 之间')
    all_tokens=json.loads((ORACLE/'manifest.json').read_text())['tokens']
    tokens=sorted(set(args.tokens)) if args.tokens else all_tokens
    if not tokens or not set(tokens)<=set(all_tokens): raise ValueError('指定场景不在 NAVSIM 已评分清单中')
    args.output.mkdir(parents=True,exist_ok=True); (args.output/'scenes').mkdir(exist_ok=True)
    meta={'tokens':tokens,'device':args.device,'top_k':args.k,'rule':'0.5*model_pred_nc + 0.5*model_pred_dac',
          'ties':'original rank ascending','actual_source':str(ORACLE/'scenes'),'old_results':str(DEFAULT_RESULTS),
          'reuse_pred':'verified shuffle_3000 and collision_190, otherwise rerun model'}
    mp=args.output/'metadata.json'
    if mp.exists() and json.loads(mp.read_text())!=meta: raise ValueError('输出目录的实验配置不一致')
    mp.write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    if args.device.startswith('npu'):
        import torch_npu  # noqa: F401
        import torch
        torch.set_num_threads(1)
        torch.backends.mha.set_fastpath_enabled(False)
    from navsim.agents.WoTE.configs.default import WoTEConfig
    from navsim.agents.WoTE.WoTE_agent import WoTEAgent
    cfg=WoTEConfig(); agent=WoTEAgent(cfg,cfg.trajectory_sampling,lr=1e-4,checkpoint_path=os.environ['WOTE_CHECKPOINT_PATH'])
    agent.initialize();agent.is_eval=True;agent.eval();agent.to(args.device)
    data=Path(os.environ['OPENSCENE_DATA_ROOT']);root=Path(os.environ['NAVSIM_DEVKIT_ROOT'])
    filt=OmegaConf.load(root/'navsim/planning/script/config/common/scene_filter/navtest.yaml');filt.tokens=tokens;filt.max_scenes=None
    loader=SceneLoader(data/'navsim_logs/test',data/'sensor_blobs/test',instantiate(filt),agent.get_sensor_config())
    missing=set(tokens)-set(loader.tokens)
    if missing: raise ValueError(f'缺失传感器场景 {len(missing)} 个')
    rows=[];failures=[]
    for n,token in enumerate(tokens,1):
        scene_path=args.output/'scenes'/f'{token}.csv'
        try:
            if scene_path.exists():
                r=read_csv(scene_path)
                if len(r)!=1 or r[0]['token']!=token: raise ValueError('已保存场景记录不完整')
                row=r[0]
            else:
                with np.load(DEFAULT_RESULTS/f'{token}.npz',allow_pickle=False) as saved:
                    saved_poses=np.asarray(saved['all_trajectory']).reshape(256,8,3)
                    saved_final=np.asarray(saved['final_rewards']).reshape(256)
                    saved_im=np.asarray(saved['im_rewards']).reshape(256)
                order=np.argsort(-saved_final,kind='stable')
                result=cached_predictions(token,order,saved_final,saved_im,args.k)
                pose_diff=0.0
                if result is None:
                    _,re=agent.compute_trajectory_with_results(loader.get_agent_input_from_token(token))
                    pose_diff=float(np.max(np.abs(saved_poses-np.asarray(re['all_trajectory']).reshape(256,8,3))))
                    final_diff=float(np.max(np.abs(saved_final-np.asarray(re['final_rewards']).reshape(256))))
                    if pose_diff > 1e-5 or final_diff > 1e-4:
                        raise ValueError(f'重推理与保存结果不一致：轨迹最大差值={pose_diff}, 综合奖励最大差值={final_diff}')
                    if int(np.argmax(re['final_rewards'])) != int(order[0]):raise ValueError('重推理 Top 1 与原结果不同')
                    pred=np.asarray(re['sim_rewards']).reshape(5,256)[:,order[:args.k]]
                    if not np.isfinite(pred).all() or np.any(pred<0) or np.any(pred>1):raise ValueError('预测分数异常')
                    source='model_rerun'
                else:
                    pred,source=result
                # 重推理输出为 float32，CSV 缓存为 float64；统一精度，
                # 避免极接近的候选在两种路径产生不同的并列结果。
                pred=np.asarray(pred,dtype=np.float64)
                combined=.5*pred[0]+.5*pred[1]
                pos=int(np.argmax(combined)); idx=int(order[pos]); original_idx=int(order[0])
                actual=read_csv(ORACLE/'scenes'/f'{token}.csv')[:args.k]
                if len(actual)!=args.k or [int(x['candidate_index']) for x in actual]!=order[:args.k].tolist():raise ValueError('NAVSIM 文件与原候选排名不一致')
                row={'token':token,'original_index':original_idx,'selected_index':idx,'selected_original_rank':pos+1,
                     'selected_changed':int(idx!=original_idx),'prediction_source':source,'max_pose_difference':pose_diff,
                     'original_pred_nc':float(pred[0,0]),'original_pred_dac':float(pred[1,0]),
                     'reranked_pred_nc':float(pred[0,pos]),'reranked_pred_dac':float(pred[1,pos]),
                     'reranked_pred_score':float(combined[pos])}
                for name,i in [('original',0),('reranked',pos)]:
                    for metric in METRICS:
                        v=float(actual[i][metric])
                        if not np.isfinite(v):raise ValueError('NAVSIM 评分非有限值')
                        row[name+'_'+metric]=v
                save_csv(scene_path,list(row),[row])
            rows.append(row)
            print(f'[{n}/{len(tokens)}] {token} 来源={row["prediction_source"]} NC={row["reranked_no_at_fault_collisions"]} score={row["reranked_score"]}',flush=True)
        except Exception as e:
            failures.append({'token':token,'error':str(e),'traceback':traceback.format_exc()})
            print(f'[{n}/{len(tokens)}] {token} 失败: {e}',flush=True)
        if n%50==0 or n==len(tokens):
            if rows:
                save_csv(args.output/'scene_comparison.csv',list(rows[0]),rows)
                s=summary(rows,len(tokens));save_csv(args.output/'overall_summary.csv',list(s[0]),s)
            (args.output/'failures.json').write_text(json.dumps(failures,ensure_ascii=False,indent=2)+'\n')
    print(f'完成 {len(rows)}/{len(tokens)} 场景，失败 {len(failures)}',flush=True)
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
