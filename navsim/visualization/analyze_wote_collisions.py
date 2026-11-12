"""读取已保存的 WoTE 轨迹，重放 NAVSIM 碰撞场景并生成分析结果。

每个场景输出碰撞摘要 JSON、逐时刻状态 CSV、BEV 仿真 GIF、相机和 BEV
轨迹图，以及 256 条候选轨迹的 CSV 和 BEV 图；不重新运行 WoTE 模型。
"""
import argparse
import csv
import json
import lzma
import math
import pickle
import shutil
import traceback
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as Patch
from PIL import Image
import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf
from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES
from nuplan.planning.metrics.utils.collision_utils import CollisionType
from navsim.common.dataloader import MetricCacheLoader, SceneLoader
from navsim.common.dataclasses import SensorConfig, Trajectory
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer_utils import get_collision_type
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import EgoAreaIndex
from navsim.visualization.bev import add_map_to_bev_ax
from navsim.visualization.plots import configure_bev_ax
from navsim.visualization.visualize_wote_scene import DEFAULT_RESULTS, load_results, plot_camera, plot_bev

# 以下路径分别指向代码仓库、NAVSIM 数据、原评分结果及评分缓存。
ROOT = Path('/home/gaoxuan_gac/code/WoTE')
DATA = Path('/mnt/sfs_turbo/gaoxuan_gac/NAVSIMv1/dataset')
EXP = Path('/mnt/sfs_turbo/gaoxuan_gac/WoTE/exp/eval/WoTE')
CSV = EXP / 'navtest_resume/2026.10.28.10.10.03.csv'
CONFIG = EXP / 'navtest_resume/code/hydra/config.yaml'
CACHE = Path('/mnt/sfs_turbo/gaoxuan_gac/NAVSIMv1/metric_cache/wote_298957c/navtest')


def collision_events(scorer):
    """复现评分器的有责碰撞判定，并记录与每个对象的首次有责碰撞。"""
    obs = scorer._observation
    # 忽略起始时已发生碰撞的对象，与 NAVSIM 评分器的处理保持一致。
    ignored = set(obs.collided_track_ids)
    events = []
    recorded = set()
    for i in range(scorer.proposal_sampling.num_poses + 1):
        occ = obs[i]
        # 下标 1 表示 WoTE 轨迹；下标 0 是同次评分使用的 PDM 参考轨迹。
        ego = scorer._ego_polygons[1, i]
        for idx in occ.query(ego, predicate='intersects'):
            token = occ.tokens[int(idx)]
            if obs.red_light_token in token or token in ignored or token not in obs.unique_objects:
                continue
            obj = obs.unique_objects[token]
            kind = get_collision_type(scorer._states[1, i], ego, obj, occ[token])
            # 前部碰撞、撞上静止物体，以及特定区域内的侧面碰撞会被判为有责。
            lateral_area = (scorer._ego_areas[1, i, EgoAreaIndex.MULTIPLE_LANES] or
                            scorer._ego_areas[1, i, EgoAreaIndex.NON_DRIVABLE_AREA])
            fault = kind in (CollisionType.ACTIVE_FRONT_COLLISION, CollisionType.STOPPED_TRACK_COLLISION) or (lateral_area and kind == CollisionType.ACTIVE_LATERAL_COLLISION)
            if fault and token not in recorded:
                events.append({'time_s': round(i * scorer.proposal_sampling.interval_length, 3),
                               'step': i, 'object_id': token, 'object_type': obj.tracked_object_type.name,
                               'collision_type': kind.name, 'penalty': 0.0 if obj.tracked_object_type in AGENT_TYPES else 0.5})
                recorded.add(token)
            elif not fault:
                ignored.add(token)
    return events


def local_xy(coords, origin):
    """将地图全局坐标转换到以自车初始位姿为原点的局部坐标系。"""
    a = np.asarray(coords, dtype=float) - np.array([origin.x, origin.y])
    c, s = math.cos(origin.heading), math.sin(origin.heading)
    return np.column_stack((c*a[:, 0]+s*a[:, 1], -s*a[:, 0]+c*a[:, 1]))


def polygon_xy(poly, origin):
    """转换多边形坐标，并按 BEV 绘图约定交换横纵坐标。"""
    xy = local_xy(poly.exterior.coords, origin)
    return np.column_stack((xy[:, 1], xy[:, 0]))


def write_timeseries(path, scorer, event):
    """保存每个仿真时刻自车与首次碰撞对象的位置、速度和多边形间距。"""
    obs = scorer._observation
    origin = scorer._states[1, 0, :3]
    target = event['object_id'] if event else None
    names = ['time_s','ego_global_x_m','ego_global_y_m','ego_heading_rad','ego_speed_mps',
             'ego_acceleration_x_mps2','ego_acceleration_y_mps2','target_id','target_global_x_m',
             'target_global_y_m','target_polygon_distance_m','target_intersects_ego','at_fault_event']
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=names); writer.writeheader()
        for i, state in enumerate(scorer._states[1]):
            occ = obs[i]; ego = scorer._ego_polygons[1, i]
            # 对象暂时不存在时留空；多边形距离表示车身边缘之间的最短距离。
            target_poly = occ[target] if target and target in occ.tokens else None
            centroid = target_poly.centroid if target_poly is not None else None
            writer.writerow({'time_s': round(i*scorer.proposal_sampling.interval_length, 3),
                'ego_global_x_m': state[0], 'ego_global_y_m': state[1], 'ego_heading_rad': state[2],
                'ego_speed_mps': float(np.hypot(state[3],state[4])), 'ego_acceleration_x_mps2': state[5],
                'ego_acceleration_y_mps2': state[6], 'target_id': target or '',
                'target_global_x_m': centroid.x if centroid else '', 'target_global_y_m': centroid.y if centroid else '',
                'target_polygon_distance_m': ego.distance(target_poly) if target_poly else '',
                'target_intersects_ego': bool(ego.intersects(target_poly)) if target_poly else '',
                'at_fault_event': bool(event and i == event['step'])})


def render_gif(path, scene, scorer, selected, human, event):
    """渲染 BEV 仿真动画：黄色为自车，红色高亮首次有责碰撞对象。"""
    from nuplan.common.actor_state.state_representation import StateSE2
    origin = StateSE2(*scene.frames[scene.scene_metadata.num_history_frames-1].ego_status.ego_pose)
    fig, ax = plt.subplots(figsize=(6,6), dpi=90)
    add_map_to_bev_ax(ax, scene.map_api, origin)
    configure_bev_ax(ax); ax.set_xlim(30,-30); ax.set_ylim(-15,45); ax.set_xticks([]); ax.set_yticks([])
    for poses, col, label in [(human,'#3aa657','Expert'), (selected,'#e06458','WoTE planned')]:
        xy=np.vstack([np.zeros((1,2)), poses[:,:2]])
        ax.plot(xy[:,1],xy[:,0],color=col,lw=2,label=label)
    ax.legend(loc='upper left', fontsize=7)
    title = ax.set_title('')
    images=[]; obs=scorer._observation
    # 周围对象的占用缓存按 0.2 秒采样，因此每隔两个 0.1 秒仿真步绘制一帧。
    for i in range(0, scorer.proposal_sampling.num_poses+1, 2):
        occ=obs[i]; patches=[]
        for token in occ.tokens:
            if token not in obs.unique_objects: continue
            poly=occ[token]
            if poly.geom_type != 'Polygon' or poly.is_empty: continue
            if poly.centroid.distance(scorer._ego_polygons[1, i].centroid)>55: continue
            hit=bool(event and token == event['object_id'] and i >= event['step'])
            patch=Patch(polygon_xy(poly,origin),closed=True,facecolor='#fb473c' if hit else '#69a4d5',edgecolor='black',lw=.5,alpha=.8,zorder=5)
            ax.add_patch(patch); patches.append(patch)
        ego=Patch(polygon_xy(scorer._ego_polygons[1,i],origin),closed=True,facecolor='#ffb225',edgecolor='black',lw=1,zorder=7)
        ax.add_patch(ego);patches.append(ego)
        t=i*scorer.proposal_sampling.interval_length
        title.set_text(f't={t:.1f}s | '+(f'COLLISION at {event["time_s"]:.1f}s' if event and i>=event['step'] else 'NAVSIM simulated'))
        # 保存当前帧后移除动态车辆，下一帧重新绘制；地图和轨迹保持不变。
        fig.canvas.draw(); rgba=np.asarray(fig.canvas.buffer_rgba()); images.append(Image.fromarray(rgba[:,:,:3].copy()))
        for patch in patches: patch.remove()
    images[0].save(path,save_all=True,append_images=images[1:],duration=200,loop=0,optimize=True)
    plt.close(fig)


def save_candidates(out, results, scene):
    """保存 256 条候选轨迹及模型奖励，并绘制候选轨迹的 BEV 图。"""
    poses=np.asarray(results['all_trajectory']); poses=poses[0] if poses.ndim==4 else poses
    scores=np.asarray(results['final_rewards']).reshape(-1); imitation=np.asarray(results['im_rewards']).reshape(-1)
    if poses.shape != (256,8,3) or len(scores)!=256 or len(imitation)!=256: raise ValueError('unexpected candidate data shape')
    # 这些分数是模型预测的奖励，不是对 256 条轨迹逐条计算的 NAVSIM 评分。
    selected=int(np.argmax(scores)); rank=np.argsort(-scores)
    # 旧 NPZ 未保存分项时留空；新模型输出包含 sim_rewards 时按原始轨迹下标写入。
    sim = np.asarray(results['sim_rewards']).reshape(5, 256) if 'sim_rewards' in results else None
    existing_csv = out / 'candidates.csv'
    if sim is None and existing_csv.is_file():
        with existing_csv.open(newline='') as existing:
            previous = csv.DictReader(existing)
            if 'sim_nc' in (previous.fieldnames or []) and any(row.get('sim_nc') for row in previous):
                raise ValueError('旧 NPZ 不包含模拟奖励，拒绝覆盖已有分项奖励的 candidates.csv')
    with (out/'candidates.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['index','rank','selected','final_reward','im_reward','trajectory_xy_heading_json',
                                    'sim_nc','sim_dac','sim_ep','sim_ttc','sim_comfort'])
        for order, idx in enumerate(rank,1):
            metric_values = sim[:, idx].tolist() if sim is not None else [''] * 5
            w.writerow([int(idx),order,int(idx)==selected,float(scores[idx]),float(imitation[idx]),json.dumps(poses[idx].tolist(),separators=(',',':'))] + metric_values)
    from navsim.visualization.plots import plot_bev_frame
    fig,ax=plot_bev_frame(scene,scene.scene_metadata.num_history_frames-1)
    # 灰色绘制全部候选，红色突出选中轨迹，橙色突出排名第 2 至第 5 的轨迹。
    for tr in poses:
        xy=np.vstack([np.zeros((1,2)),tr[:,:2]]);ax.plot(xy[:,1],xy[:,0],color='#728da6',alpha=.12,lw=.7)
    for idx in rank[:5]:
        tr=poses[idx];xy=np.vstack([np.zeros((1,2)),tr[:,:2]])
        ax.plot(xy[:,1],xy[:,0],color='red' if idx==selected else 'orange',alpha=.9,lw=2 if idx==selected else 1)
    human=scene.get_future_trajectory(num_trajectory_frames=8).poses
    xy=np.vstack([np.zeros((1,2)),human[:,:2]]);ax.plot(xy[:,1],xy[:,0],color='green',lw=2)
    ax.set_title('256 candidates: red selected, orange top 2-5, green expert')
    fig.savefig(out/'candidates_bev.png',dpi=120,bbox_inches='tight');plt.close(fig)
    return {'selected_index':selected,'top_five_indices':[int(x) for x in rank[:5]],'scores_are_model_predictions':True,
            'candidate_navsim_scores_computed':False}


def main():
    """筛选碰撞场景，逐个重放评分并输出分析文件；失败场景单独记录。"""
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--tokens',nargs='*');args=parser.parse_args()
    # 只选评分有效且 NC 为 0 或 0.5 的场景，排除汇总平均行。
    with CSV.open(newline='') as f: rows={r['token']:r for r in csv.DictReader(f) if r['token']!='average' and r['valid'].lower()=='true' and float(r['no_at_fault_collisions']) in (0,.5)}
    tokens=args.tokens if args.tokens else list(rows)
    missing=set(tokens)-rows.keys()
    if missing: raise ValueError(f'tokens not in selected CSV: {sorted(missing)}')
    cfg=OmegaConf.load(CONFIG);sim=instantiate(cfg.simulator);scorer=instantiate(cfg.scorer)
    filt=OmegaConf.load(ROOT/'navsim/planning/script/config/common/scene_filter/navtest.yaml');filt.tokens=tokens;filt.max_scenes=None
    sensors=SensorConfig(cam_f0=[3],cam_l0=[3],cam_l1=False,cam_l2=False,cam_r0=[3],cam_r1=False,cam_r2=False,cam_b0=False,lidar_pc=False)
    loader=SceneLoader(DATA/'navsim_logs/test',DATA/'sensor_blobs/test',instantiate(filt),sensors)
    cache_loader=MetricCacheLoader(CACHE);args.output.mkdir(parents=True,exist_ok=True)
    failures=[]
    for count,token in enumerate(tokens,1):
        out=args.output/token
        try:
            out.mkdir(parents=True,exist_ok=True)
            scene=loader.get_scene_from_token(token);results=load_results(DEFAULT_RESULTS/f'{token}.npz')
            with lzma.open(cache_loader.metric_cache_paths[token],'rb') as f: cache=pickle.load(f)
            # 读取已有模型轨迹与 metric cache，在 CPU 上重放评分，不重新推理。
            result=pdm_score(cache,Trajectory(results['trajectory']),sim.proposal_sampling,sim,scorer)
            if not np.isclose(result.no_at_fault_collisions,float(rows[token]['no_at_fault_collisions'])): raise ValueError('recomputed NC differs from CSV')
            events=collision_events(scorer);first=events[0] if events else None
            # 检查重建的首次碰撞时刻是否与 NAVSIM 评分器记录一致。
            expected=scorer.time_to_at_fault_collision(1)
            if first is None or not np.isclose(first['time_s'],expected): raise ValueError(f'collision reconstruction mismatch: {first} vs {expected}')
            human=scene.get_future_trajectory(num_trajectory_frames=8).poses
            plot_camera(scene,results['trajectory'],human,[],0.0,out/'front_cameras_trajectory.png')
            plot_bev(scene,results['trajectory'],human,[],out/'trajectory_bev.png')
            candidates=save_candidates(out,results,scene)
            write_timeseries(out/'simulation_states.csv',scorer,first)
            render_gif(out/'simulation_bev.gif',scene,scorer,results['trajectory'],human,first)
            # 将独热编码导航指令转换为左、直行、右或未知。
            command=scene.frames[scene.scene_metadata.num_history_frames-1].ego_status.driving_command
            cmd_index=int(np.argmax(command));cmd=['left','straight','right','unknown'][cmd_index]
            summary={'token':token,'log_name':scene.scene_metadata.log_name,'navigation_command':cmd,
                     'navigation_command_one_hot':np.asarray(command).tolist(),'metrics':asdict(result),
                     'collisions':events,'candidates':candidates,
                     'notes':['Simulation replays non-reactive cached traffic; it is not a real-world collision.',
                              'Camera overlay assumes flat road at lidar z=0; alignment has not been calibrated.',
                              'Other-agent occupancy cache is sampled every 0.2 s.','CSV polygon distance is edge-to-edge distance in meters.']}
            (out/'analysis.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
            print(f'[{count}/{len(tokens)}] OK {token} NC={result.no_at_fault_collisions} collision={first["time_s"]}s',flush=True)
        except Exception as exc:
            # 单个场景失败不影响其余场景；最后将异常写入 failures.json。
            failures.append({'token':token,'error':str(exc),'traceback':traceback.format_exc()})
            print(f'[{count}/{len(tokens)}] FAILED {token}: {exc}',flush=True)
    (args.output/'failures.json').write_text(json.dumps(failures,indent=2,ensure_ascii=False)+'\n')
    print(f'Finished: {len(tokens)-len(failures)}/{len(tokens)} successful; {len(failures)} failed',flush=True)
    if failures: raise SystemExit(1)

if __name__=='__main__':main()
