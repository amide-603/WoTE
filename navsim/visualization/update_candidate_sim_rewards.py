"""对已有碰撞场景重跑 WoTE，向候选轨迹 CSV 添加模型预测的五项模拟奖励。

输入为原有 candidates.csv、已保存的模型 NPZ 和 NAVSIM 场景；输出保留原 CSV
各列和顺序，仅追加 sim_nc、sim_dac、sim_ep、sim_ttc、sim_comfort 五列。
"""
import argparse
import csv
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from navsim.agents.WoTE.configs.default import WoTEConfig
from navsim.agents.WoTE.WoTE_agent import WoTEAgent
from navsim.common.dataloader import SceneLoader
from navsim.visualization.visualize_wote_scene import DEFAULT_RESULTS

SOURCE_DIR = Path('/mnt/sfs_turbo/gaoxuan_gac/WoTE/exp/eval/WoTE/collision_analysis_190')
SIM_COLUMNS = ('sim_nc', 'sim_dac', 'sim_ep', 'sim_ttc', 'sim_comfort')


def prepare_rows(source, token, results):
    """验证重跑与原始结果对应，并按原始轨迹下标补充奖励。"""
    with (source / token / 'candidates.csv').open(newline='') as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if len(rows) != 256 or {int(row['index']) for row in rows} != set(range(256)):
        raise ValueError(f'{token}: 原 CSV 不是完整的 256 条候选轨迹')
    sim = np.asarray(results['sim_rewards'])
    if sim.shape != (1, 5, 256) or not np.isfinite(sim).all() or np.any((sim < 0) | (sim > 1)):
        raise ValueError(f'{token}: 模拟奖励形状或取值异常: {sim.shape}')
    # 通过原始 NPZ 核对轨迹和综合奖励，避免将新推理结果错误地写到旧轨迹上。
    with np.load(DEFAULT_RESULTS / f'{token}.npz', allow_pickle=False) as saved:
        for key in ('trajectory', 'all_trajectory', 'final_rewards', 'im_rewards'):
            old, new = saved[key], results[key]
            if old.shape != new.shape or not np.allclose(old, new, rtol=1e-3, atol=1e-3):
                delta = float(np.max(np.abs(old - new))) if old.shape == new.shape else None
                raise ValueError(f'{token}: {key} 与原结果不一致，最大差异={delta}')
        if int(np.argmax(saved['final_rewards'])) != int(np.argmax(results['final_rewards'])):
            raise ValueError(f'{token}: 重跑选中的轨迹与原结果不同')
    for row in rows:
        index = int(row['index'])
        for metric, value in zip(SIM_COLUMNS, sim[0, :, index]):
            row[metric] = str(float(value))
    return fieldnames + [name for name in SIM_COLUMNS if name not in fieldnames], rows


def atomic_write(path, fieldnames, rows):
    """写入同目录临时文件，成功后原子替换，避免中断时留下半个 CSV。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    original_mode = path.stat().st_mode & 0o777 if path.exists() else 0o664
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', newline='', encoding='utf-8', dir=path.parent, delete=False) as stream:
            temp_path = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.chmod(temp_path, original_mode)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE_DIR)
    parser.add_argument('--output', type=Path, default=SOURCE_DIR)
    parser.add_argument('--tokens', nargs='*')
    parser.add_argument('--device', default='npu:0')
    args = parser.parse_args()
    tokens = args.tokens or sorted(p.name for p in args.source.iterdir() if p.is_dir() and (p/'candidates.csv').is_file())
    if not tokens:
        raise ValueError('未找到需要处理的候选轨迹 CSV')
    if args.device.startswith('npu'):
        import torch_npu  # 注册 PyTorch NPU 后端。
        torch.backends.mha.set_fastpath_enabled(False)
    config = WoTEConfig()
    agent = WoTEAgent(config, config.trajectory_sampling, lr=1e-4, checkpoint_path=os.environ['WOTE_CHECKPOINT_PATH'])
    agent.initialize()
    agent.is_eval = True
    agent.to(args.device)
    # 一次加载筛选结果，场景输入使用与原推理相同的三摄像头和激光雷达配置。
    root = Path(os.environ['NAVSIM_DEVKIT_ROOT'])
    data = Path(os.environ['OPENSCENE_DATA_ROOT'])
    filt = OmegaConf.load(root/'navsim/planning/script/config/common/scene_filter/navtest.yaml')
    filt.tokens = tokens
    filt.max_scenes = None
    loader = SceneLoader(data/'navsim_logs/test', data/'sensor_blobs/test', instantiate(filt), agent.get_sensor_config())
    missing = set(tokens) - set(loader.tokens)
    if missing:
        raise ValueError(f'数据集中缺少场景: {sorted(missing)}')
    errors = []
    for number, token in enumerate(tokens, 1):
        try:
            _, results = agent.compute_trajectory_with_results(loader.get_agent_input_from_token(token))
            fieldnames, rows = prepare_rows(args.source, token, results)
            atomic_write(args.output/token/'candidates.csv', fieldnames, rows)
            print(f'[{number}/{len(tokens)}] OK {token}', flush=True)
        except Exception as exc:
            errors.append((token, str(exc)))
            print(f'[{number}/{len(tokens)}] FAILED {token}: {exc}', flush=True)
    print(f'完成 {len(tokens)-len(errors)}/{len(tokens)}；失败 {len(errors)}', flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
