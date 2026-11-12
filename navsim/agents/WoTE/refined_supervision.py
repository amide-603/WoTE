"""为当前 refine 轨迹在线生成 PDM 与未来 BEV 监督。"""

from pathlib import Path
import os

import numpy as np
import torch


# 评分核心独立于 NPU，供 CPU 工作进程复用，原调用入口保持兼容。
from navsim.agents.WoTE.pdm_scoring import score_candidates_with_reference


class RefinedSupervision:
    """使用场景 metric cache，为每次前向生成的轨迹构造监督标签。"""

    def __init__(self, config, slice_indices):
        self.config = config
        self.future_pose_index = config.future_idx - slice_indices[0] - 1
        self._metric_loader = None
        self._sampling = None
        self._simulator = None
        self._scorer = None
        # 使用测速选出的配置；环境变量可覆盖配置，设为 0 时回退串行评分。
        self._score_workers = int(os.environ.get("WOTE_SCORE_WORKERS", getattr(config, "score_workers", 16)))
        self._score_fast = os.environ.get(
            "WOTE_SCORE_FAST", "1" if getattr(config, "score_fast_trajectory", True) else "0"
        ) == "1"
        self._score_cache_size = int(os.environ.get(
            "WOTE_SCORE_CACHE_SIZE", getattr(config, "score_cache_size", 128)
        ))
        if self._score_workers < 0:
            raise ValueError("WOTE_SCORE_WORKERS 不能为负数。")
        if self._score_cache_size < 0:
            raise ValueError("WOTE_SCORE_CACHE_SIZE 不能为负数。")
        self._score_pool = None

    def _initialize_scorer(self):
        # 延迟加载大体积仿真依赖，推理阶段无需 metric cache。
        from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
        from navsim.common.dataloader import MetricCacheLoader
        from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
        from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator

        cache_path = getattr(self.config, "metric_cache_path", None) or os.environ.get("WOTE_METRIC_CACHE_PATH")
        if not cache_path:
            raise RuntimeError("在线 PDM 评分需要配置 metric_cache_path 或环境变量 WOTE_METRIC_CACHE_PATH，指向训练集 metric cache 根目录。")
        try:
            self._metric_loader = MetricCacheLoader(Path(cache_path))
        except (OSError, IndexError) as exc:
            raise RuntimeError(f"无法读取 metric cache 元数据: {cache_path}") from exc
        # 参数与标准评测 YAML 一致，包括进度阈值 5 米。
        self._sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
        self._simulator = PDMSimulator(self._sampling)
        self._scorer = PDMScorer(self._sampling, PDMScorerConfig(progress_distance_threshold=5.0))

    def score(self, trajectories, tokens, include_total=False):
        # 仿真及标签不可微；仅将 detach 的轨迹复制到 CPU。
        if self._metric_loader is None:
            self._initialize_scorer()
        if len(tokens) != trajectories.shape[0]:
            raise ValueError("场景 token 数量与轨迹 batch 不一致。")
        trajectories_cpu = trajectories.detach().float().cpu().numpy()
        if not np.isfinite(trajectories_cpu).all():
            raise ValueError("refine 轨迹包含非有限值，无法在线评分。")
        keys = ("no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
                "time_to_collision_within_bound", "comfort")
        # 在派发前检查所有场景，避免部分评分成功后才发现缺少缓存。
        for token in tokens:
            if token not in self._metric_loader.metric_cache_paths:
                raise RuntimeError(f"场景 {token} 缺少 metric cache，无法计算在线 PDM 标签。")
        if self._score_workers:
            # 每张 NPU 的场景并发评分，CPU 子进程常驻并独立维护评分器。
            if self._score_pool is None:
                from navsim.agents.WoTE.score_pool import SceneScorePool
                self._score_pool = SceneScorePool(self._score_workers, fast=self._score_fast,
                                                  cache_size=self._score_cache_size)
            paths = [self._metric_loader.metric_cache_paths[token] for token in tokens]
            if include_total:
                batch_results = self._score_pool.score(tokens, paths, trajectories_cpu, include_total=True)
            else:
                batch_results = self._score_pool.score(tokens, paths, trajectories_cpu)
        else:
            batch_results = []
            for token, proposals in zip(tokens, trajectories_cpu):
                try:
                    cache = self._metric_loader.get_from_token(token)
                    # 训练标签采用与单轨迹评测相同的参考轨迹评分口径。
                    if include_total:
                        values = score_candidates_with_reference(cache, proposals, self._sampling, self._simulator, self._scorer, include_total=True)
                    else:
                        values = score_candidates_with_reference(cache, proposals, self._sampling, self._simulator, self._scorer)
                except Exception as exc:
                    raise RuntimeError(f"场景 {token} 的在线 PDM 评分失败。") from exc
                batch_results.append(values)
        if len(batch_results) != len(tokens):
            raise ValueError("在线 PDM 评分结果数量与场景数量不一致。")
        batch_scores, batch_totals = [], []
        for token, proposals, result in zip(tokens, trajectories_cpu, batch_results):
            values, totals = result if include_total else (result, None)
            if values.shape != (len(keys), proposals.shape[0]) or not np.isfinite(values).all():
                raise ValueError(f"场景 {token} 的在线 PDM 分数形状或数值无效。")
            batch_scores.append(values)
            if include_total:
                if totals.shape != (proposals.shape[0],) or not np.isfinite(totals).all():
                    raise ValueError(f"场景 {token} 的在线 PDM 总分形状或数值无效。")
                batch_totals.append(totals)
        # 保持原损失函数期望的 [batch, 时间切片, 指标, 候选] 形状。
        scores = torch.as_tensor(np.stack(batch_scores), device=trajectories.device).unsqueeze(1)
        if include_total:
            return scores, torch.as_tensor(np.stack(batch_totals), device=trajectories.device)
        return scores

    def future_bev_target(self, targets, trajectories):
        # 旧缓存已覆盖背景像素，必须明确提示重建，不能静默沿用旧框。
        if "fut_bev_background" not in targets:
            raise RuntimeError("目标缓存缺少 fut_bev_background，请重建 WoTE_target_refined_v1 目标缓存；已有特征缓存可复用。")
        from navsim.agents.WoTE.WoTE_targets import WoTETargetBuilder

        background = targets["fut_bev_background"].detach().cpu()
        indices = targets["sampled_trajs_index"].detach().long().cpu()
        poses = trajectories.detach().float().cpu()
        if not 0 <= self.future_pose_index < poses.shape[2]:
            raise ValueError("未来 BEV 帧索引超出 refine 轨迹范围。")
        # 复用原始目标构建器的 OpenCV 栅格化规则，保证标签尺寸与方向一致。
        painter = WoTETargetBuilder.__new__(WoTETargetBuilder)
        painter._config = self.config
        maps = []
        for batch_idx in range(poses.shape[0]):
            sample_maps = []
            for traj_idx in indices[batch_idx]:
                x, y, heading = poses[batch_idx, traj_idx, self.future_pose_index].tolist()
                box = [x, y, 0.0, 4.0, 2.0, 1.8, heading]
                sample_maps.append(painter._add_ego_box_to_bev_map(background[batch_idx].clone(), box))
            maps.append(torch.stack(sample_maps))
        return torch.stack(maps).to(trajectories.device)
