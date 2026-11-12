"""PDM 场景批量评分；供训练进程及独立 CPU 工作进程复用。"""

import numpy as np


def score_candidates_with_reference(cache, proposals, sampling, simulator, scorer,
                                    reference_states=None, fast_trajectory=False, include_total=False):
    """批量仿真，并重现标准评测中每条候选与参考轨迹的两两评分。"""
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
        BBCoordsIndex, EgoAreaIndex, MultiMetricIndex, WeightedMetricIndex,
    )

    # 所有候选共享一次仿真，但每条候选单独与参考轨迹归一化进度。
    initial = cache.ego_state
    # 同一场景的参考轨迹不随模型变化，可复用预先转换的状态数组。
    reference = (get_trajectory_as_array(cache.trajectory, sampling, initial.time_point)
                 if reference_states is None else reference_states)
    if fast_trajectory:
        # 批量转换候选，避免逐条构造轨迹与插值状态对象。
        from navsim.agents.WoTE.pdm_fast_trajectory import trajectories_to_state_arrays
        candidates = trajectories_to_state_arrays(proposals, initial, sampling)
        proposal_states = np.concatenate([reference[None], candidates], axis=0)
    else:
        candidates = [get_trajectory_as_array(transform_trajectory(Trajectory(traj), initial),
                                              sampling, initial.time_point) for traj in proposals]
        proposal_states = np.stack([reference, *candidates])
    states = simulator.simulate_proposals(proposal_states, initial)
    scorer.score_proposals(states, cache.observation, cache.centerline,
                           cache.route_lane_ids, cache.drivable_area_map)
    multi = scorer._multi_metrics.copy()
    weighted = scorer._weighted_metrics.copy()

    # 标准评测目前沿候选维计算方向指标；重现其双轨迹行为，保持旧结果可比。
    centers = scorer._ego_coords[:, :, BBCoordsIndex.CENTER]
    oncoming = np.zeros(centers.shape[:2], dtype=np.float64)
    oncoming[:, 1:] = np.linalg.norm(centers[:, 1:] - centers[:, :-1], axis=-1)
    oncoming[~scorer._ego_areas[:, :, EgoAreaIndex.ONCOMING_TRAFFIC]] = 0
    horizon = int(scorer._config.driving_direction_horizon / sampling.interval_length)
    distances = np.r_[oncoming[0].max(),
                      (oncoming[1:] + (oncoming[0] if horizon >= 1 else 0)).max(axis=-1)]
    direction = np.where(distances < scorer._config.driving_direction_compliance_threshold, 1.0,
                         np.where(distances < scorer._config.driving_direction_violation_threshold, 0.5, 0.0))
    multi[MultiMetricIndex.DRIVING_DIRECTION] = direction
    multiply = multi.prod(axis=0)
    raw = scorer._progress_raw * multiply
    denominator = np.maximum(raw[0], raw[1:])
    progress = np.ones(len(proposals), dtype=np.float64)
    valid = denominator > scorer._config.progress_distance_threshold
    progress[valid] = raw[1:][valid] / denominator[valid]
    progress[~valid & (multiply[1:] == 0)] = 0
    # 输出顺序与现有五个预测头一致。
    values = np.stack([multi[MultiMetricIndex.NO_COLLISION, 1:],
                       multi[MultiMetricIndex.DRIVABLE_AREA, 1:], progress,
                       weighted[WeightedMetricIndex.TTC, 1:],
                       weighted[WeightedMetricIndex.COMFORTABLE, 1:]]).astype(np.float32)
    if not include_total:
        return values
    weighted[WeightedMetricIndex.PROGRESS, 1:] = progress
    weights = scorer._config.weighted_metrics_array
    totals = multiply[1:] * (weighted[:, 1:] * weights[:, None]).sum(axis=0) / weights.sum()
    return values, totals.astype(np.float32)
