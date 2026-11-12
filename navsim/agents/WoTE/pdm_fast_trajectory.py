"""按现有 nuPlan 规则批量变换和插值候选轨迹。"""

import numpy as np
from scipy.interpolate import interp1d


def trajectories_to_state_arrays(proposals, initial, sampling):
    """批量生成与原 InterpolatedTrajectory 等价的状态数组。"""
    from nuplan.common.geometry.compute import principal_value
    from nuplan.common.geometry.convert import matrix_from_pose
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex

    proposals = np.asarray(proposals, dtype=np.float64)
    if proposals.ndim != 3 or proposals.shape[1:] != (8, 3):
        raise ValueError('候选轨迹应为 [数量, 8, 3]，采样间隔 0.5 秒。')
    count = len(proposals)
    origin = initial.rear_axle
    # 保持原 3×3 坐标变换及 atan2 朝向计算，兼容跨 ±π 场景。
    relative = np.zeros((count, 8, 3, 3), dtype=np.float64)
    relative[..., 0, 0] = np.cos(proposals[..., 2])
    relative[..., 0, 1] = -np.sin(proposals[..., 2])
    relative[..., 1, 0] = np.sin(proposals[..., 2])
    relative[..., 1, 1] = np.cos(proposals[..., 2])
    relative[..., 0, 2] = proposals[..., 0]
    relative[..., 1, 2] = proposals[..., 1]
    relative[..., 2, 2] = 1
    absolute = matrix_from_pose(origin) @ relative
    heading = np.arctan2(absolute[..., 1, 0], absolute[..., 0, 0])
    x, y = absolute[..., 0, 2], absolute[..., 1, 2]
    rear_offset = initial.car_footprint.vehicle_parameters.rear_axle_to_center
    # 原 EgoState 会先由后轴构造中心再取回后轴，保留相同浮点运算。
    x = (x + rear_offset * np.cos(heading)) - rear_offset * np.cos(heading)
    y = (y + rear_offset * np.sin(heading)) - rear_offset * np.sin(heading)

    # 时间戳采用原实现的秒转微秒截断和裁剪，避免大时间戳的浮点差异。
    start_s = initial.time_point.time_s
    source_s = np.arange(0.0, 4.0, 0.5) + 0.5
    source_s += start_s
    source_times = np.array([initial.time_us] + [int(t * 1e6) for t in source_s], dtype=np.int64)
    target_s = np.arange(0.0, sampling.time_horizon + sampling.interval_length, sampling.interval_length)
    target_s += start_s
    target_times = np.clip([int(t * 1e6) for t in target_s], source_times[0], source_times[-1])

    # to_split_state 只插值后轴速度、加速度和转角；未来预测状态的这些值均为零。
    linear = np.zeros((count, 9, 7), dtype=np.float64)
    linear[:, 0, :] = [origin.x, origin.y,
                       initial.dynamic_car_state.rear_axle_velocity_2d.x,
                       initial.dynamic_car_state.rear_axle_velocity_2d.y,
                       initial.dynamic_car_state.rear_axle_acceleration_2d.x,
                       initial.dynamic_car_state.rear_axle_acceleration_2d.y,
                       initial.tire_steering_angle]
    linear[:, 1:, 0] = x
    linear[:, 1:, 1] = y
    angles = np.concatenate([np.full((count, 1), origin.heading), heading], axis=1)
    interp_linear = interp1d(source_times, linear, axis=1)(target_times)
    interp_heading = principal_value(interp1d(source_times, np.unwrap(angles, axis=1), axis=1)(target_times))
    # from_split_state 再次由后轴构造中心并取回后轴，复现其坐标舍入。
    ix, iy = interp_linear[..., 0], interp_linear[..., 1]
    ix = (ix + rear_offset * np.cos(interp_heading)) - rear_offset * np.cos(interp_heading)
    iy = (iy + rear_offset * np.sin(interp_heading)) - rear_offset * np.sin(interp_heading)
    output = np.zeros((count, len(target_times), StateIndex.size()), dtype=np.float64)
    output[..., StateIndex.X] = ix
    output[..., StateIndex.Y] = iy
    output[..., StateIndex.HEADING] = interp_heading
    output[..., StateIndex.VELOCITY_X:StateIndex.STEERING_ANGLE + 1] = interp_linear[..., 2:]
    return output
