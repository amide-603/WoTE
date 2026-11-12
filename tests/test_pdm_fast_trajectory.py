"""验证批量轨迹转换与原 nuPlan 状态插值的关键边界。"""

import unittest
import numpy as np
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.agents.WoTE.pdm_fast_trajectory import trajectories_to_state_arrays
from navsim.common.dataclasses import Trajectory
from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory


class FastTrajectoryTest(unittest.TestCase):
    def test_matches_interpolator_with_heading_wrap(self):
        # 使用非零车速、加速度、转角和跨 ±π 航向验证所有状态维度。
        ego = EgoState.build_from_rear_axle(
            rear_axle_pose=StateSE2(427.2, -31.5, 3.08),
            rear_axle_velocity_2d=StateVector2D(4.1, -0.2),
            rear_axle_acceleration_2d=StateVector2D(0.7, 0.1),
            tire_steering_angle=0.18,
            time_point=TimePoint(1622345678912345),
            vehicle_parameters=get_pacifica_parameters(),
        )
        sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
        rng = np.random.default_rng(9)
        poses = rng.normal(size=(6, 8, 3)).astype(np.float32)
        poses[0, :, 2] = np.linspace(2.9, 3.5, 8)
        poses[1, :, 2] = np.linspace(-2.9, -3.5, 8)
        expected = np.stack([
            get_trajectory_as_array(transform_trajectory(Trajectory(pose), ego), sampling, ego.time_point)
            for pose in poses
        ])
        actual = trajectories_to_state_arrays(poses, ego, sampling)
        np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=0)

    def test_rejects_invalid_trajectory_shape(self):
        # 不支持的轨迹采样应明确报错，不能静默输出错误时间轴。
        with self.assertRaisesRegex(ValueError, '8, 3'):
            trajectories_to_state_arrays(np.zeros((2, 7, 3)), None, None)


if __name__ == '__main__':
    unittest.main()
