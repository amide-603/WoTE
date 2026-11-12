"""读取已保存的 WoTE 轨迹，生成三摄像头轨迹图和 BEV 图，无需重新推理。

相机投影参考 NAVSIM 社区示例，将平面局部轨迹位置视为激光雷达坐标系中的点。
路面高度采用可配置的假设值；使用前应结合实际场景检查投影是否对齐。
"""
import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SensorConfig, Trajectory
from navsim.visualization.bev import add_trajectory_to_bev_ax
from navsim.visualization.config import TRAJECTORY_CONFIG
from navsim.visualization.plots import plot_bev_frame

DEFAULT_RESULTS = Path("/mnt/sfs_turbo/gaoxuan_gac/WoTE/exp/eval/WoTE/navtest/model_results/2026.10.27.23.50.24")
DEFAULT_OUTPUT = Path("/mnt/sfs_turbo/gaoxuan_gac/WoTE/exp/eval/WoTE/visualizations")

# 候选轨迹筛选规则尚未实现，因此默认关闭候选轨迹的绘制。
DRAW_SELECTED_CANDIDATES = False


def select_candidate_indices(results):
    """预留筛选接口：后续可返回 all_trajectory 中需要绘制的轨迹下标。"""
    return []


def load_results(path):
    """读取单场景 NPZ，并检查最终轨迹是否为有效的 8×3 数组。"""
    with np.load(path, allow_pickle=False) as archive:
        results = {name: archive[name] for name in archive.files}
    poses = np.asarray(results["trajectory"])
    if poses.ndim == 3 and poses.shape[0] == 1:
        poses = poses[0]
    if poses.shape != (8, 3) or not np.isfinite(poses).all():
        raise ValueError(f"Invalid selected trajectory in {path}: {poses.shape}")
    results["trajectory"] = poses
    return results


def project_trajectory(camera, poses, road_height):
    """按假设路面高度将平面轨迹投影到单个相机；无效点保留为 NaN。"""
    rotation = np.asarray(camera.sensor2lidar_rotation)
    translation = np.asarray(camera.sensor2lidar_translation).reshape(3)
    intrinsic = np.asarray(camera.intrinsics)
    if rotation.shape != (3, 3) or intrinsic.shape != (3, 3):
        raise ValueError("Camera rotation and intrinsics must be 3x3")
    # 轨迹只提供平面位置，补入假设的高度后转换到相机坐标系。
    xyz = np.column_stack((poses[:, :2], np.full(len(poses), road_height)))
    camera_points = (xyz - translation) @ np.linalg.inv(rotation).T
    homogeneous = camera_points @ intrinsic.T
    # 只投影位于相机前方的点，避免深度为零或负值导致无效结果。
    valid = (camera_points[:, 2] > 1e-3) & (homogeneous[:, 2] > 1e-3)
    pixels = np.full((len(poses), 2), np.nan)
    pixels[valid] = homogeneous[valid, :2] / homogeneous[valid, 2:3]
    return pixels


def candidate_trajectories(results):
    """按预留筛选函数获取候选轨迹；开关关闭时不绘制任何候选轨迹。"""
    if not DRAW_SELECTED_CANDIDATES:
        return []
    candidates = np.asarray(results["all_trajectory"])
    if candidates.ndim == 4 and candidates.shape[0] == 1:
        candidates = candidates[0]
    if candidates.ndim != 3 or candidates.shape[1:] != (8, 3):
        raise ValueError(f"Invalid candidate trajectory shape: {candidates.shape}")
    indices = select_candidate_indices(results)
    for index in indices:
        if not isinstance(index, (int, np.integer)) or not 0 <= index < len(candidates):
            raise ValueError(f"Invalid candidate index: {index}")
    return [(int(index), candidates[index]) for index in indices]


def plot_camera(scene, selected, human, candidates, road_height, output):
    """拼接左前、正前、右前相机画面，并叠加模型、专家和可选候选轨迹。"""
    frame = scene.frames[scene.scene_metadata.num_history_frames - 1]
    # 使用与 WoTE 输入特征相同的相机顺序及裁剪范围。
    specs = [(frame.cameras.cam_l0, 416), (frame.cameras.cam_f0, 0),
             (frame.cameras.cam_r0, 416)]
    widths = [camera.image.shape[1] - 2 * crop for camera, crop in specs]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4),
                             gridspec_kw={"width_ratios": widths, "wspace": 0})
    paths = [(poses, "#5aa9ff", 1.2, 0.5) for _, poses in candidates]
    paths += [(human, "#50b858", 2.5, 0.95), (selected, "#ed6255", 2.5, 0.95)]
    for ax, (camera, crop) in zip(axes, specs):
        image = camera.image[28:-28, crop:camera.image.shape[1] - crop if crop else None]
        ax.imshow(image)
        height, width = image.shape[:2]
        for poses, color, linewidth, alpha in paths:
            # 对稀疏轨迹点插值，减少轨迹跨越相机画面边缘时出现的断裂。
            points = np.concatenate([np.zeros((1, 3)), poses], axis=0)
            steps = np.linspace(0, len(points) - 1, (len(points) - 1) * 20 + 1)
            dense = np.column_stack([np.interp(steps, np.arange(len(points)), points[:, i]) for i in range(3)])
            pixels = project_trajectory(camera, dense, road_height)
            pixels[:, 0] -= crop
            pixels[:, 1] -= 28
            ax.plot(pixels[:, 0], pixels[:, 1], color=color, linewidth=linewidth, alpha=alpha, clip_on=True)
        ax.set_xlim(-0.5, width - 0.5)
        ax.set_ylim(height - 0.5, -0.5)
        ax.axis("off")
    fig.legend(handles=[Line2D([0], [0], color="#ed6255", label="WoTE"),
                        Line2D([0], [0], color="#50b858", label="Expert")],
               loc="upper center", ncol=2)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_bev(scene, selected, human, candidates, output):
    """在当前场景的俯视地图上绘制并保存轨迹。"""
    frame_index = scene.scene_metadata.num_history_frames - 1
    fig, ax = plot_bev_frame(scene, frame_index)
    candidate_handles = []
    colors = plt.get_cmap("tab20")
    for position, (index, poses) in enumerate(candidates):
        xy = np.vstack([np.zeros((1, 2)), poses[:, :2]])
        color = colors(position % 20)
        ax.plot(xy[:, 1], xy[:, 0], color=color, alpha=0.85, linewidth=1.5)
        candidate_handles.append(Line2D([0], [0], color=color, label=f"Candidate {index}"))
    add_trajectory_to_bev_ax(ax, Trajectory(human), TRAJECTORY_CONFIG["human"])
    add_trajectory_to_bev_ax(ax, Trajectory(selected), TRAJECTORY_CONFIG["agent"])
    ax.legend(handles=[Line2D([0], [0], color="#DE7061", marker="o", label="WoTE"),
                       Line2D([0], [0], color="#59a14f", marker="o", label="Expert")] + candidate_handles,
              loc="upper right", fontsize="small")
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    """按场景 token 读取数据和已保存的模型输出，分别生成两张图片。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tokens", nargs="+", help="NAVSIM scene tokens")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--road-height", type=float, default=0.0,
                        help="Assumed ground height in lidar coordinates (meters)")
    parser.add_argument("--candidate-indices", type=int, nargs="+", default=None,
                        help="需要显示的候选轨迹编号，例如 14 41 144（从 0 开始）")
    parser.add_argument("--bev-only", action="store_true", help="只生成 BEV 图，不加载相机图片")
    args = parser.parse_args()
    root = Path(os.environ["NAVSIM_DEVKIT_ROOT"])
    data = Path(os.environ["OPENSCENE_DATA_ROOT"])
    filter_cfg = OmegaConf.load(root / "navsim/planning/script/config/common/scene_filter/navtest.yaml")
    filter_cfg.tokens = args.tokens
    filter_cfg.max_scenes = None
    # 只加载当前时刻的三张前向相机图片，无需加载模型或激光雷达。
    history_index = filter_cfg.num_history_frames - 1
    camera_indices = False if args.bev_only else [history_index]
    sensors = SensorConfig(cam_f0=camera_indices, cam_l0=camera_indices, cam_l1=False,
                           cam_l2=False, cam_r0=camera_indices, cam_r1=False,
                           cam_r2=False, cam_b0=False, lidar_pc=False)
    loader = SceneLoader(data / "navsim_logs/test", data / "sensor_blobs/test", instantiate(filter_cfg), sensors)
    missing = set(args.tokens) - set(loader.tokens)
    if missing:
        raise ValueError("Tokens not found in navtest: " + ", ".join(sorted(missing)))
    for token in args.tokens:
        results = load_results(args.results_dir / f"{token}.npz")
        scene = loader.get_scene_from_token(token)
        # 专家轨迹取未来 8 个点，与 WoTE 的 4 秒预测时域保持一致。
        human = scene.get_future_trajectory(num_trajectory_frames=8).poses
        selected = results["trajectory"]
        if args.candidate_indices is None:
            candidates = candidate_trajectories(results)
        else:
            poses = np.asarray(results["all_trajectory"])
            if poses.ndim == 4 and poses.shape[0] == 1:
                poses = poses[0]
            if poses.ndim != 3 or poses.shape[1:] != (8, 3):
                raise ValueError(f"Invalid candidate trajectory shape: {poses.shape}")
            indices = list(dict.fromkeys(args.candidate_indices))
            if any(index < 0 or index >= len(poses) for index in indices):
                raise ValueError(f"Candidate indices must be between 0 and {len(poses) - 1}")
            candidates = [(index, poses[index]) for index in indices]
        output = args.output / token
        output.mkdir(parents=True, exist_ok=True)
        if not args.bev_only:
            plot_camera(scene, selected, human, candidates, args.road_height, output / "front_cameras_trajectory.png")
        plot_bev(scene, selected, human, candidates, output / "trajectory_bev.png")
        print(f"{token}: {output}", flush=True)


if __name__ == "__main__":
    main()
