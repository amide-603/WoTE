"""验证 refine 轨迹与训练监督保持一致。"""

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch

from navsim.agents.WoTE.WoTE_loss import compute_im_reward_loss
from navsim.agents.WoTE.WoTE_model import WoTEModel
from navsim.agents.WoTE.configs.default import WoTEConfig
from navsim.agents.WoTE.refined_supervision import RefinedSupervision


class RefinedSupervisionTest(unittest.TestCase):
    def test_trajectory_encoding_preserves_offset_gradient(self):
        # 用轻量替身验证训练和推理都编码 refine 轨迹且可更新 offset。
        for is_eval in (False, True):
            model = WoTEModel.__new__(WoTEModel)
            torch.nn.Module.__init__(model)
            model.trajectory_anchors = torch.nn.Parameter(torch.zeros(2, 8, 3), requires_grad=False)
            model.is_eval = is_eval
            model.use_agent_loss = False
            model.use_map_loss = False
            offset = torch.ones(1, 2, 8, 3, requires_grad=True)
            model._process_backbone_features = Mock(return_value=(None, torch.zeros(1, 1, 1)))
            model._get_ego_status_feature = Mock(return_value=torch.zeros(1, 1, 1))
            model.encode_traj_into_ego_feat = Mock(side_effect=lambda status, traj, batch: (traj, traj.shape[1]))
            model._predict_offset = Mock(return_value={"trajectory_offset": offset})
            features = {name: torch.zeros(1, 1) for name in ("camera_feature", "lidar_feature", "status_feature")}
            output = model.extract_trajectory_feature(features)
            self.assertTrue(torch.equal(output["ego_feat"], offset))
            output["ego_feat"].sum().backward()
            self.assertTrue(torch.equal(offset.grad, torch.ones_like(offset)))

    def test_future_bev_and_label_gradients(self):
        # 两个位置必须生成不同自车框，标签不应反向传播到轨迹。
        config = WoTEConfig()
        helper = RefinedSupervision(config, [3])
        poses = torch.zeros(1, 2, 8, 3, requires_grad=True)
        with torch.no_grad():
            poses[0, :, 7, 0] = torch.tensor([8.0, 14.0])
        targets = {
            "fut_bev_background": torch.zeros(1, config.bev_pixel_height, config.bev_pixel_width, dtype=torch.long),
            "sampled_trajs_index": torch.tensor([[0, 1]]),
        }
        maps = helper.future_bev_target(targets, poses)
        self.assertFalse(maps.requires_grad)
        self.assertTrue((maps[0, 0] == config.ego_box_map_idx).any())
        self.assertFalse(torch.equal(maps[0, 0], maps[0, 1]))
        self.assertTrue(torch.equal(targets["fut_bev_background"], torch.zeros_like(targets["fut_bev_background"])))

        # 模仿标签按每个样本对应的 refine 轨迹生成，且不应传播标签梯度。
        logits = torch.randn(1, 2, requires_grad=True)
        loss = compute_im_reward_loss({"trajectory": torch.zeros(1, 1, 8, 3)}, logits.softmax(-1), poses.detach())
        loss.backward()
        self.assertIsNone(poses.grad)
        self.assertIsNotNone(logits.grad)

    def test_future_injection_uses_each_sample_position(self):
        # 精确检查不同 batch 的位置并验证双线性权重可反向传播。
        model = WoTEModel.__new__(WoTEModel)
        torch.nn.Module.__init__(model)
        positions = torch.tensor([[[[4.1, 1.2]]], [[[12.1, 1.2]]]], requires_grad=True)
        bev = torch.zeros(2, 64, 1)
        ego = torch.ones(2, 1, 1)
        output = model._inject_fut_ego_into_bev(bev, ego, 1, positions, fut_idx=1)
        self.assertFalse(torch.equal(output[0], output[1]))
        (output * torch.arange(64).reshape(1, 64, 1)).sum().backward()
        self.assertTrue(torch.all(positions.grad.abs().sum(dim=(-1, -2, -3)) > 0))


    def test_cache_reuses_features_and_rejects_old_targets(self):
        # 新目标缓存缺失时拒绝旧标签，并在重建时复用已有特征文件。
        from navsim.planning.training.dataset import CacheOnlyDataset, Dataset, dump_feature_target_to_pickle
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            token_dir = root / "log" / "scene"
            token_dir.mkdir(parents=True)
            feature = Mock()
            feature.get_unique_name.return_value = "feature"
            target = Mock()
            target.get_unique_name.return_value = "WoTE_target_refined_v1"
            target.compute_targets.return_value = {"fut_bev_background": torch.zeros(2, 2)}
            dump_feature_target_to_pickle(token_dir / "feature.gz", {"value": torch.tensor(7)})
            dump_feature_target_to_pickle(token_dir / "WoTE_target.gz", {"old": True})
            with self.assertRaisesRegex(RuntimeError, "重建"):
                CacheOnlyDataset(str(root), [feature], [target])
            # 普通训练入口也必须提示重建，不能隐式迁移旧目标。
            with self.assertRaisesRegex(RuntimeError, "run_dataset_caching"):
                Dataset._load_valid_caches(root, [feature], [target], scene_tokens={"scene"})
            self.assertEqual(Dataset._load_valid_caches(root, [feature], [target],
                                                       allow_target_cache_rebuild=True), {})
            scene = Mock()
            scene.scene_metadata = SimpleNamespace(log_name="log", initial_token="scene")
            loader = Mock()
            loader.get_scene_from_token.return_value = scene
            dataset = Dataset.__new__(Dataset)
            dataset._scene_loader = loader
            dataset._feature_builders = [feature]
            dataset._target_builders = [target]
            dataset._cache_path = root
            dataset._force_cache_computation = False
            dataset._valid_cache_paths = {}
            dataset._cache_scene_with_token("scene")
            feature.compute_features.assert_not_called()
            cached = CacheOnlyDataset(str(root), [feature], [target])
            features, targets = cached[0]
            self.assertEqual(targets["scene_token"], "scene")
            self.assertEqual(features["value"].item(), 7)

    def test_score_uses_refined_candidates_and_token_order(self):
        # 用可控评分替身验证每个样本的候选、token 和五个指标的排列。
        import navsim.agents.WoTE.refined_supervision as supervision
        from unittest.mock import patch
        helper = RefinedSupervision(WoTEConfig(), [3])
        helper._metric_loader = Mock()
        helper._metric_loader.metric_cache_paths = {"a": "a", "b": "b"}
        helper._metric_loader.get_from_token.side_effect = lambda token: token
        poses = torch.stack([torch.zeros(2, 8, 3), torch.ones(2, 8, 3)]).requires_grad_()
        def fake_score(cache, proposals, *_args):
            self.assertTrue(np.all(proposals == (0 if cache == "a" else 1)))
            return np.stack([np.full(2, idx) for idx in range(5)])
        # 确保训练包装层将每个样本交给逐候选参考轨迹评分函数。
        with patch.object(supervision, "score_candidates_with_reference", side_effect=fake_score):
            scores = helper.score(poses, ["a", "b"])
        self.assertEqual(tuple(scores.shape), (2, 1, 5, 2))
        self.assertFalse(scores.requires_grad)
        self.assertTrue(torch.equal(scores[1, 0, :, 0], torch.arange(5).float()))
        with self.assertRaisesRegex(RuntimeError, "缺少 metric cache"):
            helper.score(poses[:1], ["missing"])


if __name__ == "__main__":
    unittest.main()
