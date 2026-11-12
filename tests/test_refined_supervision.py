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
        # 该测试独立验证串行标签包装，避免依赖实际 CPU 子进程。
        config = WoTEConfig()
        config.score_workers = 0
        helper = RefinedSupervision(config, [3])
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

    def test_ranking_uses_online_refined_scores_and_updates_offsets(self):
        from unittest.mock import patch
        import navsim.agents.WoTE.refined_supervision as supervision
        from navsim.agents.WoTE.WoTE_agent import WoTEAgent

        config = WoTEConfig()
        config.ranking_loss_weight = 1.0
        config.score_workers = 0
        config.ranking_top_k = 3
        config.use_agent_loss = False
        config.use_map_loss = False
        config.reward_weights = [1.0, 0.0, 0.0, 0.0]
        helper = RefinedSupervision(config, [3])
        helper._metric_loader = Mock()
        helper._metric_loader.metric_cache_paths = {"scene": "scene"}
        helper._metric_loader.get_from_token.return_value = "scene"
        offset = torch.zeros(1, 3, 8, 3, requires_grad=True)
        anchors = torch.zeros(3, 8, 3)
        anchors[:, 0, 0] = torch.tensor([0.1, 0.2, 0.3])
        refined = anchors.unsqueeze(0) + offset
        predictions = {
            "refined_trajectories": refined,
            "trajectory_anchors": anchors,
            "trajectory_offset_rewards": torch.full((1, 3), 1 / 3),
            "im_rewards": (-refined[:, :, 0, 0]).softmax(-1),
            "sim_rewards": torch.full((1, 5, 3), 0.8),
        }
        calls = []
        def fake_score(cache, proposals, *_args, include_total=False):
            calls.append(proposals.copy())
            self.assertTrue(include_total)
            return np.ones((5, 3), dtype=np.float32), proposals[:, 0, 0]

        agent = WoTEAgent.__new__(WoTEAgent)
        torch.nn.Module.__init__(agent)
        agent.config = config
        agent._refined_supervision = helper
        with patch.object(supervision, "score_candidates_with_reference", side_effect=fake_score), \
             patch("navsim.agents.WoTE.WoTE_loss.compute_traj_offset_loss", return_value=torch.tensor(0.)), \
             patch("navsim.agents.WoTE.WoTE_loss.compute_im_reward_loss", return_value=torch.tensor(0.)), \
             patch("navsim.agents.WoTE.WoTE_loss.compute_sim_reward_loss", return_value=torch.tensor(0.)):
            losses = agent.compute_loss({}, {"scene_token": ["scene"]}, predictions)
            self.assertEqual(len(calls), 1)
            np.testing.assert_allclose(calls[0], refined.detach().numpy()[0])
            self.assertIn("ranking_loss", losses)
            losses["ranking_loss"].backward()
        self.assertIsNotNone(offset.grad)
        self.assertTrue(torch.isfinite(offset.grad).all())
        self.assertGreater(offset.grad.abs().sum().item(), 0)

    def test_online_ranking_scores_preserve_batch_order_and_are_detached(self):
        from unittest.mock import patch
        import navsim.agents.WoTE.refined_supervision as supervision
        config = WoTEConfig()
        config.score_workers = 0
        helper = RefinedSupervision(config, [3])
        helper._metric_loader = Mock()
        helper._metric_loader.metric_cache_paths = {"a": "a", "b": "b"}
        helper._metric_loader.get_from_token.side_effect = lambda token: token
        poses = torch.arange(4.0).reshape(2, 2, 1, 1).expand(-1, -1, 8, 3).clone().requires_grad_()
        def fake_score(cache, proposals, *_args, include_total=False):
            self.assertTrue(include_total)
            values = np.tile(proposals[:, 0, 0], (5, 1))
            return values, proposals[:, 0, 0] + (10 if cache == "b" else 0)
        with patch.object(supervision, "score_candidates_with_reference", side_effect=fake_score) as score:
            metrics, totals = helper.score(poses, ["a", "b"], include_total=True)
        self.assertEqual(score.call_count, 2)
        self.assertTrue(torch.equal(totals, torch.tensor([[0., 1.], [12., 13.]])))
        self.assertFalse(totals.requires_grad)
        self.assertFalse(metrics.requires_grad)

    def test_parallel_ranking_scores_keep_scene_and_candidate_order(self):
        config = WoTEConfig()
        config.score_workers = 2
        helper = RefinedSupervision(config, [3])
        helper._metric_loader = Mock()
        helper._metric_loader.metric_cache_paths = {"a": "path-a", "b": "path-b"}
        helper._score_pool = Mock()
        helper._score_pool.score.return_value = [
            (np.full((5, 2), 1, dtype=np.float32), np.array([0.2, 0.8], dtype=np.float32)),
            (np.full((5, 2), 2, dtype=np.float32), np.array([0.7, 0.1], dtype=np.float32)),
        ]
        poses = torch.zeros(2, 2, 8, 3, requires_grad=True)
        metrics, totals = helper.score(poses, ["a", "b"], include_total=True)
        self.assertTrue(torch.equal(totals, torch.tensor([[0.2, 0.8], [0.7, 0.1]])))
        self.assertTrue(torch.equal(metrics[:, 0, 0, 0], torch.tensor([1., 2.])))
        self.assertFalse(totals.requires_grad)
        self.assertTrue(helper._score_pool.score.call_args.kwargs['include_total'])

    def test_reference_scoring_totals_use_pairwise_progress_and_all_metrics(self):
        from unittest.mock import patch
        import navsim.evaluate.pdm_score as pdm
        from navsim.agents.WoTE.refined_supervision import score_candidates_with_reference
        from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorerConfig

        cache = SimpleNamespace(ego_state=SimpleNamespace(time_point=0), trajectory=None,
                                observation=None, centerline=None, route_lane_ids=None, drivable_area_map=None)
        simulator = Mock()
        scorer = Mock()
        scorer._config = PDMScorerConfig(progress_distance_threshold=5.0)
        scorer._multi_metrics = np.array([[1., 1., 0.5], [1., 1., 1.], [1., 1., 1.]])
        scorer._weighted_metrics = np.array([[0., 0., 0.], [1., 0.8, 0.4], [1., 0.5, 0.2]])
        scorer._progress_raw = np.array([10., 5., 16.])
        scorer._ego_coords = np.zeros((3, 2, 5, 2))
        scorer._ego_areas = np.zeros((3, 2, 3), dtype=bool)
        with patch.object(pdm, "transform_trajectory", side_effect=lambda traj, _: traj), \
             patch.object(pdm, "get_trajectory_as_array", return_value=np.zeros((2, 11))):
            values, totals = score_candidates_with_reference(
                cache, np.zeros((2, 8, 3)), SimpleNamespace(interval_length=0.1),
                simulator, scorer, include_total=True,
            )
        np.testing.assert_allclose(values[2], [0.5, 0.8])
        np.testing.assert_allclose(totals, [(5 * 0.5 + 5 * 0.8 + 2 * 0.5) / 12,
                                           0.5 * (5 * 0.8 + 5 * 0.4 + 2 * 0.2) / 12])
        simulator.simulate_proposals.assert_called_once()
        scorer.score_proposals.assert_called_once()


if __name__ == "__main__":
    unittest.main()
