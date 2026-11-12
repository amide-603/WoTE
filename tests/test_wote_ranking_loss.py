import math
import unittest

import torch
import torch.nn.functional as F

from navsim.agents.WoTE.WoTE_loss import compute_pairwise_ranking_loss


class PairwiseRankingLossTest(unittest.TestCase):
    def test_top_eight_pairs_and_gradient(self):
        logits = torch.arange(9.0, requires_grad=True)
        imitation = torch.softmax(logits, 0).unsqueeze(0)
        metrics = torch.full((1, 5, 9), 0.8)
        # Reverse the target order for the selected eight candidates.
        targets = -torch.arange(9.0).unsqueeze(0)
        actual = compute_pairwise_ranking_loss(imitation, metrics, targets, [1, 0, 0, 0], 8, 0.5)
        selected = logits[1:]
        expected = torch.stack([
            F.softplus((selected[j] - selected[i]) / 0.5)
            for i in range(8) for j in range(i + 1, 8)
        ]).mean()
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))
        actual.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad[-1].item(), 0)
        self.assertLess(logits.grad[1].item(), 0)

    def test_ties_excluded_and_all_ties_zero(self):
        imitation = torch.tensor([[0.6, 0.3, 0.1]], requires_grad=True)
        metrics = torch.full((1, 5, 3), 0.5)
        weights = [1, 0, 0, 0]
        loss = compute_pairwise_ranking_loss(imitation, metrics, torch.tensor([[1., 1., 0.]]), weights, 3, 1.)
        expected = (F.softplus(-torch.log(torch.tensor(6.))) + F.softplus(-torch.log(torch.tensor(3.)))) / 2
        self.assertTrue(torch.allclose(loss, expected))
        zero = compute_pairwise_ranking_loss(imitation, metrics, torch.ones((1, 3)), weights, 3, 1.)
        self.assertEqual(zero.item(), 0.0)
        zero.backward()
        self.assertTrue(torch.equal(imitation.grad, torch.zeros_like(imitation)))

    def test_top32_uses_all_496_pairs(self):
        logits = torch.linspace(-2, 2, 36, requires_grad=True)
        imitation = torch.softmax(logits, dim=0).unsqueeze(0)
        metrics = torch.full((1, 5, 36), 0.5)
        targets = torch.arange(36.0).unsqueeze(0)
        actual = compute_pairwise_ranking_loss(imitation, metrics, targets, [1, 0, 0, 0], 32, 0.5)
        selected = logits[4:]
        pairs = [F.softplus(-(selected[j] - selected[i]) / 0.5)
                 for i in range(32) for j in range(i + 1, 32)]
        self.assertEqual(len(pairs), 496)
        self.assertTrue(torch.allclose(actual, torch.stack(pairs).mean(), atol=1e-6))
        actual.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_parameter_validation(self):
        imitation = torch.full((1, 8), 1 / 8)
        metrics = torch.full((1, 5, 8), 0.5)
        targets = torch.zeros((1, 8))
        for k, t in ((8, 0), (1, 1), (9, 1)):
            with self.assertRaises(ValueError):
                compute_pairwise_ranking_loss(imitation, metrics, targets, [1, 1, 1, 1], k, t)


if __name__ == '__main__':
    unittest.main()
