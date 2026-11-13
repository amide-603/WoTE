import unittest

import torch
import torch.nn.functional as F

from navsim.agents.WoTE.WoTE_loss import compute_pairwise_ranking_loss


class PairwiseRankingLossTest(unittest.TestCase):
    def test_shared_topk_per_metric_and_gradient(self):
        imitation = torch.tensor([[0.05, 0.15, 0.3, 0.5]], requires_grad=True)
        metrics = torch.tensor([[[0.1, 0.2, 0.4, 0.8]] * 5], requires_grad=True)
        targets = torch.tensor([[[99., 0., 1., 2.], [99., 2., 1., 0.],
                                 [99., 1., 1., 0.], [99., 1., 1., 1.],
                                 [99., 0., 1., 2.]]], requires_grad=True)
        actual = compute_pairwise_ranking_loss(imitation, metrics, targets, [1, 0, 0, 0], 3, 0.5)
        selected = metrics[0, 0, 1:]
        forward = torch.stack([F.softplus(-(selected[j] - selected[i]) / .5)
                               for i in range(3) for j in range(i + 1, 3)]).mean()
        reverse = torch.stack([F.softplus((selected[j] - selected[i]) / .5)
                               for i in range(3) for j in range(i + 1, 3)]).mean()
        tied = (F.softplus((selected[2] - selected[0]) / .5) +
                F.softplus((selected[2] - selected[1]) / .5)) / 2
        self.assertTrue(torch.allclose(actual, torch.stack([forward, reverse, tied, forward * 0, forward])))
        actual.mean().backward()
        self.assertIsNone(imitation.grad)
        self.assertIsNone(targets.grad)
        self.assertTrue(torch.equal(metrics.grad[0, 3], torch.zeros(4)))
        self.assertTrue(torch.equal(metrics.grad[:, :, 0], torch.zeros(1, 5)))
        self.assertTrue(torch.all(metrics.grad[0, [0, 1, 2, 4]].abs().sum(dim=1) > 0))

    def test_all_ties_have_zero_loss_and_gradient(self):
        imitation = torch.full((1, 3), 1 / 3)
        metrics = torch.full((1, 5, 3), 0.5, requires_grad=True)
        loss = compute_pairwise_ranking_loss(imitation, metrics, torch.ones_like(metrics), [1, 0, 0, 0], 3, 1.)
        self.assertTrue(torch.equal(loss, torch.zeros(5)))
        loss.mean().backward()
        self.assertTrue(torch.equal(metrics.grad, torch.zeros_like(metrics)))

    def test_parameter_validation(self):
        imitation = torch.full((1, 8), 1 / 8)
        metrics = torch.full((1, 5, 8), 0.5)
        for k, temperature in ((8, 0), (1, 1), (9, 1)):
            with self.assertRaises(ValueError):
                compute_pairwise_ranking_loss(imitation, metrics, metrics, [1, 1, 1, 1], k, temperature)


if __name__ == '__main__':
    unittest.main()
