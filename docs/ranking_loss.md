# Top-k pairwise ranking loss on refined trajectories

The optional loss is disabled by default. Enable it with Hydra overrides:

```text
+agent.config.ranking_loss_weight=0.1
+agent.config.ranking_temperature=0.5
+agent.config.ranking_top_k=8
```

For each scene, the loss selects the top-k candidates by the same combined
predicted reward used at inference and ranks them using their online PDM total
scores. Both prediction and target correspond to the current forward pass's
refined trajectories. The five metric targets and total scores come from the
same simulation; the targets are detached, while gradients flow through the
reward prediction network to the trajectory offsets. Tied targets are ignored.
This also applies when CPU scoring workers and fast trajectory conversion are
enabled; each worker returns the metric targets and total from the same scoring
call, and results are restored to the input scene order.

The total uses the `e6ab3b0` reference-trajectory scoring procedure: each
candidate's progress is normalized against the reference, and all
multiplicative metrics, including driving direction, contribute to the PDM
total. No offline anchor-score labels or ranking-specific target cache are used.
The metric cache and refined target cache requirements still apply; see
`docs/ascend_training.md`.

Set `ranking_top_k=32` for Top 32 (496 unordered pairs). The weight 0.1 and
temperature 0.5 are initial experiment choices, not tuned optimum values.
