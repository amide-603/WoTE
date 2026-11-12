# Top-k pairwise ranking loss

Enable the optional loss with Hydra overrides:

```text
+agent.config.ranking_loss_weight=0.1
+agent.config.ranking_temperature=0.5
+agent.config.ranking_top_k=8
```

Set `ranking_top_k=32` to use Top 32 (496 unordered pairs). The default weight
is zero, preserving the original training behavior. For each scene the loss
chooses the largest predicted inference scores and averages
`softplus(-sign(q_i-q_j)*(s_i-s_j)/T)` over pairs
with distinct target scores. Here `s` is the inference-time combined reward,
and `q` is the precomputed `score` for each fixed trajectory anchor. That
target is a proxy for the model's offset trajectory, not a newly simulated
score for the offset trajectory. Ties provide no ordering signal.

The score is read by the target builder when ranking is enabled. Existing
training caches created without `ranking_target_score` cannot supply this
label; rebuild the cache or train directly from the dataset (`cache_path=null`).
The 0.1 weight and 0.5 temperature are initial experiment choices, not tuned
or established optimal values.
