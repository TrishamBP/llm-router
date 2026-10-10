# Training Plan

Stages are executed in order; each must be evaluated against the previous ones before moving on.

1. **Baselines** — one fixed-model baseline per enabled registry candidate (each model
   version separately), random, simple heuristics
   (e.g. query length), and the oracle upper bound.
2. **Supervised neural training** — neural router over query embeddings predicting
   per-model quality / utility (`training/train_neural.py`).
3. **XGBoost** — gradient-boosted router over text and structured features
   (planned as `training/train_router_xgboost.py`; `training/train_xgboost.py` now trains the
   easy/medium/hard complexity classifier, see `docs/complexity_classifier.md`).
4. **Fusion** — combine XGBoost and neural predictions into a single utility estimate
   (`training/train_fusion.py`).
5. **Reward optimization** — optimise the routing policy for the quality / cost / latency
   reward using preferences and rewards (`optimization/`).
6. **Evaluation** — final comparison on held-out benchmarks: quality, cost, latency,
   oracle gap, and cost–quality trade-off curves (`evaluation/`).

Local runs use CPU on small subsets; full neural training runs on Lightning AI GPUs.
