# Architecture

High-level module boundaries for `src/llm_router/`. Dependencies flow downward only:
lower layers never import from higher ones.

```text
serving / monitoring
        │
     routing
        │
 optimization ── evaluation
        │
     training
        │
      models
        │
     features
        │
       data
        │
config / schemas / logging / models.registry
```

`models.registry` is the one exception to the package-by-layer picture. It holds the candidate
schema and has no imports from `llm_router`, so it sits at the foundation level. Every layer may
import it (e.g. `data.model_catalog`, `features.structured_features`, `evaluation.pricing`).
See [model_registry.md](model_registry.md).

| Module          | Responsibility                                                                 |
|-----------------|--------------------------------------------------------------------------------|
| `config`        | Load and validate YAML configs from `configs/` and environment variables.      |
| `schemas`       | Typed records shared across modules (queries, per-model outcomes, decisions).  |
| `logging`       | Structured logging setup.                                                       |
| `data`          | Collection of model outcomes, loading, validation, splitting, dataset objects. |
| `features`      | Text features, structured query features, sentence embeddings.                 |
| `embeddings`    | Fireworks embeddings client, cached/resumable embedding generation.           |
| `models`        | Model registry; XGBoost router, neural router, utility-predictor contract, fusion. |
| `training`      | Training entry points per model, losses, checkpointing.                        |
| `optimization`  | Preference construction, reward definitions, policy optimisation.              |
| `routing`       | Candidate filter, runtime router, decision policy, calibration, escalation.    |
| `evaluation`    | Quality, cost, latency, oracle comparison, benchmarks, reporting.              |
| `inference`     | `ComplexityRouter`: query -> embedding -> complexity label.                   |
| `serving`       | Inference wrapper and HTTP API.                                                 |
| `monitoring`    | Production metrics and telemetry.                                               |

## Design principles

- **Predict utility, then decide.** Models estimate per-model outcomes; the routing policy
  turns those estimates into a choice. Cost/latency trade-offs live in the policy and reward
  configuration, not inside model weights.
- **Two complementary predictors.** XGBoost on tabular features and a neural network on
  embeddings are trained independently and combined by `models/fusion.py`.
- **CPU-first.** Every module must run on CPU locally; GPU is an optimisation used on Lightning AI.
- **Configuration over code.** Experiment choices are expressed in `configs/*.yaml`.
- **No developer tooling in the package.** Claude Code, Graphify, and SimpleMem are a separate,
  optional layer and are never imported by `llm_router`. See [developer_tooling.md](developer_tooling.md).
