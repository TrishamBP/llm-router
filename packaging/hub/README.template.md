---
license: cc-by-nc-4.0
language:
  - en
library_name: numpy
pipeline_tag: text-classification
tags:
  - llm-routing
  - prompt-complexity
  - claude-code
  - claude-code-plugin
  - fireworks
datasets:
  - regolo/brick-complexity-extractor
metrics:
  - accuracy
  - f1
---

# LLM complexity router (easy / medium / hard)

A small prompt-complexity classifier for LLM routing, packaged as a **Claude Code plugin**.
Each prompt is embedded with Fireworks AI's `fireworks/qwen3-embedding-8b`. A tiny classifier
then runs locally in NumPy and predicts `easy`, `medium`, or `hard`, and suggests a Claude
tier: easy → `haiku`, medium → `sonnet`, hard → `opus`.

- **No pickles, no PyTorch, no GPU.** The weights are plain NumPy arrays, loaded with
  `allow_pickle=False`, and the runtime needs only `numpy` and `openai`.
- **Two classifiers.** `logistic_regression` is the default. `mlp` is a 4096→256→128→3
  network that is slightly more accurate.
- **You need** [uv](https://docs.astral.sh/uv/) and a [Fireworks AI](https://fireworks.ai) API
  key. Each prompt costs one embeddings call to Fireworks.

> **Privacy:** while the plugin's hook is enabled, **every prompt you type in Claude Code is
> sent to Fireworks AI** to be embedded. Turn the hook off with `COMPLEXITY_ROUTER_HOOK=off` if
> that isn't acceptable.

## Use it with Claude Code

```bash
# 1. Download this repository
uvx --from huggingface_hub hf download Trisham97/llm-complexity-router --local-dir ~/llm-complexity-router

# 2. Provide your Fireworks key (in your shell profile, or the "env" block of ~/.claude/settings.json)
export FIREWORKS_API_KEY=fw_...            # PowerShell: $env:FIREWORKS_API_KEY = "fw_..."

# 3. Optional: warm up (installs numpy/openai into uv's cache, then classifies one prompt)
uv run --script ~/llm-complexity-router/complexity_router.py classify "Design a lock-free queue"

# 4. Start Claude Code with the plugin
claude --plugin-dir ~/llm-complexity-router
```

To install it permanently instead of passing `--plugin-dir` each time, add the downloaded folder
as a local marketplace and install from it:

```bash
claude plugin marketplace add ~/llm-complexity-router
claude plugin install llm-complexity-router@llm-complexity-router
```

### What it adds to Claude Code

| Component | What it does |
|---|---|
| `UserPromptSubmit` hook | Classifies each prompt you type and gives Claude the result as context, e.g. *"Estimated prompt complexity: HARD (easy 0.02, medium 0.13, hard 0.85). Suggested Claude tier: opus."* It **never blocks or changes your prompt**: on any error or timeout it does nothing. Slash commands and very short prompts are skipped. |
| `classify_prompt` MCP tool | Lets Claude (or you) classify any text on demand. |
| `haiku-worker`, `sonnet-worker`, `opus-worker` subagents | Worker subagents pinned to each tier, so Claude can delegate a self-contained task to the suggested model. |

**What it can't do:** Claude Code doesn't let a hook switch the main session's model, so the
router **recommends** a tier. Claude can delegate to the matching subagent, and you can switch
yourself with `/model`.

## Use it from Python or the command line

```bash
uv run --script complexity_router.py classify "What is the capital of France?"
uv run --script complexity_router.py classify --model mlp "Prove that sqrt(2) is irrational"
```

```python
from complexity_router import ComplexityRouter  # run from this folder

router = ComplexityRouter.load()  # or ComplexityRouter.load("mlp")
print(
    router.classify(["Design a distributed scheduler that minimizes GPU idle time."])[0].to_dict()
)
# {'label': ..., 'probabilities': {'easy': ..., 'medium': ..., 'hard': ...}, 'recommended_model': ..., ...}
```

| Environment variable | Meaning |
|---|---|
| `FIREWORKS_API_KEY` | Required. Never logged. |
| `FIREWORKS_BASE_URL` | Default `https://api.fireworks.ai/inference/v1`. |
| `COMPLEXITY_ROUTER_MODEL` | `logistic_regression` (default) or `mlp`. |
| `COMPLEXITY_ROUTER_HOOK` | Set to `off` to disable the prompt hook. |

`FIREWORKS_EMBEDDING_MODEL`, if set, must be `fireworks/qwen3-embedding-8b`. The classifiers only
work with that model's 4096-dimensional embeddings, and the router refuses to run with any other
embedding model.

## Evaluation

These results are on a held-out test split of {{TEST_ROWS}} prompts. The models were trained on
{{TRAIN_ROWS}} prompts and never saw the test split during training or model selection.

| Classifier | Accuracy | Macro F1 | Hard recall | Hard prompts predicted easy |
|---|---|---|---|---|
| `logistic_regression` (default) | {{LR_ACC}} | {{LR_F1}} | {{LR_HARD}} | {{LR_H2E}} |
| `mlp` | {{MLP_ACC}} | {{MLP_F1}} | {{MLP_HARD}} | {{MLP_H2E}} |

- The `mlp` published here (experiment "C") was picked over a closely matched variant *after*
  looking at test scores, so its test numbers are slightly optimistic. Logistic regression's
  numbers aren't affected by this.
- Hard recall is the weak spot. About 4% of hard prompts are predicted `easy`; most errors put
  hard prompts in `medium`.
- The predicted probabilities aren't calibrated.

## Training data and license

- **Data:** [`regolo/brick-complexity-extractor`](https://huggingface.co/datasets/regolo/brick-complexity-extractor),
  revision `{{REVISION}}`, by Regolo.ai (Seeweb S.r.l.). It holds 76,831 English prompts labelled
  by an LLM judge (Qwen3.5-122B). The label column is `text` and the files contain no confidence
  scores, contrary to the dataset card. After deduplication the data was re-split 80/10/10, with
  duplicate prompts kept in the same split.
- **License: CC BY-NC 4.0, non-commercial use only**, as required by the training data's
  license. Credit Regolo.ai's dataset when you use or share this model. For commercial use, ask
  the dataset's authors.
- **Labels are approximate.** They reflect one LLM judge's view of reasoning effort, and the
  boundary between medium and hard is subjective.
- **Out of scope:** English only. Not a safety or content filter. It doesn't judge whether a
  task *can* be done well by a given model.

## Files

| Path | Contents |
|---|---|
| `models/logistic_regression.npz`, `models/mlp.npz` | Weights (NumPy, no pickle) |
| `router_config.json` | Labels, embedding model, tier mapping, metrics |
| `complexity_router.py` | Runtime, CLI, and Claude Code hook (inline `uv` script metadata) |
| `mcp_server.py` | stdio MCP server with the `classify_prompt` tool |
| `.claude-plugin/`, `hooks/`, `agents/`, `.mcp.json` | Claude Code plugin |
