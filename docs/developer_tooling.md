# Developer Tooling: Claude Code, Graphify, SimpleMem

This document covers the **optional** local tooling used to develop this repository with Claude Code.
None of it is required to train, evaluate, or serve the router.

> **Status (2026-10-10):** documented only. Neither Graphify nor SimpleMem is installed or
> configured in this repository, and no Claude Code hooks or MCP servers have been added.
> Nothing below is considered working until it has been verified in a real Claude Code session
> (see [Verification checklist](#f-verification-checklist)).

Upstream details were checked on 2026-10-10 against each project's README and docs on GitHub.
Both projects release frequently, so re-check upstream before installing.

---

## A. Architecture separation

| Layer | Contents | Lives in | Python env |
|---|---|---|---|
| **Product** | Hybrid router, routing policy, serving API | `src/llm_router/{routing,serving,monitoring}` | project env (`uv sync`) |
| **ML research** | Datasets, features/embeddings, training, reward optimisation, evaluation, artifacts | `src/llm_router/{data,features,models,training,optimization,evaluation}`, `configs/`, `data/`, `artifacts/` | project env (`uv sync`, Lightning AI for GPU) |
| **Developer tooling** | Claude Code, Graphify, SimpleMem | `CLAUDE.md`, this document, user-level tool installs, git-ignored generated state | **separate** (`uv tool` envs / external services) |

Rules:

- `llm_router` must never import `graphify`, `simplemem`, or any other developer-tooling package.
- Developer tools are never added to `pyproject.toml` (not to core and not to any dependency group).
  `uv sync` installs nothing from this layer.
- Developer tooling must never feed into the router's features, forward pass, training loss,
  reward, or routing decision.
- Generated graph and memory state is git-ignored and can be rebuilt at any time.

---

## B. Graphify integration

### Verified upstream

| Item | Value |
|---|---|
| Repository | <https://github.com/Graphify-Labs/graphify> (Apache-2.0, default branch `v8`) |
| PyPI package | `graphifyy` (**double "y"**; the CLI command is `graphify`) |
| Version at time of writing | `0.9.84` (released 2026-10-10), requires Python ≥ 3.10 |
| Claude Code mechanisms | Skill (`/graphify`), a `CLAUDE.md` section, a `PreToolUse` hook, and an optional MCP server |

### Installation (opt-in, not part of `uv sync`)

Install with `uv tool`, which uses an isolated env outside the project's `.venv`. Pin the version:

```bash
uv tool install "graphifyy==0.9.84"        # add the [mcp] extra only if you want the MCP server
graphify install --project                 # project-scoped skill: .claude/skills/graphify/SKILL.md
```

Without `--project`, `graphify install` puts the skill in your user profile instead.
If `graphify` isn't found after installing, run `uv tool update-shell` and open a new terminal.

### Generating the graph

There are two documented paths:

1. **Inside Claude Code:** run the `/graphify .` skill. Code is parsed locally with tree-sitter.
   Docs and other non-code files are sent to the model in your Claude Code session for
   semantic extraction.
2. **Headless, code only, fully local:** `graphify extract . --code-only`. This uses no API key
   and sends nothing off the machine.

> ⚠️ Headless `graphify extract` without `--code-only` **picks a provider automatically based on
> which API key environment variables are set** (`ANTHROPIC_API_KEY` is one of them). If this
> repo's `ANTHROPIC_API_KEY` is exported in your shell, docs can be sent to the API and billed.
> Use `--code-only`, or pass an explicit `--backend`.

### Artifacts

These are written to `graphify-out/` at the repo root, which is git-ignored:

- `graph.json`: full graph, queried by the CLI and MCP server
- `GRAPH_REPORT.md`: summary with key concepts, cross-module links, and suggested questions
- `graph.html`: interactive visualisation
- Machine-local state: `cache/`, `.graphify_root`, `.graphify_python`, `manifest.json`,
  `memory/`, and others. Never commit these.

Upstream documents force-adding `graph.json` and `GRAPH_REPORT.md` for team sharing. This
project does **not** do that unless a specific need is agreed.

### How Claude Code consults the graph

- **Skill commands:** `/graphify query "<question>"`, `/graphify path "A" "B"`, `/graphify explain "X"`.
- **CLI:** `graphify query "<question>"`, also runnable from Claude Code's Bash tool.
- **Always-on nudge (optional):** `graphify claude install` writes a section into `CLAUDE.md`
  and registers a `PreToolUse` hook (`graphify hook-guard`). Before search or file-read
  tool calls, the hook nudges Claude toward the graph. `--strict` blocks the first raw file read
  of each session. **Review the diff to `CLAUDE.md` and `.claude/` before keeping it**, because it
  edits this repo's instructions. Undo with `graphify claude uninstall`.
- **MCP server (optional):** upstream documents `python -m graphify.serve graphify-out/graph.json`
  (stdio) and a `graphify-mcp` entry point, which needs the `[mcp]` extra.
  TODO: verify the exact command for a `uv tool` install and register it with
  `claude mcp add`.

### Refreshing

| When | Command |
|---|---|
| After significant code changes, `git pull`, or a merge | `graphify update .` |
| Docs changed | `/graphify . --update` (re-extracts changed files only) |
| Automatic on commit / branch switch (optional) | `graphify hook install`, which installs **git** hooks (post-commit, post-checkout). Re-run it after upgrading graphify. |

### Ignoring files

Graphify respects `.gitignore` automatically, so `.env`, `data/**`, `artifacts/**`, and
`mlruns/` are already excluded. For further exclusions, add a `.graphifyignore` file
(same syntax as `.gitignore`). None has been needed yet.

### Open TODOs

- [ ] Verify the MCP launch command under `uv tool` and decide whether MCP is needed at all, since the CLI and skill may be enough.
- [ ] Decide between the default hook and `--strict`, after reviewing what `graphify claude install` writes.
- [ ] Upstream suggests a `.claudeignore` to protect the prompt cache. This file is **not** a
      verified Claude Code feature, so it hasn't been added.
- [ ] Upstream docs contradict themselves on whether the query log (`~/.cache/graphify-queries.log`)
      is on by default. Set `GRAPHIFY_QUERY_LOG_DISABLE=1` if that matters to you.

---

## C. SimpleMem integration

### Verified upstream

| Item | Value |
|---|---|
| Repository | <https://github.com/aiming-lab/SimpleMem> (MIT, default branch `main`) |
| Latest GitHub release | `v0.3.0` (2026-05-21) |
| PyPI package | `simplemem` (PyPI shows `0.1.0`; the README now installs the unified package from a clone with `pip install -e .`) |
| Requirements | Python ≥ 3.10, plus an **external LLM and embedding provider** (OpenAI-compatible, OpenRouter, or Requesty) or a local **Ollama** server |

### Mechanisms upstream actually provides

| Mechanism | What it is | Claude Code fit |
|---|---|---|
| **MCP server** (text memory) | Streamable HTTP MCP server with Bearer-token auth. Tools: `memory_add`, `memory_add_batch`, `memory_query`, `memory_retrieve`, `memory_stats`, `memory_clear`. Either the hosted service (`https://mcp.simplemem.cloud/mcp`) or self-hosted (`MCP/run.py`, which defaults to `http://localhost:8000/mcp`). | Claude Code supports HTTP MCP servers: `claude mcp add --transport http <name> <url> --header "Authorization: Bearer …"` (checked against the local `claude mcp add --help`). **Not yet tested with SimpleMem.** |
| **Claude skill** (`SKILL/simplemem-skill`) | Skill folder copied into `~/.claude/skills/`, with LanceDB storage in the skill's `data/` folder. Uses OpenRouter for the LLM and embeddings. | Documented for Claude by upstream. Upstream installs its dependencies with `pip install -r requirements.txt`; a uv-only equivalent is still TODO. |
| **SimpleMem-Cross** (`cross/`) | Python library for session lifecycle (start → record → stop → end) with automatic context injection at session start. SQLite + LanceDB stored in `~/.simplemem-cross/`. Exposes an abstract `SessionHooks` class, a FastAPI router, and MCP tool *definitions*. | **No ready-made Claude Code hook configuration or runnable MCP server is shipped.** Using it would require writing our own adapter. |

**Conclusion:** automatic memory restore at session start is **not available out of the box**.
The realistic first step is the self-hosted MCP server, called **explicitly** (e.g. "query
project memory for prior decisions on X"). A `SessionStart` hook would be new work: write it,
review it, and verify it before enabling (see TODOs).

### Privacy implications

- When memory is written, SimpleMem sends the content to the configured LLM and embedding
  provider for compression and indexing. The **hosted** service also stores the memory on a
  third-party server, and asks for an OpenRouter key on its sign-up page.
- For this project, prefer **self-hosted with Ollama** (fully local), or at least self-hosted
  with a provider you have approved. Do not use the hosted service for unpublished research.

### Storage, backup, reset

| Mechanism | Storage location | Reset |
|---|---|---|
| Self-hosted MCP | Server `DATA_DIR` (Docker default `/app/MCP/data`; TODO: bare-metal default) | `memory_clear` tool (irreversible), or delete the data dir |
| Skill | `~/.claude/skills/simplemem-skill/data/` | Delete that directory |
| Cross | `~/.simplemem-cross/` (`cross_memory.db`, `lancedb_cross/`) | Delete that directory |

All of these are outside the repository, so nothing needs to be git-ignored. To back up, stop
the server and copy the data directory.

### What to store / not store

- **Store:** architecture decisions with their rationale, verified experimental findings (with
  run ID and config), dataset/split conventions, and known pitfalls.
- **Never store:** secrets (API keys, tokens, passwords), `.env` contents, raw dataset rows,
  model responses from collected data, or unverified speculation.

### Open TODOs

- [ ] Pick a mechanism: self-hosted MCP (recommended starting point), the skill, or a custom Cross adapter.
- [ ] Write a uv-based install for the self-hosted MCP server. Upstream uses `venv` + `pip`; this project uses `uv` only.
- [ ] Confirm Ollama works as the LLM **and** embedding backend for the self-hosted MCP server.
- [ ] Decide on the token handling. Prefer passing it with `--header` at `local` scope (stored in the
      user's Claude Code config, not in the repo). Never commit a `.mcp.json` that contains a token.
- [ ] Optional: design a `SessionStart` hook that calls `memory_retrieve` and prints a short project summary. Requires review and verification.

---

## D. Session-start workflow

Steps 1 and 4 work today. Steps 2, 3 and 5 depend on the TODOs above.

1. **Read `CLAUDE.md`.** Claude Code loads the repo-root `CLAUDE.md` automatically. ✅
2. **Retrieve project state from memory.** Once SimpleMem MCP is configured, ask Claude to call
   `memory_query` / `memory_retrieve` for a short summary. This is manual; there is no
   automatic injection yet. ⏳
3. **Consult the graph for architectural context.** Once Graphify is installed, use
   `graphify query` or `/graphify query`, and read `graphify-out/GRAPH_REPORT.md` for broad reviews. ⏳
4. **Inspect the source files** before making any change. Graph and memory are hints, not
   ground truth. ✅
5. **Update memory** only when meaningful knowledge changes (a decision made, a finding
   verified), through `memory_add`. ⏳

---

## E. Privacy and safety

- Never put secrets in memory or graph inputs. Router credentials stay in `.env`, which is
  git-ignored and therefore also skipped by Graphify. Dev-tool credentials (SimpleMem token,
  provider keys) stay in user-level config, **not** in this repo's `.env`.
- `data/`, `artifacts/`, `.env`, and MLflow output are git-ignored, so Graphify skips them.
  Do not paste dataset contents into memory.
- Don't run downloaded install scripts automatically (`curl … | sh`, `npx -y …`). Install from a
  package index with a pinned version, or from a reviewed commit.
- Pin versions: `graphifyy==<version>`; for SimpleMem, a release tag or commit SHA.
- Review before enabling: run `graphify claude install` and check the diff to `CLAUDE.md` and
  `.claude/settings.json`. Check `claude mcp list` after adding any server, and only grant MCP
  tool permissions you need. `memory_clear` is destructive.
- Keep `graphify-out/` out of Git; memory stores are outside the repo by default.
- Reset or rebuild:
  - Graph: delete `graphify-out/` (or `graphify uninstall --purge`) and regenerate it.
  - Graphify integration: `graphify claude uninstall`, `graphify hook uninstall`, `uv tool uninstall graphifyy`.
  - Memory: see [Storage, backup, reset](#storage-backup-reset). MCP registration: `claude mcp remove <name>`.

### Future setup script (not implemented)

If a helper script is added later, it must:

- be opt-in, e.g. `scripts/setup_dev_tooling.sh`, and never called from `setup_local.sh`
- install only pinned versions, using `uv tool`
- print each action before taking it, and ask for confirmation before installing third-party code
- never write secrets or register hooks without a separate confirmation

---

## F. Verification checklist

Mark the integration verified only when all of these pass in a real session:

- [ ] `graphify --version` matches the pinned version; `graphify-out/graph.json` exists after generation.
- [ ] In a new Claude Code session, a `graphify query` about module relationships returns relevant nodes.
- [ ] `claude mcp list` shows the SimpleMem server as connected.
- [ ] A fact saved in session A (`memory_add`) is retrieved in a fresh session B (`memory_query`).
- [ ] `uv run python -c "import llm_router"` works with neither tool installed in the project env.
