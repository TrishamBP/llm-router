# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26", "openai>=1.40", "mcp>=1.2,<2"]
# ///
"""stdio MCP server exposing the complexity router as a `classify_prompt` tool.

uv run --script mcp_server.py        (Claude Code starts it from the plugin's .mcp.json)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from complexity_router import ComplexityRouter, RouterError  # noqa: E402

server = FastMCP("llm-complexity-router")
_routers: dict[str, ComplexityRouter] = {}


@server.tool()
def classify_prompt(prompt: str, classifier: str = "") -> dict[str, Any]:
    """Estimate how complex a prompt is (easy / medium / hard) and suggest a Claude tier.

    Returns the label, class probabilities, the recommended model tier (haiku / sonnet /
    opus), and which classifier produced it. The prompt is sent to the Fireworks embeddings
    API. Approximate: trained on LLM-judge labels; treat it as a hint, not ground truth.
    """
    name = classifier or ""
    try:
        if name not in _routers:
            _routers[name] = ComplexityRouter.load(name or None)
        return _routers[name].classify([prompt])[0].to_dict()
    except RouterError as exc:
        return {"error": str(exc)}


if __name__ == "__main__":
    server.run()
