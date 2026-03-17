"""MCP server for the Personal Synaptic Network.

Connect your PSN to any MCP-compatible client (Claude Desktop, Claude Code,
Cursor, Cline, etc.) and use your personal thought compass directly in chat.

Usage:
    # stdio transport (default — for Claude Desktop, Cursor, etc.)
    python -m psn.mcp_server

    # HTTP transport (for remote / multi-client access)
    python -m psn.mcp_server --http --port 8430

    # Custom checkpoint
    python -m psn.mcp_server --checkpoint path/to/psn_full.pt
"""

import argparse
import json
import sys
import os
from pathlib import Path

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print("ERROR: FastMCP not installed. Run: pip install mcp[cli]", file=sys.stderr)
    sys.exit(1)

from .config import PSNConfig
from .psn_core import PSN

# ---------------------------------------------------------------------------
# Global PSN instance (loaded once at startup)
# ---------------------------------------------------------------------------

_psn: PSN | None = None
_checkpoint_path: Path | None = None


def get_psn() -> PSN:
    """Lazy-load the PSN from checkpoint."""
    global _psn
    if _psn is not None:
        return _psn

    config = PSNConfig()
    _psn = PSN(config)

    # Try checkpoint path from env, then default locations
    ckpt = _checkpoint_path
    if ckpt is None:
        ckpt_env = os.environ.get("PSN_CHECKPOINT")
        if ckpt_env:
            ckpt = Path(ckpt_env)

    if ckpt is None:
        candidates = [
            config.checkpoint_dir / "psn_full.pt",
            config.checkpoint_dir / "psn_latest.pt",
            Path("psn/checkpoints/psn_full.pt"),
            Path("psn/checkpoints/psn_latest.pt"),
        ]
        for c in candidates:
            if c.exists():
                ckpt = c
                break

    if ckpt and ckpt.exists():
        _psn.load(ckpt)

    return _psn


# ---------------------------------------------------------------------------
# MCP Application
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "Personal Synaptic Network",
    instructions=(
        "Personal Synaptic Network (PSN) — a Hopfield associative memory "
        "that stores thought patterns via Hebbian learning. Use 'recall' to "
        "query the network with a natural language cue and get back the most "
        "associated stored thoughts. Use 'store' to add new thoughts. "
        "Use 'compass' for decision validation — it retrieves patterns that "
        "reveal how the person actually thinks about a topic."
    ),
)


@mcp.tool()
def recall(cue: str, top_k: int = 5) -> str:
    """Recall associated thought patterns from the PSN.

    Give a natural language cue and the network will activate neurons,
    run attractor dynamics, and return the closest stored patterns.
    This is associative recall — not keyword search.

    Args:
        cue: Natural language query (e.g., "how should I approach hard decisions")
        top_k: Number of matches to return (default 5)
    """
    psn = get_psn()

    if psn.memory.count == 0:
        return json.dumps({
            "status": "empty",
            "message": "No thoughts stored yet. Use 'store' to add thoughts first.",
        })

    result = psn.recall(cue, top_k=min(top_k, 20), use_attractor=True)

    matches = []
    for m in result["matches"]:
        matches.append({
            "text": m["text"],
            "similarity": m["similarity"],
            "id": m["id"],
            "tags": m.get("tags", []),
        })

    return json.dumps({
        "cue": cue,
        "matches": matches,
        "attractor_steps": result["n_steps"],
        "energy_start": round(result["energy_start"], 4),
        "energy_final": round(result["energy_final"], 4),
        "elapsed_ms": round(result["elapsed_ms"], 1),
        "n_stored": psn.memory.count,
    }, ensure_ascii=False)


@mcp.tool()
def store(thought: str, tags: list[str] | None = None) -> str:
    """Store a thought into the Personal Synaptic Network.

    The thought is encoded into a sparse neural activation pattern,
    then Hebbian learning strengthens connections between co-active neurons.
    The network physically changes shape to remember this thought.

    Args:
        thought: The text to store (any natural language)
        tags: Optional labels (e.g., ["work", "decision"])
    """
    psn = get_psn()
    result = psn.store(thought, tags=tags or [])
    psn.save()

    return json.dumps({
        "status": "stored",
        "id": result["id"],
        "active_neurons": result["n_active_neurons"],
        "energy": round(result["energy"], 4),
        "total_patterns": result["learn_count"],
        "elapsed_ms": round(result["elapsed_ms"], 1),
    })


@mcp.tool()
def compass(decision: str, n_perspectives: int = 5) -> str:
    """Use the PSN as a personal decision compass.

    Given a decision or dilemma, the network retrieves your own stored
    thought patterns most associated with it. This reveals how YOU
    actually think about this kind of problem — based on patterns
    from your real conversations and thoughts.

    This is personal validation: checking a solution against who you ARE.

    Args:
        decision: The decision or dilemma to validate (natural language)
        n_perspectives: How many thought patterns to surface (default 5)
    """
    psn = get_psn()

    if psn.memory.count == 0:
        return json.dumps({
            "status": "empty",
            "message": "No thoughts stored. Feed your conversations first with 'python -m psn ingest'.",
        })

    result = psn.recall(decision, top_k=min(n_perspectives, 10), use_attractor=True)

    perspectives = []
    for m in result["matches"]:
        perspectives.append({
            "thought": m["text"],
            "resonance": m["similarity"],
        })

    avg_resonance = (
        sum(p["resonance"] for p in perspectives) / len(perspectives)
        if perspectives else 0
    )

    if avg_resonance >= 0.5:
        signal = "STRONG — your stored patterns clearly relate to this decision"
    elif avg_resonance >= 0.35:
        signal = "MODERATE — some relevant patterns found"
    else:
        signal = "WEAK — this may be outside your stored experience"

    return json.dumps({
        "decision": decision,
        "signal": signal,
        "avg_resonance": round(avg_resonance, 4),
        "perspectives": perspectives,
        "n_stored_thoughts": psn.memory.count,
    }, ensure_ascii=False)


@mcp.tool()
def status() -> str:
    """Get the current status of the Personal Synaptic Network.

    Returns network metrics: neuron count, stored patterns,
    weight norms, memory usage, and device info.
    """
    psn = get_psn()
    s = psn.status()
    return json.dumps(s)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PSN MCP Server")
    parser.add_argument("--checkpoint", "-c", help="Path to PSN checkpoint file")
    parser.add_argument("--http", action="store_true", help="Use HTTP transport instead of stdio")
    parser.add_argument("--port", type=int, default=8430, help="HTTP port (default 8430)")
    args = parser.parse_args()

    global _checkpoint_path
    if args.checkpoint:
        _checkpoint_path = Path(args.checkpoint)

    if args.http:
        mcp.settings.host = "0.0.0.0"
        mcp.settings.port = args.port
        mcp.run(transport="streamable-http")
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
