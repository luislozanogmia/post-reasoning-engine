# Personal Synaptic Network (PSN)
A blank-slate neural network that stores your thought patterns as synaptic weights and retrieves them as a decision compass.

It is not a chatbot memory, not RAG, not a knowledge graph. It is a Hebbian attractor network where neurons that fire together wire together, shaped entirely by your cognitive output. You feed it your thoughts — notes, conversations, ideas, anything in text. Each thought activates a sparse pattern across thousands of artificial neurons, about 2% firing per thought, mimicking the sparse distributed representations your biological brain uses. Co-active neurons strengthen their connections. Over time, the network builds a map of your associative structure: which ideas you connect, which domains you bridge, which patterns you repeat. Retrieval is not search — it is pattern completion. Give it a partial cue and the network settles into the closest stored pattern, the same way you remember a song from its first three notes rather than searching a database. Every AI memory system today stores and retrieves text. The PSN stores and retrieves patterns — the shape of how you think, not just what you said. When connected to an LLM, it acts as a personal validation layer: the model doesn't just know your facts, it navigates your cognitive topology before it generates a single token.

**A Personal Synaptic Network (PSN) that stores your thought patterns in a neural network and retrieves them as a decision compass.**

---

## What It Does

1. You feed it your thoughts (conversations, notes, ideas — any text)
2. It converts each thought into a sparse activation pattern across 200 artificial neurons (expandable)
3. Neurons that fire together get wired together (Hebbian learning — same rule as your biological brain)
4. When you give it a cue, it retrieves the stored patterns most associated with that cue

**It's not a search engine.** It doesn't match keywords. It activates neurons and lets the network settle into the closest stored pattern — like how you remember by association.

## How It Works

```
Your thought (text)
    → Sentence embedding (384d)
    → Sparse projection (200 neurons, top 2% fire)
    → Hebbian learning (strengthen co-active connections)
    → Synaptic store (weight matrices)

Your cue (text)
    → Same encoding
    → Attractor dynamics (network settles to nearest stored pattern)
    → Top matches returned with similarity scores
```

**Architecture**: Modern Hopfield Network with sparse block connectivity (4 blocks of 50 neurons). Energy-based attractor dynamics. Online Hebbian learning — no backprop, no optimizer, no training loop.

## Quick Start

```bash
# Clone and build
git clone https://github.com/luislozanogmia/personal_synaptic_network.git
cd personal_synaptic_network
bash buildpsn.sh

# Store a thought
python -m psn store "The best use of AI is as a partner, not a replacement"

# Store more thoughts
python -m psn store "I am more of a 0-to-1 builder than a 1-to-100 optimizer"
python -m psn store "Slow down to speed up"
python -m psn store "Structure over capability, always"

# Recall
python -m psn recall "what kind of builder am I"
python -m psn recall "how should I approach problems"

# Status
python -m psn status

# List stored thoughts
python -m psn list
```

## Feed Your Conversations

```bash
# Feed a ChatGPT export
python -m psn ingest --chatgpt path/to/chatgpt-export.zip

# Feed a Claude export
python -m psn ingest --claude-json path/to/conversations.json

# Dry run (count thoughts without storing)
python -m psn ingest --chatgpt export.zip --dry-run
```

ChatGPT exports: Settings → Data Controls → Export Data.
Claude exports: Settings → Account → Export Data.

## MCP Server — Connect to Your Chat

The PSN includes an MCP (Model Context Protocol) server so you can use your personal compass directly from Claude Desktop, Claude Code, Cursor, Cline, or any MCP-compatible client.

### Setup

**1. Build the PSN first** (see Quick Start above), then feed it your thoughts.

**2. Add to your MCP client config:**

For **Claude Desktop** (`claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "psn": {
      "command": "python",
      "args": ["-m", "psn.mcp_server"],
      "cwd": "/path/to/personal_synaptic_network",
      "env": {
        "PSN_CHECKPOINT": "/path/to/personal_synaptic_network/psn/checkpoints/psn_latest.pt"
      }
    }
  }
}
```

For **Claude Code** (`.claude/settings.json`):
```json
{
  "mcpServers": {
    "psn": {
      "command": "python",
      "args": ["-m", "psn.mcp_server"],
      "cwd": "/path/to/personal_synaptic_network",
      "env": {
        "PSN_CHECKPOINT": "/path/to/personal_synaptic_network/psn/checkpoints/psn_latest.pt"
      }
    }
  }
}
```

For **Cursor** (`~/.cursor/mcp.json`):
```json
{
  "mcpServers": {
    "psn": {
      "command": "python",
      "args": ["-m", "psn.mcp_server"],
      "cwd": "/path/to/personal_synaptic_network"
    }
  }
}
```

Replace `/path/to/personal_synaptic_network` with your actual clone path. Use the venv Python if needed (e.g., `/path/to/personal_synaptic_network/.venv/bin/python`).

**3. Restart your client.** You should see 4 new tools available.

### MCP Tools

| Tool | What it does |
|------|-------------|
| `recall` | Query the network with a natural language cue. Returns associated thought patterns with similarity scores. |
| `store` | Add a new thought to the network. Hebbian learning fires immediately. |
| `compass` | Decision validation — surfaces your own thought patterns most relevant to a decision or dilemma. Returns a signal strength (STRONG/MODERATE/WEAK). |
| `status` | Network metrics: neuron count, stored patterns, weight norms, memory usage. |

### Example Usage (in chat)

> **You**: I'm deciding whether to refactor the auth module or keep patching it. Check my PSN compass.
>
> **Claude** (calls `compass`): Your PSN shows STRONG resonance (0.52). Here are the patterns it surfaced from your own thoughts:
> - "Fix the foundation before adding floors" (0.58)
> - "Technical debt compounds faster than financial debt" (0.49)
> - "Measure twice, cut once" (0.47)
>
> Your stored patterns suggest you consistently favor structural fixes over quick patches, but your own thinking also values pragmatic timing.

### HTTP Mode

For remote or multi-client access:

```bash
python -m psn.mcp_server --http --port 8430
```

This starts a streamable-HTTP MCP server at `http://localhost:8430/mcp`.

### Environment Variables

| Variable | Description |
|----------|-------------|
| `PSN_CHECKPOINT` | Path to checkpoint file (overrides auto-detection) |

## Project Structure

```
psn/
    __init__.py         # Package init
    config.py           # All hyperparameters
    encoder.py          # Text → 384d embedding (sentence-transformers)
    projection.py       # 384d → sparse activation (k-Winners-Take-All)
    hopfield.py         # Hopfield network: blocks, weights, attractor dynamics
    hebbian.py          # Hebbian learning: co-activation strengthening + decay
    memory_store.py     # Thought metadata index
    psn_core.py         # Orchestrator: store / recall / status / save / load
    persistence.py      # Checkpoint save/load
    cli.py              # Command-line interface
    ingest.py           # Conversation extraction and chunking
    mcp_server.py       # MCP server (stdio + HTTP transports)
buildpsn.sh             # One-command setup
requirements.txt        # Dependencies
```

## Requirements

- Python 3.10+
- PyTorch 2.0+ (CPU only — no GPU needed)
- FastMCP 1.0+ (for MCP server)
- ~50MB disk for the network + embedding model

## Validation Results

Tested against 33 historical decision inflection points from one person's conversation history (86,505 stored thoughts):

**Round 1 — 15 life + career decisions:**

| Metric | Result |
|---|---|
| Decisions tested | 15 |
| Aligned with actual outcome | 13 (86.7%) |
| Partially aligned | 2 (13.3%) |
| Wrong direction | 0 (0%) |

**Round 2 — 18 architectural + technical decisions:**

| Metric | Result |
|---|---|
| Decisions tested | 18 |
| Strong alignment | 17 (94.4%) |
| Moderate alignment | 1 (5.6%) |
| Weak / wrong | 0 (0%) |

The compass never pointed the wrong way across 33 tested decisions.

### Example: Technical Decision

**Cue**: "should I rewrite the pipeline or keep patching"

**PSN returned**:
- `[0.465]` "Every patch adds one more thing to remember — rewrites reset the complexity clock"
- `[0.424]` "Measure the cost of NOT rewriting — if it keeps growing, the rewrite pays for itself"

**What actually happened**: Rewrote the core module, eliminated three recurring bug classes, shipping velocity doubled.

### Example: Architecture Decision

**Cue**: "catastrophic forgetting is collision not decay slot competition"

**PSN returned**:
- `[0.737]` "Catastrophic forgetting? more like Catastrophic collisions..."
- `[0.563]` "Stage 3 — Early Attention Layers (Pattern Hooking)..."

The network surfaced the person's own reasoning — patterns stored months earlier — that aligned with decisions they eventually made.

## The Science

- **Modern Hopfield Network**: Exponential storage capacity, energy-based retrieval. Patterns are energy minima; recall is gradient descent on the energy landscape.
- **Hebbian Learning**: "Neurons that fire together wire together." No gradients, no loss function. Pure co-activation strengthening.
- **Sparse Distributed Representations**: Only 2% of neurons fire per thought. Similar to biological cortical coding.
- **Block Connectivity**: 4 modules of 50 neurons (default), dense within blocks, sparse between blocks. Mirrors cortical column architecture.

## Why 200 Neurons Is Enough

We tested 50,000 neurons (462 MB checkpoint, GPU required) against 200 neurons (0.3 MB checkpoint, CPU only) on the same 13K+ thought corpus. Recall quality is identical — the sentence-transformer embedding (384d) does the heavy lifting for similarity matching, while the Hopfield attractor dynamics work just as well with sparse 200-neuron activations. The 50K network used 1,500x more memory for no measurable recall improvement.

| Config | Neurons | Checkpoint | RAM | Device | Recall quality |
|--------|---------|-----------|-----|--------|---------------|
| Original | 50,000 | 462 MB | 1.8 GB | GPU | baseline |
| Current | 200 | 0.3 MB | ~36 MB | CPU | same |

Start with 200. Scale up only if you have evidence of attractor collapse with your specific corpus.

## Combining with Pre-Reasoning

For structural analysis before LLM commitment, see the [Pre-Reasoning Engine](https://www.mia-labs.com/pre-reasoning).

Together:
```
Problem → Pre-Reasoning (structure) → LLM (solution) → PSN (personal validation)
```

Pre-reasoning prevents structural drift. PSN prevents personally wrong answers.

## License

MIT — see [LICENSE](LICENSE).

## Built By

[Mia Labs](https://mia-labs.com) — Luis Lozano + Dr. Shannon (Claude Opus 4.6)

Built in one session. March 10-11, 2026.
