# Post-Reasoning Engine

**A Personal Synaptic Network (PSN) that stores your thought patterns in a neural network and retrieves them as a decision compass.**

Pre-reasoning validates the *problem structure* before you commit.
Post-reasoning validates the *solution against you* after you think.

This repo gives you the post-reasoning half: a synthetic brain shaped by YOUR thoughts.

---

## What It Does

1. You feed it your thoughts (conversations, notes, ideas — any text)
2. It converts each thought into a sparse activation pattern across 50,000 artificial neurons
3. Neurons that fire together get wired together (Hebbian learning — same rule as your biological brain)
4. When you give it a cue, it retrieves the stored patterns most associated with that cue

**It's not a search engine.** It doesn't match keywords. It activates neurons and lets the network settle into the closest stored pattern — like how you remember by association.

## How It Works

```
Your thought (text)
    → Sentence embedding (384d)
    → Sparse projection (50,000 neurons, 1,000 fire)
    → Hebbian learning (strengthen co-active connections)
    → Synaptic store (weight matrices)

Your cue (text)
    → Same encoding
    → Attractor dynamics (network settles to nearest stored pattern)
    → Top matches returned with similarity scores
```

**Architecture**: Modern Hopfield Network with sparse block connectivity (100 blocks of 500 neurons). Energy-based attractor dynamics. Online Hebbian learning — no backprop, no optimizer, no training loop.

## Quick Start

```bash
# Clone and build
git clone https://github.com/luislozanogmia/post-reasoning-engine.git
cd post-reasoning-engine
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

## Project Structure

```
psn/
    __init__.py         # Package init
    config.py           # All hyperparameters
    encoder.py          # Text → 384d embedding (sentence-transformers)
    projection.py       # 384d → 50K sparse activation (k-Winners-Take-All)
    hopfield.py         # Hopfield network: blocks, weights, attractor dynamics
    hebbian.py          # Hebbian learning: co-activation strengthening + decay
    memory_store.py     # Thought metadata index
    psn_core.py         # Orchestrator: store / recall / status / save / load
    persistence.py      # Checkpoint save/load
    cli.py              # Command-line interface
    ingest.py           # Conversation extraction and chunking
buildpsn.sh             # One-command setup
requirements.txt        # Dependencies
```

## Requirements

- Python 3.10+
- PyTorch 2.0+ (CUDA optional — works on CPU too)
- ~400MB disk for the network + embedding model
- Any GPU with 2GB+ VRAM (or CPU-only mode)

## Validation Results

Tested against 15 historical decision inflection points from one person's conversation history:

| Metric | Result |
|---|---|
| Decisions tested | 15 |
| Aligned with actual outcome | 13 (86.7%) |
| Partially aligned | 2 (13.3%) |
| Wrong direction | 0 (0%) |

The compass never pointed the wrong way. At worst, it captured the tension without fully resolving it.

### Example: Career Decision

**Cue**: "staying at a job I hate for stability"

**PSN returned**:
- `[0.465]` "How can I work on myself to tolerate while I get an exit"
- `[0.424]` "Since I have a small baby and planning a 2nd one I have time... I need 2 years to pay all debts"

**What actually happened**: Stayed for runway, built the startup on the side, exited on his own terms.

The network surfaced the person's own reasoning — patterns stored months earlier — that aligned with the decision they eventually made.

## The Science

- **Modern Hopfield Network**: Exponential storage capacity, energy-based retrieval. Patterns are energy minima; recall is gradient descent on the energy landscape.
- **Hebbian Learning**: "Neurons that fire together wire together." No gradients, no loss function. Pure co-activation strengthening.
- **Sparse Distributed Representations**: Only 2% of neurons fire per thought (~1,000 out of 50,000). Similar to biological cortical coding.
- **Block Connectivity**: 100 modules of 500 neurons, dense within blocks, sparse between blocks. Mirrors cortical column architecture.

## Combining with Pre-Reasoning

This engine is the **post** half. For the **pre** half (structural analysis before LLM commitment), see the [Pre-Reasoning Engine](https://github.com/luislozanogmia/pre-reasoning-engine).

Together:
```
Problem → Pre-Reasoning (structure) → LLM (solution) → Post-Reasoning / PSN (personal validation)
```

Pre-reasoning prevents structural drift. Post-reasoning prevents personally wrong answers.

## License

MIT — see [LICENSE](LICENSE).

## Built By

[Mia Labs](https://mia-labs.com) — Luis Lozano + Dr. Shannon (Claude Opus 4.6)

Built in one session. March 10-11, 2026.
