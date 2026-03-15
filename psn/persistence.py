"""Checkpoint save/load for the full PSN state."""

from pathlib import Path
import torch
from .config import PSNConfig


PSN_CHECKPOINT_VERSION = 1


def save_checkpoint(path: Path, config: PSNConfig, hopfield_state: dict,
                    projection_state: dict, memory_state: dict,
                    learner_count: int):
    """Save complete PSN state to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "version": PSN_CHECKPOINT_VERSION,
        "config": {
            "n_neurons": config.n_neurons,
            "n_blocks": config.n_blocks,
            "block_size": config.block_size,
            "d_embedding": config.d_embedding,
            "k_winners_pct": config.k_winners_pct,
            "beta": config.beta,
            "eta": config.eta,
            "decay_rate": config.decay_rate,
            "inter_block_k": config.inter_block_k,
            "inter_block_density": config.inter_block_density,
        },
        "hopfield": hopfield_state,
        "projection": projection_state,
        "memory": memory_state,
        "learner_count": learner_count,
    }
    torch.save(checkpoint, path)


def load_checkpoint(path: Path) -> dict:
    """Load PSN checkpoint from disk.

    Returns:
        dict with keys: version, config, hopfield, projection, memory, learner_count
    """
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    if checkpoint.get("version", 0) != PSN_CHECKPOINT_VERSION:
        raise ValueError(f"Checkpoint version mismatch: expected {PSN_CHECKPOINT_VERSION}, "
                         f"got {checkpoint.get('version', 'unknown')}")

    return checkpoint
