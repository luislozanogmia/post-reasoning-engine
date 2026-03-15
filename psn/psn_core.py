"""PSN: the orchestrator. Store thoughts, retrieve associations."""

import time
from pathlib import Path
import torch
from .config import PSNConfig
from .encoder import TextEncoder
from .projection import SparseProjection
from .hopfield import HopfieldNetwork
from .hebbian import HebbianLearner
from .memory_store import MemoryStore
from .persistence import save_checkpoint, load_checkpoint


class PSN:
    """Personal Synaptic Network — store and retrieve one person's thought patterns."""

    def __init__(self, config: PSNConfig = None):
        self.config = config or PSNConfig()
        self.encoder = TextEncoder(self.config)
        self.projection = SparseProjection(self.config)
        self.network = HopfieldNetwork(self.config)
        self.learner = HebbianLearner(self.config)
        self.memory = MemoryStore()

    def store(self, text: str, tags: list[str] = None) -> dict:
        """Store a thought into the synaptic network."""
        t0 = time.perf_counter()
        embedding = self.encoder.encode(text)
        activation, active_indices = self.projection.project(embedding)
        self.learner.learn(self.network, activation, active_indices)
        memory_id = self.memory.add(text, embedding, activation, active_indices, tags)
        energy = self.network.energy(activation)
        elapsed = time.perf_counter() - t0
        return {
            "id": memory_id,
            "n_active_neurons": len(active_indices),
            "energy": energy,
            "learn_count": self.learner.learn_count,
            "elapsed_ms": elapsed * 1000,
        }

    def recall(self, cue: str, top_k: int = 5, use_attractor: bool = True) -> dict:
        """Retrieve associated patterns from a text cue."""
        t0 = time.perf_counter()
        embedding = self.encoder.encode(cue)
        activation, _ = self.projection.project(embedding)

        if use_attractor and self.memory.count > 0:
            converged, energy_history, n_steps = self.network.attract(activation)
        else:
            converged = activation
            energy_history = [self.network.energy(activation)]
            n_steps = 0

        # Primary: embedding-based matching
        embed_matches = self.memory.find_nearest_by_embedding(embedding, top_k)

        # Secondary: Jaccard similarity on attractor output
        jaccard_raw = self.memory.find_nearest(converged, top_k) if use_attractor else []

        matches = []
        for memory_id, sim in embed_matches:
            entry = self.memory.get(memory_id)
            if entry:
                entry.retrieval_count += 1
                matches.append({
                    "id": entry.id,
                    "text": entry.text,
                    "similarity": round(sim, 4),
                    "tags": entry.tags,
                    "retrieval_count": entry.retrieval_count,
                })

        jaccard_matches = []
        for memory_id, sim in jaccard_raw:
            entry = self.memory.get(memory_id)
            if entry:
                jaccard_matches.append({
                    "id": entry.id, "text": entry.text, "similarity": round(sim, 4),
                })

        elapsed = time.perf_counter() - t0
        return {
            "matches": matches,
            "jaccard_matches": jaccard_matches,
            "n_steps": n_steps,
            "energy_start": energy_history[0] if energy_history else 0,
            "energy_final": energy_history[-1] if energy_history else 0,
            "elapsed_ms": elapsed * 1000,
        }

    def status(self) -> dict:
        intra_norms = self.network.W_intra.norm(dim=(1, 2)).cpu().tolist()
        avg_norm = sum(intra_norms) / len(intra_norms) if intra_norms else 0
        n_inter = self.network._inter_values.numel()
        inter_active = (self.network._inter_values.abs() > 1e-8).sum().item()
        intra_bytes = self.network.W_intra.numel() * 4
        inter_bytes = (self.network._inter_indices.numel() + self.network._inter_values.numel()) * 4
        proj_bytes = self.projection.W_proj.numel() * 4
        total_mb = (intra_bytes + inter_bytes + proj_bytes) / 1e6
        return {
            "n_neurons": self.config.n_neurons,
            "n_blocks": self.config.n_blocks,
            "n_stored_patterns": self.memory.count,
            "learn_count": self.learner.learn_count,
            "avg_intra_weight_norm": round(avg_norm, 4),
            "inter_connections_active": inter_active,
            "memory_mb": round(total_mb, 1),
            "device": self.config.device,
        }

    def save(self, path: str | Path = None):
        if path is None:
            path = self.config.checkpoint_dir / "psn_latest.pt"
        path = Path(path)
        save_checkpoint(path, self.config, self.network.state_dict(),
                        self.projection.state_dict(), self.memory.state_dict(),
                        self.learner.learn_count)
        return str(path)

    def load(self, path: str | Path):
        path = Path(path)
        checkpoint = load_checkpoint(path)
        self.network.load_state_dict(checkpoint["hopfield"])
        self.projection.load_state_dict(checkpoint["projection"])
        self.memory.load_state_dict(checkpoint["memory"])
        self.learner.learn_count = checkpoint.get("learner_count", 0)
