"""Ingest conversations into the PSN — extract human messages, chunk, and store."""

import json
import zipfile
import re
import time
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Thought:
    """A single extracted thought from a conversation."""
    text: str
    source: str  # "chatgpt" or "claude"
    conversation: str  # conversation name/id
    timestamp: float  # unix timestamp
    tags: list[str]


def extract_chatgpt_thoughts(zip_path: str) -> list[Thought]:
    """Extract human messages from ChatGPT export zip.

    ChatGPT format: list of conversations, each with 'mapping' containing
    nested message nodes. Human messages have author.role == 'user'.
    """
    thoughts = []
    zf = zipfile.ZipFile(zip_path)

    conv_files = sorted([n for n in zf.namelist() if n.startswith("conversations-") and n.endswith(".json")])
    print(f"ChatGPT: found {len(conv_files)} conversation files")

    for cf in conv_files:
        print(f"  Processing {cf}...")
        data = json.loads(zf.read(cf))

        for conv in data:
            conv_name = conv.get("title", conv.get("conversation_id", "untitled"))
            conv_time = conv.get("create_time", 0) or 0
            mapping = conv.get("mapping", {})

            for node_id, node in mapping.items():
                msg = node.get("message")
                if not msg:
                    continue
                if msg.get("author", {}).get("role") != "user":
                    continue

                # Extract text from content parts
                parts = msg.get("content", {}).get("parts", [])
                text = " ".join(str(p) for p in parts if isinstance(p, str))
                text = text.strip()

                if len(text) < 15:
                    continue  # skip trivial messages ("ok", "yes", etc.)

                msg_time = msg.get("create_time", conv_time) or conv_time

                # Chunk long messages into thought-sized pieces
                for chunk in chunk_text(text):
                    thoughts.append(Thought(
                        text=chunk,
                        source="chatgpt",
                        conversation=str(conv_name),
                        timestamp=float(msg_time),
                        tags=["chatgpt", "mia"],
                    ))

    print(f"ChatGPT: extracted {len(thoughts)} thoughts")
    return thoughts


def extract_claude_thoughts(source_path: str) -> list[Thought]:
    """Extract human messages from Claude conversations.

    Handles both:
    - ZIP exports (conversations.json inside zip)
    - Raw JSON files (all_conversations_raw.json or individual chat files)
    """
    thoughts = []

    path = Path(source_path)

    if path.suffix == ".zip":
        zf = zipfile.ZipFile(source_path)
        data = json.loads(zf.read("conversations.json"))
    elif path.suffix == ".json":
        with open(source_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unknown format: {path.suffix}")

    if not isinstance(data, list):
        data = [data]

    print(f"Claude: found {len(data)} conversations")

    for conv in data:
        conv_name = conv.get("name", conv.get("uuid", "untitled"))
        messages = conv.get("chat_messages", conv.get("messages", []))

        for msg in messages:
            sender = msg.get("sender", msg.get("role", ""))
            if sender != "human":
                continue

            # Get text content
            text = msg.get("text", "")
            if not text:
                content = msg.get("content", "")
                if isinstance(content, list):
                    text = " ".join(
                        b.get("text", "") for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                elif isinstance(content, str):
                    text = content

            text = text.strip()
            if len(text) < 15:
                continue

            msg_time = msg.get("created_at", "")
            if isinstance(msg_time, str) and msg_time:
                try:
                    from datetime import datetime
                    ts = datetime.fromisoformat(msg_time.replace("Z", "+00:00")).timestamp()
                except Exception:
                    ts = 0.0
            elif isinstance(msg_time, (int, float)):
                ts = float(msg_time)
            else:
                ts = 0.0

            for chunk in chunk_text(text):
                thoughts.append(Thought(
                    text=chunk,
                    source="claude",
                    conversation=str(conv_name),
                    timestamp=ts,
                    tags=["claude"],
                ))

    print(f"Claude: extracted {len(thoughts)} thoughts")
    return thoughts


def chunk_text(text: str, max_len: int = 500, min_len: int = 15) -> list[str]:
    """Split text into thought-sized chunks.

    Strategy:
    - Split on sentence boundaries (. ! ? followed by space/newline)
    - Merge short sentences into chunks up to max_len
    - Each chunk should be a coherent thought unit
    """
    # Clean up
    text = re.sub(r'\s+', ' ', text).strip()

    if len(text) <= max_len:
        return [text] if len(text) >= min_len else []

    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current = ""

    for sent in sentences:
        if not sent.strip():
            continue
        if len(current) + len(sent) + 1 <= max_len:
            current = (current + " " + sent).strip() if current else sent
        else:
            if len(current) >= min_len:
                chunks.append(current)
            current = sent

    if current and len(current) >= min_len:
        chunks.append(current)

    # Handle case where a single sentence is longer than max_len
    final = []
    for chunk in chunks:
        if len(chunk) > max_len * 2:
            # Force split at word boundaries
            words = chunk.split()
            sub = ""
            for w in words:
                if len(sub) + len(w) + 1 > max_len:
                    if len(sub) >= min_len:
                        final.append(sub)
                    sub = w
                else:
                    sub = (sub + " " + w).strip() if sub else w
            if len(sub) >= min_len:
                final.append(sub)
        else:
            final.append(chunk)

    return final


def deduplicate_thoughts(thoughts: list[Thought], threshold: float = 0.95) -> list[Thought]:
    """Remove near-duplicate thoughts (exact or near-exact matches)."""
    seen = set()
    unique = []
    for t in thoughts:
        # Normalize for dedup
        key = re.sub(r'\s+', ' ', t.text.lower().strip())
        if key not in seen:
            seen.add(key)
            unique.append(t)
    return unique


def ingest_to_psn(thoughts: list[Thought], psn, batch_size: int = 50, max_thoughts: int = None):
    """Feed extracted thoughts into the PSN.

    Args:
        thoughts: list of Thought objects
        psn: PSN instance
        batch_size: print progress every N thoughts
        max_thoughts: limit ingestion (for testing)
    """
    if max_thoughts:
        thoughts = thoughts[:max_thoughts]

    total = len(thoughts)
    print(f"Ingesting {total} thoughts into PSN...")
    t0 = time.perf_counter()

    for i, thought in enumerate(thoughts):
        result = psn.store(thought.text, tags=thought.tags)

        if (i + 1) % batch_size == 0:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{total}] {rate:.1f} thoughts/sec, ETA: {eta:.0f}s, "
                  f"energy={result['energy']:.4f}")

    elapsed = time.perf_counter() - t0
    print(f"Done. {total} thoughts in {elapsed:.1f}s ({total/elapsed:.1f}/sec)")


if __name__ == "__main__":
    print("Use: python -m psn ingest --chatgpt <path> or --claude-json <path>")
