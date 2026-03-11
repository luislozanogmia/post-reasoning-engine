"""CLI for the Personal Synaptic Network."""

import argparse
from pathlib import Path


def get_psn(checkpoint=None):
    from .psn_core import PSN
    from .config import PSNConfig
    config = PSNConfig()
    psn = PSN(config)
    ckpt = Path(checkpoint) if checkpoint else config.checkpoint_dir / "psn_latest.pt"
    if ckpt.exists():
        psn.load(ckpt)
    return psn


def cmd_store(args):
    psn = get_psn(args.checkpoint)
    result = psn.store(args.text, tags=args.tags)
    print(f"Stored thought #{result['id']}")
    print(f"  Active neurons: {result['n_active_neurons']}")
    print(f"  Energy: {result['energy']:.4f}")
    print(f"  Total patterns: {result['learn_count']}")
    print(f"  Time: {result['elapsed_ms']:.1f}ms")
    psn.save()


def cmd_recall(args):
    psn = get_psn(args.checkpoint)
    result = psn.recall(args.cue, top_k=args.top_k)
    print(f'Recall for: "{args.cue}"')
    print(f"  Attractor steps: {result['n_steps']}, Time: {result['elapsed_ms']:.1f}ms")
    print()
    if result["matches"]:
        for m in result["matches"]:
            print(f"  [{m['similarity']:.4f}] #{m['id']}: {m['text'][:100]}")
    else:
        print("  No stored patterns yet. Use 'store' to add thoughts first.")


def cmd_status(args):
    psn = get_psn(args.checkpoint)
    print("PSN Status:")
    for k, v in psn.status().items():
        print(f"  {k}: {v}")


def cmd_list(args):
    psn = get_psn(args.checkpoint)
    entries = psn.memory.metadata_dict()
    if not entries:
        print("No stored thoughts.")
        return
    print(f"Stored thoughts ({len(entries)}):")
    for e in entries:
        text = e['text'][:80]
        print(f"  #{e['id']}: {text}{'...' if len(e['text']) > 80 else ''}")


def cmd_ingest(args):
    from .ingest import extract_chatgpt_thoughts, extract_claude_thoughts, deduplicate_thoughts
    import time

    all_thoughts = []
    if args.chatgpt:
        all_thoughts.extend(extract_chatgpt_thoughts(args.chatgpt))
    if args.claude_json:
        all_thoughts.extend(extract_claude_thoughts(args.claude_json))
    if not all_thoughts:
        print("No sources specified. Use --chatgpt or --claude-json")
        return

    all_thoughts.sort(key=lambda t: t.timestamp)
    before = len(all_thoughts)
    all_thoughts = deduplicate_thoughts(all_thoughts)
    print(f"Extracted: {before} -> {len(all_thoughts)} unique thoughts")

    if args.dry_run:
        print("Dry run. Sample:")
        for t in all_thoughts[:5]:
            print(f"  [{t.source}] {t.text[:100]}")
        return

    psn = get_psn(args.checkpoint)
    total = min(len(all_thoughts), args.max) if args.max else len(all_thoughts)
    t0 = time.perf_counter()

    for i, t in enumerate(all_thoughts[:total]):
        result = psn.store(t.text, tags=t.tags)
        if (i + 1) % 500 == 0:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / elapsed
            print(f"  [{i+1}/{total}] {rate:.1f}/sec, energy={result['energy']:.4f}")

    elapsed = time.perf_counter() - t0
    print(f"Done: {total} thoughts in {elapsed:.1f}s ({total/elapsed:.1f}/sec)")
    psn.save()
    print(f"Saved. Status: {psn.status()}")


def main():
    parser = argparse.ArgumentParser(description="Personal Synaptic Network")
    parser.add_argument("--checkpoint", "-c")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("store")
    p.add_argument("text")
    p.add_argument("--tags", nargs="*", default=[])

    p = sub.add_parser("recall")
    p.add_argument("cue")
    p.add_argument("--top-k", type=int, default=5)

    sub.add_parser("status")
    sub.add_parser("list")

    p = sub.add_parser("ingest")
    p.add_argument("--chatgpt", help="ChatGPT export zip path")
    p.add_argument("--claude-json", help="Claude conversations JSON path")
    p.add_argument("--max", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    {"store": cmd_store, "recall": cmd_recall, "status": cmd_status,
     "list": cmd_list, "ingest": cmd_ingest}[args.command](args)


if __name__ == "__main__":
    main()
