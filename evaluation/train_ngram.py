"""Train and persist a token n-gram LM for speculative decoding."""

import argparse
import json
import os

from transformers import AutoTokenizer

from model.ngram import train_token_ngram_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-path", type=str, required=True)
    parser.add_argument("--corpus-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--order", type=int, default=4, help="N-gram order (e.g., 4 for 4-gram).")
    parser.add_argument(
        "--max-texts",
        type=int,
        default=200000,
        help="Maximum texts to ingest; set <=0 to use all available texts.",
    )
    parser.add_argument(
        "--jsonl-text-field",
        type=str,
        default="text",
        help="JSONL field used as text when corpus rows are JSON objects.",
    )
    parser.add_argument(
        "--stats-path",
        type=str,
        default=None,
        help="Optional stats JSON output path. Defaults to <output-path>.stats.json.",
    )

    args = parser.parse_args()

    if args.order < 1:
        raise ValueError("--order must be >= 1")

    max_texts = None if args.max_texts <= 0 else args.max_texts
    stats_path = args.stats_path or f"{args.output_path}.stats.json"

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    model, stats = train_token_ngram_model(
        tokenizer=tokenizer,
        corpus_path=args.corpus_path,
        order=args.order,
        max_texts=max_texts,
        jsonl_text_field=args.jsonl_text_field,
    )

    model.save(args.output_path)

    stats_payload = {
        "tokenizer_path": args.tokenizer_path,
        "corpus_path": args.corpus_path,
        "output_path": args.output_path,
        "jsonl_text_field": args.jsonl_text_field,
        "max_texts": max_texts,
        "order": stats["order"],
        "texts": stats["texts"],
        "tokens": stats["tokens"],
        "total_tokens": model.total_tokens,
    }

    stats_dir = os.path.dirname(stats_path)
    if stats_dir:
        os.makedirs(stats_dir, exist_ok=True)

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_payload, f, ensure_ascii=True, indent=2, sort_keys=True)

    print(
        "Saved token n-gram model:",
        f"order={stats['order']}",
        f"texts={stats['texts']}",
        f"tokens={stats['tokens']}",
        f"model={args.output_path}",
        f"stats={stats_path}",
    )
