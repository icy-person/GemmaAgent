#!/usr/bin/env python3
"""Build a real text corpus for GemmaAgent pretraining.

Downloads public-domain books plus the Tiny Shakespeare corpus, removes common
Project Gutenberg boilerplate, normalizes text, removes duplicate paragraphs,
and creates deterministic document-level train/validation splits.

The downloader is resumable at the source level: successfully downloaded files
are cached under data/raw and are reused on later runs. Transient and partial
HTTP downloads are retried before a source is skipped.

Outputs:
  data/train.txt
  data/val.txt
  data/corpus_manifest.json

Usage:
  python3 scripts/prepare_real_corpus.py
  python3 scripts/prepare_real_corpus.py --min-chars 200
  python3 scripts/prepare_real_corpus.py --val-docs 2
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path


SOURCES = [
    {
        "name": "Tiny Shakespeare",
        "url": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "kind": "github",
        "license": "public domain source texts; dataset packaging by source project",
    },
    {
        "name": "Alice's Adventures in Wonderland",
        "url": "https://www.gutenberg.org/files/11/11-0.txt",
        "kind": "gutenberg",
        "license": "public domain in the United States",
    },
    {
        "name": "Pride and Prejudice",
        "url": "https://www.gutenberg.org/files/1342/1342-0.txt",
        "kind": "gutenberg",
        "license": "public domain in the United States",
    },
    {
        "name": "Frankenstein",
        "url": "https://www.gutenberg.org/files/84/84-0.txt",
        "kind": "gutenberg",
        "license": "public domain in the United States",
    },
    {
        "name": "The Adventures of Sherlock Holmes",
        "url": "https://www.gutenberg.org/files/1661/1661-0.txt",
        "kind": "gutenberg",
        "license": "public domain in the United States",
    },
    {
        "name": "Moby Dick",
        "url": "https://www.gutenberg.org/files/2701/2701-0.txt",
        "kind": "gutenberg",
        "license": "public domain in the United States",
    },
    {
        "name": "A Tale of Two Cities",
        "url": "https://www.gutenberg.org/files/98/98-0.txt",
        "kind": "gutenberg",
        "license": "public domain in the United States",
    },
    {
        "name": "The Adventures of Tom Sawyer",
        "url": "https://www.gutenberg.org/files/74/74-0.txt",
        "kind": "gutenberg",
        "license": "public domain in the United States",
    },
]


@dataclass
class Document:
    name: str
    source_url: str
    license: str
    characters: int
    sha256: str


def fetch(url: str, timeout: int = 45, retries: int = 5) -> str:
    """Download UTF-8 text with retries for transient/partial HTTP reads."""
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "GemmaAgent-corpus-builder/1.1",
                    "Accept-Encoding": "identity",
                },
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
            return raw.decode("utf-8", errors="replace")
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            http.client.IncompleteRead,
            TimeoutError,
            OSError,
        ) as exc:
            last_error = exc
            if attempt < retries:
                delay = min(8.0, 1.5 * attempt)
                print(
                    f"  retry {attempt + 1}/{retries} after download error: {exc}",
                    file=sys.stderr,
                )
                time.sleep(delay)
    raise RuntimeError(f"download failed after {retries} attempts: {url}: {last_error}")


def strip_gutenberg(text: str) -> str:
    start_markers = (
        "*** START OF THE PROJECT GUTENBERG EBOOK",
        "*** START OF THIS PROJECT GUTENBERG EBOOK",
    )
    end_markers = (
        "*** END OF THE PROJECT GUTENBERG EBOOK",
        "*** END OF THIS PROJECT GUTENBERG EBOOK",
    )

    lines = text.splitlines()
    start = 0
    end = len(lines)
    for i, line in enumerate(lines):
        if any(marker in line.upper() for marker in start_markers):
            start = i + 1
            break
    for i in range(start, len(lines)):
        if any(marker in lines[i].upper() for marker in end_markers):
            end = i
            break
    return "\n".join(lines[start:end])


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\ufeff", "")
    text = re.sub(r"[\t\x0b\x0c]+", " ", text)
    text = re.sub(r" +", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def paragraphs(text: str, min_chars: int) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for block in re.split(r"\n\s*\n", text):
        block = re.sub(r"\s+", " ", block).strip()
        if len(block) < min_chars:
            continue
        key = hashlib.sha256(block.lower().encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        result.append(block)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--val-docs", type=int, default=2)
    parser.add_argument("--min-chars", type=int, default=120)
    parser.add_argument("--timeout", type=int, default=45)
    args = parser.parse_args()

    if args.val_docs < 1 or args.val_docs >= len(SOURCES):
        parser.error("--val-docs must be at least 1 and smaller than the number of sources")
    if args.min_chars < 1:
        parser.error("--min-chars must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    cleaned: list[tuple[dict, str]] = []
    manifest_docs: list[Document] = []

    print(f"Downloading {len(SOURCES)} real sources...")
    for index, source in enumerate(SOURCES, start=1):
        print(f"[{index}/{len(SOURCES)}] {source['name']}")
        cache_name = (
            re.sub(r"[^a-z0-9]+", "_", source["name"].lower()).strip("_") + ".txt"
        )
        cache_path = raw_dir / cache_name
        try:
            if cache_path.exists() and cache_path.stat().st_size > 0:
                text = cache_path.read_text(encoding="utf-8", errors="replace")
                print(f"  using cache: {cache_path}")
            else:
                text = fetch(source["url"], timeout=args.timeout)
                cache_path.write_text(text, encoding="utf-8")
                print(f"  cached: {cache_path}")
        except Exception as exc:
            print(f"warning: skipping {source['name']}: {exc}", file=sys.stderr)
            continue

        if source["kind"] == "gutenberg":
            text = strip_gutenberg(text)
        text = normalize(text)
        if len(text) < args.min_chars:
            print(
                f"warning: {source['name']} is too small after cleaning",
                file=sys.stderr,
            )
            continue

        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        manifest_docs.append(
            Document(
                name=source["name"],
                source_url=source["url"],
                license=source["license"],
                characters=len(text),
                sha256=digest,
            )
        )
        cleaned.append((source, text))

    if len(cleaned) <= args.val_docs:
        raise RuntimeError("not enough sources downloaded to create train/validation splits")

    # Reserve whole documents for validation. This avoids near-duplicate passages
    # from the same book appearing on both sides of the split.
    val_sources = {source["name"] for source, _ in cleaned[-args.val_docs:]}
    train_parts: list[str] = []
    val_parts: list[str] = []
    seen_global: set[str] = set()

    for source, text in cleaned:
        blocks = paragraphs(text, args.min_chars)
        destination = val_parts if source["name"] in val_sources else train_parts
        destination.append(f"\n\n### {source['name']}\n\n")
        for block in blocks:
            key = hashlib.sha256(block.lower().encode("utf-8")).hexdigest()
            if key in seen_global:
                continue
            seen_global.add(key)
            destination.append(block)

    train_text = "\n\n".join(part.strip() for part in train_parts if part.strip()) + "\n"
    val_text = "\n\n".join(part.strip() for part in val_parts if part.strip()) + "\n"

    train_path = output_dir / "train.txt"
    val_path = output_dir / "val.txt"
    train_path.write_text(train_text, encoding="utf-8")
    val_path.write_text(val_text, encoding="utf-8")

    manifest = {
        "name": "GemmaAgent Real Public-Domain Corpus",
        "version": 2,
        "description": "Deterministically prepared corpus from public-domain literature and Tiny Shakespeare for pretraining pipeline validation.",
        "generated_at_unix": int(time.time()),
        "train_file": str(train_path),
        "validation_file": str(val_path),
        "train_characters": len(train_text),
        "validation_characters": len(val_text),
        "unique_paragraphs": len(seen_global),
        "validation_sources": sorted(val_sources),
        "sources": [asdict(doc) for doc in manifest_docs],
        "warning": "This corpus is real and useful for pipeline/bootstrap training, but it is not a broad modern instruction corpus and will not produce a general-purpose assistant by itself.",
    }
    (output_dir / "corpus_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print(f"train: {train_path} ({len(train_text):,} chars)")
    print(f"val:   {val_path} ({len(val_text):,} chars)")
    print(f"unique paragraphs: {len(seen_global):,}")
    print("ready for GemmaAgent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
