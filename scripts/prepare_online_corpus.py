#!/usr/bin/env python3
"""Build a cached, attributed online corpus from Wikimedia topic searches.

The script intentionally uses a small, rate-limited public API footprint so a
training job can refresh knowledge without acting like a crawler. Every page is
cached by page id/revision metadata and all downloaded text is represented in a
manifest with source URL, license, revision id, and SHA-256.

Outputs:
  data/online/*.txt
  data/online_manifest.json
  data/online/train.txt
  data/online/val.txt

Examples:
  python3 scripts/prepare_online_corpus.py
  python3 scripts/prepare_online_corpus.py --pages-per-topic 20 --max-pages 120
  python3 scripts/prepare_online_corpus.py --language en --min-chars 300
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path


API = "https://en.wikipedia.org/w/api.php"
LICENSE = "CC BY-SA 4.0 / Wikimedia Foundation terms and source attribution"
TOPICS = [
    "computer science",
    "programming language",
    "Rust programming language",
    "machine learning",
    "artificial intelligence",
    "algorithms",
    "data structures",
    "mathematics",
    "physics",
    "chemistry",
    "biology",
    "astronomy",
    "economics",
    "history",
    "literature",
    "linguistics",
    "engineering",
    "statistics",
]


@dataclass
class Page:
    pageid: int
    title: str
    url: str
    revision_id: int | None
    characters: int
    sha256: str


def api(params: dict[str, str], timeout: int, retries: int, delay: float) -> dict:
    query = urllib.parse.urlencode(params)
    url = f"{API}?{query}"
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "GemmaAgent-online-corpus-bot/1.0 (https://github.com/icy-person/GemmaAgent)",
                    "Accept": "application/json",
                    "Accept-Encoding": "identity",
                },
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8", errors="replace"))
            if "error" in payload:
                raise RuntimeError(str(payload["error"]))
            time.sleep(delay)
            return payload
        except Exception as exc:
            last = exc
            if attempt < retries:
                time.sleep(min(12.0, delay * (2**attempt)))
    raise RuntimeError(f"API request failed: {last}")


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def safe_slug(title: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", title).strip("_").lower()
    return slug[:120] or "page"


def search_topic(topic: str, limit: int, timeout: int, retries: int, delay: float) -> list[dict]:
    payload = api(
        {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "generator": "search",
            "gsrsearch": topic,
            "gsrnamespace": "0",
            "gsrlimit": str(min(limit, 20)),
            "prop": "extracts|info|revisions",
            "explaintext": "1",
            "exchars": "1200",
            "inprop": "url",
            "rvprop": "ids",
            "rvlimit": "1",
        },
        timeout,
        retries,
        delay,
    )
    return payload.get("query", {}).get("pages", [])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/online")
    parser.add_argument("--pages-per-topic", type=int, default=12)
    parser.add_argument("--max-pages", type=int, default=160)
    parser.add_argument("--min-chars", type=int, default=300)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--delay", type=float, default=0.35)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--retries", type=int, default=4)
    args = parser.parse_args()

    if args.pages_per_topic < 1 or args.max_pages < 1:
        parser.error("page counts must be positive")
    if args.min_chars < 1 or not 0 < args.val_fraction < 0.5:
        parser.error("invalid min-chars or val-fraction")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pages_dir = out / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out.parent / "online_manifest.json"

    existing: dict[int, dict] = {}
    if manifest_path.exists():
        try:
            existing = {int(row["pageid"]): row for row in json.loads(manifest_path.read_text(encoding="utf-8"))}
        except (OSError, ValueError, KeyError, TypeError):
            existing = {}

    found: dict[int, dict] = dict(existing)
    for topic in TOPICS:
        if len(found) >= args.max_pages:
            break
        try:
            pages = search_topic(topic, args.pages_per_topic, args.timeout, args.retries, args.delay)
        except Exception as exc:
            print(f"warning: topic {topic!r} failed: {exc}", file=sys.stderr)
            continue
        for page in pages:
            pageid = int(page.get("pageid", 0))
            if not pageid or pageid in found:
                continue
            extract = normalize(page.get("extract", ""))
            if len(extract) < args.min_chars:
                continue
            revision_id = None
            revisions = page.get("revisions") or []
            if revisions:
                revision_id = revisions[0].get("revid")
            source_url = page.get("fullurl") or f"https://en.wikipedia.org/wiki/{urllib.parse.quote(page['title'].replace(' ', '_'))}"
            record = {
                "pageid": pageid,
                "title": page["title"],
                "url": source_url,
                "revision_id": revision_id,
                "extract": extract,
            }
            filename = f"{pageid}_{safe_slug(page['title'])}.txt"
            path = pages_dir / filename
            path.write_text(extract + "\n\n", encoding="utf-8")
            found[pageid] = {
                "pageid": pageid,
                "title": page["title"],
                "url": source_url,
                "revision_id": revision_id,
                "characters": len(extract),
                "sha256": hashlib.sha256(extract.encode("utf-8")).hexdigest(),
                "license": LICENSE,
                "path": str(path),
                "topic": topic,
            }
            if len(found) >= args.max_pages:
                break

    rows = sorted(found.values(), key=lambda x: (x["title"].lower(), x["pageid"]))
    if not rows:
        raise SystemExit("no online documents available; keep using the offline corpus")

    rng = random.Random(args.seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * args.val_fraction))) if len(shuffled) > 1 else 0
    val_ids = {row["pageid"] for row in shuffled[:val_count]}

    train_parts: list[str] = []
    val_parts: list[str] = []
    for row in rows:
        text = Path(row["path"]).read_text(encoding="utf-8")
        if row["pageid"] in val_ids:
            val_parts.append(text)
        else:
            train_parts.append(text)

    (out / "train.txt").write_text("\n".join(train_parts), encoding="utf-8")
    (out / "val.txt").write_text("\n".join(val_parts), encoding="utf-8")
    metadata = {
        "schema": "gemmaagent.online-corpus.v1",
        "generated_at_unix": int(time.time()),
        "api": API,
        "license": LICENSE,
        "topics": TOPICS,
        "pages": rows,
        "train_pages": len(rows) - val_count,
        "val_pages": val_count,
        "train_characters": sum(len(x) for x in train_parts),
        "val_characters": sum(len(x) for x in val_parts),
    }
    manifest_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"online corpus: pages={len(rows)} train_chars={metadata['train_characters']} val_chars={metadata['val_characters']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
