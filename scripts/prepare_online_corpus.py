#!/usr/bin/env python3
"""Build a cached, attributed online corpus from Wikimedia topic searches.

The collector deliberately uses a small, rate-limited public API footprint.
Each downloaded extract is cached by page id and tracked with its revision id,
source URL, license and SHA-256. A restored manifest is reused before making
new API requests; pass --refresh to query the configured topics again.

Outputs:
  data/online/pages/*.txt
  data/online_manifest.json
  data/online/train.txt
  data/online/val.txt

Examples:
  python3 scripts/prepare_online_corpus.py
  python3 scripts/prepare_online_corpus.py --pages-per-topic 20 --max-pages 120
  python3 scripts/prepare_online_corpus.py --refresh --max-pages 160
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
    # Do not request `revisions` here. MediaWiki forbids rvlimit/rvprop on a
    # generator such as `generator=search`. `info` already exposes lastrevid.
    payload = api(
        {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "generator": "search",
            "gsrsearch": topic,
            "gsrnamespace": "0",
            "gsrlimit": str(min(limit, 20)),
            "prop": "extracts|info",
            "explaintext": "1",
            "exchars": "1200",
            "inprop": "url",
        },
        timeout,
        retries,
        delay,
    )
    return payload.get("query", {}).get("pages", [])


def load_manifest(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        pages = raw.get("pages", []) if isinstance(raw, dict) else raw
        if not isinstance(pages, list):
            return {}
        return {int(row["pageid"]): row for row in pages if isinstance(row, dict) and "pageid" in row}
    except (OSError, ValueError, TypeError, KeyError):
        return {}


def write_page(out: Path, page: dict, topic: str) -> dict | None:
    pageid = int(page.get("pageid", 0))
    title = str(page.get("title", "")).strip()
    extract = normalize(str(page.get("extract", "")))
    if not pageid or not title or not extract:
        return None

    # `lastrevid` is supplied by the `info` property and works with a
    # generator=search query, unlike the revisions module parameters.
    revision_id = page.get("lastrevid")
    source_url = page.get("fullurl") or f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
    path = out / "pages" / f"{pageid}_{safe_slug(title)}.txt"
    path.write_text(extract + "\n\n", encoding="utf-8")
    return {
        "pageid": pageid,
        "title": title,
        "url": source_url,
        "revision_id": revision_id,
        "characters": len(extract),
        "sha256": hashlib.sha256(extract.encode("utf-8")).hexdigest(),
        "license": LICENSE,
        "path": str(path),
        "topic": topic,
    }


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
    parser.add_argument("--refresh", action="store_true", help="query Wikimedia even when a cached corpus exists")
    args = parser.parse_args()

    if args.pages_per_topic < 1 or args.max_pages < 1:
        parser.error("page counts must be positive")
    if args.min_chars < 1 or not 0 < args.val_fraction < 0.5:
        parser.error("invalid min-chars or val-fraction")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pages").mkdir(parents=True, exist_ok=True)
    manifest_path = out.parent / "online_manifest.json"

    found = {} if args.refresh else load_manifest(manifest_path)
    if len(found) < args.max_pages or args.refresh:
        for topic in TOPICS:
            if len(found) >= args.max_pages and not args.refresh:
                break
            try:
                pages = search_topic(topic, args.pages_per_topic, args.timeout, args.retries, args.delay)
            except Exception as exc:
                print(f"warning: topic {topic!r} failed: {exc}", file=sys.stderr)
                continue
            for page in pages:
                candidate = write_page(out, page, topic)
                if candidate is None or candidate["characters"] < args.min_chars:
                    continue
                found[candidate["pageid"]] = candidate
                if len(found) >= args.max_pages and not args.refresh:
                    break

    rows = sorted(found.values(), key=lambda x: (x["title"].lower(), x["pageid"]))[: args.max_pages]
    if not rows:
        raise SystemExit("no online documents available; keep using the offline corpus")

    for row in rows:
        path = Path(row["path"])
        if not path.exists():
            path.write_text("", encoding="utf-8")
    rows = [row for row in rows if Path(row["path"]).stat().st_size >= args.min_chars]
    if not rows:
        raise SystemExit("online manifest exists but cached page files are missing")

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
        "schema": "gemmaagent.online-corpus.v2",
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
    print(
        f"online corpus: pages={len(rows)} train_chars={metadata['train_characters']} "
        f"val_chars={metadata['val_characters']} refresh={args.refresh}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
