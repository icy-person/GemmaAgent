#!/usr/bin/env python3
"""Prepare a small multi-domain curriculum for CPU pretraining.

The curriculum intentionally mixes replay from the literature corpus into every
stage to reduce catastrophic forgetting while exposing the model to different
patterns: prose, Rust/code, mathematics, reasoning, and a balanced mixture.
"""
from __future__ import annotations

import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CURRICULUM = DATA / "curriculum"
SEED = 20260907


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")
    print(f"{path}: {len(text):,} chars")


def collect_code() -> str:
    parts: list[str] = []
    for pattern in ("*.rs", "*.toml", "*.md", "*.yml", "*.yaml", "*.py"):
        for path in sorted(ROOT.rglob(pattern)):
            if any(part in {"target", ".git", "data"} for part in path.parts):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if text.strip():
                parts.append(f"\n\n### FILE: {path.relative_to(ROOT)}\n\n{text}")
    return "".join(parts)


def make_math() -> str:
    rng = random.Random(SEED)
    out: list[str] = []
    for _ in range(60000):
        a = rng.randint(-999, 999)
        b = rng.randint(-999, 999)
        op = rng.choice(["+", "-", "*"])
        if op == "+": ans = a + b
        elif op == "-": ans = a - b
        else: ans = a * b
        out.append(f"Question: What is {a} {op} {b}?\nAnswer: {ans}")

    for _ in range(20000):
        x = rng.randint(-30, 30)
        m = rng.randint(2, 12)
        c = rng.randint(-50, 50)
        rhs = m * x + c
        out.append(f"Solve for x: {m}x + {c} = {rhs}.\nAnswer: x = {x}.")

    for _ in range(10000):
        a = rng.randint(1, 30)
        b = rng.randint(1, 30)
        out.append(f"A rectangle has width {a} and height {b}.\nArea: {a*b}.\nPerimeter: {2*(a+b)}.")
    return "\n\n".join(out)


def make_reasoning() -> str:
    rows = [
        "A sequence is 2, 4, 6, 8. The next number is 10 because the difference is +2.",
        "A sequence is 3, 6, 12, 24. The next number is 48 because each term is doubled.",
        "If all cats are animals and Luna is a cat, then Luna is an animal.",
        "If every square is a rectangle and this shape is a square, then this shape is a rectangle.",
        "A hypothesis is a testable explanation; an observation is information obtained from measurement.",
        "A function maps inputs to outputs. A deterministic function gives the same output for the same input.",
        "An algorithm is a finite sequence of steps used to solve a problem or compute a result.",
        "A variable stores a value that can be read and, when mutable, changed during computation.",
        "A loop repeats a block of instructions while a condition holds or for a specified number of iterations.",
        "A tokenizer converts text into discrete tokens that a language model can process numerically.",
    ]
    return "\n".join(rows * 4000)


def sample_blocks(text: str, limit: int, rng: random.Random) -> list[str]:
    blocks = [b.strip() for b in text.split("\n\n") if len(b.strip()) > 40]
    if len(blocks) <= limit:
        return blocks
    return rng.sample(blocks, limit)


def mix(name: str, literature: str, code: str, math: str, reasoning: str, weights: tuple[float, float, float, float], seed: int) -> None:
    rng = random.Random(seed)
    pools = [
        sample_blocks(literature, 1600, rng),
        sample_blocks(code, 700, rng),
        sample_blocks(math, 1200, rng),
        sample_blocks(reasoning, 900, rng),
    ]
    labels = ["LITERATURE", "CODE", "MATH", "REASONING"]
    counts = [max(1, int(len(pool) * weight)) for pool, weight in zip(pools, weights)]
    blocks: list[str] = []
    for label, pool, count in zip(labels, pools, counts):
        chosen = pool[:]
        rng.shuffle(chosen)
        blocks.extend([f"### {label}\n{b}" for b in chosen[:count]])
    rng.shuffle(blocks)
    write(CURRICULUM / f"{name}.txt", "\n\n".join(blocks))


def main() -> None:
    literature = read(DATA / "train.txt")
    if not literature:
        raise SystemExit("data/train.txt is missing; run prepare_real_corpus.py first")
    code = collect_code()
    math = make_math()
    reasoning = make_reasoning()

    write(CURRICULUM / "literature.txt", literature)
    write(CURRICULUM / "code.txt", code)
    write(CURRICULUM / "math.txt", math)
    write(CURRICULUM / "reasoning.txt", reasoning)

    mix("01-literature", literature, code, math, reasoning, (0.75, 0.05, 0.10, 0.10), SEED + 1)
    mix("02-code", literature, code, math, reasoning, (0.20, 0.60, 0.10, 0.10), SEED + 2)
    mix("03-math", literature, code, math, reasoning, (0.20, 0.10, 0.60, 0.10), SEED + 3)
    mix("04-reasoning", literature, code, math, reasoning, (0.25, 0.10, 0.20, 0.45), SEED + 4)
    mix("05-balanced", literature, code, math, reasoning, (0.30, 0.20, 0.25, 0.25), SEED + 5)

    print("curriculum ready")


if __name__ == "__main__":
    main()
