#!/usr/bin/env python3
"""Build a deterministic multi-domain curriculum without giant repeated templates."""
from __future__ import annotations
import argparse
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260907

def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""

def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")
    print(f"{path}: {len(text):,} chars")

def collect_code(root: Path) -> str:
    parts: list[str] = []
    for pattern in ("*.rs", "*.toml", "*.md", "*.yml", "*.yaml", "*.py"):
        for path in sorted(root.rglob(pattern)):
            if any(part in {"target", ".git", "data", "curriculum"} for part in path.parts):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if text.strip():
                parts.append(f"\n\n### FILE: {path.relative_to(root)}\n\n{text}")
    return "".join(parts)

def make_math() -> str:
    rng = random.Random(SEED)
    out: list[str] = []
    for _ in range(30_000):
        a, b = rng.randint(-999, 999), rng.randint(-999, 999)
        op = rng.choice(["+", "-", "*"])
        ans = a + b if op == "+" else a - b if op == "-" else a * b
        out.append(f"Question: What is {a} {op} {b}?\nAnswer: {ans}")
    for _ in range(15_000):
        x, m, c = rng.randint(-50, 50), rng.randint(2, 20), rng.randint(-100, 100)
        rhs = m * x + c
        out.append(f"Solve for x: {m}x + {c} = {rhs}.\nAnswer: x = {x}.")
    for _ in range(10_000):
        a, b = rng.randint(1, 100), rng.randint(1, 100)
        out.append(f"A rectangle has width {a} and height {b}. Area = {a*b}. Perimeter = {2*(a+b)}.")
    for _ in range(5_000):
        total, groups = rng.randint(20, 500), rng.randint(2, 10)
        q, r = divmod(total, groups)
        out.append(f"Distribute {total} items among {groups} groups as evenly as possible. Each group gets {q}; remainder {r}.")
    return "\n\n".join(out)

def make_reasoning() -> str:
    rng = random.Random(SEED + 1)
    names = ["Luna", "Aria", "Nora", "Mika", "Omid", "Sara", "Kian", "Rin"]
    out: list[str] = []
    for _ in range(30_000):
        n = rng.choice(names)
        a, b = rng.randint(2, 40), rng.randint(1, 20)
        patterns = [
            f"All {n.lower()}s in this example are machines. {n} is one of them. Therefore {n} is a machine.",
            f"A sequence starts at {a} and increases by {b} each time. The next term is {a+b}.",
            f"A sequence starts at {a} and doubles each time. The next term is {a*2}.",
            f"A function receives input {a} and adds {b}. Its output is {a+b}.",
            f"An experiment changes one controlled variable while measuring an outcome; the measured outcome is evidence for evaluating the hypothesis.",
            f"An algorithm repeats a finite set of operations until a stopping condition is reached; the stopping condition prevents an infinite loop.",
        ]
        out.append(rng.choice(patterns))
    return "\n\n".join(out)

def sample_blocks(text: str, limit: int, rng: random.Random) -> list[str]:
    blocks = [b.strip() for b in text.split("\n\n") if len(b.strip()) > 40]
    if len(blocks) <= limit:
        return blocks
    return rng.sample(blocks, limit)

def mix(output_dir: Path, name: str, literature: str, code: str, math: str, reasoning: str, weights: tuple[float, float, float, float], seed: int) -> None:
    rng = random.Random(seed)
    pools = [sample_blocks(literature, 1800, rng), sample_blocks(code, 1000, rng), sample_blocks(math, 1600, rng), sample_blocks(reasoning, 1400, rng)]
    labels = ["LITERATURE", "CODE", "MATH", "REASONING"]
    counts = [max(1, int(len(pool) * weight)) for pool, weight in zip(pools, weights)]
    blocks: list[str] = []
    for label, pool, count in zip(labels, pools, counts):
        chosen = pool[:]
        rng.shuffle(chosen)
        blocks.extend(f"### {label}\n{b}" for b in chosen[:count])
    rng.shuffle(blocks)
    write(output_dir / f"{name}.txt", "\n\n".join(blocks))

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    data = args.root_dir / "data"
    curriculum = args.output_dir or data / "curriculum"
    literature = read(data / "train.txt")
    if not literature:
        raise SystemExit(f"{data / 'train.txt'} is missing; run prepare_real_corpus.py first")
    code, math, reasoning = collect_code(args.root_dir), make_math(), make_reasoning()
    write(curriculum / "literature.txt", literature); write(curriculum / "code.txt", code); write(curriculum / "math.txt", math); write(curriculum / "reasoning.txt", reasoning)
    mix(curriculum, "01-literature", literature, code, math, reasoning, (0.75, 0.05, 0.10, 0.10), SEED + 1)
    mix(curriculum, "02-code", literature, code, math, reasoning, (0.20, 0.60, 0.10, 0.10), SEED + 2)
    mix(curriculum, "03-math", literature, code, math, reasoning, (0.20, 0.10, 0.60, 0.10), SEED + 3)
    mix(curriculum, "04-reasoning", literature, code, math, reasoning, (0.25, 0.10, 0.20, 0.45), SEED + 4)
    mix(curriculum, "05-balanced", literature, code, math, reasoning, (0.30, 0.20, 0.25, 0.25), SEED + 5)
    print("curriculum ready")

if __name__ == "__main__":
    main()
