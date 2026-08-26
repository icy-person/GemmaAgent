#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/rust-core"
command -v cargo >/dev/null || { echo "cargo is required"; exit 1; }
command -v cargo-ndk >/dev/null || { echo "Install cargo-ndk: cargo install cargo-ndk"; exit 1; }
mkdir -p "$ROOT/app/src/main/jniLibs/arm64-v8a" "$ROOT/app/src/main/jniLibs/x86_64"
cargo ndk -t arm64-v8a -t x86_64 -o "$ROOT/app/src/main/jniLibs" build --release
