# GemmaAgent / gemma-rs

A clean-room educational language-model implementation in pure Rust. This repository does **not** include Google's Gemma source code or model weights.

## Included

- differentiable 2-D tensor graph and reverse-mode autograd
- matrix multiplication, elementwise math, ReLU, softmax, concat and gather
- causal multi-head self-attention with triangular context
- residual decoder blocks and feed-forward network
- byte-level tokenizer
- AdamW optimizer
- next-token cross-entropy training loop
- binary checkpoint save/load
- greedy autoregressive inference
- deterministic initialization and unit tests

## Debug model

The executable trains a small CPU model so the complete pipeline can run locally:

`vocab=258, context=128, d_model=64, layers=2, heads=4, ffn=128`

The larger target architecture is kept in `Config::target()` for subsequent optimized backends. The debug model is intentionally much smaller because this repository currently uses an educational scalar Rust autograd engine rather than SIMD/GPU kernels.

## Run

```bash
cargo test
cargo run --release -- train 500
cargo run --release -- infer gemma-agent.ckpt "Rust is"
```

## Roadmap to a serious model

1. BPE/SentencePiece-compatible tokenizer and real corpus streaming.
2. fused tensor kernels, SIMD and multithreading.
3. mixed precision and memory-efficient attention.
4. KV cache and sampling (temperature/top-k/top-p).
5. sharded checkpoints and distributed training.
6. Vulkan backend and optional CUDA/ROCm kernels.
7. scale the configuration and train on a large curated corpus.
