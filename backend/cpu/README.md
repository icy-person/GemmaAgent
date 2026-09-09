# CPU backend

Portable scalar Rust/autograd training backend.

- `autograd.rs`: reverse-mode autodiff engine (matmul parallelizes across
  threads for the dominant 1xN row-vector shape; std only, no new deps)
- `model.rs`: decoder-only causal language model with RoPE on Q/K, RMSNorm,
  and a SiLU-gated FFN — architecturally aligned with the AMD/Android Vulkan
  backend
- `optim.rs`: AdamW with gradient clipping
- `checkpoint.rs`: model/optimizer/RNG resume state
- `config.rs` and `tokenizer.rs`: adapters to `src/core`
- `bin/cpu-train.rs`: canonical CPU trainer

Example:

```bash
cargo run --release --no-default-features --features runner-cpu --bin cpu-train -- --large --steps 1000 --data data/train.txt --val-data data/val.txt
```
