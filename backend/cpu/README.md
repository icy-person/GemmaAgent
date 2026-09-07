# CPU backend

Portable scalar Rust/autograd training backend.

- `autograd.rs`: reverse-mode autodiff engine
- `model.rs`: decoder-only causal language model
- `optim.rs`: AdamW with gradient clipping
- `checkpoint.rs`: model/optimizer/RNG resume state
- `config.rs` and `tokenizer.rs`: adapters to `src/core`
- `bin/cpu-train.rs`: canonical CPU trainer

Example:

```bash
cargo run --release --no-default-features --features runner-cpu --bin cpu-train -- --large --steps 1000 --data data/train.txt --val-data data/val.txt
```
