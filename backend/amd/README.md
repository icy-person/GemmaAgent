# AMD Vulkan backend

Canonical desktop Radeon/Vulkan backend.

- `amd.rs`: Burn/WGPU Vulkan model, training, checkpointing and KV-cache inference.
- `amd_tokenizer.rs`: AMD tokenizer for target checkpoints.
- `bin/amd-train.rs`: GPU training entrypoint.
- `bin/amd-bench.rs`: throughput benchmark.
- `bin/amd-infer.rs`: KV-cache inference.

```bash
cargo run --release --no-default-features --features amd-vulkan --bin amd-train -- --target
cargo run --release --no-default-features --features amd-vulkan --bin amd-bench -- --target
cargo run --release --no-default-features --features amd-vulkan --bin amd-infer -- --target --checkpoint model.bin --tokenizer model.bin.tok --prompt "Rust is"
```
