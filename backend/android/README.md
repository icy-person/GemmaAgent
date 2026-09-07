# Android Vulkan backend

Android arm64 Vulkan inference backend.

- `amd.rs` is a small adapter to the canonical Vulkan engine in `backend/amd`.
- `amd_tokenizer.rs` adapts the AMD tokenizer implementation.
- `bin/android-infer.rs` is the canonical Android inference entrypoint.
- `config.rs` and `tokenizer.rs` use the backend-neutral `src/core` definitions.

Build:

```bash
rustup target add aarch64-linux-android
cargo install cargo-ndk
cargo ndk -t arm64-v8a build --release --no-default-features --features android-vulkan --bin android-infer
```
