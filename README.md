# GemmaAgent → gemma-rs

یک پروژهٔ clean-room برای ساخت یک مدل زبانی کوچک با Rust، بدون استفاده از source یا weights مدل‌های Google.

## مدل فعلی

- Decoder-only Transformer
- 6 لایه
- `d_model = 416`
- 8 attention heads
- `head_dim = 52`
- `FFN = 1664`
- context = 1024 token
- vocabulary = 16384
- RoPE
- RMSNorm
- SwiGLU-style feed-forward block
- tied input/output embeddings
- CPU و بدون dependency خارجی

## هستهٔ آموزش

- Tensor/Matrix primitives
- Cross-entropy + logits gradient
- AdamW
- Autograd گرافی با backward برای `add` و `matmul`
- تست گرادیان برای matmul

وزن‌های Transformer هنوز تصادفی هستند و مدل آموزش‌دیده نیست. مرحلهٔ بعدی، اتصال autograd به تمام عملیات Transformer و ساخت data loader و training loop واقعی است.

## اجرا

```bash
cargo run --release
```

بررسی کد:

```bash
cargo fmt --check
cargo check
cargo test
```

## نقشهٔ راه

1. اتصال autograd به Transformer
2. cross-entropy روی sequence logits
3. AdamW روی تمام پارامترهای مدل
4. tokenizer واقعی BPE/SentencePiece-compatible
5. dataset و streaming data loader
6. checkpoint format
7. gradient accumulation و mixed precision
8. KV cache برای inference
9. SIMD و Vulkan backend
