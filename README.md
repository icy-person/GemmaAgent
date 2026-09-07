# GemmaAgent → gemma-rs

این مخزن از پایه به یک پروژهٔ Rust برای ساخت یک مدل زبانی کوچک تبدیل شده است.

## مدل v0.1

- Decoder-only Transformer
- 6 لایه
- `d_model = 416`
- 8 attention heads
- `FFN = 1664`
- context = 1024 token
- vocabulary = 16384
- RoPE positional encoding
- RMSNorm
- SwiGLU-style feed-forward block
- tied input/output embeddings
- اجرای اولیه با CPU و بدون وابستگی خارجی

وزن‌ها فعلاً به‌صورت deterministic و تصادفی مقداردهی می‌شوند؛ این نسخه هنوز مدل آموزش‌دیده نیست. هدف این commit ساخت هستهٔ درست معماری است تا مرحلهٔ بعدی، یعنی tokenizer واقعی، loss و backpropagation در Rust روی آن اضافه شود.

## اجرا

```bash
cargo run --release
```

برای بررسی:

```bash
cargo fmt --check
cargo check
```

## نقشهٔ راه

1. Tensor/autograd کامل
2. cross-entropy loss
3. AdamW
4. tokenizer با BPE/SentencePiece-compatible vocabulary
5. data loader و training loop
6. checkpoint format
7. mixed precision / memory optimizations
8. inference با KV cache
9. backend های SIMD و Vulkan
