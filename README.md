# GemmaAgent / gemma-rs

یک موتور آموزشی مدل زبانی از پایه و با Rust خالص. این پروژه شامل کد یا وزن‌های Google Gemma نیست.

## اکنون چه چیزی داریم؟

- reverse-mode Autograd با تست گرادیان
- ماتریس و `matmul` با گرادیان
- ReLU، Softmax، Log، Gather، concat و slicing
- causal multi-head self-attention
- decoder blocks با residual + feed-forward
- embeddingهای مشترک ورودی/خروجی
- tokenizer بایتی
- next-token cross-entropy
- AdamW
- حلقهٔ آموزش واقعی CPU
- ذخیره و بارگذاری checkpoint باینری
- autoregressive greedy inference
- initialization قطعی برای بازتولیدپذیری

## مدل قابل آموزش

برای اینکه کل مسیر روی CPU قابل آزمایش باشد، اجرای پیش‌فرض از مدل کوچک استفاده می‌کند:

`vocab=258 | context=128 | d_model=64 | layers=2 | heads=4 | ffn=128`

پروفایل هدف بزرگ‌تر نیز در `Config::target()` وجود دارد:

`vocab=16384 | context=1024 | d_model=416 | layers=6 | heads=8 | ffn=1664`

این پروفایل حدود ۱۹ میلیون پارامتر دارد، اما موتور فعلی عمداً آموزشی و scalar است و برای آموزش این اندازه هنوز kernelهای SIMD/GPU ندارد.

## اجرا

```bash
cargo test
cargo run --release -- train 300
cargo run --release -- infer gemma-agent.ckpt "Rust is"
```

بعد از آموزش، checkpoint در `gemma-agent.ckpt` ساخته می‌شود.

## گام بعدی برای نسخهٔ جدی

بهینه‌سازی tensor kernels، threading، mixed precision، KV cache، sampling و سپس backendهای Vulkan/CUDA/ROCm باید بعد از تثبیت این هسته اضافه شوند.
