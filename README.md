# GemmaAgent / gemma-rs

یک موتور آموزشی مدل زبانی از پایه و با Rust خالص. این پروژه clean-room است و شامل کد یا وزن‌های Google Gemma نیست.

## هستهٔ فعلی

- reverse-mode Autograd با `matmul`, `transpose`, `softmax`, `log`, `gather`, row-gather و slicing
- گرادیان‌گیری برای SiLU و concatهای چندسطره
- causal multi-head self-attention با 8 head در پروفایل هدف
- decoder block با residual و feed-forward
- SiLU در FFN
- embedding مشترک ورودی/خروجی به‌صورت یک ماتریس واحد
- positional encoding سینوسی بدون پارامتر اضافی
- tokenizer بایتی با BOS/EOS
- next-token cross-entropy
- AdamW با weight decay و bias correction
- حلقهٔ آموزش واقعی CPU
- checkpoint باینری shape-safe
- autoregressive greedy inference با سقف token قابل تنظیم
- initialization قطعی برای بازتولیدپذیری
- تست‌های regression برای autograd، مدل، optimizer، tokenizer و checkpoint

## پروفایل‌ها

مدل کوچک برای تست سریع مسیر کامل:

`vocab=258 | context=128 | d_model=64 | layers=2 | heads=4 | ffn=128`

مدل هدف:

`vocab=16384 | context=1024 | d_model=416 | layers=6 | heads=8 | ffn=1664`

فرمول شمارش پارامترها:

`vocab*d_model + layers*(4*d_model^2 + 2*d_model*ffn)`

که برای پروفایل هدف دقیقاً `19,275,776` پارامتر است. در FP32 فقط وزن‌ها حدود `73.53 MiB` فضا می‌گیرند؛ activationها، گرادیان‌ها و state مربوط به AdamW جدا هستند.

## اجرا

تست‌ها:

```bash
cargo test
```

آموزش سریع برای صحت‌سنجی کل pipeline:

```bash
cargo run --release -- train 300
```

آموزش پروفایل هدف:

```bash
cargo run --release -- train 300 gemma-agent-target.ckpt --target
```

> هشدار: kernelهای فعلی عمداً scalar و CPU-only هستند. بنابراین پروفایل ۱۹ میلیون پارامتری از نظر زمانی آموزشی و برای توسعهٔ سریع مناسب نیست. هدف این نسخه، تثبیت correctness و architecture است.

Inference:

```bash
cargo run --release -- infer gemma-agent.ckpt "Rust is"
cargo run --release -- infer gemma-agent.ckpt "Rust is" --tokens 128
```

برای پروفایل هدف باید همان فلگ `--target` را در inference هم بدهید:

```bash
cargo run --release -- infer gemma-agent-target.ckpt "Rust is" --target --tokens 128
```

## وضعیت مهندسی

این repository اکنون یک هستهٔ مستقل و قابل تست دارد، اما هنوز یک runtime سریع تولیدی نیست. مرحلهٔ بعدی منطقی عبارت است از tensorهای contiguous و kernelهای SIMD، threading، memory planning، mixed precision، KV cache، sampling و سپس backendهای Vulkan/CUDA/ROCm.
