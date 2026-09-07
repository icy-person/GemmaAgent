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
- آموزش از corpus داخلی یا فایل متن UTF-8 با `--data`
- gradient accumulation قابل تنظیم برای batch مؤثر بزرگ‌تر
- checkpoint باینری shape-safe و corruption-aware
- checkpoint دوره‌ای در طول آموزش
- runtime مستقل و مستقیم CPU برای inference
- KV cache واقعی برای decoding افزایشی
- autoregressive inference با greedy و sampling
- sampling با `temperature` و `top-k` بدون dependency جدید
- benchmark داخلی برای prefill و decode throughput
- initialization قطعی برای بازتولیدپذیری
- regression test برای برابری runtime مستقیم با forward مرجع و صحت incremental KV cache

## پروفایل‌ها

مدل کوچک برای تست سریع مسیر کامل:

`vocab=258 | context=128 | d_model=64 | layers=2 | heads=4 | ffn=128`

مدل هدف:

`vocab=16384 | context=1024 | d_model=416 | layers=6 | heads=8 | ffn=1664`

فرمول شمارش پارامترها:

`vocab*d_model + layers*(4*d_model^2 + 2*d_model*ffn)`

که برای پروفایل هدف دقیقاً `19,275,776` پارامتر است. در FP32 فقط وزن‌ها حدود `73.53 MiB` فضا می‌گیرند؛ activationها، گرادیان‌ها و state مربوط به AdamW جدا هستند.

نکته: tokenizer فعلی بایتی است و فقط 258 شناسهٔ واقعی تولید می‌کند؛ vocab بزرگ‌تر در پروفایل هدف عمداً برای آزمایش معماری و پارامترشمارش نگه داشته شده است.

## اجرا

تست‌ها:

```bash
cargo test
```

lint و regression CI:

```bash
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --all-features
```

آموزش سریع با corpus داخلی:

```bash
cargo run --release -- train 300
```

آموزش از فایل متن UTF-8:

```bash
cargo run --release -- train 5000 gemma-agent.ckpt --data ./train.txt --grad-accum 8 --checkpoint-every 250
```

فایل با tokenizer بایتی فعلی به BOS/EOS + byte IDs تبدیل می‌شود. بنابراین این قابلیت برای ادامهٔ توسعه و آزمایش pipeline واقعی است؛ برای یک مدل زبانی جدی، tokenizer subword/BPE باید در مرحلهٔ بعد اضافه شود.

برای batch مؤثر بزرگ‌تر و checkpoint دوره‌ای:

```bash
cargo run --release -- train 10000 gemma-agent.ckpt --grad-accum 8 --checkpoint-every 100
```

آموزش پروفایل هدف:

```bash
cargo run --release -- train 300 gemma-agent-target.ckpt --target --grad-accum 4 --checkpoint-every 25
```

> هشدار: kernelهای آموزشی فعلی عمداً scalar و CPU-only هستند. بنابراین پروفایل ۱۹ میلیون پارامتری از نظر زمانی آموزشی مناسب توسعهٔ سریع نیست. هدف این نسخه، تثبیت correctness و architecture است.

## Inference

Inference checkpoint را به runtime مستقیم CPU منتقل می‌کند؛ graph مربوط به autograd در هر token ساخته نمی‌شود و KV cache برای decoding افزایشی نگهداری می‌شود.

Greedy:

```bash
cargo run --release -- infer gemma-agent.ckpt "Rust is"
```

Sampling:

```bash
cargo run --release -- infer gemma-agent.ckpt "Rust is" --temperature 0.8 --top-k 40 --tokens 128
```

برای پروفایل هدف باید همان فلگ `--target` را در inference هم بدهید:

```bash
cargo run --release -- infer gemma-agent-target.ckpt "Rust is" --target --temperature 0.8 --top-k 40 --tokens 128
```

`--temperature 0` یا مقدار بسیار نزدیک به صفر، greedy decoding را فعال می‌کند. `--top-k 0` یعنی محدودسازی top-k غیرفعال است. `--checkpoint-every 0` یا حذف این گزینه، checkpoint دوره‌ای را غیرفعال می‌کند. `--grad-accum 1` یعنی یک update برای هر window آموزشی.

## Benchmark

برای سنجش runtime مستقیم روی CPU:

```bash
cargo run --release -- bench
cargo run --release -- bench --prompt-tokens 128 --tokens 64
cargo run --release -- bench --target --prompt-tokens 128 --tokens 64
```

خروجی، زمان و token/s برای prefill و incremental decode را جداگانه گزارش می‌کند. این benchmark با وزن‌های deterministic اجرا می‌شود و به checkpoint نیاز ندارد.

## وضعیت مهندسی

هستهٔ مدل و runtime مستقیم اکنون قابل تست و قابل بازتولید هستند. inference از graph autograd جدا شده و KV cache دارد، ولی runtime هنوز production-grade نیست. گام‌های بعدی عبارت‌اند از tokenizer subword/BPE، tensorهای contiguous واقعی، SIMD/threading برای matmul، mixed precision، memory planning، batching واقعی در سطح tensor، rotary position embeddings، RMSNorm، samplingهای پیشرفته‌تر و سپس backendهای Vulkan/CUDA/ROCm.

CI فعلی compilation/lint و regression tests را روی Rust stable اجرا می‌کند.
