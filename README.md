# GemmaAgent / gemma-rs

یک موتور آموزشی مدل زبانی از پایه و با Rust خالص. پروژه clean-room است و شامل کد یا وزن‌های Google Gemma نیست.

## هستهٔ فعلی

- reverse-mode Autograd در مسیر CPU
- causal multi-head self-attention و FFN با SiLU در مسیر CPU
- tokenizer بایتی CPU با BOS/EOS
- tokenizer هدف AMD با subwordهای آموخته‌شده از corpus و byte fallback
- next-token cross-entropy روی تمام موقعیت‌های causal
- AdamW، gradient clipping و warmup/cosine schedule
- train/validation split و validation perplexity
- latest و best checkpoint
- checkpoint مدل + optimizer state + RNG/training metadata برای resume
- backend CUDA اختیاری با Candle
- backend **AMD/Vulkan** با Burn + WGPU
- decoder-only GPU با pre-norm RMSNorm
- RoPE روی Q/K
- native scaled dot-product attention با causal mode
- SwiGLU feed-forward
- gradient accumulation با GradientsAccumulator
- KV-cache واقعی برای prefill و incremental decoding

## پروفایل‌ها

مدل کوچک:

`vocab=258 | context=128 | d_model=64 | layers=2 | heads=4 | ffn=128`

مدل هدف:

`vocab=16384 | context=1024 | d_model=416 | layers=6 | heads=8 | ffn=1664`

پروفایل هدف CPU در وزن‌های اصلی دقیقاً `19,275,776` پارامتر دارد. backend AMD معماری GPU-محور مستقل دارد و به‌دلیل RoPE، SwiGLU و head مستقل، شمارش پارامتر GPU دقیقاً برابر نیست.

## آموزش CPU

```bash
cargo run --release -- train 5000 gemma-agent.ckpt \
  --data ./train.txt \
  --grad-accum 8 \
  --targets-per-step 32 \
  --lr 0.001 \
  --checkpoint-every 250
```

## آموزش NVIDIA / CUDA

```bash
nvidia-smi
cargo run --release --features cuda --bin gpu-train -- \
  --steps 5000 \
  --data ./train.txt \
  --checkpoint gemma-agent-gpu.safetensors \
  --batch-size 8 \
  --grad-accum 2 \
  --lr 0.0003 \
  --checkpoint-every 250 \
  --gpu 0
```

## آموزش AMD / Vulkan

برای Radeon روی لینوکس، مسیر AMD با **Burn + WGPU + Vulkan** ساخته شده است. Burn 0.21.0 primitive attention، RoPE، SwiGLU، GradientsAccumulator و optimizer-state records را ارائه می‌کند؛ WGPU نیز Vulkan و انتخاب GPU مجتمع/مستقل را پشتیبانی می‌کند. citeturn966778search0turn966778search1turn966778search2turn815236search0turn966778search5

بررسی GPU:

```bash
vulkaninfo --summary
lspci | grep -Ei 'vga|3d|display'
```

تنظیم پیشنهادی روی iGPU با 8GB RAM:

```bash
cargo run --release --features amd-vulkan --bin amd-train -- \
  --target \
  --steps 20000 \
  --data ./train.txt \
  --checkpoint gemma-agent-target-amd.bin \
  --best-checkpoint gemma-agent-target-amd.bin.best \
  --tokenizer gemma-agent-target-amd.bin.tok \
  --batch-size 1 \
  --grad-accum 1 \
  --lr 0.0003 \
  --checkpoint-every 250 \
  --eval-every 250 \
  --gpu-kind integrated \
  --gpu 0
```

برای کارت مستقل AMD از `--gpu-kind discrete` و برای انتخاب خودکار از `--gpu-kind best` استفاده می‌شود.

### Resume کامل

checkpoint اصلی سه فایل جانبی ایجاد می‌کند:

- `gemma-agent-target-amd.bin` وزن‌های مدل
- `gemma-agent-target-amd.bin.opt` وضعیت AdamW
- `gemma-agent-target-amd.bin.state` شمارهٔ update، بهترین validation loss و RNG state

در نتیجه resume فقط وزن نیست:

```bash
cargo run --release --features amd-vulkan --bin amd-train -- \
  --target \
  --steps 50000 \
  --data ./train.txt \
  --checkpoint gemma-agent-target-amd.bin \
  --best-checkpoint gemma-agent-target-amd.bin.best \
  --tokenizer gemma-agent-target-amd.bin.tok \
  --resume gemma-agent-target-amd.bin \
  --batch-size 1 \
  --grad-accum 1 \
  --lr 0.0003 \
  --checkpoint-every 250 \
  --eval-every 250 \
  --gpu-kind integrated \
  --gpu 0
```

در resume، `--steps` تعداد کل updateهای هدف است؛ اگر checkpoint روی update 20,000 باشد و `--steps 50,000` بدهی، 30,000 update دیگر اجرا می‌شود.

## معماری AMD

ساختار هر decoder block:

`RMSNorm -> fused QKV -> RoPE(Q,K) -> native causal SDPA -> output projection -> residual -> RMSNorm -> SwiGLU -> down projection -> residual`

RoPE در خود Burn برای ورودی‌های 4D با شکل `(batch, heads, seq, head_dim)` پشتیبانی می‌شود و `apply(x, start)` برای ادامهٔ موقعیت‌های cache شده دارد. citeturn815236search0turn815236search1

attention از primitive بومی Burn استفاده می‌کند و `is_causal=true` را مستقیماً به backend می‌دهد؛ این مسیر برای backendهای بهینه‌شده از explicit mask مناسب‌تر است. citeturn966778search2

SwiGLU نیز به‌صورت native از `SwiGluConfig` استفاده می‌شود. citeturn966778search5turn966778search15

برای gradient accumulation، هر micro-batch جداگانه backward می‌شود و `GradientsAccumulator` گرادیان‌ها را در یک update ادغام می‌کند؛ این نسبت به ساختن یک graph واحد برای همهٔ micro-batchها حافظهٔ کمتری مصرف می‌کند. citeturn966778search1

## KV cache

AMD inference ابتدا کل prompt را یک‌بار prefill می‌کند و K/V هر لایه را ذخیره می‌کند. در decode هر token فقط attention به K/V قبلی را انجام می‌دهد و prefix دوباره محاسبه نمی‌شود. RoPE با offset موقعیت cache شده اعمال می‌شود. API فعلی `amd-infer` از `temperature` و `top-k` نیز پشتیبانی می‌کند.

نمونه:

```bash
cargo run --release --features amd-vulkan --bin amd-infer -- \
  --target \
  --checkpoint gemma-agent-target-amd.bin \
  --tokenizer gemma-agent-target-amd.bin.tok \
  --prompt "Rust is a" \
  --tokens 128 \
  --temperature 0.7 \
  --top-k 40 \
  --gpu-kind integrated \
  --gpu 0
```

## سرعت و حافظه

قبل از آموزش مدل هدف، سرعت مدل را اندازه بگیر:

```bash
cargo run --release --features amd-vulkan --bin amd-bench -- \
  --batch-size 1 \
  --context 256 \
  --iterations 50 \
  --gpu-kind integrated \
  --gpu 0
```

برای 8GB RAM، `batch-size=1` نقطهٔ شروع امن است. بعد از مشخص‌شدن مصرف واقعی حافظه می‌توان `--grad-accum` را بالا برد تا batch مؤثر بیشتر شود بدون افزایش batch فیزیکی GPU.

GPU trainer فعلاً F32 است تا correctness و پایداری اولویت داشته باشند. mixed precision و activation checkpointing واقعی باید بعد از benchmark سخت‌افزار و تأیید API/backend اضافه شوند؛ در کد ادعای fake checkpointing وجود ندارد.

## کیفیت مدل

فناوری آموزش به‌تنهایی مدل را باهوش نمی‌کند. ترتیب اثرگذاری عملی این است: corpus بزرگ و تمیز، tokenizer مناسب، معماری پایدار، تعداد tokenهای آموزشی کافی، validation، scheduler و سپس بهینه‌سازی سرعت.

`train.txt` باید حاوی متن واقعی و متنوع باشد؛ یک corpus کوچک تکراری فقط باعث حفظ‌کردن همان متن می‌شود و توان استدلال عمومی ایجاد نمی‌کند.

## CI

CI مسیرهای CPU و AMD را compile/test می‌کند. اجرای واقعی Vulkan روی GitHub Actions بدون GPU معادل سخت‌افزار کاربر نیست، بنابراین benchmark نهایی باید روی خود Radeon انجام شود.

## وضعیت فعلی

هستهٔ AMD اکنون مسیر واقعی آموزش decoder-only شامل **RoPE + native SDPA + SwiGLU + gradient accumulation + validation + optimizer-state resume + KV-cache inference** دارد. optimizer در Burn نیز state قابل‌ذخیره و قابل‌بازیابی دارد. citeturn966778search0turn966778search9
