# GemmaAgent / gemma-rs

یک موتور آموزشی مدل زبانی از پایه و با Rust خالص. پروژه clean-room است و شامل کد یا وزن‌های Google Gemma نیست.

## سه backend اصلی

پروژه اکنون سه مسیر اجرایی مشخص دارد:

| Backend | هدف | فناوری | خروجی اصلی |
|---|---|---|---|
| `runner-cpu` | GitHub Actions / CPU runner | Rust scalar CPU + Autograd | آموزش کامل و قابل resume |
| `amd-vulkan` | لپ‌تاپ/دسکتاپ Radeon یا GPU سازگار | Burn + WGPU + Vulkan | آموزش + inference + KV-cache |
| `android-vulkan` | Android arm64 با GPU Vulkan | Burn + WGPU + Vulkan | inference روی دستگاه |

WGPU به‌صورت رسمی Vulkan را روی Linux و Android پشتیبانی می‌کند، بنابراین مسیر Android از همان هستهٔ Vulkan استفاده می‌کند و مدل/توکنایزر را با backend لپ‌تاپ به اشتراک می‌گذارد. citeturn650859search0turn650859search2

## هستهٔ فعلی

- reverse-mode Autograd در مسیر CPU
- causal multi-head self-attention و FFN با SiLU در مسیر CPU
- tokenizer بایتی CPU با BOS/EOS
- tokenizer هدف AMD/Android با subwordهای آموخته‌شده از corpus و byte fallback
- next-token cross-entropy روی تمام موقعیت‌های causal در backend GPU
- AdamW، gradient clipping و warmup/cosine schedule
- train/validation split و validation perplexity
- latest و best checkpoint
- checkpoint مدل + optimizer state + RNG/training metadata برای resume
- backend CUDA اختیاری با Candle
- backend **Vulkan** مشترک برای AMD desktop و Android
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

پروفایل هدف CPU در وزن‌های اصلی دقیقاً `19,275,776` پارامتر دارد. پروفایل long-training CPU نیز دقیقاً `49,807,360` پارامتر دارد.

## GitHub Runner / CPU

backend پیش‌فرض `runner-cpu` است و برای GitHub Actions طراحی شده است:

```bash
cargo build --release --features runner-cpu --bin cpu-train
cargo run --release --features runner-cpu --bin cpu-train -- \
  --large \
  --steps 5000 \
  --data ./train.txt \
  --val-data ./val.txt \
  --checkpoint checkpoints/gemma-agent.cpu.ckpt \
  --tokenizer checkpoints/gemma-agent.cpu.ckpt.tok \
  --grad-accum 8 \
  --targets-per-step 128 \
  --eval-samples 16 \
  --lr 0.0002 \
  --checkpoint-every 25
```

resume کامل از وزن، optimizer state، RNG و tokenizer انجام می‌شود.

## لپ‌تاپ / AMD Vulkan

برای Radeon روی لینوکس، مسیر `amd-vulkan` با **Burn + WGPU + Vulkan** ساخته شده است.

بررسی GPU:

```bash
vulkaninfo --summary
lspci | grep -Ei 'vga|3d|display'
```

آموزش:

```bash
cargo build --release --features amd-vulkan --bin amd-train
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

برای انتخاب خودکار GPU از `--gpu-kind best` استفاده می‌شود. `--gpu-util 50` نیز duty-cycle تقریبی workload را محدود می‌کند.

## Android / Vulkan

Android از backend مستقل `android-vulkan` استفاده می‌کند ولی هستهٔ Transformer، tokenizer و checkpoint آن با مسیر AMD مشترک است. مسیر رسمی WGPU برای Linux/Android شامل Vulkan است. citeturn650859search0turn650859search3

برای arm64-v8a:

```bash
rustup target add aarch64-linux-android
cargo ndk -t arm64-v8a build --release --features android-vulkan --bin android-infer
```

سپس binary را روی دستگاه اجرا کن یا به اپ Android خودت بسته‌بندی کن. نمونهٔ inference:

```bash
./android-infer \
  --target \
  --checkpoint /data/local/tmp/model.bin \
  --tokenizer /data/local/tmp/model.bin.tok \
  --prompt "Rust is" \
  --tokens 64 \
  --temperature 0.7 \
  --top-k 40
```

برای build و بررسی خودکار Android در CI:

```bash
cargo check --target aarch64-linux-android --features android-vulkan --bin android-infer
```

## CUDA

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

## معماری Vulkan

ساختار هر decoder block:

`RMSNorm -> fused QKV -> RoPE(Q,K) -> native causal SDPA -> output projection -> residual -> RMSNorm -> SwiGLU -> down projection -> residual`

attention از primitive بومی Burn استفاده می‌کند و `is_causal=true` را مستقیماً به backend می‌دهد. gradient accumulation نیز با `GradientsAccumulator` انجام می‌شود.

KV-cache در inference ابتدا prompt را prefill می‌کند و سپس هر token جدید را incremental پردازش می‌کند.

## سرعت و حافظه

قبل از آموزش مدل هدف روی لپ‌تاپ:

```bash
cargo run --release --features amd-vulkan --bin amd-bench -- \
  --batch-size 1 \
  --context 256 \
  --iterations 50 \
  --gpu-kind integrated \
  --gpu 0
```

برای لپ‌تاپ 8GB RAM، `batch-size=1` نقطهٔ شروع امن است و `grad-accum` برای افزایش batch مؤثر استفاده می‌شود.

GPU trainer فعلاً F32 است تا correctness و پایداری اولویت داشته باشند؛ mixed precision بعد از benchmark واقعی سخت‌افزار اضافه می‌شود.

## کیفیت مدل

فناوری آموزش به‌تنهایی مدل را باهوش نمی‌کند. ترتیب اثرگذاری عملی این است: corpus بزرگ و تمیز، tokenizer مناسب، معماری پایدار، تعداد tokenهای آموزشی کافی، validation، scheduler و سپس بهینه‌سازی سرعت.

`train.txt` باید حاوی متن واقعی و متنوع باشد؛ یک corpus کوچک تکراری فقط باعث حفظ‌کردن همان متن می‌شود و توان استدلال عمومی ایجاد نمی‌کند.

## CI

CI سه مسیر را به‌صورت جدا بررسی می‌کند:

1. `Runner / CPU backend`
2. `Laptop / AMD Vulkan backend`
3. `Android / arm64 Vulkan backend`

روی GitHub-hosted runner امکان benchmark واقعی Radeon یا GPU موبایل وجود ندارد؛ CI فقط compile/test correctness را بررسی می‌کند. benchmark نهایی باید روی سخت‌افزار واقعی انجام شود.

## وضعیت فعلی

هستهٔ Vulkan اکنون بین AMD desktop و Android مشترک است و هر دو از **RoPE + native SDPA + SwiGLU + gradient accumulation + validation + optimizer-state resume + KV-cache inference** استفاده می‌کنند.
