# GemmaAgent / gemma-rs

یک موتور آموزشی مدل زبانی از پایه و با Rust خالص. پروژه clean-room است و شامل کد یا وزن‌های Google Gemma نیست.

## سه backend اصلی

| Backend | هدف | فناوری | خروجی اصلی |
|---|---|---|---|
| `runner-cpu` | GitHub Actions / CPU runner | Rust scalar CPU + Autograd | آموزش کامل و قابل resume |
| `amd-vulkan` | لپ‌تاپ/دسکتاپ Radeon یا GPU سازگار | Burn + WGPU + Vulkan | آموزش + inference + KV-cache |
| `android-vulkan` | Android arm64 با GPU Vulkan | Burn + WGPU + Vulkan | inference روی دستگاه |

فقط یکی از این سه feature را برای هر build فعال کن. برای دو backend GPU از `--no-default-features` استفاده می‌شود تا CPU backend ناخواسته نیز وارد build نشود. `build.rs` در زمان build این قانون را enforce می‌کند و فعال‌شدن صفر یا چند backend را خطا می‌دهد. مسیر Android و AMD هستهٔ Transformer، tokenizer و فرمت checkpoint مشترک دارند.

## هستهٔ فعلی

- reverse-mode Autograd در مسیر CPU (matmul برای شکل غالب مدل — بردار ۱×N ضرب در ماتریس — بین چند thread موازی می‌شود)
- causal multi-head self-attention با RoPE روی Q/K و FFN با SiLU در مسیر CPU — هم‌راستا با معماری backend GPU
- مقداردهی اولیهٔ وزن‌ها متناسب با fan-in (LeCun-style) به‌جای بازهٔ ثابت
- tokenizer بایتی برای profile کوچک و tokenizer subword آموخته‌شده برای profileهای بزرگ
- byte fallback و ذخیره/بازیابی tokenizer
- next-token cross-entropy با نمونه‌گیری بدون تکرار از موقعیت‌های context در CPU
- next-token cross-entropy روی تمام موقعیت‌های causal در backend GPU
- AdamW، gradient clipping و warmup/cosine schedule
- train/validation split و validation perplexity
- latest و best checkpoint
- checkpoint مدل + optimizer state + RNG/training metadata برای resume
- backend **Vulkan** مشترک برای AMD desktop و Android
- decoder-only GPU با pre-norm RMSNorm
- RoPE روی Q/K
- native scaled dot-product attention با causal mode
- SwiGLU feed-forward
- gradient accumulation با GradientsAccumulator
- KV-cache واقعی برای prefill و incremental decoding
- CPU LR scheduler پیشرفتهٔ warmup→cosine

## پروفایل‌ها

مدل کوچک:

`vocab=258 | context=128 | d_model=64 | layers=2 | heads=4 | ffn=128`

مدل هدف:

`vocab=16384 | context=1024 | d_model=416 | layers=6 | heads=8 | ffn=1664`

پروفایل هدف CPU در وزن‌های اصلی دقیقاً `19,275,776` پارامتر دارد. پروفایل long-training CPU نیز دقیقاً `49,807,360` پارامتر دارد.

## GitHub Runner / CPU

```bash
cargo build --release --no-default-features --features runner-cpu --bin cpu-train
cargo run --release --no-default-features --features runner-cpu --bin cpu-train -- \
  --large \
  --steps 5000 \
  --data ./train.txt \
  --val-data ./val.txt \
  --checkpoint checkpoints/gemma-agent.cpu.ckpt \
  --tokenizer checkpoints/gemma-agent.cpu.ckpt.tok \
  --grad-accum 8 \
  --targets-per-step 256 \
  --warmup 100 \
  --min-lr-ratio 0.1 \
  --lr 0.0002 \
  --checkpoint-every 25
```

`targets-per-step` اکنون targetهای causal یکتا را انتخاب می‌کند و با افزایش آن، گرادیان نمایندهٔ بهتری از کل context می‌شود. `warmup` و `min-lr-ratio` روی LR scheduler اعمال می‌شوند. checkpoint کامل شامل وزن، optimizer state، RNG و tokenizer است.

## آموزش با اینترنت و corpus آنلاین

Pipeline آموزشی یک مرحلهٔ آنلاین دارد که از **Wikimedia/Wikipedia** دادهٔ متنی می‌گیرد. این جمع‌آوری crawler آزاد نیست؛ دامنه و API مشخص است، درخواست‌ها rate-limited هستند، داده cache می‌شود و برای هر صفحه title، URL، revision id، hash و license داخل `data/online_manifest.json` ثبت می‌شود. برای درخواست‌های ماشینی به Wikimedia، User-Agent توصیفی و رعایت throttling ضروری است. urlراهنمای API Policy و User-Agent و TextExtractshttps://meta.wikimedia.org/wiki/API_Policy_Update_2024/en

اجرای محلی:

```bash
python3 scripts/prepare_online_corpus.py \
  --refresh \
  --pages-per-topic 12 \
  --max-pages 160 \
  --min-chars 300 \
  --delay 0.35
```

خروجی:

```text
data/online/train.txt
data/online/val.txt
data/online/pages/
data/online_manifest.json
```

برای ساخت tokenizer اولیه، pipeline ترکیبی از corpus آفلاین و آنلاین را به `data/bootstrap.txt` می‌سازد. بعد از ساخته‌شدن tokenizer، tokenizer ثابت می‌ماند تا checkpoint قبلی با واژگان متفاوت خراب نشود.

چرخهٔ آموزش GitHub Actions شامل این مراحل است:

`bootstrap -> literature -> code -> math -> reasoning -> balanced -> online-wikipedia -> repeat`

مرحلهٔ آنلاین از `data/online/train.txt` و validation متناظر خودش استفاده می‌کند؛ بنابراین مدل فقط روی دادهٔ آنلاین train نمی‌شود و drift شدید به سمت یک منبع واحد کمتر می‌شود.

## Laptop / AMD Vulkan — آموزش واقعی روی GPU

GitHub-hosted runner به GPU واقعی AMD کاربر دسترسی ندارد. برای training واقعی روی Radeon، یک **self-hosted GitHub runner** با labelهای `self-hosted`, `linux`, `amd-vulkan` آماده شده است.

بعد از نصب runner روی لپ‌تاپ:

```bash
vulkaninfo --summary
lspci | grep -Ei 'vga|3d|display'
```

Workflow `Train GemmaAgent AMD Vulkan` را از GitHub Actions به‌صورت دستی اجرا کن. پارامترهای پیش‌فرض برای دستگاه‌های کم‌حافظه محافظه‌کارانه‌اند:

```text
profile=target-19,275,776
batch-size=2
grad-accum=2
gpu-util=100
```

برای **بیشترین سرعت** `gpu-util=100` استفاده می‌شود. مقدار `50` فقط duty-cycle تقریبی workload را پایین می‌آورد و باعث سریع‌تر شدن آموزش نمی‌شود.

برای اجرای مستقیم محلی:

```bash
cargo build --release --no-default-features --features amd-vulkan --bin amd-train
cargo run --release --no-default-features --features amd-vulkan --bin amd-train -- \
  --target \
  --steps 20000 \
  --data data/online/train.txt \
  --checkpoint checkpoints/gemma-agent-target-amd.bin \
  --best-checkpoint checkpoints/gemma-agent-target-amd.bin.best \
  --tokenizer checkpoints/gemma-agent-target-amd.bin.tok \
  --batch-size 2 \
  --grad-accum 2 \
  --lr 0.00015 \
  --checkpoint-every 250 \
  --eval-every 250 \
  --gpu-kind integrated \
  --gpu 0 \
  --gpu-util 100
```

scheduler AMD از قبل warmup + cosine و gradient clipping دارد و loss روی تمام tokenهای batch محاسبه می‌شود. برای throughput بالاتر، اول `batch-size` را تا مرز امن حافظه بالا ببر، سپس از `grad-accum` استفاده کن.

benchmark:

```bash
cargo run --release --no-default-features --features amd-vulkan --bin amd-bench -- \
  --batch-size 2 \
  --context 256 \
  --iterations 50 \
  --gpu-kind integrated \
  --gpu 0
```

## Android / Vulkan

Android با feature `android-vulkan` و target `aarch64-linux-android` همان موتور Vulkan مسیر AMD را استفاده می‌کند و برای inference ساخته شده است.

```bash
rustup target add aarch64-linux-android
cargo install cargo-ndk
cargo ndk -t arm64-v8a build --release --no-default-features --features android-vulkan --bin android-infer
```

نمونهٔ اجرا روی دستگاه:

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

این backend برای **inference** طراحی شده و CI آن را برای `aarch64-linux-android` compile-check می‌کند؛ آموزش روی Android در این نسخه فعال نشده است چون مسیر فعلی optimizer/training به عنوان desktop/runner workload طراحی شده است.

## معماری مشترک Vulkan

ساختار هر decoder block:

`RMSNorm -> fused QKV -> RoPE(Q,K) -> native causal SDPA -> output projection -> residual -> RMSNorm -> SwiGLU -> down projection -> residual`

در inference، prompt یک‌بار prefill می‌شود و K/V هر لایه در KV-cache نگه‌داری می‌شود؛ tokenهای بعدی incremental پردازش می‌شوند.

مسیر CPU از همین idea برای RoPE(Q,K) استفاده می‌کند (بدون fused QKV و بدون KV-cache، چون هدفش portability و correctness برای CI است نه throughput)، اما دیگر معماری متفاوتی نسبت به GPU ندارد.

## انتخاب backend

Runner:

```bash
cargo test --all-targets --no-default-features --features runner-cpu
```

Laptop:

```bash
cargo check --all-targets --no-default-features --features amd-vulkan
```

Android:

```bash
cargo check --target aarch64-linux-android --no-default-features --features android-vulkan --bin android-infer
```

هر build باید با یک backend مشخص ساخته شود. این قید هم در مستندات و هم در `build.rs` enforce شده است.

## سرعت و حافظه

روی لپ‌تاپ 8GB RAM، برای CPU `grad-accum=8` و برای AMD `batch-size=2, grad-accum=2` نقطهٔ شروع محافظه‌کارانه هستند. backend Vulkan فعلاً F32 است تا correctness و پایداری اولویت داشته باشند.

برای رسیدن هم‌زمان به سرعت و دقت بیشتر، ترتیب بهینه‌سازی این پروژه این است: افزایش throughput با batch/fusion، افزایش diversity داده، target sampling یکتا در CPU، warmup/cosine LR، validation منظم، و ادامهٔ آموزش از checkpoint به‌جای شروع دوباره.

## کیفیت مدل

فناوری backend به‌تنهایی مدل را باهوش نمی‌کند. کیفیت بیشتر به corpus تمیز و بزرگ، tokenizer مناسب، تعداد token کافی، validation صحیح و آموزش طولانی وابسته است. رسیدن به validation loss زیر 1 روی corpus کوچک ممکن است صرفاً نشانهٔ memorization باشد و نباید به‌عنوان تضمین توانایی عمومی مدل تفسیر شود.

## CI

CI سه مسیر را مستقل بررسی می‌کند:

1. `Runner / CPU backend`
2. `Laptop / AMD Vulkan backend`
3. `Android / arm64 Vulkan backend`

همچنین CI یک build با دو backend هم‌زمان را عمداً امتحان می‌کند و انتظار دارد `build.rs` آن را رد کند. GitHub-hosted runner سخت‌افزار Radeon یا GPU موبایل کاربر را شبیه‌سازی نمی‌کند؛ بنابراین benchmark نهایی AMD باید روی دستگاه واقعی انجام شود.

Workflow آموزش CPU به اینترنت متصل است، corpus آنلاین را refresh می‌کند، manifest آن را validate می‌کند و مرحلهٔ `online-wikipedia` را در curriculum اجرا می‌کند. Workflow جداگانهٔ AMD نیز همین corpus را روی GPU واقعی self-hosted آموزش می‌دهد.

## وضعیت فعلی

پروژه اکنون سه backend اصلی و جدا دارد: **CPU Runner + AMD Vulkan + Android Vulkan**. هر build فقط یک backend runtime دارد، backendهای GPU بر پایهٔ Vulkan هستند، GPU desktop و Android از هستهٔ مشترک Vulkan استفاده می‌کنند، CPU و GPU اکنون هر دو RoPE روی Q/K دارند (دیگر معماری متفاوتی ندارند)، CPU training scheduler و sampling بهینه‌تری دارد، matmul مسیر CPU برای شکل غالب مدل بین چند thread موازی می‌شود، مقداردهی اولیهٔ وزن‌ها متناسب با fan-in است، و pipeline آموزش علاوه بر corpus آفلاین از corpus آنلاینِ قابل‌ردیابی و cache‌شده نیز استفاده می‌کند.