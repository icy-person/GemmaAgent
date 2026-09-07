# GemmaAgent / gemma-rs

یک موتور آموزشی مدل زبانی از پایه و با Rust خالص. پروژه clean-room است و شامل کد یا وزن‌های Google Gemma نیست.

## سه backend اصلی

پروژه دقیقاً سه مسیر اجرایی دارد:

| Backend | هدف | فناوری | خروجی اصلی |
|---|---|---|---|
| `runner-cpu` | GitHub Actions / CPU runner | Rust scalar CPU + Autograd | آموزش کامل و قابل resume |
| `amd-vulkan` | لپ‌تاپ/دسکتاپ Radeon یا GPU سازگار | Burn + WGPU + Vulkan | آموزش + inference + KV-cache |
| `android-vulkan` | Android arm64 با GPU Vulkan | Burn + WGPU + Vulkan | inference روی دستگاه |

فقط یکی از این سه feature را برای هر build فعال کن. برای دو backend GPU از `--no-default-features` استفاده می‌شود تا CPU backend ناخواسته نیز وارد build نشود. `build.rs` در زمان build این قانون را enforce می‌کند و فعال‌شدن صفر یا چند backend را خطا می‌دهد. مسیر Android و AMD هستهٔ Transformer، tokenizer و فرمت checkpoint مشترک دارند.

## هستهٔ فعلی

- reverse-mode Autograd در مسیر CPU
- causal multi-head self-attention و FFN با SiLU در مسیر CPU
- tokenizer بایتی برای profile کوچک و tokenizer subword آموخته‌شده برای profileهای بزرگ
- byte fallback و ذخیره/بازیابی tokenizer
- next-token cross-entropy با نمونه‌گیری کنترل‌شده از موقعیت‌های context در CPU
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
- نرمال‌سازی صحیح sampled CPU loss، طوری که loss دقیقاً بر تعداد targetهای واقعاً مصرف‌شده تقسیم می‌شود

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
  --targets-per-step 128 \
  --eval-samples 16 \
  --lr 0.0002 \
  --checkpoint-every 25
```

checkpoint کامل شامل وزن، optimizer state، RNG و tokenizer است. trainer از checkpoint قبلی resume می‌کند و loss sampled را با تعداد واقعی targetها نرمال می‌کند.

## آموزش با اینترنت و corpus آنلاین

Pipeline آموزشی اکنون یک مرحلهٔ آنلاین دارد که از **Wikimedia/Wikipedia** دادهٔ متنی می‌گیرد. این جمع‌آوری یک crawler آزاد نیست؛ دامنه و API مشخص است، درخواست‌ها rate-limited هستند، داده cache می‌شود و برای هر صفحه title، URL، revision id، hash و license داخل `data/online_manifest.json` ثبت می‌شود. برای درخواست‌های ماشینی به Wikimedia، User-Agent توصیفی و رعایت throttling ضروری است. urlراهنمای API Policy و User-Agent و TextExtractshttps://meta.wikimedia.org/wiki/API_Policy_Update_2024/en

اجرای محلی:

```bash
python3 scripts/prepare_online_corpus.py \
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

برای ساخت tokenizer اولیه، pipeline ترکیبی از corpus آفلاین و آنلاین را به `data/bootstrap.txt` می‌سازد. بعد از ساخته‌شدن tokenizer، همان tokenizer از checkpoint ادامه پیدا می‌کند و تغییرات وزن مدل روی مرحلهٔ آنلاین نیز اعمال می‌شود.

چرخهٔ آموزش GitHub Actions شامل این مراحل است:

`bootstrap -> literature -> code -> math -> reasoning -> balanced -> online-wikipedia -> repeat`

مرحلهٔ آنلاین از `data/online/train.txt` و validation متناظر خودش استفاده می‌کند؛ بنابراین مدل فقط روی دادهٔ آنلاین train نمی‌شود و دچار drift شدید به سمت یک منبع واحد نمی‌شود.

اجرای دستی یک دور آنلاین:

```bash
cargo run --release --no-default-features --features runner-cpu --bin cpu-train -- \
  --large \
  --steps 500 \
  --data data/online/train.txt \
  --val-data data/online/val.txt \
  --checkpoint checkpoints/gemma-agent.cpu.ckpt \
  --tokenizer checkpoints/gemma-agent.cpu.ckpt.tok \
  --resume checkpoints/gemma-agent.cpu.ckpt \
  --train-context 512 \
  --grad-accum 8 \
  --targets-per-step 128 \
  --eval-samples 16 \
  --lr 0.00006
```

دادهٔ آنلاین به‌صورت artifact نیز کنار checkpoint ذخیره می‌شود تا اجرای بعدی بتواند cache را بازیابی کند. خود corpus generated در git commit نمی‌شود.

## Laptop / AMD Vulkan

برای Radeon روی لینوکس، مسیر `amd-vulkan` با **Burn + WGPU + Vulkan** اجرا می‌شود.

بررسی GPU:

```bash
vulkaninfo --summary
lspci | grep -Ei 'vga|3d|display'
```

build و آموزش:

```bash
cargo build --release --no-default-features --features amd-vulkan --bin amd-train
cargo run --release --no-default-features --features amd-vulkan --bin amd-train -- \
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

برای انتخاب خودکار GPU از `--gpu-kind best` استفاده می‌شود. `--gpu-util 50` duty-cycle تقریبی workload را محدود می‌کند؛ این مقدار محدودکنندهٔ سخت‌افزاری driver نیست.

benchmark:

```bash
cargo run --release --no-default-features --features amd-vulkan --bin amd-bench -- \
  --batch-size 1 \
  --context 256 \
  --iterations 50 \
  --gpu-kind integrated \
  --gpu 0
```

برای throughput بالاتر، افزایش `batch-size` و سپس استفاده از `grad-accum` روی دستگاه دارای حافظهٔ کافی مناسب‌تر از syncهای زیاد با batch=1 است.

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

این backend در CI به‌صورت مستقل برای `aarch64-linux-android` compile-check می‌شود؛ اجرای Vulkan روی یک گوشی واقعی باید روی همان دستگاه benchmark شود.

## معماری مشترک Vulkan

ساختار هر decoder block:

`RMSNorm -> fused QKV -> RoPE(Q,K) -> native causal SDPA -> output projection -> residual -> RMSNorm -> SwiGLU -> down projection -> residual`

در inference، prompt یک‌بار prefill می‌شود و K/V هر لایه در KV-cache نگه‌داری می‌شود؛ tokenهای بعدی incremental پردازش می‌شوند.

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

روی لپ‌تاپ 8GB RAM، `batch-size=1` نقطهٔ شروع امن است و `grad-accum` برای افزایش batch مؤثر استفاده می‌شود. backend Vulkan فعلاً F32 است تا correctness و پایداری اولویت داشته باشند.

در مسیر CPU، `targets-per-step` هزینهٔ هر forward/backward را کنترل می‌کند. برای شروع پایدار `128` مناسب است؛ اگر زمان هر update زیاد شد، این عدد را پایین بیاور و با `grad-accum` تعداد updateهای مؤثر را حفظ کن.

## کیفیت مدل

فناوری backend به‌تنهایی مدل را باهوش نمی‌کند. کیفیت بیشتر به corpus تمیز و بزرگ، tokenizer مناسب، تعداد token کافی، validation صحیح و آموزش طولانی وابسته است. رسیدن به validation loss زیر 1 روی corpus کوچک ممکن است صرفاً نشانهٔ memorization باشد و نباید به‌عنوان تضمین توانایی عمومی مدل تفسیر شود.

## CI

CI سه مسیر را مستقل بررسی می‌کند:

1. `Runner / CPU backend`
2. `Laptop / AMD Vulkan backend`
3. `Android / arm64 Vulkan backend`

همچنین CI یک build با دو backend هم‌زمان را عمداً امتحان می‌کند و انتظار دارد `build.rs` آن را رد کند. GitHub-hosted runner سخت‌افزار Radeon یا GPU موبایل کاربر را شبیه‌سازی نمی‌کند؛ بنابراین benchmark نهایی باید روی دستگاه واقعی انجام شود.

Workflow آموزش CPU نیز به اینترنت متصل است، corpus آنلاین را refresh می‌کند، manifest آن را validate می‌کند و سپس مرحلهٔ `online-wikipedia` را در curriculum اجرا می‌کند.

## وضعیت فعلی

پروژه اکنون سه backend اصلی و جدا دارد: **CPU Runner + AMD Vulkan + Android Vulkan**. هر build فقط یک backend runtime دارد، backendهای GPU بر پایهٔ Vulkan هستند، GPU desktop و Android از هستهٔ مشترک Vulkan استفاده می‌کنند، و pipeline آموزش CPU اکنون علاوه بر corpus آفلاین از یک corpus آنلاینِ قابل‌ردیابی و cache‌شده نیز استفاده می‌کند.