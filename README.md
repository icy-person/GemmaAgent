# GemmaAgent / gemma-rs

یک موتور آموزشی مدل زبانی از پایه و با Rust خالص. این پروژه clean-room است و شامل کد یا وزن‌های Google Gemma نیست.

## هستهٔ فعلی

- reverse-mode Autograd در مسیر CPU
- causal multi-head self-attention
- decoder block و FFN با SiLU
- tokenizer بایتی با BOS/EOS برای مسیر CPU
- tokenizer هدف AMD با subwordهای آموخته‌شده از corpus و byte fallback
- next-token cross-entropy
- آموزش چندهدفهٔ causal در مسیر CPU
- AdamW و global gradient clipping در مسیر CPU
- checkpoint امن برای مسیر CPU
- runtime مستقیم CPU و KV cache برای inference
- sampling با `temperature` و `top-k`
- backend آموزشی CUDA اختیاری با Candle
- backend آموزشی **AMD/Vulkan** با Burn + WGPU
- GPU autograd، pre-norm transformer و dense causal next-token loss
- GPU batch training و gradient accumulation
- curriculum context برای شروع سریع‌تر و انتقال تدریجی به context کامل
- random window sampling برای کاهش هم‌پوشانی و تکرار داده
- train/validation split و validation perplexity
- best-checkpoint و latest-checkpoint جداگانه
- warmup + cosine learning-rate schedule
- GPU checkpoint با Burn binary recorder

## پروفایل‌ها

مدل کوچک:

`vocab=258 | context=128 | d_model=64 | layers=2 | heads=4 | ffn=128`

مدل هدف:

`vocab=16384 | context=1024 | d_model=416 | layers=6 | heads=8 | ffn=1664`

پروفایل هدف CPU در وزن‌های اصلی دقیقاً `19,275,776` پارامتر دارد. backend AMD شامل embedding موقعیت، normalization و output head مستقل است، بنابراین شمارش پارامتر GPU دقیقاً برابر با عدد CPU نیست.

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

مسیر NVIDIA با Candle و CUDA پیاده شده است.

```bash
nvidia-smi
```

```bash
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

برای GPUهای AMD روی لینوکس، مسیر اصلی پروژه **Burn + WGPU + Vulkan** است. این پروژه فعلاً Burn `0.20.1` را پین می‌کند تا مسیر optimizer به نسخه‌ای با ریسک regression شناخته‌شده در `0.21.0` وابسته نباشد. WGPU در این مسیر می‌تواند Vulkan را به‌کار بگیرد و device مجتمع یا مستقل را انتخاب کند. citeturn176319search0turn120file0

ابتدا:

```bash
vulkaninfo --summary
lspci | grep -Ei 'vga|3d|display'
```

### benchmark مدل کوچک روی Radeon مجتمع

```bash
cargo run --release --features amd-vulkan --bin amd-bench -- \
  --batch-size 2 \
  --context 128 \
  --iterations 50 \
  --gpu-kind integrated \
  --gpu 0
```

### آموزش مدل هدف — تنظیم پیشنهادی برای iGPU با 8GB RAM

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

در اولین اجرا، tokenizer هدف از خود `train.txt` ساخته و در فایل `.tok` ذخیره می‌شود؛ اجرای مجدد از همان tokenizer استفاده می‌کند تا IDهای توکن با وزن‌های مدل جابه‌جا نشوند.

آموزش هدف با سه مرحلهٔ context انجام می‌شود: شروع با 256، سپس 512 و در پایان 1024. این کار هزینهٔ محاسباتی ابتدای آموزش را پایین می‌آورد و در انتهای آموزش مدل را روی context کامل تثبیت می‌کند.

`--resume` فقط وزن‌های checkpoint قبلی را برمی‌گرداند و وضعیت optimizer فعلاً از صفر شروع می‌شود:

```bash
cargo run --release --features amd-vulkan --bin amd-train -- \
  --target \
  --steps 20000 \
  --data ./train.txt \
  --checkpoint gemma-agent-target-amd.bin \
  --resume gemma-agent-target-amd.bin \
  --batch-size 1 \
  --grad-accum 1 \
  --lr 0.0003 \
  --checkpoint-every 250 \
  --eval-every 250 \
  --gpu-kind integrated \
  --gpu 0
```

### آموزش مدل کوچک برای تست pipeline

```bash
cargo run --release --features amd-vulkan --bin amd-train -- \
  --steps 5000 \
  --data ./train.txt \
  --checkpoint gemma-agent-amd.bin \
  --batch-size 2 \
  --grad-accum 1 \
  --lr 0.0003 \
  --checkpoint-every 250 \
  --eval-every 250 \
  --gpu-kind integrated \
  --gpu 0
```

`integrated` برای iGPUهای Radeon مناسب است. برای کارت مستقل AMD از `--gpu-kind discrete` استفاده کن؛ `--gpu-kind best` هم اجازه می‌دهد WGPU بهترین device موجود را انتخاب کند.

## معماری AMD

مسیر AMD فعلی شامل:

- token embedding
- learned positional embedding
- causal attention mask
- Transformer pre-norm
- multi-head self-attention
- FFN با activation پیش‌فرض Transformer backend
- RMSNorm نهایی
- output projection بدون bias
- dense next-token loss روی تمام موقعیت‌های context
- random windows + train/validation split
- context curriculum: `256 -> 512 -> 1024`
- batch واقعی روی WGPU/Vulkan
- gradient accumulation
- AdamW با `beta1=0.9`, `beta2=0.95`, `weight_decay=0.1`
- warmup + cosine LR
- latest + best Burn binary checkpoint

برای پروفایل هدف، tokenizer دیگر byte-only نیست: corpus به یک واژگان subword آموزش‌دیده با byte fallback تبدیل می‌شود تا ظرفیت 16,384 خروجی واقعاً استفاده شود. این تغییر برای کیفیت مدل مهم است، چون داشتن یک output head شانزده‌هزارکلاسه در حالی که فقط 258 ID تولید می‌شود، بخش بزرگی از ظرفیت خروجی را بلااستفاده می‌گذارد.

این backend از معماری آموزشی CPU جداست تا مسیر tensor-level واقعی روی GPU داشته باشد. GPU trainer فعلاً F32 است؛ mixed precision و fused attention بعد از تثبیت correctness و benchmark واقعی این سخت‌افزار اضافه خواهند شد.

## صحت optimizer

نسخهٔ Burn `0.21.0` یک گزارش upstream دربارهٔ no-op شدن `Optimizer::step` برای مدل‌های `#[derive(Module)]` دارد؛ در همان گزارش `Param::map` به‌عنوان workaround مؤثر ثبت شده است. بنابراین این مخزن فعلاً `0.20.1` را استفاده می‌کند و trainer علاوه بر آن، قبل از optimizer step وجود gradient را به‌صورت صریح بررسی می‌کند تا آموزش خاموش و بی‌اثر رخ ندهد. citeturn316115view0

## CI

CI علاوه بر مسیر عادی CPU، `cargo check --all-targets --features amd-vulkan` را نیز اجرا می‌کند تا compile مسیر AMD به regression تبدیل نشود. اجرای واقعی Vulkan به GPU runner نیاز دارد.

## وضعیت مهندسی

در حال حاضر مسیر AMD از نظر آموزش از پایه شامل دادهٔ واقعی corpus، tokenizer هدف، split ارزیابی، curriculum context، AdamW، scheduler، checkpoint و معیار perplexity است. برای گام‌های بعدی کیفیت و سرعت، اولویت‌های منطقی عبارت‌اند از fused/SDPA attention، mixed precision، gradient checkpointing، optimizer-state checkpoint/resume کامل و سپس GPU KV-cache inference.
