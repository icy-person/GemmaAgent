# GemmaAgent / gemma-rs

یک موتور آموزشی مدل زبانی از پایه و با Rust خالص. این پروژه clean-room است و شامل کد یا وزن‌های Google Gemma نیست.

## هستهٔ فعلی

- reverse-mode Autograd در مسیر CPU
- causal multi-head self-attention
- decoder block و FFN با SiLU
- tokenizer بایتی با BOS/EOS
- next-token cross-entropy
- آموزش چندهدفهٔ causal در مسیر CPU
- AdamW و global gradient clipping در مسیر CPU
- checkpoint امن برای مسیر CPU
- runtime مستقیم CPU و KV cache برای inference
- sampling با `temperature` و `top-k`
- backend آموزشی CUDA اختیاری با Candle
- backend آموزشی **AMD/Vulkan** اختیاری با Burn + WGPU
- GPU autograd، pre-norm transformer و dense causal next-token loss
- GPU batch training و gradient accumulation
- warmup + cosine learning-rate schedule
- GPU checkpoint با Burn binary recorder

## پروفایل‌ها

مدل کوچک:

`vocab=258 | context=128 | d_model=64 | layers=2 | heads=4 | ffn=128`

مدل هدف:

`vocab=16384 | context=1024 | d_model=416 | layers=6 | heads=8 | ffn=1664`

پروفایل هدف CPU در وزن‌های اصلی دقیقاً `19,275,776` پارامتر دارد. backendهای GPU شامل embedding موقعیت، normalization و output head مستقل هستند، بنابراین شمارش پارامتر GPU دقیقاً برابر با عدد CPU نیست.

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

برای GPUهای AMD روی لینوکس، مسیر اصلی پروژه **Burn + WGPU + Vulkan** است. Burn 0.21، Autodiff و AdamW و Transformer را ارائه می‌کند و burn-wgpu، WGPU را برای backendهای GPU در اختیار می‌گذارد. WGPU در لینوکس می‌تواند از Vulkan استفاده کند و `WgpuDevice` انتخاب `IntegratedGpu` و `DiscreteGpu` را فراهم می‌کند. citeturn137906search0turn137906search1turn137906search3turn137906search7

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

### آموزش مدل کوچک

```bash
cargo run --release --features amd-vulkan --bin amd-train -- \
  --steps 5000 \
  --data ./train.txt \
  --checkpoint gemma-agent-amd.bin \
  --batch-size 2 \
  --grad-accum 2 \
  --lr 0.0003 \
  --checkpoint-every 250 \
  --gpu-kind integrated \
  --gpu 0
```

`integrated` برای iGPUهای Radeon مناسب است. برای کارت مستقل AMD از `--gpu-kind discrete` استفاده کن؛ `--gpu-kind best` هم اجازه می‌دهد WGPU بهترین device موجود را انتخاب کند.

### آموزش مدل هدف

```bash
cargo run --release --features amd-vulkan --bin amd-train -- \
  --target \
  --steps 5000 \
  --data ./train.txt \
  --checkpoint gemma-agent-target-amd.bin \
  --batch-size 1 \
  --grad-accum 1 \
  --lr 0.0003 \
  --checkpoint-every 100 \
  --gpu-kind integrated \
  --gpu 0
```

مدل هدف با `context=1024` و attention متراکم حافظهٔ قابل‌توجهی می‌خواهد؛ روی iGPU بهتر است با `batch-size=1` و `grad-accum=1` شروع شود.

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
- batch واقعی روی WGPU/Vulkan
- gradient accumulation با یک graph تجمیعی
- AdamW با `beta1=0.9`, `beta2=0.95`, `weight_decay=0.1`
- warmup + cosine LR
- Burn binary checkpoint

این backend از معماری آموزشی CPU جداست تا مسیر tensor-level واقعی روی GPU داشته باشد. GPU trainer فعلاً F32 است؛ mixed precision و fused attention عمداً بعد از تثبیت correctness اضافه خواهند شد.

## CI

CI علاوه بر مسیر عادی CPU، `cargo check --all-targets --features amd-vulkan` را نیز اجرا می‌کند تا compile مسیر AMD به regression تبدیل نشود. اجرای واقعی Vulkan به GPU runner نیاز دارد.

## وضعیت مهندسی

مسیر CPU، CUDA و AMD/Vulkan در یک مخزن نگهداری می‌شوند. گام‌های بعدی کیفیت مدل عبارت‌اند از tokenizer BPE/subword، dataset بزرگ و تمیز، document packing، mixed precision، fused/SDPA attention، memory planning، gradient checkpointing، optimizer-state checkpoint/resume کامل، validation perplexity و سپس GPU KV-cache inference.
