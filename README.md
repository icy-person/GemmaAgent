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

پروفایل هدف CPU در وزن‌های اصلی دقیقاً `19,275,776` پارامتر دارد. backendهای GPU شامل embedding موقعیت و normalization اضافی هستند، بنابراین شمارش پارامترهای آن‌ها دقیقاً برابر با این عدد نیست.

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

این مسیر با Candle و CUDA پیاده شده است.

```bash
nvidia-smi
```

سپس:

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

برای کارت‌های AMD روی لینوکس، مسیر اصلی پروژه **Burn + WGPU + Vulkan** است. WGPU در لینوکس می‌تواند از Vulkan استفاده کند و `WgpuDevice::DiscreteGpu(n)` برای انتخاب GPU مجزا در دسترس است. citeturn137906search0turn137906search3

ابتدا مطمئن شو Vulkan و GPU دیده می‌شوند:

```bash
vulkaninfo --summary
```

و:

```bash
lspci | grep -Ei 'vga|3d|display'
```

### benchmark مدل کوچک

```bash
cargo run --release --features amd-vulkan --bin amd-bench -- \
  --batch-size 4 \
  --context 128 \
  --iterations 100 \
  --gpu 0
```

### آموزش مدل کوچک

```bash
cargo run --release --features amd-vulkan --bin amd-train -- \
  --steps 5000 \
  --data ./train.txt \
  --checkpoint gemma-agent-amd.bin \
  --batch-size 4 \
  --grad-accum 2 \
  --lr 0.0003 \
  --checkpoint-every 250 \
  --gpu 0
```

### آموزش مدل هدف

```bash
cargo run --release --features amd-vulkan --bin amd-train -- \
  --target \
  --steps 5000 \
  --data ./train.txt \
  --checkpoint gemma-agent-target-amd.bin \
  --batch-size 1 \
  --grad-accum 2 \
  --lr 0.0003 \
  --checkpoint-every 100 \
  --gpu 0
```

مدل هدف با `context=1024` از attention متراکم استفاده می‌کند؛ بنابراین با `batch-size=1` شروع کن و فقط در صورت کافی بودن VRAM آن را افزایش بده.

## معماری AMD

مسیر AMD فعلی شامل:

- embedding توکن
- learned positional embedding
- causal attention mask
- Transformer pre-norm
- multi-head self-attention
- FFN با activation پیش‌فرض Transformer backend
- RMSNorm نهایی
- dense next-token loss روی تمام موقعیت‌های context
- batch واقعی روی WGPU
- gradient accumulation با یک graph برای چند micro-batch
- AdamW با `beta1=0.9`, `beta2=0.95`, `weight_decay=0.1`
- warmup + cosine LR
- checkpoint با Burn `BinFileRecorder`

Burn 0.21 برای Transformer، causal attention mask، Autodiff، WGPU و AdamW APIهای لازم را دارد. citeturn137906search1turn137906search2turn137906search5turn137906search7

## چرا AMD از Vulkan/WGPU استفاده می‌کند؟

هدف این backend، اجرای tensor operations و autograd روی GPUهای AMD بدون وابستگی به CUDA است. `burn-wgpu` backend را از طریق wgpu به APIهای گرافیکی مختلف متصل می‌کند و Vulkan روی Linux یکی از مسیرهای پشتیبانی‌شده است. citeturn137906search3

## Checkpoint

`*.ckpt` و `*.safetensors` و checkpointهای Burn AMD در git نادیده گرفته می‌شوند. checkpointهای CPU، CUDA و AMD فرمت و معماری یکسانی ندارند و مستقیماً قابل جابه‌جایی نیستند.

## وضعیت مهندسی

مسیر CPU برای correctness و آموزش از پایه حفظ شده است. مسیر CUDA و AMD/Vulkan نیز training tensor-level با autograd، normalization، causal attention، AdamW و batch training دارند.

برای افزایش جدی کیفیت مدل، گام‌های اصلی بعدی این‌ها هستند: tokenizer BPE/subword، dataset بزرگ و تمیز، document packing، mixed precision، fused/SDPA attention، memory planning، gradient checkpointing، optimizer-state checkpoint/resume کامل، validation perplexity و سپس GPU KV-cache inference.
