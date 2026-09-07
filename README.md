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
- **backend آموزشی CUDA اختیاری با Candle**
- GPU autograd، pre-norm RMSNorm، RoPE، causal attention، AdamW و batch training
- GPU gradient accumulation
- warmup + cosine learning-rate schedule
- GPU checkpoint با Safetensors

## پروفایل‌ها

مدل کوچک:

`vocab=258 | context=128 | d_model=64 | layers=2 | heads=4 | ffn=128`

مدل هدف:

`vocab=16384 | context=1024 | d_model=416 | layers=6 | heads=8 | ffn=1664`

پروفایل هدف CPU در وزن‌های اصلی دقیقاً `19,275,776` پارامتر دارد. مسیر GPU به‌دلیل gainهای RMSNorm تعداد پارامتر بیشتری دارد.

## آموزش CPU

```bash
cargo run --release -- train 5000 gemma-agent.ckpt \
  --data ./train.txt \
  --grad-accum 8 \
  --targets-per-step 32 \
  --lr 0.001 \
  --checkpoint-every 250
```

## آموزش GPU / CUDA

Backend GPU با Candle 0.11 پیاده شده است. Candle برای CPU/CUDA، backpropagation، RMSNorm، RoPE و AdamW API رسمی دارد. citeturn687109search2turn170084search0turn170084search1turn313133search4

ابتدا بررسی کن GPU دیده می‌شود:

```bash
nvidia-smi
```

### تست سریع روی مدل کوچک

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

این مسیر تمام موقعیت‌های context را در loss وارد می‌کند و optimizer updateها را روی GPU انجام می‌دهد. `grad-accum` چند microbatch را قبل از هر update جمع می‌کند. learning rate از warmup عبور کرده و سپس cosine decay می‌شود.

### مدل هدف ۱۹ میلیون پارامتری

```bash
cargo run --release --features cuda --bin gpu-train -- \
  --target \
  --steps 5000 \
  --data ./train.txt \
  --checkpoint gemma-agent-target-gpu.safetensors \
  --batch-size 1 \
  --grad-accum 4 \
  --lr 0.0003 \
  --checkpoint-every 100 \
  --gpu 0
```

برای مدل هدف، به‌علت `context=1024` و attention متراکم، از `--batch-size 1` شروع کن و فقط در صورت کافی بودن VRAM مقدار آن را بالا ببر. GPU trainer در حال حاضر F32 است تا پایداری عددی training حفظ شود.

### GPU benchmark

مدل debug:

```bash
cargo run --release --features cuda --bin gpu-bench -- \
  --batch-size 8 --context 128 --iterations 100 --gpu 0
```

مدل هدف:

```bash
cargo run --release --features cuda --bin gpu-bench -- \
  --target --batch-size 1 --context 1024 --iterations 20 --gpu 0
```

## معماری GPU

مسیر GPU شامل:

- embedding و output projection مشترک
- pre-norm RMSNorm قابل‌آموزش
- RoPE با مسیر differentiable برای backward
- Q/K/V و output projection بدون bias
- causal multi-head attention
- SiLU FFN
- dense next-token loss روی تمام موقعیت‌ها
- batch واقعی روی CUDA
- gradient accumulation با `GradStore`
- AdamW با `beta1=0.9`, `beta2=0.95`, `weight_decay=0.1`
- warmup + cosine LR
- checkpoint در Safetensors

Candle برای `GradStore.extend` و optimizer `step` API رسمی دارد؛ بنابراین accumulation در سطح gradient قبل از optimizer update انجام می‌شود. citeturn544034search1turn544034search0

## چرا CUDA اختیاری است؟

پروژه بدون feature `cuda` همچنان مسیر CPU و CI را حفظ می‌کند. feature مربوط به CUDA در `candle-core` و `candle-nn` فعال می‌شود. citeturn544034search6

## Checkpoint

`*.ckpt` برای مسیر CPU و `*.safetensors` برای مسیر GPU در git نادیده گرفته می‌شوند. checkpoint GPU را نمی‌توان با runtime CPU فعلی مستقیماً بارگذاری کرد.

## وضعیت مهندسی

مسیر GPU اکنون training واقعی روی CUDA را دارد و از نظر pipeline، objective، normalization، RoPE، optimizer و checkpoint یک پله جدی‌تر از kernel scalar قبلی است. برای کیفیت مدل واقعی، مهم‌ترین ارتقاء بعدی tokenizer BPE/subword، dataset بزرگ و تمیز، packing، mixed precision پس از تثبیت F32، fused/SDPA attention، memory planning و سپس GPU KV-cache inference است.
