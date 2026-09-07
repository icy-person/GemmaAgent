# GemmaAgent / gemma-rs

یک موتور آموزشی مدل زبانی از پایه و با Rust خالص. این پروژه clean-room است و شامل کد یا وزن‌های Google Gemma نیست.

## هستهٔ فعلی

- reverse-mode Autograd با `matmul`, `transpose`, `softmax`, `log`, `gather`, row-gather و slicing
- RMSNorm پارامتر-free با backward اختصاصی در مسیر CPU
- causal multi-head self-attention
- decoder block با pre-norm، residual و feed-forward
- SiLU در FFN
- embedding مشترک ورودی/خروجی
- positional encoding سینوسی در مسیر CPU
- tokenizer بایتی با BOS/EOS
- next-token cross-entropy
- آموزش چندهدفهٔ causal از چند موقعیت داخل هر context window
- AdamW با weight decay، bias correction و global gradient clipping در مسیر CPU
- checkpoint باینری shape-safe و corruption-aware برای مسیر CPU
- runtime مستقیم CPU و KV cache
- greedy decoding و sampling با `temperature` و `top-k`
- benchmark داخلی برای prefill/decode
- **backend آموزشی CUDA اختیاری با Candle**
- GPU autograd، RMSNorm، RoPE، dense causal attention، AdamW و batch training
- GPU checkpoint با Safetensors

## پروفایل‌ها

مدل کوچک CPU:

`vocab=258 | context=128 | d_model=64 | layers=2 | heads=4 | ffn=128`

مدل هدف CPU:

`vocab=16384 | context=1024 | d_model=416 | layers=6 | heads=8 | ffn=1664`

فرمول شمارش پارامترهای وزن‌های اصلی:

`vocab*d_model + layers*(4*d_model^2 + 2*d_model*ffn)`

که برای پروفایل هدف `19,275,776` پارامتر است. مسیر GPU به‌دلیل RMSNormهای قابل‌آموزش gainهای اضافی دارد.

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

Backend GPU با Candle 0.11 پیاده شده است. Candle برای Rust backendهای CPU/CUDA و backpropagation دارد و candle-nn نیز AdamW، RMSNorm، RoPE و عملیات attention را ارائه می‌کند. citeturn687109search2turn170084search0turn170084search1turn912734search0

ابتدا بررسی کن کارت NVIDIA و driver دیده می‌شوند:

```bash
nvidia-smi
```

سپس training را روی GPU صفر اجرا کن:

```bash
cargo run --release --features cuda --bin gpu-train -- \
  --steps 5000 \
  --data ./train.txt \
  --checkpoint gemma-agent-gpu.safetensors \
  --batch-size 8 \
  --lr 0.0003 \
  --checkpoint-every 250 \
  --gpu 0
```

در این مسیر loss روی تمام موقعیت‌های `context` محاسبه می‌شود؛ بنابراین آموزش، next-token objective متراکم واقعی دارد. `batch-size` تعداد windowهای هم‌زمان روی GPU را تعیین می‌کند.

برای GPU با حافظه محدود:

```bash
--batch-size 2
```

یا:

```bash
--batch-size 4
```

برای throughput بیشتر، `8` یا بالاتر را تا سقف حافظهٔ کارت افزایش بده.

### GPU benchmark

```bash
cargo run --release --features cuda --bin gpu-bench -- \
  --batch-size 8 \
  --context 128 \
  --iterations 100 \
  --gpu 0
```

## چرا CUDA اختیاری است؟

نسخهٔ عادی پروژه بدون CUDA همچنان با Rust stable ساخته می‌شود. برای GPU باید feature `cuda` فعال شود. candle-core و candle-nn هر دو feature رسمی CUDA دارند. citeturn463911search0turn463911search4

## معماری GPU

مسیر GPU برای training از نظر پایداری و کیفیت یک مرحله جلوتر از kernel سادهٔ CPU است:

- embedding و output projection مشترک
- pre-norm با RMSNorm قابل‌آموزش
- RoPE برای موقعیت
- causal multi-head self-attention
- SiLU FFN
- dense next-token loss روی تمام موقعیت‌ها
- batch training واقعی روی GPU
- AdamW با `beta1=0.9`, `beta2=0.95`, `weight_decay=0.1`
- checkpoint در قالب Safetensors

برای training از نسخه‌های slow/graph-based عملیات RoPE در جاهایی که به gradient نیاز است استفاده شده تا مسیر backward حفظ شود؛ APIهای Candle این عملیات را روی همان device اجرا می‌کنند. citeturn912734search1turn734540search0

## Inference CPU

```bash
cargo run --release -- infer gemma-agent.ckpt "Rust is" \
  --temperature 0.8 \
  --top-k 40 \
  --tokens 128
```

## Checkpoint

`gemma-agent.ckpt` مربوط به مسیر CPU است و `gemma-agent-gpu.safetensors` مربوط به مدل GPU. این دو checkpoint format مشترک ندارند.

## وضعیت مهندسی

مسیر CPU برای correctness و آموزش از پایه حفظ شده است. مسیر CUDA اکنون training واقعی tensor-level روی GPU، autograd، RMSNorm، RoPE و batch loss متراکم دارد. مرحلهٔ بعدی برای کیفیت مدل شامل tokenizer BPE/subword، dataset بزرگ و تمیز، mixed precision، flash/fused attention، memory planning و سپس runtime GPU با KV cache است.
