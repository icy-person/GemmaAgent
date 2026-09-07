# GemmaAgent — راهنمای کامل آموزش مدل

این فایل مسیر پیشنهادی برای آموزش `GemmaAgent` از آماده‌سازی corpus تا آموزش روی CPU یا AMD/Vulkan، ارزیابی، checkpoint/resume و inference را مستند می‌کند.

> **نکتهٔ مهم:** این پروژه یک پیاده‌سازی clean-room آموزشی است و کد یا وزن‌های Google Gemma را شامل نمی‌شود. پروفایل فعلی مدل هدف در `src/config.rs` برابر `vocab=16384`, `context=1024`, `d_model=416`, `layers=6`, `heads=8`, `ffn=1664` و 19,275,776 پارامتر است.

---

## 1. معماری فعلی

مدل AMD یک decoder-only Transformer است:

```text
Token IDs
   │
   ▼
Embedding
   │
   ├── RMSNorm
   ├── fused QKV projection
   ├── RoPE(Q,K)
   ├── causal scaled dot-product attention
   ├── output projection
   └── residual
   │
   ├── RMSNorm
   ├── SwiGLU
   ├── down projection
   └── residual
   │
   ▼
final RMSNorm
   │
   ▼
LM Head
   │
   ▼
next-token logits
```

در inference، cache واقعی K/V برای هر لایه نگهداری می‌شود؛ prompt ابتدا prefill می‌شود و سپس هر token با `step` ادامه پیدا می‌کند.

---

## 2. پروفایل‌ها

### Debug

```text
vocab   = 258
context = 128
d_model = 64
layers  = 2
heads   = 4
ffn     = 128
```

برای تست سریع build و correctness استفاده شود.

### Target

```text
vocab   = 16384
context = 1024
d_model = 416
layers  = 6
heads   = 8
ffn     = 1664
```

این پروفایل حدود 19.3M پارامتر دارد و برای آموزش اصلی پروژه در نظر گرفته شده است.

---

## 3. پیش‌نیازهای AMD/Vulkan

روی لینوکس مطمئن شو Vulkan و درایور AMD درست کار می‌کنند:

```bash
vulkaninfo --summary
lspci | grep -Ei 'vga|3d|display'
```

ابتدا build بدون آموزش:

```bash
cargo check --release --features amd-vulkan
```

سپس تست‌ها:

```bash
cargo test --release --features amd-vulkan
```

---

## 4. آماده‌سازی corpus

فایل اصلی آموزش:

```text
./train.txt
```

باید UTF-8 و تا حد ممکن تمیز و متنوع باشد.

### corpus ضعیف

```text
سلام
سلام
سلام
من یک مدل زبان هستم
من یک مدل زبان هستم
```

این نوع داده باعث overfit سریع می‌شود و مدل عمومی خوبی نمی‌سازد.

### corpus بهتر

داده‌ها را از چند حوزه ترکیب کن:

- فارسی عمومی و ادبی
- انگلیسی عمومی
- مستندات فنی
- Rust و Linux
- علوم و ریاضی
- پرسش و پاسخ تمیز
- متن آموزشی با ساختار مناسب

دادهٔ تکراری، HTML خام، متن خراب و محتوای بسیار کم‌کیفیت را قبل از آموزش حذف کن.

### ساختار پیشنهادی

هر نمونه بهتر است مرز مشخصی داشته باشد. برای corpus آموزشی ساده، separatorهای ثابت بین اسناد کمک می‌کنند:

```text
<document>
...
</document>

<document>
...
</document>
```

در صورت استفاده از قالب سفارشی، tokenizer و dataset باید همان قرارداد را حفظ کنند.

---

## 5. Tokenizer هدف

وقتی از پروفایل target استفاده می‌شود، `AmdTokenizer` در صورت نبود tokenizer آن را از corpus آموزش می‌دهد و در مسیر `--tokenizer` ذخیره می‌کند. corpus بعداً با همان tokenizer encode می‌شود.

بنابراین اجرای اول target ممکن است tokenizer را نیز بسازد:

```bash
cargo run --release --features amd-vulkan --bin amd-train -- \
  --target \
  --steps 100 \
  --data ./train.txt \
  --checkpoint ./checkpoints/smoke.bin \
  --tokenizer ./checkpoints/target.tok \
  --batch-size 1 \
  --grad-accum 1 \
  --lr 0.0003 \
  --checkpoint-every 50 \
  --eval-every 50 \
  --gpu-kind integrated \
  --gpu 0
```

بعد از ساخته‌شدن tokenizer، آن را نگه دار و برای runهای بعدی دوباره استفاده کن. تغییر tokenizer در میانهٔ آموزش مدل قبلی مجاز نیست.

---

## 6. تست سریع قبل از آموزش اصلی

قبل از اجرای طولانی، با run کوچک مطمئن شو همه‌چیز درست است:

```bash
mkdir -p checkpoints

cargo run --release --features amd-vulkan --bin amd-train -- \
  --target \
  --steps 100 \
  --data ./train.txt \
  --checkpoint ./checkpoints/smoke.bin \
  --best-checkpoint ./checkpoints/smoke.best \
  --tokenizer ./checkpoints/target.tok \
  --batch-size 1 \
  --grad-accum 1 \
  --lr 0.0003 \
  --checkpoint-every 50 \
  --eval-every 50 \
  --gpu-kind integrated \
  --gpu 0
```

در خروجی باید حداقل این اطلاعات را ببینی:

```text
GPU=integrated:0
params=...
loss ... lr ... tok/s ... ctx ...
validation ... loss ... perplexity ...
checkpoint -> ...
```

اگر `loss` یا `perplexity` برابر `NaN`/`Inf` شد، آموزش را ادامه نده و مشکل را قبل از run اصلی بررسی کن.

---

## 7. آموزش پیشنهادی AMD با حافظه محدود

برای GPUهایی که حافظه اشتراکی با RAM دارند، نقطهٔ شروع محافظه‌کارانه:

```bash
cargo run --release --features amd-vulkan --bin amd-train -- \
  --target \
  --steps 20000 \
  --data ./train.txt \
  --checkpoint ./checkpoints/gemma-agent.bin \
  --best-checkpoint ./checkpoints/gemma-agent.best \
  --tokenizer ./checkpoints/target.tok \
  --batch-size 1 \
  --grad-accum 1 \
  --lr 0.0003 \
  --checkpoint-every 250 \
  --eval-every 250 \
  --gpu-kind integrated \
  --gpu 0
```

`batch-size=1` فشار حافظه را کم می‌کند. برای بالا بردن batch مؤثر بدون بالا بردن batch فیزیکی می‌توان `--grad-accum` را افزایش داد:

```bash
--batch-size 1 --grad-accum 4
```

در این حالت، چهار micro-batch قبل از هر optimizer update در یک gradient accumulation ادغام می‌شوند.

---

## 8. GPU مستقل AMD

برای GPU مستقل:

```bash
cargo run --release --features amd-vulkan --bin amd-train -- \
  --target \
  --steps 20000 \
  --data ./train.txt \
  --checkpoint ./checkpoints/gemma-agent.bin \
  --best-checkpoint ./checkpoints/gemma-agent.best \
  --tokenizer ./checkpoints/target.tok \
  --batch-size 2 \
  --grad-accum 2 \
  --lr 0.0003 \
  --checkpoint-every 250 \
  --eval-every 250 \
  --gpu-kind discrete \
  --gpu 0
```

برای انتخاب خودکار بهترین device:

```bash
--gpu-kind best
```

مقادیر معتبر `--gpu-kind` عبارت‌اند از:

```text
integrated

discrete

best
```

---

## 9. برنامهٔ یادگیری

Trainer فعلی warmup و cosine decay دارد. مقدار warmup بر اساس تعداد کل updateها محاسبه می‌شود و نرخ یادگیری بعد از warmup به‌صورت cosine تا floor مشخص‌شده کاهش می‌یابد.

همچنین در آموزش target، context به‌صورت curriculum رشد می‌کند:

```text
ابتدا: 256
سپس:   512
در انتها: 1024
```

این کار اجازه می‌دهد مدل در مراحل ابتدایی با sequence کوتاه‌تر آموزش ببیند و در ادامه به context کامل برسد.

---

## 10. تنظیمات optimizer

optimizer فعلی AdamW با تنظیمات زیر ساخته می‌شود:

```text
beta1         = 0.9
beta2         = 0.95
epsilon       = 1e-8
weight_decay  = 0.1
grad clipping = global norm 1.0
```

نقطهٔ شروع پیشنهادی برای همین مدل:

```text
lr = 3e-4
```

اگر دادهٔ بسیار کوچک داری، بالا بردن بیش از حد learning rate می‌تواند overfit و نوسان را بیشتر کند. اگر loss خیلی آهسته کاهش یافت، ابتدا کیفیت و اندازهٔ corpus را بررسی کن و بعد learning rate را تغییر بده.

---

## 11. loss و validation

هدف آموزش next-token prediction است:

```text
X = token[0 .. n-1]
Y = token[1 .. n]
```

مدل logits برای هر موقعیت تولید می‌کند و Cross Entropy روی target بعدی محاسبه می‌شود.

برای validation، corpus به train و validation split تقسیم می‌شود و validation loss و perplexity گزارش می‌شوند:

```text
perplexity = exp(validation_loss)
```

معیار اصلی انتخاب checkpoint، کمترین validation loss است.

---

## 12. checkpoint کامل

checkpoint فقط وزن‌های مدل نیست.

برای فایل اصلی مثلاً:

```text
checkpoints/gemma-agent.bin
```

سه بخش ذخیره می‌شوند:

```text
gemma-agent.bin
 gemma-agent.bin.opt
 gemma-agent.bin.state
```

`*.opt` وضعیت AdamW را ذخیره می‌کند و `*.state` شامل update، بهترین validation loss و state تصادفی آموزش است.

این موضوع برای resume مهم است؛ صرفاً بارگذاری weights معادل ادامهٔ واقعی optimizer نیست.

---

## 13. ادامه دادن آموزش

مثلاً اگر checkpoint در update 20,000 ساخته شده و هدف جدید 50,000 است:

```bash
cargo run --release --features amd-vulkan --bin amd-train -- \
  --target \
  --steps 50000 \
  --data ./train.txt \
  --checkpoint ./checkpoints/gemma-agent.bin \
  --best-checkpoint ./checkpoints/gemma-agent.best \
  --tokenizer ./checkpoints/target.tok \
  --resume ./checkpoints/gemma-agent.bin \
  --batch-size 1 \
  --grad-accum 1 \
  --lr 0.0003 \
  --checkpoint-every 250 \
  --eval-every 250 \
  --gpu-kind integrated \
  --gpu 0
```

`--steps` تعداد کل updateهای هدف است؛ بنابراین از 20,000 تا 50,000، فقط updateهای باقی‌مانده اجرا می‌شوند.

---

## 14. benchmark قبل از run طولانی

قبل از آموزش طولانی، throughput را اندازه بگیر:

```bash
cargo run --release --features amd-vulkan --bin amd-bench -- \
  --batch-size 1 \
  --context 256 \
  --iterations 50 \
  --gpu-kind integrated \
  --gpu 0
```

برای GPU مستقل:

```bash
--gpu-kind discrete --gpu 0
```

این benchmark forward throughput را گزارش می‌کند. throughput آموزش واقعی به backward، optimizer، memory traffic و اندازهٔ context نیز وابسته است.

---

## 15. تشخیص رفتار loss

### حالت خوب

```text
loss:   8.2 -> 7.5 -> 6.9 -> 6.4 -> ...
val:    به‌تدریج کاهش
```

بهتر است روند چند صد update را ببینی، نه یک عدد منفرد.

### loss تقریباً ثابت

موارد محتمل:

1. corpus بسیار کوچک یا ضعیف است.
2. learning rate نامناسب است.
3. tokenizer اطلاعات مفید را از بین می‌برد.
4. تعداد update کم است.
5. مدل نسبت به corpus بیش از حد بزرگ یا نسبت به task بیش از حد کوچک است.

### train loss پایین ولی validation loss بالا

این معمولاً نشانهٔ overfitting است. دادهٔ متنوع‌تر، deduplication بهتر و regularization/weight decay مناسب‌تر می‌تواند کمک کند.

### NaN/Inf

ابتدا این موارد را بررسی کن:

```text
learning rate
هم‌گرایی optimizer
gradient clipping
دادهٔ خراب
NaN در ورودی
```

run را متوقف کن و از checkpoint خراب ادامه نده.

---

## 16. انتخاب تعداد update

فقط با «تعداد step» دربارهٔ کیفیت قضاوت نکن. یک update با `batch=1`, `context=256` بسیار کمتر از یک update با batch و context بزرگ‌تر token می‌بیند.

برای همین، مقدار مهم‌تر:

```text
training tokens seen
```

است.

در خروجی trainer نیز throughput بر اساس token گزارش می‌شود.

قاعدهٔ عملی:

```text
تست اولیه       : 100–500 update
آموزش آزمایشی   : 2k–5k update
آموزش جدی       : 20k+ update
```

این اعداد الزاماً optimal نیستند؛ corpus و سخت‌افزار تعیین‌کننده‌اند.

---

## 17. برای مدل واقعاً بهتر چه چیزی مهم‌تر است؟

اولویت‌ها:

```text
1. کیفیت corpus
2. تعداد tokenهای آموزشی
3. tokenizer مناسب
4. validation درست
5. معماری پایدار
6. learning-rate schedule
7. throughput سخت‌افزار
8. fine-tuning و دادهٔ instruction
```

بالا بردن فقط parameter count بدون corpus مناسب معمولاً کیفیت عمومی را به‌طور متناسب زیاد نمی‌کند.

---

## 18. pretraining در مقابل instruction tuning

`GemmaAgent` در این مرحله برای next-token pretraining طراحی شده است.

برای دستیار مکالمه‌ای بهتر، مسیر منطقی دو مرحله‌ای است:

```text
مرحله 1: pretraining روی corpus متنوع
                 ↓
مرحله 2: instruction tuning روی prompt/response تمیز
                 ↓
مرحله 3: ارزیابی و انتخاب checkpoint
```

دادهٔ instruction بهتر است ساختار ثابتی داشته باشد، مثلاً:

```text
<user>
توضیح بده borrow checker چیست.
</user>
<assistant>
Borrow checker در Rust ...
</assistant>
```

فرمت دقیق باید با tokenizer و dataset loader هماهنگ شود.

---

## 19. inference با KV-cache

بعد از آموزش، می‌توان از binary مربوط به inference استفاده کرد. نمونه:

```bash
cargo run --release --features amd-vulkan --bin amd-infer -- \
  --target \
  --checkpoint ./checkpoints/gemma-agent.bin \
  --tokenizer ./checkpoints/target.tok \
  --prompt "Rust is a" \
  --tokens 128 \
  --temperature 0.7 \
  --top-k 40 \
  --gpu-kind integrated \
  --gpu 0
```

KV-cache باعث می‌شود در decode افزایشی prefix دوباره از ابتدا محاسبه نشود.

---

## 20. پیشنهاد اجرای پایدار روی لپ‌تاپ

یک ترتیب امن برای توسعه:

```text
1. cargo check
2. cargo test
3. benchmark
4. smoke training با 100 update
5. 1k–5k update و بررسی validation
6. بررسی RAM/GPU و tok/s
7. افزایش grad-accum در صورت نیاز
8. run طولانی با checkpoint دوره‌ای
9. resume test از یک checkpoint
10. inference test روی best checkpoint
```

resume را حتماً یک‌بار عمداً آزمایش کن؛ checkpointی که فقط weights را ذخیره کند برای ادامهٔ deterministic آموزش کافی نیست.

---

## 21. دستور پیشنهادی نهایی

برای شروع یک run نسبتاً جدی روی AMD:

```bash
mkdir -p checkpoints

cargo run --release --features amd-vulkan --bin amd-train -- \
  --target \
  --steps 20000 \
  --data ./train.txt \
  --checkpoint ./checkpoints/gemma-agent.bin \
  --best-checkpoint ./checkpoints/gemma-agent.best \
  --tokenizer ./checkpoints/target.tok \
  --batch-size 1 \
  --grad-accum 4 \
  --lr 0.0003 \
  --checkpoint-every 250 \
  --eval-every 250 \
  --gpu-kind integrated \
  --gpu 0
```

برای RAM/GPU محدود، همین `batch-size=1` را نگه دار و در صورت جا داشتن، ابتدا `grad-accum` را زیاد کن.

---

## 22. چک‌لیست نهایی

قبل از آموزش:

```text
[ ] train.txt بزرگ و تمیز است
[ ] tokenizer ثابت و ذخیره‌شده است
[ ] Vulkan سالم است
[ ] cargo test موفق است
[ ] benchmark انجام شده
[ ] checkpoint directory ساخته شده
[ ] batch size با RAM/VRAM سازگار است
```

در حین آموزش:

```text
[ ] loss در چند صد update روند نزولی دارد
[ ] validation loss بدتر نمی‌شود
[ ] perplexity منطقی است
[ ] tok/s پایدار است
[ ] NaN/Inf وجود ندارد
[ ] checkpointها ایجاد می‌شوند
```

بعد از آموزش:

```text
[ ] best checkpoint مشخص است
[ ] resume آزمایش شده
[ ] inference کار می‌کند
[ ] خروجی روی promptهای جدید بررسی شده
```

---

## 23. ارتباط فایل با کد فعلی

فایل اجرایی AMD در `src/bin/amd-train.rs` پارامترهای آموزش مانند `--steps`, `--data`, `--checkpoint`, `--resume`, `--batch-size`, `--grad-accum`, `--lr`, `--checkpoint-every`, `--eval-every`, `--gpu-kind` و `--gpu` را می‌پذیرد.

منطق اصلی آموزش در `src/amd.rs` قرار دارد و شامل batching پنجره‌ای، tokenizer target، train/validation split، warmup/cosine LR، gradient accumulation، validation، best/latest checkpoint و state مربوط به resume است.

---

## 24. نکتهٔ مهم دربارهٔ کیفیت

این repository یک مدل کوچک آموزشی است. رسیدن به loss پایین‌تر به‌تنهایی به معنی «باهوش شدن» مدل نیست. برای کیفیت واقعی، corpus بزرگ و متنوع، دادهٔ تمیز، tokenهای آموزشی کافی و سپس instruction tuning مهم‌تر از صرفاً زیاد کردن step هستند.

همچنین benchmark forward با benchmark کامل training یکسان نیست؛ throughput واقعی را از logهای خود trainer ارزیابی کن.
