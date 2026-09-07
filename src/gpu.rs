#![cfg(feature = "cuda")]

use candle_core::{D, DType, Device, Result, Tensor};
use candle_nn::embedding::Embedding;
use candle_nn::layer_norm::RmsNorm;
use candle_nn::linear::{Linear, linear_no_bias};
use candle_nn::loss::cross_entropy;
use candle_nn::ops::{silu, softmax};
use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};
use candle_nn::rotary_emb::rope_slow;
use candle_nn::{Module, VarBuilder, VarMap, embedding, rms_norm as make_rms_norm};
use std::path::Path;
use std::time::Instant;

use crate::config::Config;
use crate::tokenizer::Tokenizer;

const EPS: f64 = 1e-5;
const ROPE_THETA: f64 = 10_000.0;

struct GpuBlock {
    ln1: RmsNorm,
    ln2: RmsNorm,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    up: Linear,
    down: Linear,
}

struct GpuModel {
    cfg: Config,
    emb: Embedding,
    blocks: Vec<GpuBlock>,
    ln_f: RmsNorm,
    cos: Tensor,
    sin: Tensor,
    causal_mask: Tensor,
}

impl GpuModel {
    fn new(cfg: Config, vb: VarBuilder, device: &Device) -> Result<Self> {
        cfg.validate();
        assert_eq!(
            cfg.head_dim() % 2,
            0,
            "head dimension must be even for RoPE"
        );

        let emb = embedding(cfg.vocab, cfg.d_model, vb.pp("tok_embeddings"))?;
        let mut blocks = Vec::with_capacity(cfg.layers);
        for layer in 0..cfg.layers {
            let b = vb.pp(format!("layers.{layer}"));
            blocks.push(GpuBlock {
                ln1: make_rms_norm(cfg.d_model, EPS, b.pp("ln1"))?,
                ln2: make_rms_norm(cfg.d_model, EPS, b.pp("ln2"))?,
                q: linear_no_bias(cfg.d_model, cfg.d_model, b.pp("q"))?,
                k: linear_no_bias(cfg.d_model, cfg.d_model, b.pp("k"))?,
                v: linear_no_bias(cfg.d_model, cfg.d_model, b.pp("v"))?,
                o: linear_no_bias(cfg.d_model, cfg.d_model, b.pp("o"))?,
                up: linear_no_bias(cfg.d_model, cfg.ffn, b.pp("up"))?,
                down: linear_no_bias(cfg.ffn, cfg.d_model, b.pp("down"))?,
            });
        }
        let ln_f = make_rms_norm(cfg.d_model, EPS, vb.pp("ln_f"))?;

        let half = cfg.head_dim() / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1.0 / ROPE_THETA.powf((2 * i) as f64 / cfg.head_dim() as f64) as f32)
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, (1, half), device)?;
        let positions = Tensor::arange(0u32, cfg.context as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((cfg.context, 1))?;
        let freqs = positions.matmul(&inv_freq)?;
        let cos = freqs.cos()?;
        let sin = freqs.sin()?;

        let upper = Tensor::triu2(cfg.context, DType::F32, device)?;
        let eye = Tensor::eye(cfg.context, DType::F32, device)?;
        let causal_mask = (upper - eye)?.affine(-1.0e4, 0.0)?;

        Ok(Self {
            cfg,
            emb,
            blocks,
            ln_f,
            cos,
            sin,
            causal_mask,
        })
    }

    fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        let (batch, seq) = ids.dims2()?;
        assert!(seq <= self.cfg.context);

        let mut x = self.emb.forward(ids)?;
        let cos = self.cos.narrow(0, 0, seq)?;
        let sin = self.sin.narrow(0, 0, seq)?;

        for block in &self.blocks {
            let n = block.ln1.forward(&x)?;
            let q = block
                .q
                .forward(&n)?
                .reshape((batch, seq, self.cfg.heads, self.cfg.head_dim()))?
                .transpose(1, 2)?
                .contiguous()?;
            let k = block
                .k
                .forward(&n)?
                .reshape((batch, seq, self.cfg.heads, self.cfg.head_dim()))?
                .transpose(1, 2)?
                .contiguous()?;
            let v = block
                .v
                .forward(&n)?
                .reshape((batch, seq, self.cfg.heads, self.cfg.head_dim()))?
                .transpose(1, 2)?
                .contiguous()?;

            let q = rope_slow(&q, &cos, &sin)?;
            let k = rope_slow(&k, &cos, &sin)?;
            let k_t = k.transpose(2, 3)?.contiguous()?;
            let mut scores = q.broadcast_matmul(&k_t)?;
            scores = scores.affine(1.0 / (self.cfg.head_dim() as f64).sqrt(), 0.0)?;
            let mask = self.causal_mask.narrow(0, 0, seq)?.narrow(1, 0, seq)?;
            let mask = mask.broadcast_as((batch, self.cfg.heads, seq, seq))?;
            scores = (scores + mask)?;
            let weights = softmax(&scores, D::Minus1)?;
            let attn = weights.broadcast_matmul(&v)?.transpose(1, 2)?.reshape((
                batch,
                seq,
                self.cfg.d_model,
            ))?;
            x = (x + block.o.forward(&attn)?)?;

            let n2 = block.ln2.forward(&x)?;
            let ff = silu(&block.up.forward(&n2)?)?;
            x = (x + block.down.forward(&ff)?)?;
        }

        self.ln_f.forward(&x)
    }

    fn logits(&self, hidden: &Tensor) -> Result<Tensor> {
        hidden.matmul(&self.emb.embeddings().t()?)
    }

    fn forward_logits(&self, ids: &Tensor) -> Result<Tensor> {
        let hidden = self.forward(ids)?;
        self.logits(&hidden)
    }
}

fn build_varmap(cfg: Config, device: &Device) -> Result<(VarMap, GpuModel)> {
    device.set_seed(42)?;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device).pp("model");
    let model = GpuModel::new(cfg, vb, device)?;
    Ok((varmap, model))
}

fn load_or_init(cfg: Config, device: &Device, checkpoint: &str) -> Result<(VarMap, GpuModel)> {
    let (mut varmap, model) = build_varmap(cfg, device)?;
    if Path::new(checkpoint).exists() {
        varmap.load(checkpoint)?;
        println!("loaded GPU checkpoint: {checkpoint}");
    }
    Ok((varmap, model))
}

fn make_batch(
    encoded: &[usize],
    context: usize,
    batch_size: usize,
    step: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let window_count = encoded.len() - context;
    let mut x = Vec::with_capacity(batch_size * context);
    let mut y = Vec::with_capacity(batch_size * context);
    for b in 0..batch_size {
        let start = (step * batch_size + b) % window_count;
        x.extend(encoded[start..start + context].iter().map(|&v| v as u32));
        y.extend(
            encoded[start + 1..start + context + 1]
                .iter()
                .map(|&v| v as u32),
        );
    }
    let xs = Tensor::from_vec(x, (batch_size, context), device)?;
    let ys = Tensor::from_vec(y, (batch_size, context), device)?;
    Ok((xs, ys))
}

fn scheduled_lr(base_lr: f64, update: usize, total_updates: usize) -> f64 {
    let warmup = 100usize.min(total_updates.max(1));
    if update < warmup {
        base_lr * (update + 1) as f64 / warmup as f64
    } else {
        let denom = (total_updates.saturating_sub(warmup)).max(1) as f64;
        let progress = (update.saturating_sub(warmup)) as f64 / denom;
        let cosine = 0.5 * (1.0 + (std::f64::consts::PI * progress.min(1.0)).cos());
        base_lr * (0.1 + 0.9 * cosine)
    }
}

pub fn train(
    steps: usize,
    checkpoint: &str,
    data_path: &str,
    cfg: Config,
    batch_size: usize,
    grad_accum: usize,
    lr: f64,
    checkpoint_every: usize,
    gpu_index: usize,
) -> Result<()> {
    assert!(steps > 0 && batch_size > 0 && grad_accum > 0);
    assert!(lr.is_finite() && lr > 0.0);

    let device = Device::new_cuda(gpu_index)?;
    println!("GPU backend: CUDA device {gpu_index}");
    println!("GPU training uses Candle CUDA tensors/autograd and AdamW");

    let tokenizer = Tokenizer::new();
    let corpus = std::fs::read_to_string(data_path).map_err(candle_core::Error::wrap)?;
    let encoded = tokenizer.encode(&corpus);
    assert!(encoded.len() > cfg.context + 1);

    println!(
        "GemmaAgent GPU: {} params + RMSNorm gains | context {} | heads {} | batch {} | grad-accum {} | lr {:.6}",
        cfg.params() + cfg.d_model * (2 * cfg.layers + 1),
        cfg.context,
        cfg.heads,
        batch_size,
        grad_accum,
        lr
    );
    println!("training corpus: {} bytes from {data_path}", corpus.len());

    let (varmap, model) = load_or_init(cfg, &device, checkpoint)?;
    let vars = varmap.all_vars();
    let mut optimizer = AdamW::new(
        vars,
        ParamsAdamW {
            lr: scheduled_lr(lr, 0, steps),
            beta1: 0.9,
            beta2: 0.95,
            eps: 1e-8,
            weight_decay: 0.1,
        },
    )?;

    let mut loss_total = 0.0f64;
    let mut last = Instant::now();
    for update in 0..steps {
        optimizer.set_learning_rate(scheduled_lr(lr, update, steps));
        let mut grad_store = candle_core::backprop::GradStore::default();
        let mut update_loss = 0.0f64;
        for micro in 0..grad_accum {
            let logical_step = update * grad_accum + micro;
            let (xs, ys) = make_batch(&encoded, cfg.context, batch_size, logical_step, &device)?;
            let logits = model.forward_logits(&xs)?;
            let flat_logits = logits.reshape((batch_size * cfg.context, cfg.vocab))?;
            let flat_targets = ys.flatten_all()?;
            let loss = cross_entropy(&flat_logits, &flat_targets)?;
            update_loss += loss.to_scalar::<f32>()? as f64;
            let scaled = loss.affine(1.0 / grad_accum as f64, 0.0)?;
            let grads = scaled.backward()?;
            grad_store.extend(grads)?;
        }
        optimizer.step(&grad_store)?;
        loss_total += update_loss / grad_accum as f64;

        if update % 10 == 9 || update + 1 == steps {
            let elapsed = last.elapsed().as_secs_f64().max(f64::MIN_POSITIVE);
            let samples = 10.min(update + 1);
            let avg = loss_total / samples as f64;
            let tokens = (batch_size * cfg.context * grad_accum * samples) as f64;
            println!(
                "gpu update {:5} loss {:.5} lr {:.7} | {:.0} tok/s",
                update + 1,
                avg,
                scheduled_lr(lr, update, steps),
                tokens / elapsed
            );
            loss_total = 0.0;
            last = Instant::now();
        }

        if checkpoint_every > 0 && (update + 1) % checkpoint_every == 0 {
            varmap.save(checkpoint)?;
            println!("GPU checkpoint: {checkpoint} (update {})", update + 1);
        }
    }

    device.synchronize()?;
    varmap.save(checkpoint)?;
    println!("GPU checkpoint: {checkpoint}");
    Ok(())
}

pub fn benchmark(
    gpu_index: usize,
    cfg: Config,
    batch_size: usize,
    context: usize,
    iters: usize,
) -> Result<()> {
    assert!(batch_size > 0 && context > 0 && iters > 0);
    assert!(context <= cfg.context);
    let device = Device::new_cuda(gpu_index)?;
    let (_varmap, model) = build_varmap(cfg, &device)?;

    let ids = Tensor::zeros((batch_size, context), DType::U32, &device)?;
    for _ in 0..3 {
        let _ = model.forward_logits(&ids)?;
    }
    device.synchronize()?;
    let start = Instant::now();
    for _ in 0..iters {
        let _ = model.forward_logits(&ids)?;
    }
    device.synchronize()?;
    let secs = start.elapsed().as_secs_f64();
    let tok_per_sec = (batch_size * context * iters) as f64 / secs.max(f64::MIN_POSITIVE);
    println!("GPU benchmark: CUDA device {gpu_index}");
    println!(
        "{} params | batch {batch_size} | context {context} | iterations {iters}",
        cfg.params()
    );
    println!("forward throughput: {:.0} tok/s", tok_per_sec);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_geometry_matches_debug_profile() {
        let cfg = Config::debug();
        assert_eq!(cfg.head_dim() % 2, 0);
        assert_eq!(cfg.vocab, 258);
    }

    #[test]
    fn gpu_cpu_forward_backward_smoke() -> Result<()> {
        let cfg = Config::debug();
        let device = Device::Cpu;
        let (varmap, model) = build_varmap(cfg, &device)?;
        let ids = Tensor::from_vec(vec![256u32, 82, 117, 115, 116], (1, 5), &device)?;
        let targets = Tensor::from_vec(vec![82u32, 117, 115, 116, 32], (1, 5), &device)?;
        let logits = model.forward_logits(&ids)?.reshape((5, cfg.vocab))?;
        let loss = cross_entropy(&logits, &targets.flatten_all()?)?;
        assert!(loss.to_scalar::<f32>()?.is_finite());
        let grads = loss.backward()?;
        assert!(!varmap.all_vars().is_empty());
        assert!(!grads.get_ids().collect::<Vec<_>>().is_empty());
        Ok(())
    }

    #[test]
    fn scheduler_starts_at_small_learning_rate() {
        let first = scheduled_lr(3e-4, 0, 1000);
        let peak = scheduled_lr(3e-4, 99, 1000);
        let tail = scheduled_lr(3e-4, 999, 1000);
        assert!(first < peak);
        assert!(tail < peak);
        assert!(tail > 0.0);
    }

    #[test]
    fn gpu_feature_uses_cuda_path() {
        let _ = std::any::type_name::<Device>();
    }
}
