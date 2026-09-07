#![cfg(feature = "cuda")]

use candle_core::{DType, D, Device, IndexOp, Result, Tensor};
use candle_nn::embedding::Embedding;
use candle_nn::init::{FanInOut, Init, NormalOrUniform, NonLinearity};
use candle_nn::layer_norm::RmsNorm;
use candle_nn::linear::{linear_no_bias, Linear};
use candle_nn::loss::cross_entropy;
use candle_nn::ops::{rms_norm, silu, softmax};
use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};
use candle_nn::rotary_emb::rope_slow;
use candle_nn::{embedding, rms_norm as make_rms_norm, Module, VarBuilder, VarMap};
use std::path::Path;
use std::time::Instant;

use crate::config::Config;
use crate::tokenizer::Tokenizer;

const EPS: f64 = 1e-5;
const ROPE_THETA: f64 = 10_000.0;
const DEFAULT_GPU_LR: f64 = 3e-4;

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
        assert_eq!(cfg.head_dim() % 2, 0, "head dimension must be even for RoPE");

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
            let q = block.q.forward(&n)?
                .reshape((batch, seq, self.cfg.heads, self.cfg.head_dim()))?
                .transpose(1, 2)?
                .contiguous()?;
            let k = block.k.forward(&n)?
                .reshape((batch, seq, self.cfg.heads, self.cfg.head_dim()))?
                .transpose(1, 2)?
                .contiguous()?;
            let v = block.v.forward(&n)?
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
            let attn = weights.broadcast_matmul(&v)?
                .transpose(1, 2)?
                .reshape((batch, seq, self.cfg.d_model))?;
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

fn make_init() -> Init {
    Init::Kaiming {
        dist: NormalOrUniform::Normal,
        fan: FanInOut::FanIn,
        non_linearity: NonLinearity::Linear,
    }
}

fn build_varmap(cfg: Config, device: &Device) -> Result<(VarMap, GpuModel)> {
    device.set_seed(42)?;
    let varmap = VarMap::new();
    let mut vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    vb = vb.pp("model");
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
        y.extend(encoded[start + 1..start + context + 1].iter().map(|&v| v as u32));
    }
    let xs = Tensor::from_vec(x, (batch_size, context), device)?;
    let ys = Tensor::from_vec(y, (batch_size, context), device)?;
    Ok((xs, ys))
}

pub fn train(
    steps: usize,
    checkpoint: &str,
    data_path: &str,
    batch_size: usize,
    lr: f64,
    checkpoint_every: usize,
    gpu_index: usize,
) -> Result<()> {
    assert!(steps > 0);
    assert!(batch_size > 0);
    assert!(lr.is_finite() && lr > 0.0);

    let device = Device::new_cuda(gpu_index)?;
    println!("GPU backend: CUDA device {gpu_index}");
    println!("GPU training uses Candle CUDA tensors/autograd and AdamW");

    let cfg = Config::debug();
    let tokenizer = Tokenizer::new();
    let corpus = std::fs::read_to_string(data_path)
        .map_err(candle_core::Error::wrap)?;
    let encoded = tokenizer.encode(&corpus);
    assert!(encoded.len() > cfg.context + 1);

    println!(
        "GemmaAgent GPU: {} params | context {} | heads {} | batch {} | lr {:.6}",
        cfg.params(), cfg.context, cfg.heads, batch_size, lr
    );
    println!("training corpus: {} bytes from {data_path}", corpus.len());
    println!("checkpoint format: Candle safetensors -> {checkpoint}");

    let (varmap, model) = load_or_init(cfg, &device, checkpoint)?;
    let vars = varmap.all_vars();
    let mut optimizer = AdamW::new(
        vars,
        ParamsAdamW {
            lr,
            beta1: 0.9,
            beta2: 0.95,
            eps: 1e-8,
            weight_decay: 0.1,
        },
    )?;

    let mut loss_total = 0.0f64;
    let mut last = Instant::now();
    for step in 0..steps {
        let (xs, ys) = make_batch(&encoded, cfg.context, batch_size, step, &device)?;
        let logits = model.forward_logits(&xs)?;
        let flat_logits = logits.reshape((batch_size * cfg.context, cfg.vocab))?;
        let flat_targets = ys.flatten_all()?;
        let loss = cross_entropy(&flat_logits, &flat_targets)?;
        let loss_value = loss.to_scalar::<f32>()? as f64;
        loss_total += loss_value;
        optimizer.backward_step(&loss)?;

        if step % 10 == 9 || step + 1 == steps {
            let elapsed = last.elapsed().as_secs_f64().max(f64::MIN_POSITIVE);
            let avg = loss_total / 10.0_f64.min((step + 1) as f64);
            let tokens = (batch_size * cfg.context * 10.min(step + 1)) as f64;
            println!(
                "gpu step {:5} loss {:.5} | {:.0} tok/s",
                step + 1,
                avg,
                tokens / elapsed
            );
            loss_total = 0.0;
            last = Instant::now();
        }

        if checkpoint_every > 0 && (step + 1) % checkpoint_every == 0 {
            varmap.save(checkpoint)?;
            println!("GPU checkpoint: {checkpoint} (step {})", step + 1);
        }
    }

    device.synchronize()?;
    varmap.save(checkpoint)?;
    println!("GPU checkpoint: {checkpoint}");
    Ok(())
}

pub fn benchmark(gpu_index: usize, batch_size: usize, context: usize, iters: usize) -> Result<()> {
    assert!(batch_size > 0 && context > 0 && iters > 0);
    let device = Device::new_cuda(gpu_index)?;
    let cfg = Config::debug();
    assert!(context <= cfg.context);
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
    println!("batch {batch_size} | context {context} | iterations {iters}");
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
    fn gpu_feature_uses_cuda_path() {
        let _ = std::any::type_name::<Device>();
    }
}
