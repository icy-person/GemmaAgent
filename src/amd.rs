#![cfg(feature = "amd-vulkan")]

use burn::{
    backend::Autodiff,
    module::{AutodiffModule, Module},
    nn::{
        Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig,
        loss::CrossEntropyLossConfig,
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
    },
    optim::{AdamWConfig, GradientsParams, Optimizer},
    prelude::*,
    tensor::{Int, Tensor, TensorData},
};
use burn_wgpu::{graphics::Vulkan, init_setup, Wgpu, WgpuDevice};
use std::time::Instant;

use crate::{amd_tokenizer::AmdTokenizer, config::Config, tokenizer::Tokenizer};

pub type AmdBase = Wgpu<f32, i32>;
pub type AmdBackend = Autodiff<AmdBase>;

const EPSILON: f64 = 1e-5;
const VAL_FRACTION_NUM: usize = 9;
const VAL_FRACTION_DEN: usize = 10;

#[derive(Module, Debug)]
pub struct AmdModel<B: Backend> {
    pub token_embedding: Embedding<B>,
    pub position_embedding: Embedding<B>,
    pub transformer: TransformerEncoder<B>,
    pub final_norm: RmsNorm<B>,
    pub output: Linear<B>,
    pub vocab: usize,
    pub context: usize,
}

pub struct AmdModelConfig {
    cfg: Config,
}

impl AmdModelConfig {
    pub fn new(cfg: Config) -> Self {
        cfg.validate();
        Self { cfg }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> AmdModel<B> {
        let transformer = TransformerEncoderConfig::new(
            self.cfg.d_model,
            self.cfg.ffn,
            self.cfg.heads,
            self.cfg.layers,
        )
        .with_dropout(0.0)
        .with_norm_first(true)
        .with_layer_norm_eps(EPSILON)
        .init(device);

        AmdModel {
            token_embedding: EmbeddingConfig::new(self.cfg.vocab, self.cfg.d_model).init(device),
            position_embedding: EmbeddingConfig::new(self.cfg.context, self.cfg.d_model).init(device),
            transformer,
            final_norm: RmsNormConfig::new(self.cfg.d_model)
                .with_epsilon(EPSILON)
                .init(device),
            output: LinearConfig::new(self.cfg.d_model, self.cfg.vocab)
                .with_bias(false)
                .init(device),
            vocab: self.cfg.vocab,
            context: self.cfg.context,
        }
    }
}

impl<B: Backend> AmdModel<B> {
    pub fn forward_hidden(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch, seq] = tokens.dims();
        assert!(seq > 0 && seq <= self.context);
        let device = tokens.device();
        let positions = Tensor::<B, 1, Int>::arange(0..seq as i64, &device)
            .reshape([1, seq])
            .repeat_dim(0, batch);
        let x = self.token_embedding.forward(tokens) + self.position_embedding.forward(positions);
        let mask = burn::nn::attention::generate_autoregressive_mask(batch, seq, &device);
        let encoded = self
            .transformer
            .forward(TransformerEncoderInput::new(x).mask_attn(mask));
        self.final_norm.forward(encoded)
    }

    pub fn forward_logits(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.output.forward(self.forward_hidden(tokens))
    }
}

fn make_device(gpu_index: usize, gpu_kind: &str) -> WgpuDevice {
    let device = match gpu_kind {
        "integrated" => WgpuDevice::IntegratedGpu(gpu_index),
        "discrete" => WgpuDevice::DiscreteGpu(gpu_index),
        "best" => WgpuDevice::BestAvailable,
        other => panic!("invalid --gpu-kind '{other}', expected integrated, discrete, or best"),
    };
    init_setup::<Vulkan>(&device, Default::default());
    device
}

fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn random_start(state: &mut u64, window_count: usize) -> usize {
    (xorshift64(state) as usize) % window_count
}

fn make_batch(
    encoded: &[usize],
    context: usize,
    batch_size: usize,
    rng: &mut u64,
    device: &<AmdBackend as Backend>::Device,
) -> (Tensor<AmdBackend, 2, Int>, Tensor<AmdBackend, 2, Int>) {
    let window_count = encoded.len().saturating_sub(context);
    assert!(window_count > 0, "not enough tokens for the requested context");
    let mut xs = Vec::with_capacity(batch_size * context);
    let mut ys = Vec::with_capacity(batch_size * context);
    for _ in 0..batch_size {
        let start = random_start(rng, window_count);
        xs.extend(encoded[start..start + context].iter().map(|&v| v as i64));
        ys.extend(encoded[start + 1..start + context + 1].iter().map(|&v| v as i64));
    }
    (
        Tensor::from_data(TensorData::new(xs, [batch_size, context]), device),
        Tensor::from_data(TensorData::new(ys, [batch_size, context]), device),
    )
}

fn make_eval_batch(
    encoded: &[usize],
    context: usize,
    batch_size: usize,
    batch_index: usize,
    device: &<AmdBase as Backend>::Device,
) -> (Tensor<AmdBase, 2, Int>, Tensor<AmdBase, 2, Int>) {
    let window_count = encoded.len().saturating_sub(context);
    assert!(window_count > 0, "validation split is shorter than context");
    let mut xs = Vec::with_capacity(batch_size * context);
    let mut ys = Vec::with_capacity(batch_size * context);
    for b in 0..batch_size {
        let start = (batch_index * batch_size + b) % window_count;
        xs.extend(encoded[start..start + context].iter().map(|&v| v as i64));
        ys.extend(encoded[start + 1..start + context + 1].iter().map(|&v| v as i64));
    }
    (
        Tensor::from_data(TensorData::new(xs, [batch_size, context]), device),
        Tensor::from_data(TensorData::new(ys, [batch_size, context]), device),
    )
}

fn cosine_lr(base: f64, min_ratio: f64, step: usize, warmup: usize, total: usize) -> f64 {
    if step < warmup {
        return base * (step + 1) as f64 / warmup.max(1) as f64;
    }
    let decay_steps = total.saturating_sub(warmup).max(1);
    let progress = (step.saturating_sub(warmup) as f64 / decay_steps as f64).clamp(0.0, 1.0);
    let cosine = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
    base * (min_ratio + (1.0 - min_ratio) * cosine)
}

fn curriculum_context(full: usize, update: usize, total: usize) -> usize {
    if full <= 256 {
        return full;
    }
    let third = total.max(1) / 3;
    if update < third.max(1) {
        256.min(full)
    } else if update < (third * 2).max(2) {
        512.min(full)
    } else {
        full
    }
}

fn save_model(model: &AmdModel<AmdBackend>, path: &str) {
    let recorder = burn::record::BinFileRecorder::<burn::record::FullPrecisionSettings>::default();
    model
        .clone()
        .save_file(path, &recorder)
        .unwrap_or_else(|e| panic!("failed to save AMD checkpoint '{path}': {e}"));
}

fn evaluate(
    model: &AmdModel<AmdBase>,
    encoded: &[usize],
    context: usize,
    batch_size: usize,
    batches: usize,
    device: &<AmdBase as Backend>::Device,
) -> f64 {
    let loss_fn = CrossEntropyLossConfig::new().init(device);
    let mut total = 0.0f64;
    for batch in 0..batches {
        let (xs, ys) = make_eval_batch(encoded, context, batch_size, batch, device);
        let logits = model.forward_logits(xs);
        let flat_logits = logits.reshape([batch_size * context, model.vocab]);
        let flat_targets = ys.reshape([batch_size * context]);
        total += loss_fn.forward(flat_logits, flat_targets).into_scalar::<f32>() as f64;
    }
    total / batches.max(1) as f64
}

pub fn train(
    cfg: Config,
    steps: usize,
    checkpoint: &str,
    best_checkpoint: &str,
    data_path: &str,
    tokenizer_path: &str,
    resume: Option<&str>,
    batch_size: usize,
    grad_accum: usize,
    lr: f64,
    checkpoint_every: usize,
    eval_every: usize,
    gpu_index: usize,
    gpu_kind: &str,
) {
    cfg.validate();
    assert!(steps > 0 && batch_size > 0 && grad_accum > 0);
    assert!(lr.is_finite() && lr > 0.0);

    let device = make_device(gpu_index, gpu_kind);
    let model_cfg = AmdModelConfig::new(cfg);
    let mut model: AmdModel<AmdBackend> = model_cfg.init(&device);
    let recorder = burn::record::BinFileRecorder::<burn::record::FullPrecisionSettings>::default();
    if let Some(path) = resume {
        model = model
            .load_file(path, &recorder, &device)
            .unwrap_or_else(|e| panic!("failed to load resume checkpoint '{path}': {e}"));
        println!("resumed weights from {path} (optimizer state starts fresh)");
    }

    let tokenizer;
    let corpus = std::fs::read_to_string(data_path)
        .unwrap_or_else(|e| panic!("failed to read training data '{data_path}': {e}"));
    let encoded = if cfg.vocab == Config::target().vocab {
        tokenizer = if std::path::Path::new(tokenizer_path).exists() {
            AmdTokenizer::load(tokenizer_path)
                .unwrap_or_else(|e| panic!("failed to load AMD tokenizer '{tokenizer_path}': {e}"))
        } else {
            let tok = AmdTokenizer::train(&corpus, cfg.vocab);
            tok.save(tokenizer_path)
                .unwrap_or_else(|e| panic!("failed to save AMD tokenizer '{tokenizer_path}': {e}"));
            tok
        };
        assert_eq!(tokenizer.vocab_size(), cfg.vocab);
        tokenizer.encode(&corpus)
    } else {
        let byte_tokenizer = Tokenizer::new();
        byte_tokenizer.encode(&corpus)
    };
    assert!(encoded.len() > cfg.context + 2, "training corpus is shorter than context");

    let split = (encoded.len() * VAL_FRACTION_NUM / VAL_FRACTION_DEN)
        .max(cfg.context + 2)
        .min(encoded.len().saturating_sub(cfg.context + 2));
    let train_tokens = &encoded[..split];
    let val_tokens = &encoded[split.saturating_sub(cfg.context)..];
    assert!(train_tokens.len() > cfg.context + 1, "training split is shorter than context");
    assert!(val_tokens.len() > cfg.context + 1, "validation split is shorter than context");

    let warmup = (steps / 20).max(20).min(steps);
    println!("AMD Vulkan backend: Burn WGPU");
    println!("GPU kind: {gpu_kind} | index: {gpu_index}");
    println!(
        "model: vocab={} context={} d_model={} layers={} heads={} ffn={}",
        cfg.vocab, cfg.context, cfg.d_model, cfg.layers, cfg.heads, cfg.ffn
    );
    println!("tokens={} | train={} | validation={}", encoded.len(), train_tokens.len(), val_tokens.len());
    if cfg.vocab == Config::target().vocab {
        println!("tokenizer: corpus-trained subword pieces + byte fallback -> {tokenizer_path}");
    } else {
        println!("tokenizer: byte fallback debug tokenizer");
    }
    println!("batch-size={batch_size} grad-accum={grad_accum}");
    println!("base-lr={lr:.6} warmup={warmup} min-lr-ratio=0.1");
    println!("curriculum: 256 -> 512 -> {} tokens", cfg.context);
    println!("checkpoint={checkpoint} | best={best_checkpoint}");

    let mut optimizer = AdamWConfig::new()
        .with_beta_1(0.9)
        .with_beta_2(0.95)
        .with_epsilon(1e-8)
        .with_weight_decay(0.1)
        .init();
    let loss_fn = CrossEntropyLossConfig::new().init(&device);
    let mut rng_state = 0xD1B5_4A32_9F6C_71E3u64;
    let mut interval_loss = 0.0f64;
    let mut interval_tokens = 0usize;
    let mut last = Instant::now();
    let mut best_val = f64::INFINITY;

    for update in 0..steps {
        let current_context = curriculum_context(cfg.context, update, steps);
        let current_lr = cosine_lr(lr, 0.1, update, warmup, steps);
        let mut combined_loss: Option<Tensor<AmdBackend, 1>> = None;
        for _ in 0..grad_accum {
            let (xs, ys) = make_batch(train_tokens, current_context, batch_size, &mut rng_state, &device);
            let logits = model.forward_logits(xs);
            let flat_logits = logits.reshape([batch_size * current_context, cfg.vocab]);
            let flat_targets = ys.reshape([batch_size * current_context]);
            let loss = loss_fn.forward(flat_logits, flat_targets);
            let scaled = loss / grad_accum as f64;
            combined_loss = Some(match combined_loss {
                Some(previous) => previous + scaled,
                None => scaled,
            });
            interval_tokens += batch_size * current_context;
        }

        let loss = combined_loss.expect("at least one micro-batch is required");
        let loss_value = loss.clone().into_scalar::<f32>() as f64;
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        let grad_count = grads.len();
        assert!(
            grad_count > 0,
            "no gradients reached the optimizer; refusing to continue with a silent no-op update"
        );
        model = optimizer.step(current_lr, model, grads);

        interval_loss += loss_value;
        if update % 10 == 9 || update + 1 == steps {
            let elapsed = last.elapsed().as_secs_f64().max(1e-9);
            let avg_loss = interval_loss / 10.0f64.min((update + 1) as f64);
            println!(
                "amd update {:5} loss {:.5} lr {:.6} | {:.0} tok/s | ctx {}",
                update + 1,
                avg_loss,
                current_lr,
                interval_tokens as f64 / elapsed,
                current_context
            );
            interval_loss = 0.0;
            interval_tokens = 0;
            last = Instant::now();
        }

        if eval_every > 0 && ((update + 1) % eval_every == 0 || update + 1 == steps) {
            let eval_context = cfg.context;
            if val_tokens.len() > eval_context + 1 {
                let eval_model = model.valid();
                let val_loss = evaluate(
                    &eval_model,
                    val_tokens,
                    eval_context,
                    batch_size.min(2),
                    4,
                    &device,
                );
                let perplexity = val_loss.exp();
                println!("validation @ {:5}: loss {:.5} | ppl {:.3}", update + 1, val_loss, perplexity);
                if val_loss < best_val {
                    best_val = val_loss;
                    save_model(&model, best_checkpoint);
                    println!("best AMD checkpoint: {best_checkpoint} (val loss {best_val:.5})");
                }
            }
        }

        if checkpoint_every > 0 && (update + 1) % checkpoint_every == 0 {
            save_model(&model, checkpoint);
            println!("AMD checkpoint: {checkpoint} (update {})", update + 1);
        }
    }

    save_model(&model, checkpoint);
    println!("AMD final checkpoint: {checkpoint}");
    if best_val.is_finite() {
        println!("best validation loss: {best_val:.5} -> {best_checkpoint}");
    }
}

pub fn benchmark(
    cfg: Config,
    gpu_index: usize,
    gpu_kind: &str,
    batch_size: usize,
    context: usize,
    iterations: usize,
) {
    cfg.validate();
    assert!(batch_size > 0 && context > 0 && context <= cfg.context && iterations > 0);
    let device = make_device(gpu_index, gpu_kind);
    let model_cfg = AmdModelConfig::new(cfg);
    let model: AmdModel<AmdBase> = model_cfg.init(&device);
    let ids = Tensor::<AmdBase, 2, Int>::zeros([batch_size, context], &device);
    for _ in 0..3 {
        let _ = model.forward_logits(ids.clone());
    }
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = model.forward_logits(ids.clone());
    }
    let seconds = start.elapsed().as_secs_f64().max(1e-9);
    println!(
        "AMD Vulkan forward ({gpu_kind} GPU {gpu_index}): {:.0} tok/s",
        (batch_size * context * iterations) as f64 / seconds
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scheduler_warms_and_reaches_floor() {
        assert!(cosine_lr(1.0, 0.1, 0, 10, 100) < cosine_lr(1.0, 0.1, 9, 10, 100));
        assert!((cosine_lr(1.0, 0.1, 99, 10, 100) - 0.1).abs() < 1e-6);
    }

    #[test]
    fn curriculum_is_monotonic() {
        assert!(curriculum_context(1024, 0, 900) <= curriculum_context(1024, 400, 900));
        assert!(curriculum_context(1024, 400, 900) <= curriculum_context(1024, 899, 900));
    }
}
