#![cfg(feature = "amd-vulkan")]

use burn::{
    backend::Autodiff,
    grad_clipping::GradientClippingConfig,
    module::{AutodiffModule, Module},
    nn::{
        Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig, RotaryEncoding,
        RotaryEncodingConfig, SwiGlu, SwiGluConfig, loss::CrossEntropyLossConfig,
    },
    optim::{
        AdamW, AdamWConfig, GradientsAccumulator, GradientsParams, Optimizer,
        adaptor::OptimizerAdaptor,
    },
    prelude::*,
    record::{BinFileRecorder, FullPrecisionSettings, Recorder},
    tensor::{Device, Int, Tensor, TensorData, backend::ops::AttentionModuleOptions},
};
use burn_wgpu::{Wgpu, WgpuDevice, graphics::Vulkan, init_setup};
use std::{fs, path::Path, time::Instant};

use crate::{amd_tokenizer::AmdTokenizer, config::Config, tokenizer::Tokenizer};

pub type AmdBase = Wgpu<f32, i32>;
pub type AmdBackend = Autodiff<AmdBase>;
pub type AmdOptimizer = OptimizerAdaptor<AdamW, AmdModel<AmdBackend>, AmdBackend>;

const EPSILON: f64 = 1e-5;
const ROPE_THETA: f32 = 10_000.0;
const VAL_FRACTION_NUM: usize = 9;
const VAL_FRACTION_DEN: usize = 10;
const META_HEADER: &str = "AMDSTATE3";

#[derive(Clone, Debug, Default)]
pub struct AmdLayerCache<B: Backend> {
    pub key: Option<Tensor<B, 4>>,
    pub value: Option<Tensor<B, 4>>,
}

#[derive(Clone, Debug)]
pub struct AmdCache<B: Backend> {
    pub layers: Vec<AmdLayerCache<B>>,
    pub position: usize,
}

impl<B: Backend> AmdCache<B> {
    pub fn new(layers: usize) -> Self {
        Self {
            layers: (0..layers).map(|_| AmdLayerCache::default()).collect(),
            position: 0,
        }
    }

    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.key = None;
            layer.value = None;
        }
        self.position = 0;
    }
}

#[derive(Module, Debug)]
pub struct AmdBlock<B: Backend> {
    pub norm_attn: RmsNorm<B>,
    pub qkv: Linear<B>,
    pub attn_out: Linear<B>,
    pub norm_ffn: RmsNorm<B>,
    pub swiglu: SwiGlu<B>,
    pub down: Linear<B>,
    pub rope: RotaryEncoding<B>,
    pub heads: usize,
    pub d_model: usize,
    pub head_dim: usize,
}

impl<B: Backend> AmdBlock<B> {
    fn new(device: &B::Device, d_model: usize, ffn: usize, heads: usize, context: usize) -> Self {
        assert_eq!(
            d_model % heads,
            0,
            "d_model must divide evenly across heads"
        );
        let head_dim = d_model / heads;
        assert_eq!(head_dim % 2, 0, "RoPE requires an even head dimension");
        Self {
            norm_attn: RmsNormConfig::new(d_model)
                .with_epsilon(EPSILON)
                .init(device),
            qkv: LinearConfig::new(d_model, d_model * 3)
                .with_bias(false)
                .init(device),
            attn_out: LinearConfig::new(d_model, d_model)
                .with_bias(false)
                .init(device),
            norm_ffn: RmsNormConfig::new(d_model)
                .with_epsilon(EPSILON)
                .init(device),
            swiglu: SwiGluConfig::new(d_model, ffn)
                .with_bias(false)
                .init(device),
            down: LinearConfig::new(ffn, d_model)
                .with_bias(false)
                .init(device),
            rope: RotaryEncodingConfig::new(context, head_dim)
                .with_theta(ROPE_THETA)
                .init(device),
            heads,
            d_model,
            head_dim,
        }
    }

    fn split_qkv(&self, x: Tensor<B, 3>) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        let [batch, seq, _] = x.dims();
        let qkv = self
            .qkv
            .forward(x)
            .reshape([batch, seq, 3, self.heads, self.head_dim]);
        let q = qkv
            .clone()
            .slice([0..batch, 0..seq, 0..1, 0..self.heads, 0..self.head_dim])
            .reshape([batch, self.heads, seq, self.head_dim]);
        let k = qkv
            .clone()
            .slice([0..batch, 0..seq, 1..2, 0..self.heads, 0..self.head_dim])
            .reshape([batch, self.heads, seq, self.head_dim]);
        let v = qkv
            .slice([0..batch, 0..seq, 2..3, 0..self.heads, 0..self.head_dim])
            .reshape([batch, self.heads, seq, self.head_dim]);
        (q, k, v)
    }

    fn attention(
        &self,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
        causal: bool,
    ) -> Tensor<B, 4> {
        burn::tensor::module::attention(
            q,
            k,
            v,
            None,
            None,
            AttentionModuleOptions {
                scale: None,
                softcap: None,
                is_causal: causal,
            },
        )
    }

    pub fn forward_prefill(&self, x: Tensor<B, 3>, cache: &mut AmdLayerCache<B>) -> Tensor<B, 3> {
        let residual = x.clone();
        let (q, k, v) = self.split_qkv(self.norm_attn.forward(x));
        let q = self.rope.apply(q, 0);
        let k = self.rope.apply(k, 0);
        let y = self.attention(q, k.clone(), v.clone(), true);
        cache.key = Some(k);
        cache.value = Some(v);
        let [batch, heads, seq, head_dim] = y.dims();
        let attn = self
            .attn_out
            .forward(y.reshape([batch, seq, heads * head_dim]));
        let x = residual + attn;
        let ffn = self.norm_ffn.forward(x.clone());
        x + self.down.forward(self.swiglu.forward(ffn))
    }

    pub fn forward_step(
        &self,
        x: Tensor<B, 3>,
        cache: &mut AmdLayerCache<B>,
        position: usize,
    ) -> Tensor<B, 3> {
        let residual = x.clone();
        let (q, k, v) = self.split_qkv(self.norm_attn.forward(x));
        let q = self.rope.apply(q, position);
        let k = self.rope.apply(k, position);
        let (keys, values) = match (cache.key.take(), cache.value.take()) {
            (Some(old_k), Some(old_v)) => (
                Tensor::cat(vec![old_k, k.clone()], 2),
                Tensor::cat(vec![old_v, v.clone()], 2),
            ),
            _ => (k.clone(), v.clone()),
        };
        let y = self.attention(q, keys.clone(), values.clone(), false);
        cache.key = Some(keys);
        cache.value = Some(values);
        let [batch, heads, seq, head_dim] = y.dims();
        let attn = self
            .attn_out
            .forward(y.reshape([batch, seq, heads * head_dim]));
        let x = residual + attn;
        let ffn = self.norm_ffn.forward(x.clone());
        x + self.down.forward(self.swiglu.forward(ffn))
    }
}

#[derive(Module, Debug)]
pub struct AmdModel<B: Backend> {
    pub token_embedding: Embedding<B>,
    pub blocks: Vec<AmdBlock<B>>,
    pub final_norm: RmsNorm<B>,
    pub output: Linear<B>,
    pub vocab: usize,
    pub context: usize,
    pub heads: usize,
    pub d_model: usize,
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
        let blocks = (0..self.cfg.layers)
            .map(|_| {
                AmdBlock::new(
                    device,
                    self.cfg.d_model,
                    self.cfg.ffn,
                    self.cfg.heads,
                    self.cfg.context,
                )
            })
            .collect();
        AmdModel {
            token_embedding: EmbeddingConfig::new(self.cfg.vocab, self.cfg.d_model).init(device),
            blocks,
            final_norm: RmsNormConfig::new(self.cfg.d_model)
                .with_epsilon(EPSILON)
                .init(device),
            output: LinearConfig::new(self.cfg.d_model, self.cfg.vocab)
                .with_bias(false)
                .init(device),
            vocab: self.cfg.vocab,
            context: self.cfg.context,
            heads: self.cfg.heads,
            d_model: self.cfg.d_model,
        }
    }
}

impl<B: Backend> AmdModel<B> {
    pub fn num_layers(&self) -> usize {
        self.blocks.len()
    }

    pub fn forward_hidden(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [_, seq] = tokens.dims();
        assert!(seq > 0 && seq <= self.context);
        let mut x = self.token_embedding.forward(tokens);
        for block in &self.blocks {
            x = block.forward_prefill(x, &mut AmdLayerCache::default());
        }
        self.final_norm.forward(x)
    }

    pub fn forward_logits(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.output.forward(self.forward_hidden(tokens))
    }

    pub fn prefill(&self, tokens: Tensor<B, 2, Int>, cache: &mut AmdCache<B>) -> Tensor<B, 3> {
        let [batch, seq] = tokens.dims();
        assert_eq!(batch, 1, "KV-cache prefill supports batch=1");
        assert!(seq > 0 && seq <= self.context);
        cache.clear();
        let mut x = self.token_embedding.forward(tokens);
        for (idx, block) in self.blocks.iter().enumerate() {
            x = block.forward_prefill(x, &mut cache.layers[idx]);
        }
        cache.position = seq;
        self.final_norm.forward(x)
    }

    pub fn step(&self, token: Tensor<B, 2, Int>, cache: &mut AmdCache<B>) -> Tensor<B, 3> {
        let [batch, seq] = token.dims();
        assert_eq!(batch, 1);
        assert_eq!(seq, 1);
        assert!(cache.position < self.context);
        let position = cache.position;
        let mut x = self.token_embedding.forward(token);
        for (idx, block) in self.blocks.iter().enumerate() {
            x = block.forward_step(x, &mut cache.layers[idx], position);
        }
        cache.position += 1;
        self.final_norm.forward(x)
    }
}

#[allow(deprecated)]
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
    device: &Device<AmdBase>,
) -> (Tensor<AmdBackend, 2, Int>, Tensor<AmdBackend, 2, Int>) {
    let window_count = encoded.len().saturating_sub(context);
    assert!(window_count > 0);
    let mut xs = Vec::with_capacity(batch_size * context);
    let mut ys = Vec::with_capacity(batch_size * context);
    for _ in 0..batch_size {
        let start = random_start(rng, window_count);
        xs.extend(encoded[start..start + context].iter().map(|&v| v as i64));
        ys.extend(
            encoded[start + 1..start + context + 1]
                .iter()
                .map(|&v| v as i64),
        );
    }
    (
        Tensor::from_data(TensorData::new(xs, [batch_size, context]), device),
        Tensor::from_data(TensorData::new(ys, [batch_size, context]), device),
    )
}

fn make_eval_batch(
    encoded: &[usize],
    context: usize,
    batch_index: usize,
    device: &Device<AmdBase>,
) -> (Tensor<AmdBase, 2, Int>, Tensor<AmdBase, 2, Int>) {
    let window_count = encoded.len().saturating_sub(context);
    assert!(window_count > 0);
    let start = (batch_index * context) % window_count;
    let xs = encoded[start..start + context]
        .iter()
        .map(|&v| v as i64)
        .collect::<Vec<_>>();
    let ys = encoded[start + 1..start + context + 1]
        .iter()
        .map(|&v| v as i64)
        .collect::<Vec<_>>();
    (
        Tensor::from_data(TensorData::new(xs, [1, context]), device),
        Tensor::from_data(TensorData::new(ys, [1, context]), device),
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
fn meta_path(path: &str) -> String {
    format!("{path}.state")
}
fn optimizer_path(path: &str) -> String {
    format!("{path}.opt")
}
fn save_state(path: &str, update: usize, best_val: f64, rng_state: u64) {
    let body = format!(
        "{META_HEADER}\nupdate={update}\nbest_val_bits={}\nrng_state={rng_state}\n",
        best_val.to_bits()
    );
    fs::write(meta_path(path), body)
        .unwrap_or_else(|e| panic!("failed to save training state: {e}"));
}
fn load_state(path: &str) -> Option<(usize, f64, u64)> {
    let text = fs::read_to_string(meta_path(path)).ok()?;
    let mut lines = text.lines();
    if lines.next() != Some(META_HEADER) {
        return None;
    }
    let mut update = None;
    let mut best = None;
    let mut rng = None;
    for line in lines {
        let (key, value) = line.split_once('=')?;
        match key {
            "update" => update = value.parse().ok(),
            "best_val_bits" => best = value.parse::<u64>().ok().map(f64::from_bits),
            "rng_state" => rng = value.parse().ok(),
            _ => {}
        }
    }
    Some((update?, best?, rng?))
}
fn make_optimizer() -> AmdOptimizer {
    AdamWConfig::new()
        .with_beta_1(0.9)
        .with_beta_2(0.95)
        .with_epsilon(1e-8)
        .with_weight_decay(0.1)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init()
}

fn save_checkpoint(
    model: &AmdModel<AmdBackend>,
    optimizer: &AmdOptimizer,
    checkpoint: &str,
    update: usize,
    best_val: f64,
    rng_state: u64,
) {
    let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
    model
        .clone()
        .save_file(checkpoint, &recorder)
        .unwrap_or_else(|e| panic!("failed to save model checkpoint: {e}"));
    BinFileRecorder::<FullPrecisionSettings>::default()
        .record(optimizer.to_record(), optimizer_path(checkpoint).into())
        .unwrap_or_else(|e| panic!("failed to save optimizer checkpoint: {e}"));
    save_state(checkpoint, update, best_val, rng_state);
}

fn load_optimizer(
    optimizer: AmdOptimizer,
    checkpoint: &str,
    device: &WgpuDevice,
) -> AmdOptimizer {
    let path = optimizer_path(checkpoint);

    if !Path::new(&path).exists() {
        return optimizer;
    }

    let recorder = BinFileRecorder::<FullPrecisionSettings>::default();

    let record = recorder
        .load(path.into(), device)
        .unwrap_or_else(|e| panic!("failed to load optimizer checkpoint: {e}"));

    optimizer.load_record(record)
}

fn evaluate(
    model: &AmdModel<AmdBase>,
    encoded: &[usize],
    context: usize,
    batches: usize,
    device: &Device<AmdBase>,
) -> f64 {
    let loss_fn = CrossEntropyLossConfig::new().init(device);
    let mut total = 0.0;
    let batches = batches.max(1);
    for batch in 0..batches {
        let (xs, ys) = make_eval_batch(encoded, context, batch, device);
        let logits = model.forward_logits(xs);
        total += loss_fn
            .forward(
                logits.reshape([context, model.vocab]),
                ys.reshape([context]),
            )
            .into_scalar() as f64;
    }
    total / batches as f64
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
    assert!(steps > 0 && batch_size > 0 && grad_accum > 0 && lr.is_finite() && lr > 0.0);
    let device = make_device(gpu_index, gpu_kind);
    let model_cfg = AmdModelConfig::new(cfg);
    let mut model: AmdModel<AmdBackend> = model_cfg.init(&device);
    let mut optimizer = make_optimizer();
    let mut start_update = 0usize;
    let mut best_val = f64::INFINITY;
    let mut rng_state = 0xD1B5_4A32_9F6C_71E3u64;
    let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
    if let Some(path) = resume {
        model = model
            .load_file(path, &recorder, &device)
            .unwrap_or_else(|e| panic!("failed to load resume checkpoint: {e}"));
        optimizer = load_optimizer(optimizer, path, &device);
        if let Some((update, best, rng)) = load_state(path) {
            start_update = update.min(steps);
            best_val = best;
            rng_state = rng;
            println!("resumed: update={start_update} best-val={best_val:.5}");
        }
    }
    let corpus = fs::read_to_string(data_path)
        .unwrap_or_else(|e| panic!("failed to read training data '{data_path}': {e}"));
    let encoded = if cfg.vocab == Config::target().vocab {
        let tok = if Path::new(tokenizer_path).exists() {
            AmdTokenizer::load(tokenizer_path)
                .unwrap_or_else(|e| panic!("failed to load tokenizer: {e}"))
        } else {
            let tok = AmdTokenizer::train(&corpus, cfg.vocab);
            tok.save(tokenizer_path)
                .unwrap_or_else(|e| panic!("failed to save tokenizer: {e}"));
            tok
        };
        assert_eq!(tok.vocab_size(), cfg.vocab);
        tok.encode(&corpus)
    } else {
        Tokenizer::new().encode(&corpus)
    };
    assert!(encoded.len() > cfg.context + 2);
    let split = (encoded.len() * VAL_FRACTION_NUM / VAL_FRACTION_DEN)
        .max(cfg.context + 2)
        .min(encoded.len().saturating_sub(cfg.context + 2));
    let train_tokens = &encoded[..split];
    let val_tokens = &encoded[split.saturating_sub(cfg.context)..];
    assert!(train_tokens.len() > cfg.context + 1 && val_tokens.len() > cfg.context + 1);
    let warmup = (steps / 20).max(20).min(steps);
    println!("AMD Vulkan: Burn WGPU native SDPA + RoPE + SwiGLU + KV-cache");
    println!(
        "GPU={gpu_kind}:{gpu_index} | params={} | vocab={} ctx={} d_model={} layers={} heads={} ffn={}",
        model.num_params(),
        cfg.vocab,
        cfg.context,
        cfg.d_model,
        cfg.layers,
        cfg.heads,
        cfg.ffn
    );
    println!(
        "tokens={} train={} val={} | batch={} accum={} | lr={lr:.7} warmup={warmup}",
        encoded.len(),
        train_tokens.len(),
        val_tokens.len(),
        batch_size,
        grad_accum
    );
    let loss_fn = CrossEntropyLossConfig::new().init(&device);
    let mut interval_loss = 0.0;
    let mut interval_tokens = 0usize;
    let mut interval_updates = 0usize;
    let mut last = Instant::now();
    for update in start_update..steps {
        let current_context = curriculum_context(cfg.context, update, steps);
        let current_lr = cosine_lr(lr, 0.1, update, warmup, steps);
        let mut accumulator = GradientsAccumulator::new();
        let mut update_loss = 0.0;
        for _ in 0..grad_accum {
            let (xs, ys) = make_batch(
                train_tokens,
                current_context,
                batch_size,
                &mut rng_state,
                &device,
            );
            let logits = model.forward_logits(xs);
            let loss = loss_fn.forward(
                logits.reshape([batch_size * current_context, cfg.vocab]),
                ys.reshape([batch_size * current_context]),
            );
            update_loss += loss.clone().into_scalar() as f64 / grad_accum as f64;
            let grads = loss.div_scalar(grad_accum as f64).backward();
            accumulator.accumulate(&model, GradientsParams::from_grads(grads, &model));
            interval_tokens += batch_size * current_context;
        }
        let grads = accumulator.grads();
        assert!(!grads.is_empty(), "no gradients reached optimizer");
        model = optimizer.step(current_lr, model, grads);
        interval_loss += update_loss;
        interval_updates += 1;
        if update % 10 == 9 || update + 1 == steps {
            let elapsed = last.elapsed().as_secs_f64().max(1e-9);
            println!(
                "update {:6} loss {:.5} lr {:.7} | {:.0} tok/s | ctx {}",
                update + 1,
                interval_loss / interval_updates as f64,
                current_lr,
                interval_tokens as f64 / elapsed,
                current_context
            );
            interval_loss = 0.0;
            interval_tokens = 0;
            interval_updates = 0;
            last = Instant::now();
        }
        if eval_every > 0 && ((update + 1) % eval_every == 0 || update + 1 == steps) {
            let eval_model = model.valid();
            let val_loss = evaluate(&eval_model, val_tokens, cfg.context, 2, &device);
            println!(
                "validation {:6} loss {:.5} perplexity {:.3}",
                update + 1,
                val_loss,
                val_loss.exp()
            );
            if val_loss < best_val {
                best_val = val_loss;
                save_checkpoint(
                    &model,
                    &optimizer,
                    best_checkpoint,
                    update + 1,
                    best_val,
                    rng_state,
                );
                println!("best checkpoint -> {best_checkpoint}");
            }
        }
        if checkpoint_every > 0 && (update + 1) % checkpoint_every == 0 {
            save_checkpoint(
                &model,
                &optimizer,
                checkpoint,
                update + 1,
                best_val,
                rng_state,
            );
            println!("checkpoint -> {checkpoint} (update {})", update + 1);
        }
    }
    save_checkpoint(&model, &optimizer, checkpoint, steps, best_val, rng_state);
    println!("final checkpoint -> {checkpoint}");
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
    let model: AmdModel<AmdBase> = AmdModelConfig::new(cfg).init(&device);
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
        "AMD Vulkan forward {}:{}, {:.0} tok/s | params={}",
        gpu_kind,
        gpu_index,
        (batch_size * context * iterations) as f64 / seconds,
        model.num_params()
    );
}

pub fn inference_prefill(
    model: &AmdModel<AmdBase>,
    tokens: &[usize],
    device: &WgpuDevice,
) -> (AmdCache<AmdBase>, Vec<f32>) {
    assert!(!tokens.is_empty() && tokens.len() <= model.context);
    let input = Tensor::<AmdBase, 2, Int>::from_data(
        TensorData::new(
            tokens.iter().map(|&v| v as i64).collect::<Vec<_>>(),
            [1, tokens.len()],
        ),
        device,
    );
    let mut cache = AmdCache::new(model.num_layers());
    let hidden = model.prefill(input, &mut cache);
    let logits = model.output.forward(hidden);
    let last = logits
        .slice([0..1, tokens.len() - 1..tokens.len(), 0..model.vocab])
        .reshape([model.vocab]);
    let values = last
        .into_data()
        .to_vec::<f32>()
        .expect("failed to read logits");
    (cache, values)
}

pub fn inference_step(
    model: &AmdModel<AmdBase>,
    token: usize,
    cache: &mut AmdCache<AmdBase>,
    device: &WgpuDevice,
) -> Vec<f32> {
    let input =
        Tensor::<AmdBase, 2, Int>::from_data(TensorData::new(vec![token as i64], [1, 1]), device);
    let hidden = model.step(input, cache);
    model
        .output
        .forward(hidden)
        .reshape([model.vocab])
        .into_data()
        .to_vec::<f32>()
        .expect("failed to read logits")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scheduler_reaches_floor() {
        assert!(cosine_lr(1.0, 0.1, 0, 10, 100) < cosine_lr(1.0, 0.1, 9, 10, 100));
        assert!((cosine_lr(1.0, 0.1, 99, 10, 100) - 0.1).abs() < 1e-6);
    }
    #[test]
    fn curriculum_reaches_full_context() {
        assert_eq!(curriculum_context(1024, 0, 300), 256);
        assert_eq!(curriculum_context(1024, 120, 300), 512);
        assert_eq!(curriculum_context(1024, 299, 300), 1024);
    }
}
