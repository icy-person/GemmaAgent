#![cfg(feature = "amd-vulkan")]

use burn::{
    backend::Autodiff,
    module::Module,
    nn::{
        Embedding, EmbeddingConfig, RmsNorm, RmsNormConfig,
        loss::CrossEntropyLossConfig,
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
    },
    optim::{AdamWConfig, GradientsParams, Optimizer},
    prelude::*,
    tensor::{Int, Tensor, TensorData, activation::log_softmax},
};
use burn_wgpu::Wgpu;
use std::{path::Path, time::Instant};

use crate::{config::Config, tokenizer::Tokenizer};

pub type AmdBase = Wgpu<f32, i32>;
pub type AmdBackend = Autodiff<AmdBase>;

const EPS: f64 = 1e-5;

#[derive(Module, Debug)]
pub struct AmdModel<B: Backend> {
    pub token_embedding: Embedding<B>,
    pub position_embedding: Embedding<B>,
    pub transformer: TransformerEncoder<B>,
    pub final_norm: RmsNorm<B>,
}

impl AmdModelConfig {
    fn new(cfg: Config) -> Self {
        Self { cfg }
    }
}

pub struct AmdModelConfig {
    cfg: Config,
}

impl AmdModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> AmdModel<B> {
        let transformer = TransformerEncoderConfig::new(
            self.cfg.d_model,
            self.cfg.ffn,
            self.cfg.heads,
            self.cfg.layers,
        )
        .with_dropout(0.0)
        .with_norm_first(true)
        .init(device);

        AmdModel {
            token_embedding: EmbeddingConfig::new(self.cfg.vocab, self.cfg.d_model).init(device),
            position_embedding: EmbeddingConfig::new(self.cfg.context, self.cfg.d_model).init(device),
            transformer,
            final_norm: RmsNormConfig::new(self.cfg.d_model)
                .with_epsilon(EPS)
                .init(device),
        }
    }
}

impl<B: Backend> AmdModel<B> {
    pub fn forward(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch, seq] = tokens.dims();
        assert!(seq <= self.position_embedding.devices()[0].clone().into());
        let positions = Tensor::<B, 1, Int>::arange(0..seq as i64, &tokens.device())
            .reshape([1, seq])
            .repeat_dim(0, batch);
        let x = self.token_embedding.forward(tokens) + self.position_embedding.forward(positions);
        let mask = burn::nn::attention::generate_autoregressive_mask(batch, seq, &x.device());
        let x = self.transformer.forward(TransformerEncoderInput::new(x).mask_attn(mask));
        self.final_norm.forward(x)
    }

    pub fn logits(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq, d] = hidden.dims();
        let flat = hidden.reshape([batch * seq, d]);
        let weight = self.token_embedding.weight.val().transpose();
        flat.matmul(weight).reshape([batch, seq, self.cfg().vocab])
    }

    pub fn cfg(&self) -> Config {
        // The AMD path is constructed from Config::debug or Config::target externally.
        // This helper is overridden by the caller through stored dimensions in tensors.
        unimplemented!("AmdModel::cfg is only a placeholder and should not be called")
    }
}

fn make_batch(
    encoded: &[usize],
    context: usize,
    batch_size: usize,
    step: usize,
    device: &<AmdBackend as Backend>::Device,
) -> (Tensor<AmdBackend, 2, Int>, Tensor<AmdBackend, 2, Int>) {
    let window_count = encoded.len() - context;
    let mut x = Vec::with_capacity(batch_size * context);
    let mut y = Vec::with_capacity(batch_size * context);
    for b in 0..batch_size {
        let start = (step * batch_size + b) % window_count;
        x.extend(encoded[start..start + context].iter().map(|&v| v as i64));
        y.extend(encoded[start + 1..start + context + 1].iter().map(|&v| v as i64));
    }
    (
        Tensor::from_data(TensorData::new(x, [batch_size, context]), device),
        Tensor::from_data(TensorData::new(y, [batch_size, context]), device),
    )
}

pub fn train(
    cfg: Config,
    steps: usize,
    checkpoint: &str,
    data_path: &str,
    batch_size: usize,
    grad_accum: usize,
    lr: f64,
    checkpoint_every: usize,
    gpu_index: usize,
) {
    cfg.validate();
    assert!(steps > 0 && batch_size > 0 && grad_accum > 0);
    assert!(lr.is_finite() && lr > 0.0);

    let device = burn::backend::wgpu::WgpuDevice::DiscreteGpu(gpu_index);
    let model_cfg = AmdModelConfig::new(cfg);
    let mut model: AmdModel<AmdBackend> = model_cfg.init(&device);
    let mut optimizer = AdamWConfig::new()
        .with_beta1(0.9)
        .with_beta2(0.95)
        .with_epsilon(1e-8)
        .with_weight_decay(0.1)
        .init();

    let tokenizer = Tokenizer::new();
    let corpus = std::fs::read_to_string(data_path).expect("failed to read training data");
    let encoded = tokenizer.encode(&corpus);
    assert!(encoded.len() > cfg.context + 1, "training corpus is too short");

    println!("AMD Vulkan backend: Burn WGPU device {gpu_index}");
    println!("model: vocab={} context={} d_model={} layers={} heads={} ffn={}", cfg.vocab, cfg.context, cfg.d_model, cfg.layers, cfg.heads, cfg.ffn);
    println!("batch={} grad_accum={} lr={:.6} corpus={} bytes", batch_size, grad_accum, lr, corpus.len());
    println!("checkpoint: {checkpoint}");

    if Path::new(checkpoint).exists() {
        println!("warning: AMD checkpoint loading is intentionally disabled until optimizer-state compatible resume is implemented");
    }

    let mut interval_loss = 0.0f64;
    let mut interval_steps = 0usize;
    let mut last = Instant::now();

    for update in 0..steps {
        let mut grads_all = None;
        let mut loss_sum = 0.0f64;

        for micro in 0..grad_accum {
            let logical = update * grad_accum + micro;
            let (xs, ys) = make_batch(&encoded, cfg.context, batch_size, logical, &device);
            let hidden = model.forward(xs);
            let logits = {
                let [b, t, d] = hidden.dims();
                let flat = hidden.reshape([b * t, d]);
                flat.matmul(model.token_embedding.weight.val().transpose())
                    .reshape([b, t, cfg.vocab])
            };
            let flat_logits = logits.reshape([batch_size * cfg.context, cfg.vocab]);
            let flat_targets = ys.reshape([batch_size * cfg.context]);
            let loss = CrossEntropyLossConfig::new()
                .init(&flat_logits.device())
                .forward(flat_logits, flat_targets)
                / grad_accum as f64;
            loss_sum += loss.clone().into_scalar::<f32>() as f64 * grad_accum as f64;
            let grads = loss.backward();
            let param_grads = GradientsParams::from_grads(grads, &model);
            grads_all = Some(match grads_all {
                Some(existing) => existing.merge(param_grads),
                None => param_grads,
            });
        }

        if let Some(grads) = grads_all {
            model = optimizer.step(lr, model, grads);
        }

        interval_loss += loss_sum / grad_accum as f64;
        interval_steps += 1;

        if update % 10 == 9 || update + 1 == steps {
            let seconds = last.elapsed().as_secs_f64().max(1e-9);
            let avg = interval_loss / interval_steps as f64;
            let tok = (batch_size * cfg.context * grad_accum * interval_steps) as f64;
            println!("amd update {:5} loss {:.5} | {:.0} tok/s", update + 1, avg, tok / seconds);
            interval_loss = 0.0;
            interval_steps = 0;
            last = Instant::now();
        }

        if checkpoint_every > 0 && (update + 1) % checkpoint_every == 0 {
            let recorder = burn::record::BinFileRecorder::<burn::record::FullPrecisionSettings>::default();
            model.clone().save_file(checkpoint, &recorder).expect("failed to save AMD checkpoint");
            println!("AMD checkpoint: {checkpoint} (update {})", update + 1);
        }
    }

    let recorder = burn::record::BinFileRecorder::<burn::record::FullPrecisionSettings>::default();
    model.save_file(checkpoint, &recorder).expect("failed to save AMD checkpoint");
    println!("AMD checkpoint: {checkpoint}");
}

pub fn benchmark(cfg: Config, gpu_index: usize, batch_size: usize, context: usize, iterations: usize) {
    let device = burn::backend::wgpu::WgpuDevice::DiscreteGpu(gpu_index);
    let model_cfg = AmdModelConfig::new(cfg);
    let model: AmdModel<AmdBase> = model_cfg.init(&device);
    let ids = Tensor::<AmdBase, 2, Int>::zeros([batch_size, context], &device);

    for _ in 0..3 {
        let _ = model.forward(ids.clone());
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = model.forward(ids.clone());
    }
    let secs = start.elapsed().as_secs_f64().max(1e-9);
    println!("AMD Vulkan benchmark: {} tok/s", (batch_size * context * iterations) as f64 / secs);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn amd_backend_types_are_distinct() {
        assert_ne!(std::any::type_name::<AmdBase>(), std::any::type_name::<AmdBackend>());
    }
}
