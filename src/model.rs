use crate::config::Config;
use crate::tensor::{rms_norm, silu, softmax};

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self { Self(seed) }
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        let bits = ((self.0 >> 32) as u32) | 0x3f80_0000;
        f32::from_bits(bits) - 1.0
    }
    fn normalish(&mut self, scale: f32) -> f32 {
        (self.next() + self.next() - 1.0) * scale
    }
}

struct Linear {
    out: usize,
    input: usize,
    weight: Vec<f32>,
}

impl Linear {
    fn new(rng: &mut Rng, input: usize, out: usize, scale: f32) -> Self {
        let mut weight = vec![0.0; input * out];
        for v in &mut weight { *v = rng.normalish(scale); }
        Self { out, input, weight }
    }

    fn apply(&self, x: &[f32], y: &mut [f32]) {
        debug_assert_eq!(x.len(), self.input);
        debug_assert_eq!(y.len(), self.out);
        for o in 0..self.out {
            let row = &self.weight[o * self.input..(o + 1) * self.input];
            let mut sum = 0.0;
            for i in 0..self.input { sum += row[i] * x[i]; }
            y[o] = sum;
        }
    }
}

struct Block {
    qkv: Linear,
    out: Linear,
    gate: Linear,
    up: Linear,
    down: Linear,
    norm1: Vec<f32>,
    norm2: Vec<f32>,
}

pub struct Model {
    pub config: Config,
    token_embedding: Vec<f32>,
    blocks: Vec<Block>,
    final_norm: Vec<f32>,
}

impl Model {
    pub fn new(config: Config, seed: u64) -> Self {
        assert_eq!(config.d_model % config.n_heads, 0, "d_model must be divisible by n_heads");
        assert_eq!(config.head_dim() % 2, 0, "head_dim must be even for RoPE");
        let d = config.d_model;
        let mut rng = Rng::new(seed);
        let scale = (2.0 / d as f32).sqrt();
        let mut token_embedding = vec![0.0; config.vocab_size * d];
        for v in &mut token_embedding { *v = rng.normalish(0.02); }

        let mut blocks = Vec::with_capacity(config.n_layers);
        for _ in 0..config.n_layers {
            blocks.push(Block {
                qkv: Linear::new(&mut rng, d, d * 3, scale),
                out: Linear::new(&mut rng, d, d, scale),
                gate: Linear::new(&mut rng, d, config.ffn_dim, scale),
                up: Linear::new(&mut rng, d, config.ffn_dim, scale),
                down: Linear::new(&mut rng, config.ffn_dim, d, scale),
                norm1: vec![1.0; d],
                norm2: vec![1.0; d],
            });
        }

        Self { final_norm: vec![1.0; d], config, token_embedding, blocks }
    }

    #[inline]
    fn embedding(&self, id: usize) -> &[f32] {
        let start = id * self.config.d_model;
        &self.token_embedding[start..start + self.config.d_model]
    }

    fn apply_rope(&self, q: &mut [f32], k: &mut [f32], position: usize) {
        let half = self.config.head_dim() / 2;
        for head in 0..self.config.n_heads {
            let base = head * self.config.head_dim();
            for i in 0..half {
                let freq = self.config.rope_theta.powf(-(2.0 * i as f32 / self.config.head_dim() as f32));
                let (sin, cos) = (position as f32 * freq).sin_cos();
                let j = base + 2 * i;
                let (qa, qb) = (q[j], q[j + 1]);
                q[j] = qa * cos - qb * sin;
                q[j + 1] = qa * sin + qb * cos;
                let (ka, kb) = (k[j], k[j + 1]);
                k[j] = ka * cos - kb * sin;
                k[j + 1] = ka * sin + kb * cos;
            }
        }
    }

    pub fn forward(&self, tokens: &[usize]) -> Vec<f32> {
        assert!(!tokens.is_empty(), "input cannot be empty");
        assert!(tokens.len() <= self.config.context, "context length exceeded");
        assert!(tokens.iter().all(|&t| t < self.config.vocab_size), "token out of range");

        let d = self.config.d_model;
        let hd = self.config.head_dim();
        let n = tokens.len();
        let mut x = Vec::with_capacity(n * d);
        for &token in tokens { x.extend_from_slice(self.embedding(token)); }

        for block in &self.blocks {
            let mut normalized = vec![0.0; n * d];
            for t in 0..n {
                let dst = &mut normalized[t * d..(t + 1) * d];
                dst.copy_from_slice(&x[t * d..(t + 1) * d]);
                rms_norm(dst, &block.norm1, 1e-6);
            }

            let mut q = vec![0.0; n * d];
            let mut k = vec![0.0; n * d];
            let mut v = vec![0.0; n * d];
            for t in 0..n {
                let mut packed = vec![0.0; d * 3];
                block.qkv.apply(&normalized[t * d..(t + 1) * d], &mut packed);
                q[t * d..(t + 1) * d].copy_from_slice(&packed[..d]);
                k[t * d..(t + 1) * d].copy_from_slice(&packed[d..2 * d]);
                v[t * d..(t + 1) * d].copy_from_slice(&packed[2 * d..]);
                self.apply_rope(&mut q[t * d..(t + 1) * d], &mut k[t * d..(t + 1) * d], t);
            }

            let mut attended = vec![0.0; n * d];
            let scale = (hd as f32).sqrt().recip();
            for t in 0..n {
                for head in 0..self.config.n_heads {
                    let base = head * hd;
                    let mut scores = vec![0.0; t + 1];
                    for j in 0..=t {
                        let mut dot = 0.0;
                        for i in 0..hd { dot += q[t * d + base + i] * k[j * d + base + i]; }
                        scores[j] = dot * scale;
                    }
                    softmax(&mut scores);
                    for j in 0..=t {
                        let weight = scores[j];
                        for i in 0..hd { attended[t * d + base + i] += weight * v[j * d + base + i]; }
                    }
                }
            }

            let mut projected = vec![0.0; n * d];
            for t in 0..n {
                block.out.apply(&attended[t * d..(t + 1) * d], &mut projected[t * d..(t + 1) * d]);
                for i in 0..d { x[t * d + i] += projected[t * d + i]; }
            }

            let mut norm2 = vec![0.0; n * d];
            for t in 0..n {
                norm2[t * d..(t + 1) * d].copy_from_slice(&x[t * d..(t + 1) * d]);
                rms_norm(&mut norm2[t * d..(t + 1) * d], &block.norm2, 1e-6);
            }

            for t in 0..n {
                let inp = &norm2[t * d..(t + 1) * d];
                let mut gate = vec![0.0; self.config.ffn_dim];
                let mut up = vec![0.0; self.config.ffn_dim];
                block.gate.apply(inp, &mut gate);
                block.up.apply(inp, &mut up);
                for i in 0..gate.len() { gate[i] = silu(gate[i]) * up[i]; }
                let mut down = vec![0.0; d];
                block.down.apply(&gate, &mut down);
                for i in 0..d { x[t * d + i] += down[i]; }
            }
        }

        let last = n - 1;
        let mut hidden = x[last * d..(last + 1) * d].to_vec();
        rms_norm(&mut hidden, &self.final_norm, 1e-6);

        let mut logits = vec![0.0; self.config.vocab_size];
        for token in 0..self.config.vocab_size {
            let e = self.embedding(token);
            let mut sum = 0.0;
            for i in 0..d { sum += hidden[i] * e[i]; }
            logits[token] = sum;
        }
        logits
    }
}
