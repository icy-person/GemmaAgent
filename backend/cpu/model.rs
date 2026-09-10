use crate::{autograd::Value, config::Config};

const RMS_EPS: f32 = 1e-5;
const ROPE_THETA: f32 = 10_000.0;
const CAUSAL_MASK_VALUE: f32 = -1.0e9;

pub struct Linear {
    pub w: Value,
}
impl Linear {
    fn new(input: usize, output: usize, seed: &mut u64) -> Self {
        Self {
            w: Value::parameter(output, input, seed),
        }
    }
    fn forward(&self, x: &Value) -> Value {
        x.matmul(&self.w.transpose())
    }
}

pub struct Block {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    up: Linear,
    down: Linear,
}
impl Block {
    fn new(cfg: &Config, seed: &mut u64) -> Self {
        let d = cfg.d_model;
        let f = cfg.ffn;
        Self {
            q: Linear::new(d, d, seed),
            k: Linear::new(d, d, seed),
            v: Linear::new(d, d, seed),
            o: Linear::new(d, d, seed),
            up: Linear::new(d, f, seed),
            down: Linear::new(f, d, seed),
        }
    }
}

pub struct Model {
    pub cfg: Config,
    pub emb: Value,
    pub blocks: Vec<Block>,
}
impl Model {
    pub fn new(cfg: Config, seed: u64) -> Self {
        cfg.validate();
        let mut seed = seed;
        let emb = Value::parameter(cfg.vocab, cfg.d_model, &mut seed);
        let blocks = (0..cfg.layers)
            .map(|_| Block::new(&cfg, &mut seed))
            .collect();
        Self { cfg, emb, blocks }
    }

    /// Builds batched RoPE tensors for a whole sequence. Keeping the sequence as one matrix is
    /// important: the old token-by-token implementation created several autograd nodes per
    /// head *per token*. This version creates the same rotations with matrix-shaped ops.
    fn rope_angles(&self, rows: usize) -> (Value, Value) {
        let head_dim = self.cfg.head_dim();
        let half = head_dim / 2;
        let mut cos = Vec::with_capacity(rows * half);
        let mut sin = Vec::with_capacity(rows * half);
        for pos in 0..rows {
            for i in 0..half {
                let exponent = (2 * i) as f32 / head_dim as f32;
                let freq = 1.0 / ROPE_THETA.powf(exponent);
                let angle = pos as f32 * freq;
                cos.push(angle.cos());
                sin.push(angle.sin());
            }
        }
        (Value::leaf(rows, half, cos), Value::leaf(rows, half, sin))
    }

    /// Applies GPT-NeoX/LLaMA rotate-half RoPE to every position in a `(rows, d_model)` matrix.
    fn apply_rope_batch(&self, x: &Value) -> Value {
        let (rows, cols) = x.shape();
        assert_eq!(cols, self.cfg.d_model);
        let head_dim = self.cfg.head_dim();
        let half = head_dim / 2;
        let (cos, sin) = self.rope_angles(rows);
        let mut heads = Vec::with_capacity(self.cfg.heads);
        for head in 0..self.cfg.heads {
            let offset = head * head_dim;
            let first = x.slice_cols(offset, half);
            let second = x.slice_cols(offset + half, half);
            let rotated_first = first.mul(&cos).add(&second.mul(&sin).neg());
            let rotated_second = first.mul(&sin).add(&second.mul(&cos));
            heads.push(Value::concat_cols(&[rotated_first, rotated_second]));
        }
        Value::concat_cols(&heads)
    }

    /// Causal attention is now evaluated as batched matrix attention:
    ///
    ///   Q [T,D] @ K^T [D,T] -> scores [T,T]
    ///   causal mask -> row-wise softmax -> weights [T,T]
    ///   weights [T,T] @ V [T,H] -> output [T,H]
    ///
    /// The previous implementation materialized one autograd graph node per query/key pair.
    /// At T=1024 that was ~25 million pairs across this 6-layer/8-head model. The matrix form
    /// keeps the O(T^2) data in a handful of dense tensors instead of millions of tiny nodes,
    /// reducing both RAM and allocator overhead while also making matmul much more CPU-friendly.
    fn attention(&self, block: &Block, q: &Value, k: &Value, v: &Value, causal_mask: &Value) -> Value {
        let head_dim = self.cfg.head_dim();
        let mut heads = Vec::with_capacity(self.cfg.heads);
        for head in 0..self.cfg.heads {
            let offset = head * head_dim;
            let qh = q.slice_cols(offset, head_dim);
            let kh = k.slice_cols(offset, head_dim);
            let vh = v.slice_cols(offset, head_dim);
            let scores = qh
                .matmul(&kh.transpose())
                .div_scalar((head_dim as f32).sqrt())
                .add(causal_mask);
            let weights = scores.softmax();
            heads.push(weights.matmul(&vh));
        }
        block.o.forward(&Value::concat_cols(&heads))
    }

    fn causal_mask(rows: usize) -> Value {
        let mut data = vec![0.0f32; rows * rows];
        for row in 0..rows {
            for col in (row + 1)..rows {
                data[row * rows + col] = CAUSAL_MASK_VALUE;
            }
        }
        Value::leaf(rows, rows, data)
    }

    pub fn forward_all_hidden(&self, tokens: &[usize]) -> Vec<Value> {
        assert!(!tokens.is_empty() && tokens.len() <= self.cfg.context);
        let rows = tokens.len();
        let mut embedded = Vec::with_capacity(rows);
        for &token in tokens {
            assert!(token < self.cfg.vocab);
            embedded.push(self.emb.row(token));
        }
        let mut states = Value::concat_rows(&embedded);
        let causal_mask = Self::causal_mask(rows);

        for block in &self.blocks {
            // Batch every position through each projection/MLP. This both reduces graph size and
            // lets the row-parallel CPU matmul implementation do substantially larger kernels.
            let normed = states.rms_norm(RMS_EPS);
            let queries = self.apply_rope_batch(&block.q.forward(&normed));
            let keys = self.apply_rope_batch(&block.k.forward(&normed));
            let values = block.v.forward(&normed);
            let attention = self.attention(block, &queries, &keys, &values, &causal_mask);
            let residual = states.add(&attention);
            let hidden = block.up.forward(&residual.rms_norm(RMS_EPS)).silu();
            states = residual.add(&block.down.forward(&hidden));
        }

        (0..rows).map(|row| states.row(row)).collect()
    }

    pub fn forward_hidden(&self, tokens: &[usize]) -> Value {
        self.forward_all_hidden(tokens)
            .pop()
            .expect("non-empty token sequence")
    }

    pub fn logits(&self, hidden: &Value) -> Value {
        hidden
            .rms_norm(RMS_EPS)
            .matmul(&self.emb.transpose())
    }

    pub fn parameters(&self) -> Vec<Value> {
        let mut parameters = vec![self.emb.clone()];
        for block in &self.blocks {
            parameters.extend([
                block.q.w.clone(),
                block.k.w.clone(),
                block.v.w.clone(),
                block.o.w.clone(),
                block.up.w.clone(),
                block.down.w.clone(),
            ]);
        }
        parameters
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_parameter_count_is_exact() {
        let cfg = Config::target();
        let model = Model::new(cfg, 42);
        let total: usize = model.parameters().iter().map(|p| p.data().len()).sum();
        assert_eq!(total, 19_275_776);
        assert_eq!(cfg.params(), total);
    }

    #[test]
    fn forward_and_logits_have_expected_shapes() {
        let cfg = Config::debug();
        let model = Model::new(cfg, 42);
        let hidden = model.forward_hidden(&[256, b'R' as usize, b'u' as usize]);
        assert_eq!(hidden.shape(), (1, cfg.d_model));
        assert_eq!(model.logits(&hidden).shape(), (1, cfg.vocab));
    }

    #[test]
    fn forward_all_hidden_returns_every_position() {
        let cfg = Config::debug();
        let model = Model::new(cfg, 42);
        let hidden = model.forward_all_hidden(&[256, 65, 66, 67]);
        assert_eq!(hidden.len(), 4);
        assert!(hidden.iter().all(|value| value.shape() == (1, cfg.d_model)));
        assert_eq!(hidden[3].data(), model.forward_hidden(&[256, 65, 66, 67]).data());
    }

    #[test]
    fn forward_is_deterministic_for_fixed_seed() {
        let a = Model::new(Config::debug(), 123).forward_hidden(&[256, 65, 66, 257]);
        let b = Model::new(Config::debug(), 123).forward_hidden(&[256, 65, 66, 257]);
        assert_eq!(a.data(), b.data());
    }

    #[test]
    fn rope_rotation_preserves_vector_norm() {
        let cfg = Config::debug();
        let model = Model::new(cfg, 7);
        let mut seed = 1;
        let x = Value::parameter(1, cfg.d_model, &mut seed);
        let rotated = model.apply_rope_batch(&x);
        let norm = |v: &Value| v.data().iter().map(|d| d * d).sum::<f32>().sqrt();
        assert!((norm(&rotated) - norm(&x)).abs() < 1e-4);
    }

    #[test]
    fn rope_is_position_dependent() {
        let cfg = Config::debug();
        let model = Model::new(cfg, 7);
        let mut seed = 1;
        let row = Value::parameter(1, cfg.d_model, &mut seed);
        let x = Value::concat_rows(&[row.clone(), row]);
        let rotated = model.apply_rope_batch(&x).data();
        let cols = cfg.d_model;
        assert_ne!(&rotated[..cols], &rotated[cols..2 * cols]);
    }

    #[test]
    fn causal_mask_blocks_future_positions() {
        let mask = Model::causal_mask(4).data();
        for row in 0..4 {
            for col in 0..4 {
                let value = mask[row * 4 + col];
                if col > row {
                    assert_eq!(value, CAUSAL_MASK_VALUE);
                } else {
                    assert_eq!(value, 0.0);
                }
            }
        }
    }
}
