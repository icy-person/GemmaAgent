use crate::{autograd::Value, config::Config};

const RMS_EPS: f32 = 1e-5;
const ROPE_THETA: f32 = 10_000.0;

pub struct Linear { pub w: Value }
impl Linear {
    fn new(input: usize, output: usize, seed: &mut u64) -> Self { Self { w: Value::parameter(output, input, seed) } }
    /// `x` is `(rows, input)` — every row (one token's state) goes through the same weight
    /// matrix in a single matmul, so a whole sequence costs one big matmul instead of `rows`
    /// separate `1 x input` matmuls.
    fn forward(&self, x: &Value) -> Value { x.matmul(&self.w.transpose()) }
}

pub struct Block { q: Linear, k: Linear, v: Linear, o: Linear, up: Linear, down: Linear }
impl Block {
    fn new(cfg: &Config, seed: &mut u64) -> Self {
        let d = cfg.d_model; let f = cfg.ffn;
        Self { q: Linear::new(d, d, seed), k: Linear::new(d, d, seed), v: Linear::new(d, d, seed), o: Linear::new(d, d, seed), up: Linear::new(d, f, seed), down: Linear::new(f, d, seed) }
    }
}

pub struct Model { pub cfg: Config, pub emb: Value, pub blocks: Vec<Block> }

impl Model {
    pub fn new(cfg: Config, seed: u64) -> Self {
        cfg.validate();
        let mut seed = seed;
        let emb = Value::parameter(cfg.vocab, cfg.d_model, &mut seed);
        let blocks = (0..cfg.layers).map(|_| Block::new(&cfg, &mut seed)).collect();
        Self { cfg, emb, blocks }
    }

    /// RoPE angle tables for every position `0..len`: `(cos, sin)`, each `(len, head_dim/2)`.
    /// Built once per forward call and shared (cheap `Rc` clones, via `Value::clone`) across
    /// every layer and head, since the rotation only depends on absolute position and
    /// `head_dim`, not on the layer. Matches the AMD/Android Vulkan backend's
    /// `RotaryEncodingConfig` (theta=10000), so CPU and GPU rotate Q/K the same way.
    fn rope_tables(&self, len: usize) -> (Value, Value) {
        let head_dim = self.cfg.head_dim();
        let half = head_dim / 2;
        let mut cos = Vec::with_capacity(len * half);
        let mut sin = Vec::with_capacity(len * half);
        for pos in 0..len {
            for i in 0..half {
                let exponent = (2 * i) as f32 / head_dim as f32;
                let freq = 1.0 / ROPE_THETA.powf(exponent);
                let angle = pos as f32 * freq;
                cos.push(angle.cos());
                sin.push(angle.sin());
            }
        }
        (Value::leaf(len, half, cos), Value::leaf(len, half, sin))
    }

    /// Causal mask `(len, len)`: 0 on/below the diagonal, a large negative above it. Added to
    /// raw attention scores before softmax so a query never attends to a future position. Built
    /// once per forward call and shared across every layer/head (same sequence length
    /// throughout one call).
    fn causal_mask(len: usize) -> Value {
        let mut d = vec![0.0f32; len * len];
        for i in 0..len { for j in (i + 1)..len { d[i * len + j] = -1.0e9; } }
        Value::leaf(len, len, d)
    }

    /// Applies RoPE to every head slice of a `(len, d_model)` projection matrix in one shot per
    /// head, using the GPT-NeoX/LLaMA "rotate half" convention (contiguous halves instead of
    /// interleaved pairs). `cos`/`sin` are the shared `(len, head_dim/2)` tables from
    /// `rope_tables`, so this needs no per-position loop: one elementwise mul/add pair per head
    /// rotates every position in the sequence at once. Each `(first[i], second[i])` pair is an
    /// independent 2D rotation, so this preserves the per-head, per-position vector norm
    /// exactly, same as any orthogonal rotation.
    fn apply_rope(&self, x: &Value, cos: &Value, sin: &Value) -> Value {
        let head_dim = self.cfg.head_dim();
        let half = head_dim / 2;
        let mut heads = Vec::with_capacity(self.cfg.heads);
        for head in 0..self.cfg.heads {
            let offset = head * head_dim;
            let first = x.slice_cols(offset, half);
            let second = x.slice_cols(offset + half, half);
            let rotated_first = first.mul(cos).add(&second.mul(sin).neg());
            let rotated_second = first.mul(sin).add(&second.mul(cos));
            heads.push(Value::concat_cols(&[rotated_first, rotated_second]));
        }
        Value::concat_cols(&heads)
    }

    /// Causal multi-head self-attention over the whole sequence at once. `q`/`k`/`v` are each
    /// `(len, d_model)`. Every head does exactly one `Q·Kᵀ` matmul (`(len, len)`), one masked
    /// row-wise softmax, and one `weights·V` matmul across all `len` positions together — a
    /// constant number of autograd nodes per layer regardless of sequence length, instead of
    /// the old per-position-pair loop that built `O(len²)` separate nodes.
    fn attention(&self, block: &Block, q: &Value, k: &Value, v: &Value, mask: &Value) -> Value {
        let head_dim = self.cfg.head_dim();
        let scale = (head_dim as f32).sqrt();
        let mut heads = Vec::with_capacity(self.cfg.heads);
        for head in 0..self.cfg.heads {
            let offset = head * head_dim;
            let qh = q.slice_cols(offset, head_dim);
            let kh = k.slice_cols(offset, head_dim);
            let vh = v.slice_cols(offset, head_dim);
            let scores = qh.matmul(&kh.transpose()).div_scalar(scale).add(mask);
            let weights = scores.softmax();
            heads.push(weights.matmul(&vh));
        }
        block.o.forward(&Value::concat_cols(&heads))
    }

    /// Runs the full stack and returns every position's final hidden state. The whole sequence
    /// is carried as a single `(len, d_model)` matrix through every layer — one batched matmul
    /// per projection instead of one per position, and one `(len, len)` attention matrix per
    /// head instead of `len` growing per-position score vectors — and is only split back into
    /// per-position rows at the very end, for callers that sample a subset of positions (e.g.
    /// the training loss, which only needs `targets_per_step` of them).
    pub fn forward_all_hidden(&self, tokens: &[usize]) -> Vec<Value> {
        assert!(!tokens.is_empty() && tokens.len() <= self.cfg.context);
        let len = tokens.len();
        let rows: Vec<Value> = tokens.iter().map(|&token| { assert!(token < self.cfg.vocab); self.emb.row(token) }).collect();
        let mut states = Value::concat_rows(&rows);
        let (cos, sin) = self.rope_tables(len);
        let mask = Self::causal_mask(len);
        for block in &self.blocks {
            let normed = states.rms_norm(RMS_EPS);
            let q = self.apply_rope(&block.q.forward(&normed), &cos, &sin);
            let k = self.apply_rope(&block.k.forward(&normed), &cos, &sin);
            let v = block.v.forward(&normed);
            let attn = self.attention(block, &q, &k, &v, &mask);
            let residual = states.add(&attn);
            let hidden = block.up.forward(&residual.rms_norm(RMS_EPS)).silu();
            states = residual.add(&block.down.forward(&hidden));
        }
        (0..len).map(|i| states.row(i)).collect()
    }
    pub fn forward_hidden(&self, tokens: &[usize]) -> Value { self.forward_all_hidden(tokens).pop().expect("non-empty token sequence") }
    pub fn logits(&self, hidden: &Value) -> Value { hidden.rms_norm(RMS_EPS).matmul(&self.emb.transpose()) }
    pub fn parameters(&self) -> Vec<Value> {
        let mut parameters = vec![self.emb.clone()];
        for block in &self.blocks { parameters.extend([block.q.w.clone(), block.k.w.clone(), block.v.w.clone(), block.o.w.clone(), block.up.w.clone(), block.down.w.clone()]); }
        parameters
    }
}

#[cfg(test)]
mod tests{use super::*;
#[test]fn target_parameter_count_is_exact(){let cfg=Config::target();let model=Model::new(cfg,42);let total:usize=model.parameters().iter().map(|p|p.data().len()).sum();assert_eq!(total,19_275_776);assert_eq!(cfg.params(),total);}
#[test]fn forward_and_logits_have_expected_shapes(){let cfg=Config::debug();let model=Model::new(cfg,42);let hidden=model.forward_hidden(&[256,b'R' as usize,b'u' as usize]);assert_eq!(hidden.shape(),(1,cfg.d_model));assert_eq!(model.logits(&hidden).shape(),(1,cfg.vocab));}
#[test]fn forward_all_hidden_returns_every_position(){let cfg=Config::debug();let model=Model::new(cfg,42);let hidden=model.forward_all_hidden(&[256,65,66,67]);assert_eq!(hidden.len(),4);assert!(hidden.iter().all(|value|value.shape()==(1,cfg.d_model)));assert_eq!(hidden[3].data(),model.forward_hidden(&[256,65,66,67]).data());}
#[test]fn forward_is_deterministic_for_fixed_seed(){let cfg=Config::debug();let a=Model::new(cfg,123).forward_hidden(&[256,65,66,257]);let b=Model::new(cfg,123).forward_hidden(&[256,65,66,257]);assert_eq!(a.data(),b.data());}
#[test]fn causal_masking_prevents_future_leakage(){
    // Changing only the LAST token must never change the hidden state at any EARLIER position —
    // that's the whole point of causal masking, and the property most at risk from switching to
    // a batched (len, len) attention matrix instead of the old strictly-sequential per-position
    // loop.
    let cfg=Config::debug();
    let model=Model::new(cfg,11);
    let a=model.forward_all_hidden(&[10,20,30,40]);
    let b=model.forward_all_hidden(&[10,20,30,99]);
    for i in 0..3 { assert_eq!(a[i].data(), b[i].data(), "position {i} must not be affected by a later token"); }
    assert_ne!(a[3].data(), b[3].data(), "the changed position itself should differ");
}
#[test]fn rope_rotation_preserves_vector_norm(){let cfg=Config::debug();let model=Model::new(cfg,7);let mut seed=1;let x=Value::parameter(1,cfg.d_model,&mut seed);let(cos,sin)=model.rope_tables(1);let rotated=model.apply_rope(&x,&cos,&sin);let norm=|v:&Value|v.data().iter().map(|d|d*d).sum::<f32>().sqrt();assert!((norm(&rotated)-norm(&x)).abs()<1e-4,"RoPE is a rotation and must preserve vector norm");}
#[test]fn rope_is_position_dependent(){let cfg=Config::debug();let model=Model::new(cfg,7);let mut seed=1;let x=Value::parameter(1,cfg.d_model,&mut seed);let(cos0,sin0)=model.rope_tables(1);let at0=model.apply_rope(&x,&cos0,&sin0);let(cos_seq,sin_seq)=model.rope_tables(6);let cos5=cos_seq.row(5);let sin5=sin_seq.row(5);let at5=model.apply_rope(&x,&cos5,&sin5);assert_ne!(at0.data(),at5.data());}
#[test]fn rope_at_position_zero_is_identity(){let cfg=Config::debug();let model=Model::new(cfg,7);let mut seed=1;let x=Value::parameter(1,cfg.d_model,&mut seed);let(cos,sin)=model.rope_tables(1);let rotated=model.apply_rope(&x,&cos,&sin);for(a,b) in rotated.data().iter().zip(x.data().iter()){assert!((a-b).abs()<1e-5);}}
}
