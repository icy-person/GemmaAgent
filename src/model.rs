use crate::{autograd::Value, config::Config};

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

    fn positional_encoding(&self, pos: usize) -> Value {
        let d = self.cfg.d_model;
        let mut values = Vec::with_capacity(d);
        for i in 0..d {
            let exponent = (2 * (i / 2)) as f32 / d as f32;
            let angle = pos as f32 / 10000.0_f32.powf(exponent);
            values.push(if i % 2 == 0 { angle.sin() } else { angle.cos() });
        }
        Value::leaf(1, d, values)
    }

    fn attention(
        &self,
        block: &Block,
        query: &Value,
        keys: &[Value],
        values: &[Value],
        pos: usize,
    ) -> Value {
        let head_dim = self.cfg.head_dim();
        let mut heads = Vec::with_capacity(self.cfg.heads);

        for head in 0..self.cfg.heads {
            let offset = head * head_dim;
            let qh = query.slice_cols(offset, head_dim);
            let mut scores = Vec::with_capacity(pos + 1);
            let mut head_values = Vec::with_capacity(pos + 1);
            for (key, value) in keys[..=pos].iter().zip(&values[..=pos]) {
                scores.push(
                    qh.matmul(&key.slice_cols(offset, head_dim).transpose())
                        .div_scalar((head_dim as f32).sqrt()),
                );
                head_values.push(value.slice_cols(offset, head_dim));
            }
            let weights = Value::concat_cols(&scores).softmax();
            heads.push(weights.matmul(&Value::concat_rows(&head_values)));
        }

        block.o.forward(&Value::concat_cols(&heads))
    }

    pub fn forward_hidden(&self, tokens: &[usize]) -> Value {
        assert!(!tokens.is_empty() && tokens.len() <= self.cfg.context);
        let mut states: Vec<Value> = tokens
            .iter()
            .enumerate()
            .map(|(pos, &token)| {
                assert!(token < self.cfg.vocab);
                self.emb.row(token).add(&self.positional_encoding(pos))
            })
            .collect();

        for block in &self.blocks {
            // Project Q/K/V once per token. The previous implementation rebuilt
            // all K/V projections for every query position, multiplying work by
            // another O(T) factor during every decoder block.
            let queries: Vec<Value> = states.iter().map(|x| block.q.forward(x)).collect();
            let keys: Vec<Value> = states.iter().map(|x| block.k.forward(x)).collect();
            let values: Vec<Value> = states.iter().map(|x| block.v.forward(x)).collect();

            let mut next = Vec::with_capacity(states.len());
            for pos in 0..states.len() {
                let attention = self.attention(block, &queries[pos], &keys, &values, pos);
                let residual = states[pos].add(&attention);
                let hidden = block.up.forward(&residual).silu();
                next.push(residual.add(&block.down.forward(&hidden)));
            }
            states = next;
        }

        states.pop().expect("non-empty token sequence")
    }

    pub fn logits(&self, hidden: &Value) -> Value {
        hidden.matmul(&self.emb.transpose())
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
    fn forward_is_deterministic_for_fixed_seed() {
        let cfg = Config::debug();
        let a = Model::new(cfg, 123).forward_hidden(&[256, 65, 66, 257]);
        let b = Model::new(cfg, 123).forward_hidden(&[256, 65, 66, 257]);
        assert_eq!(a.data(), b.data());
    }
}
