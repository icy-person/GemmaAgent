use crate::{autograd::Value, config::Config};

struct LayerWeights {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    o: Vec<f32>,
    up: Vec<f32>,
    down: Vec<f32>,
}

struct CacheLayer {
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
}

pub struct RuntimeModel {
    cfg: Config,
    emb: Vec<f32>,
    layers: Vec<LayerWeights>,
}

pub struct KvCache {
    layers: Vec<CacheLayer>,
    len: usize,
}

impl RuntimeModel {
    pub fn from_parameters(cfg: Config, parameters: &[Value]) -> Self {
        cfg.validate();
        assert_eq!(parameters.len(), 1 + cfg.layers * 6);
        let emb = parameters[0].data();
        let mut layers = Vec::with_capacity(cfg.layers);
        let mut index = 1;
        for _ in 0..cfg.layers {
            layers.push(LayerWeights {
                q: parameters[index].data(),
                k: parameters[index + 1].data(),
                v: parameters[index + 2].data(),
                o: parameters[index + 3].data(),
                up: parameters[index + 4].data(),
                down: parameters[index + 5].data(),
            });
            index += 6;
        }
        Self { cfg, emb, layers }
    }

    pub fn new_cache(&self) -> KvCache {
        KvCache {
            layers: (0..self.cfg.layers)
                .map(|_| CacheLayer {
                    keys: Vec::with_capacity(self.cfg.context),
                    values: Vec::with_capacity(self.cfg.context),
                })
                .collect(),
            len: 0,
        }
    }

    fn positional(&self, pos: usize) -> Vec<f32> {
        let d = self.cfg.d_model;
        let mut out = vec![0.0; d];
        for (i, value) in out.iter_mut().enumerate() {
            let exponent = (2 * (i / 2)) as f32 / d as f32;
            let angle = pos as f32 / 10000.0_f32.powf(exponent);
            *value = if i % 2 == 0 { angle.sin() } else { angle.cos() };
        }
        out
    }

    fn matvec(rows: usize, cols: usize, weight: &[f32], x: &[f32]) -> Vec<f32> {
        assert_eq!(weight.len(), rows * cols);
        assert_eq!(x.len(), cols);
        let mut out = vec![0.0; rows];
        for r in 0..rows {
            let base = r * cols;
            let mut sum = 0.0;
            for c in 0..cols {
                sum += weight[base + c] * x[c];
            }
            out[r] = sum;
        }
        out
    }

    fn project_into(&self, weight: &[f32], x: &[f32]) -> Vec<f32> {
        Self::matvec(self.cfg.d_model, self.cfg.d_model, weight, x)
    }

    fn feed_forward(&self, layer: &LayerWeights, x: &[f32]) -> Vec<f32> {
        let up = Self::matvec(self.cfg.ffn, self.cfg.d_model, &layer.up, x);
        let mut activated = up;
        for value in &mut activated {
            *value /= 1.0 + (-*value).exp();
        }
        Self::matvec(self.cfg.d_model, self.cfg.ffn, &layer.down, &activated)
    }

    fn attention(
        &self,
        layer: &LayerWeights,
        query: &[f32],
        cache: &CacheLayer,
    ) -> Vec<f32> {
        let h = self.cfg.heads;
        let head_dim = self.cfg.head_dim();
        let mut output = vec![0.0; self.cfg.d_model];

        for head in 0..h {
            let offset = head * head_dim;
            let mut scores = Vec::with_capacity(cache.keys.len());
            for key in &cache.keys {
                let mut dot = 0.0;
                for j in 0..head_dim {
                    dot += query[offset + j] * key[offset + j];
                }
                scores.push(dot / (head_dim as f32).sqrt());
            }
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut total = 0.0;
            for score in &mut scores {
                *score = (*score - max).exp();
                total += *score;
            }
            let inv = 1.0 / total.max(1e-20);
            for score in &mut scores {
                *score *= inv;
            }

            for (weight, value) in scores.iter().zip(&cache.values) {
                for j in 0..head_dim {
                    output[offset + j] += *weight * value[offset + j];
                }
            }
        }
        Self::matvec(self.cfg.d_model, self.cfg.d_model, &layer.o, &output)
    }

    fn step_layer(&self, layer_index: usize, state: &[f32], cache: &mut CacheLayer) -> Vec<f32> {
        let layer = &self.layers[layer_index];
        let q = self.project_into(&layer.q, state);
        let k = self.project_into(&layer.k, state);
        let v = self.project_into(&layer.v, state);
        cache.keys.push(k);
        cache.values.push(v);
        let attention = self.attention(layer, &q, cache);
        let residual: Vec<f32> = state
            .iter()
            .zip(attention)
            .map(|(a, b)| a + b)
            .collect();
        let ff = self.feed_forward(layer, &residual);
        residual.iter().zip(ff).map(|(a, b)| a + b).collect()
    }

    pub fn prime(&self, tokens: &[usize], cache: &mut KvCache) -> Vec<f32> {
        assert!(!tokens.is_empty());
        assert!(tokens.len() <= self.cfg.context);
        let mut states = Vec::with_capacity(tokens.len());
        for (pos, &token) in tokens.iter().enumerate() {
            assert!(token < self.cfg.vocab);
            let mut state = self.emb[token * self.cfg.d_model..(token + 1) * self.cfg.d_model].to_vec();
            for (a, b) in state.iter_mut().zip(self.positional(pos)) {
                *a += b;
            }
            states.push(state);
        }

        for layer_index in 0..self.cfg.layers {
            let mut next = Vec::with_capacity(states.len());
            cache.layers[layer_index].keys.clear();
            cache.layers[layer_index].values.clear();
            for state in &states {
                next.push(self.step_layer(layer_index, state, &mut cache.layers[layer_index]));
            }
            states = next;
        }
        cache.len = tokens.len();
        states.pop().unwrap()
    }

    pub fn next(&self, token: usize, cache: &mut KvCache) -> Vec<f32> {
        assert!(token < self.cfg.vocab);
        assert!(cache.len < self.cfg.context);
        let pos = cache.len;
        let mut state = self.emb[token * self.cfg.d_model..(token + 1) * self.cfg.d_model].to_vec();
        for (a, b) in state.iter_mut().zip(self.positional(pos)) {
            *a += b;
        }
        for layer_index in 0..self.cfg.layers {
            state = self.step_layer(layer_index, &state, &mut cache.layers[layer_index]);
        }
        cache.len += 1;
        state
    }

    pub fn logits(&self, hidden: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0; self.cfg.vocab];
        for (token, value) in out.iter_mut().enumerate() {
            let row = &self.emb[token * self.cfg.d_model..(token + 1) * self.cfg.d_model];
            *value = row.iter().zip(hidden).map(|(a, b)| a * b).sum();
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_tracks_sequence_length() {
        let model = crate::model::Model::new(Config::debug(), 42);
        let parameters = model.parameters();
        let runtime = RuntimeModel::from_parameters(Config::debug(), &parameters);
        let mut cache = runtime.new_cache();
        let hidden = runtime.prime(&[256, 65, 66], &mut cache);
        assert_eq!(hidden.len(), Config::debug().d_model);
        assert_eq!(cache.len, 3);
        let _ = runtime.next(67, &mut cache);
        assert_eq!(cache.len, 4);
    }

    #[test]
    fn direct_runtime_matches_reference_forward() {
        let cfg = Config::debug();
        let model = crate::model::Model::new(cfg, 123);
        let parameters = model.parameters();
        let runtime = RuntimeModel::from_parameters(cfg, &parameters);
        let tokens = [256, b'R' as usize, b'u' as usize, b's' as usize];
        let reference = model.forward_hidden(&tokens).data();
        let mut cache = runtime.new_cache();
        let actual = runtime.prime(&tokens, &mut cache);
        assert_eq!(reference.len(), actual.len());
        let max_error = reference
            .iter()
            .zip(actual)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_error < 1e-5, "runtime mismatch: max error {max_error}");
    }
}
