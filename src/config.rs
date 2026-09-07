#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub vocab_size: usize,
    pub context: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub ffn_dim: usize,
    pub rope_theta: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            vocab_size: 16_384,
            context: 1_024,
            d_model: 416,
            n_layers: 6,
            n_heads: 8,
            ffn_dim: 1_664,
            rope_theta: 10_000.0,
        }
    }
}

impl Config {
    pub fn head_dim(&self) -> usize { self.d_model / self.n_heads }

    pub fn parameter_count(&self) -> usize {
        let embedding = self.vocab_size * self.d_model;
        let per_layer =
            3 * self.d_model * self.d_model +
            self.d_model * self.d_model +
            3 * self.d_model +
            self.d_model * self.ffn_dim +
            self.ffn_dim * self.d_model +
            2 * self.d_model;
        embedding + self.n_layers * per_layer
    }
}
