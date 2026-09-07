#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Config {
    pub vocab: usize,
    pub context: usize,
    pub d_model: usize,
    pub layers: usize,
    pub heads: usize,
    pub ffn: usize,
}

impl Config {
    pub fn debug() -> Self {
        Self {
            vocab: 258,
            context: 128,
            d_model: 64,
            layers: 2,
            heads: 4,
            ffn: 128,
        }
    }

    pub fn target() -> Self {
        Self {
            vocab: 16_384,
            context: 1_024,
            d_model: 416,
            layers: 6,
            heads: 8,
            ffn: 1_664,
        }
    }

    pub fn head_dim(&self) -> usize {
        assert_eq!(
            self.d_model % self.heads,
            0,
            "d_model must divide evenly across heads"
        );
        self.d_model / self.heads
    }

    pub fn params(&self) -> usize {
        self.vocab * self.d_model
            + self.layers * (4 * self.d_model * self.d_model + 2 * self.d_model * self.ffn)
    }

    pub fn validate(&self) {
        assert!(self.vocab >= 2, "vocab must contain at least BOS and EOS");
        assert!(self.context > 0, "context must be non-zero");
        assert!(self.d_model > 0 && self.layers > 0 && self.heads > 0 && self.ffn > 0);
        assert_eq!(
            self.d_model % self.heads,
            0,
            "d_model must divide evenly across heads"
        );
    }

    pub fn approx_parameter_memory_mb(&self) -> f32 {
        self.params() as f32 * 4.0 / (1024.0 * 1024.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_profile_matches_documented_shape() {
        let c = Config::target();
        c.validate();
        assert_eq!(
            (c.context, c.heads, c.d_model, c.layers, c.ffn),
            (1024, 8, 416, 6, 1664)
        );
        assert_eq!(c.params(), 19_275_776);
        assert!((c.approx_parameter_memory_mb() - 73.530).abs() < 0.01);
    }
}
