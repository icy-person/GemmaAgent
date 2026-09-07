pub const BOS: usize = 256;
pub const EOS: usize = 257;
pub const VOCAB_SIZE: usize = 258;

pub struct Tokenizer;

impl Tokenizer {
    pub fn new() -> Self {
        Self
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut tokens = Vec::with_capacity(text.len() + 2);
        tokens.push(BOS);
        tokens.extend(text.bytes().map(|byte| byte as usize));
        tokens.push(EOS);
        tokens
    }

    pub fn decode(&self, tokens: &[usize]) -> String {
        let bytes = tokens
            .iter()
            .filter_map(|&token| (token < 256).then_some(token as u8))
            .collect::<Vec<_>>();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

pub fn tiny_corpus() -> String {
    "Rust is a systems programming language. A small language model learns to predict the next token. Rust makes the engine fast and explicit. ".repeat(64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_round_trip() {
        let tokenizer = Tokenizer::new();
        let text = "Rust + UTF-8: café";
        let tokens = tokenizer.encode(text);
        assert_eq!(tokens.first().copied(), Some(BOS));
        assert_eq!(tokens.last().copied(), Some(EOS));
        assert_eq!(tokenizer.decode(&tokens), text);
    }

    #[test]
    fn vocabulary_is_target_compatible() {
        assert_eq!(VOCAB_SIZE, 258);
        assert!(BOS < VOCAB_SIZE && EOS < VOCAB_SIZE);
    }
}
