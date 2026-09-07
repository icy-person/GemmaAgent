use std::{collections::HashMap, fs, path::Path};

pub const BOS: usize = 256;
pub const EOS: usize = 257;
const BYTE_VOCAB: usize = 256;
const FIRST_LEARNED_ID: usize = 256;

#[derive(Clone, Debug)]
pub struct Tokenizer {
    vocab_size: usize,
    bos: usize,
    eos: usize,
    pieces: Vec<Vec<u8>>,
    by_first: Vec<Vec<usize>>,
}

impl Tokenizer {
    pub fn new() -> Self { Self::for_vocab(258) }
    pub fn for_vocab(vocab_size: usize) -> Self {
        assert!(vocab_size >= 258, "vocab must contain 256 byte ids plus BOS/EOS");
        let bos = vocab_size - 2;
        let eos = vocab_size - 1;
        Self { vocab_size, bos, eos, pieces: vec![Vec::new(); vocab_size], by_first: (0..BYTE_VOCAB).map(|_| Vec::new()).collect() }
    }
    pub fn train(text: &str, vocab_size: usize) -> Self {
        let mut tokenizer = Self::for_vocab(vocab_size);
        let learned_limit = vocab_size.saturating_sub(BYTE_VOCAB + 2);
        if learned_limit == 0 { return tokenizer; }
        let mut counts: HashMap<Vec<u8>, u32> = HashMap::new();
        for word in text.split_whitespace() {
            let bytes = word.as_bytes();
            if bytes.len() >= 2 {
                *counts.entry(bytes.to_vec()).or_default() += 1;
                let mut spaced = Vec::with_capacity(bytes.len() + 1); spaced.push(b' '); spaced.extend_from_slice(bytes);
                *counts.entry(spaced).or_default() += 1;
            }
            let max_piece = bytes.len().min(12);
            for start in 0..bytes.len() {
                for len in 2..=max_piece.min(bytes.len() - start) { *counts.entry(bytes[start..start + len].to_vec()).or_default() += 1; }
            }
        }
        let raw = text.as_bytes();
        for start in 0..raw.len() {
            for len in 2..=4.min(raw.len() - start) {
                let piece = &raw[start..start + len];
                if piece.iter().all(|b| !b.is_ascii_control()) { *counts.entry(piece.to_vec()).or_default() += 1; }
            }
        }
        let mut candidates: Vec<(Vec<u8>, u64)> = counts.into_iter().filter(|(piece, count)| piece.len() >= 2 && *count >= 2 && piece.len() <= 64).map(|(piece, count)| (piece, u64::from(count) * piece.len().saturating_sub(1) as u64)).collect();
        candidates.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| b.0.len().cmp(&a.0.len())).then_with(|| a.0.cmp(&b.0)));
        candidates.truncate(learned_limit);
        for (offset, (piece, _)) in candidates.into_iter().enumerate() { tokenizer.pieces[FIRST_LEARNED_ID + offset] = piece; }
        tokenizer.rebuild_index(); tokenizer
    }
    fn rebuild_index(&mut self) {
        self.by_first = (0..BYTE_VOCAB).map(|_| Vec::new()).collect();
        for id in FIRST_LEARNED_ID..self.vocab_size.saturating_sub(2) {
            if self.pieces[id].is_empty() { continue; }
            self.by_first[self.pieces[id][0] as usize].push(id);
        }
        for ids in &mut self.by_first { ids.sort_unstable_by(|&a, &b| self.pieces[b].len().cmp(&self.pieces[a].len()).then_with(|| a.cmp(&b))); }
    }
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let bytes = text.as_bytes(); let mut tokens = Vec::with_capacity(bytes.len() / 2 + 2); tokens.push(self.bos); let mut pos = 0usize;
        while pos < bytes.len() {
            let first = bytes[pos] as usize; let mut matched = None;
            for &id in &self.by_first[first] { let piece = &self.pieces[id]; if piece.len() <= bytes.len() - pos && bytes[pos..pos + piece.len()] == piece[..] { matched = Some(id); break; } }
            if let Some(id) = matched { tokens.push(id); pos += self.pieces[id].len(); } else { tokens.push(bytes[pos] as usize); pos += 1; }
        }
        tokens.push(self.eos); tokens
    }
    pub fn decode(&self, tokens: &[usize]) -> String {
        let mut bytes = Vec::new();
        for &token in tokens { if token < BYTE_VOCAB { bytes.push(token as u8); } else if token != self.bos && token != self.eos && token < self.pieces.len() { bytes.extend_from_slice(&self.pieces[token]); } }
        String::from_utf8_lossy(&bytes).into_owned()
    }
    pub fn save<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let mut out = format!("GATOK1\n{}\n", self.vocab_size);
        for id in FIRST_LEARNED_ID..self.vocab_size.saturating_sub(2) {
            if self.pieces[id].is_empty() { continue; }
            out.push_str(&id.to_string()); out.push(' ');
            for byte in &self.pieces[id] { out.push_str(&format!("{byte:02x}")); }
            out.push('\n');
        }
        fs::write(path, out)
    }
    pub fn load<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let text = fs::read_to_string(path)?; let mut lines = text.lines();
        if lines.next() != Some("GATOK1") { return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid tokenizer header")); }
        let vocab_size: usize = lines.next().ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "missing vocab size"))?.parse().map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid vocab size"))?;
        if vocab_size < 258 { return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "tokenizer vocab is too small")); }
        let mut tokenizer = Self::for_vocab(vocab_size);
        for line in lines.filter(|line| !line.trim().is_empty()) {
            let (id_text, hex) = line.split_once(' ').ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "malformed tokenizer entry"))?;
            let id: usize = id_text.parse().map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid token id"))?;
            if id < FIRST_LEARNED_ID || id >= vocab_size.saturating_sub(2) || hex.len() % 2 != 0 || hex.is_empty() { return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "token id out of range")); }
            let mut bytes = Vec::with_capacity(hex.len() / 2);
            for chunk in hex.as_bytes().chunks(2) {
                let s = std::str::from_utf8(chunk).map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid token bytes"))?;
                bytes.push(u8::from_str_radix(s, 16).map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid token hex"))?);
            }
            tokenizer.pieces[id] = bytes;
        }
        tokenizer.rebuild_index(); Ok(tokenizer)
    }
    pub fn vocab_size(&self) -> usize { self.vocab_size }
    pub fn bos(&self) -> usize { self.bos }
    pub fn eos(&self) -> usize { self.eos }
    pub fn tiny_corpus() -> String { "Rust is a systems programming language. A small language model learns to predict the next token. Rust makes the engine fast and explicit. ".repeat(64) }
}

#[cfg(test)]
mod tests {
    use super::*; use std::{env, fs};
    #[test] fn byte_round_trip() { let tokenizer = Tokenizer::new(); let text = "Rust + UTF-8: café"; let tokens = tokenizer.encode(text); assert_eq!(tokens.first().copied(), Some(BOS)); assert_eq!(tokens.last().copied(), Some(EOS)); assert_eq!(tokenizer.decode(&tokens), text); }
    #[test] fn learned_tokens_reduce_sequence_length() { let tokenizer = Tokenizer::train("hello world hello world hello world", 1024); let tokens = tokenizer.encode("hello world hello world"); assert!(tokens.len() < "hello world hello world".len() + 2); assert_eq!(tokenizer.decode(&tokens), "hello world hello world"); }
    #[test] fn tokenizer_round_trip_survives_save_load() { let path = env::temp_dir().join(format!("gemma-agent-tokenizer-{}.tok", std::process::id())); let tokenizer = Tokenizer::train("hello world hello Rust", 1024); tokenizer.save(&path).unwrap(); let loaded = Tokenizer::load(&path).unwrap(); let text = "Rust + UTF-8: café\nhello world"; assert_eq!(loaded.decode(&loaded.encode(text)), text); assert_eq!(loaded.vocab_size(), 1024); fs::remove_file(path).ok(); }
    #[test] fn special_tokens_are_last_two_ids() { let tokenizer = Tokenizer::for_vocab(16_384); assert_eq!(tokenizer.bos(), 16_382); assert_eq!(tokenizer.eos(), 16_383); }
}
