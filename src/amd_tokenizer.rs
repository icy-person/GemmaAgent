use std::{collections::HashMap, fs, path::Path};

pub const BOS: usize = 16_382;
pub const EOS: usize = 16_383;
const FIRST_LEARNED_ID: usize = 256;

#[derive(Clone, Debug)]
pub struct AmdTokenizer {
    vocab_size: usize,
    pieces: Vec<Vec<u8>>,
    by_first: Vec<Vec<usize>>,
}

impl AmdTokenizer {
    pub fn train(text: &str, vocab_size: usize) -> Self {
        assert!(
            vocab_size >= 258,
            "vocab must leave room for byte + special tokens"
        );

        let mut counts: HashMap<Vec<u8>, u32> = HashMap::new();
        for word in text.split_whitespace() {
            let bytes = word.as_bytes();
            if bytes.len() >= 2 {
                *counts.entry(bytes.to_vec()).or_default() += 1;
                let mut prefixed = Vec::with_capacity(bytes.len() + 1);
                prefixed.push(b' ');
                prefixed.extend_from_slice(bytes);
                *counts.entry(prefixed).or_default() += 1;
            }

            let max_piece = bytes.len().min(8);
            for start in 0..bytes.len() {
                for len in 2..=max_piece.min(bytes.len() - start) {
                    let piece = &bytes[start..start + len];
                    *counts.entry(piece.to_vec()).or_default() += 1;
                }
            }
        }

        // Common whitespace/punctuation fragments improve fallback efficiency without
        // requiring a full BPE implementation during startup.
        let raw = text.as_bytes();
        let max_raw_piece = 4usize;
        for start in 0..raw.len() {
            for len in 2..=max_raw_piece.min(raw.len() - start) {
                let piece = &raw[start..start + len];
                if piece.iter().all(|b| !b.is_ascii_control()) {
                    *counts.entry(piece.to_vec()).or_default() += 1;
                }
            }
        }

        let learned_limit = vocab_size.saturating_sub(258);
        let mut candidates: Vec<(Vec<u8>, u64)> = counts
            .into_iter()
            .filter(|(piece, count)| piece.len() >= 2 && *count >= 2 && piece.len() <= 64)
            .map(|(piece, count)| {
                let score = u64::from(count) * (piece.len().saturating_sub(1) as u64);
                (piece, score)
            })
            .collect();
        candidates.sort_unstable_by(|a, b| {
            b.1.cmp(&a.1)
                .then_with(|| b.0.len().cmp(&a.0.len()))
                .then_with(|| a.0.cmp(&b.0))
        });
        candidates.truncate(learned_limit);

        let mut pieces = vec![Vec::<u8>::new(); vocab_size];
        for (offset, (piece, _)) in candidates.into_iter().enumerate() {
            pieces[FIRST_LEARNED_ID + offset] = piece;
        }

        let mut by_first: Vec<Vec<usize>> = (0..256).map(|_| Vec::new()).collect();
        for id in FIRST_LEARNED_ID..vocab_size.saturating_sub(2) {
            if pieces[id].is_empty() {
                continue;
            }
            by_first[pieces[id][0] as usize].push(id);
        }
        for ids in &mut by_first {
            ids.sort_unstable_by(|&a, &b| {
                pieces[b]
                    .len()
                    .cmp(&pieces[a].len())
                    .then_with(|| a.cmp(&b))
            });
        }

        Self {
            vocab_size,
            pieces,
            by_first,
        }
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        let bytes = text.as_bytes();
        let mut tokens = Vec::with_capacity(bytes.len() / 2 + 2);
        tokens.push(BOS);
        let mut pos = 0usize;
        while pos < bytes.len() {
            let first = bytes[pos] as usize;
            let mut matched = None;
            for &id in &self.by_first[first] {
                let piece = &self.pieces[id];
                if piece.len() <= bytes.len() - pos && bytes[pos..pos + piece.len()] == piece[..] {
                    matched = Some(id);
                    break;
                }
            }
            if let Some(id) = matched {
                tokens.push(id);
                pos += self.pieces[id].len();
            } else {
                tokens.push(bytes[pos] as usize);
                pos += 1;
            }
        }
        tokens.push(EOS);
        tokens
    }

    pub fn decode(&self, tokens: &[usize]) -> String {
        let mut bytes = Vec::new();
        for &token in tokens {
            if token < 256 {
                bytes.push(token as u8);
            } else if token != BOS && token != EOS && token < self.pieces.len() {
                bytes.extend_from_slice(&self.pieces[token]);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let mut out = format!("AMDTOK1\n{}\n", self.vocab_size);
        for id in FIRST_LEARNED_ID..self.vocab_size.saturating_sub(2) {
            if self.pieces[id].is_empty() {
                continue;
            }
            out.push_str(&id.to_string());
            out.push(' ');
            for byte in &self.pieces[id] {
                out.push_str(&format!("{byte:02x}"));
            }
            out.push('\n');
        }
        fs::write(path, out)
    }

    pub fn load<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let text = fs::read_to_string(path)?;
        let mut lines = text.lines();
        if lines.next() != Some("AMDTOK1") {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "invalid AMD tokenizer header",
            ));
        }
        let vocab_size: usize = lines
            .next()
            .ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "missing vocab size")
            })?
            .parse()
            .map_err(|_| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid vocab size")
            })?;
        if vocab_size < 258 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "AMD tokenizer vocab is too small",
            ));
        }
        let mut pieces = vec![Vec::<u8>::new(); vocab_size];
        for line in lines.filter(|line| !line.trim().is_empty()) {
            let (id_text, hex) = line.split_once(' ').ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "malformed tokenizer entry")
            })?;
            let id: usize = id_text.parse().map_err(|_| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid token id")
            })?;
            if id < FIRST_LEARNED_ID || id >= vocab_size.saturating_sub(2) || hex.len() % 2 != 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "token id out of range",
                ));
            }
            let mut bytes = Vec::with_capacity(hex.len() / 2);
            for chunk in hex.as_bytes().chunks_exact(2) {
                let s = std::str::from_utf8(chunk).map_err(|_| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid token bytes")
                })?;
                let byte = u8::from_str_radix(s, 16).map_err(|_| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid token hex")
                })?;
                bytes.push(byte);
            }
            pieces[id] = bytes;
        }

        let mut by_first: Vec<Vec<usize>> = (0..256).map(|_| Vec::new()).collect();
        for id in FIRST_LEARNED_ID..vocab_size.saturating_sub(2) {
            if pieces[id].is_empty() {
                continue;
            }
            by_first[pieces[id][0] as usize].push(id);
        }
        for ids in &mut by_first {
            ids.sort_unstable_by(|&a, &b| {
                pieces[b]
                    .len()
                    .cmp(&pieces[a].len())
                    .then_with(|| a.cmp(&b))
            });
        }

        Ok(Self {
            vocab_size,
            pieces,
            by_first,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_is_exact() {
        let tokenizer = AmdTokenizer::train("hello world hello Rust", 1024);
        let text = "Rust + UTF-8: café\nhello world";
        let encoded = tokenizer.encode(text);
        assert_eq!(tokenizer.decode(&encoded), text);
    }

    #[test]
    fn ids_fit_vocab() {
        let tokenizer = AmdTokenizer::train("a b c Rust language model", 512);
        let encoded = tokenizer.encode("Rust model");
        assert!(encoded.into_iter().all(|id| id < tokenizer.vocab_size()));
    }
}
