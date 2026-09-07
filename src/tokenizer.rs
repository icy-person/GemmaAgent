pub const BOS: usize = 256;
pub const EOS: usize = 257;
pub const VOCAB_USED: usize = 258;

#[derive(Default)]
pub struct ByteTokenizer;

impl ByteTokenizer {
    pub fn new() -> Self { Self }

    pub fn encode(&self, text: &str, add_bos: bool, add_eos: bool) -> Vec<usize> {
        let mut ids = Vec::with_capacity(text.len() + 2);
        if add_bos { ids.push(BOS); }
        ids.extend(text.as_bytes().iter().map(|&b| b as usize));
        if add_eos { ids.push(EOS); }
        ids
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        let bytes: Vec<u8> = ids.iter()
            .filter_map(|&id| if id < 256 { Some(id as u8) } else { None })
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}
