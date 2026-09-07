pub const BOS:usize=256;pub const EOS:usize=257;
pub struct Tokenizer;
impl Tokenizer{pub fn new()->Self{Self}pub fn encode(&self,s:&str)->Vec<usize>{let mut v=Vec::with_capacity(s.len()+2);v.push(BOS);v.extend(s.bytes().map(|b|b as usize));v.push(EOS);v}pub fn decode(&self,t:&[usize])->String{String::from_utf8_lossy(&t.iter().filter_map(|&x|if x<256{Some(x as u8)}else{None}).collect::<Vec<_>>()).into_owned()}}
pub fn tiny_corpus()->String{"Rust is a systems programming language. A small language model learns to predict the next token. Rust makes the engine fast and explicit. ".repeat(64)}
