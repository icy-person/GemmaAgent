mod autograd;
mod config;
mod model;
mod tensor;
mod tokenizer;

use autograd::Value;
use config::Config;
use model::Model;
use tokenizer::ByteTokenizer;

fn argmax(values: &[f32]) -> usize {
    values.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i).unwrap_or(0)
}

fn main() {
    let config = Config::default();
    println!("gemma-rs: Rust language model foundation");
    println!("architecture: decoder-only Transformer");
    println!("layers: {} | d_model: {} | heads: {} | FFN: {}", config.n_layers, config.d_model, config.n_heads, config.ffn_dim);
    println!("context: {} tokens | vocabulary: {}", config.context, config.vocab_size);
    println!("parameters (approx): {:.2}M", config.parameter_count() as f64 / 1_000_000.0);

    let tokenizer = ByteTokenizer::new();
    let prompt = "Hello, Rust";
    let tokens = tokenizer.encode(prompt, true, false);
    println!("prompt: {prompt:?}");
    println!("tokens: {}", tokens.len());

    let model = Model::new(config, 42);
    let logits = model.forward(&tokens);
    let next = argmax(&logits);
    println!("next token id: {next}");

    let a = Value::leaf(1, 2, vec![2.0, 3.0]);
    let b = Value::leaf(2, 1, vec![5.0, 7.0]);
    let y = a.matmul(&b);
    y.backward();
    println!("autograd check: y={:?}, da={:?}, db={:?}", y.data(), a.grad(), b.grad());
}
