mod autograd;
mod checkpoint;
mod config;
mod model;
mod optim;
mod runtime;
mod tokenizer;

use autograd::Value;
use config::Config;
use model::Model;
use optim::AdamW;
use runtime::RuntimeModel;
use std::time::Instant;
use tokenizer::Tokenizer;

fn argmax(values: &[f32]) -> usize { values.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(index, _)| index).unwrap_or(0) }
fn parse_float_arg(args: &[String], name: &str, default: f32) -> f32 { args.iter().position(|arg| arg == name).and_then(|i| args.get(i + 1)).and_then(|x| x.parse().ok()).unwrap_or(default) }
fn parse_usize_arg(args: &[String], name: &str, default: usize) -> usize { args.iter().position(|arg| arg == name).and_then(|i| args.get(i + 1)).and_then(|x| x.parse().ok()).unwrap_or(default) }
fn parse_string_arg<'a>(args: &'a [String], name: &str) -> Option<&'a str> { args.iter().position(|arg| arg == name).and_then(|i| args.get(i + 1)).map(String::as_str) }

fn cross_entropy_from_hidden(model: &Model, hidden: &Value, target: usize) -> Value { model.logits(hidden).softmax().gather(target).log().neg() }

fn cross_entropy(model: &Model, input: &[usize], target: usize, targets_per_step: usize, rng: &mut u64) -> Value {
    assert!(input.len() >= 2 && targets_per_step > 0);
    let hidden = model.forward_all_hidden(input);
    let usable = input.len() - 1;
    let count = targets_per_step.min(usable).max(1);
    let mut total = cross_entropy_from_hidden(model, &hidden[0], input[1]);
    let mut selected = 1usize;
    for _ in 1..count {
        let pos = (xorshift64(rng) as usize) % usable;
        total = total.add(&cross_entropy_from_hidden(model, &hidden[pos], input[pos + 1]));
        selected += 1;
    }
    total.add(&cross_entropy_from_hidden(model, hidden.last().expect("non-empty hidden sequence"), target)).div_scalar(selected as f32)
}

fn config_from_args(args: &[String]) -> Config { if args.iter().any(|arg| arg == "--target") { Config::target() } else { Config::debug() } }
fn xorshift64(state: &mut u64) -> u64 { let mut x = *state; x ^= x << 13; x ^= x >> 7; x ^= x << 17; *state = x; x }
fn uniform01(state: &mut u64) -> f32 { ((xorshift64(state) >> 40) as f32) / ((1u64 << 24) - 1) as f32 }

fn sample_token(logits: &[f32], temperature: f32, top_k: usize, rng_state: &mut u64) -> usize {
    assert!(temperature.is_finite() && temperature > 0.0);
    let mut indices: Vec<usize> = (0..logits.len()).collect();
    indices.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    if top_k > 0 && top_k < indices.len() { indices.truncate(top_k); }
    let max_logit = indices.iter().map(|&i| logits[i] / temperature).fold(f32::NEG_INFINITY, f32::max);
    let mut weights = Vec::with_capacity(indices.len()); let mut total = 0.0;
    for &index in &indices { let weight = (logits[index] / temperature - max_logit).exp(); weights.push(weight); total += weight; }
    if !total.is_finite() || total <= 0.0 { return indices[0]; }
    let threshold = uniform01(rng_state) * total; let mut cumulative = 0.0;
    for (index, weight) in indices.iter().zip(weights) { cumulative += weight; if cumulative >= threshold { return *index; } }
    *indices.last().expect("non-empty candidate list")
}

fn train(steps: usize, path: &str, cfg: Config, checkpoint_every: usize, grad_accum: usize, data_path: Option<&str>, targets_per_step: usize, lr: Option<f32>) {
    cfg.validate(); assert!(grad_accum > 0 && targets_per_step > 0);
    let corpus = match data_path { Some(path) => std::fs::read_to_string(path).unwrap_or_else(|e| panic!("failed to read training data '{path}': {e}")), None => Tokenizer::tiny_corpus() };
    let tokenizer = if cfg.vocab == 258 { Tokenizer::new() } else { Tokenizer::train(&corpus, cfg.vocab) };
    let encoded = tokenizer.encode(&corpus); assert!(encoded.len() > cfg.context + 1);
    println!("GemmaAgent: {} params | context {} | {} heads | {:.2} MiB fp32", cfg.params(), cfg.context, cfg.heads, cfg.approx_parameter_memory_mb());
    println!("training corpus: {} bytes{}", corpus.len(), data_path.map(|p| format!(" from {p}")).unwrap_or_default());
    let lr = lr.unwrap_or(if cfg == Config::target() { 0.0003 } else { 0.001 });
    let model = Model::new(cfg, 42); let parameters = model.parameters(); let mut optimizer = AdamW::new(lr);
    let mut rng = 0x00C0_FFEE_2026_0907_u64 ^ encoded.len() as u64;
    let mut completed_updates = 0usize; let mut accumulated = 0usize; let mut loss_sum = 0.0f32;
    for step in 1..=steps {
        let start = (xorshift64(&mut rng) as usize) % (encoded.len() - cfg.context);
        let loss = cross_entropy(&model, &encoded[start..start + cfg.context], encoded[start + cfg.context], targets_per_step, &mut rng);
        loss_sum += loss.data()[0]; accumulated += 1;
        let divisor = if step == steps { accumulated } else { grad_accum }; loss.div_scalar(divisor as f32).backward();
        if accumulated == grad_accum || step == steps { optimizer.step(&parameters); completed_updates += 1; println!("update={completed_updates} train_loss={:.6}", loss_sum / accumulated as f32); accumulated = 0; loss_sum = 0.0; }
        if checkpoint_every > 0 && step % checkpoint_every == 0 { checkpoint::save_full(path, &parameters, &optimizer, completed_updates, rng).expect("failed to save checkpoint"); tokenizer.save(format!("{path}.tok")).expect("failed to save tokenizer"); }
    }
    checkpoint::save_full(path, &parameters, &optimizer, completed_updates, rng).expect("failed to save checkpoint"); tokenizer.save(format!("{path}.tok")).expect("failed to save tokenizer");
}

fn infer(path: &str, prompt: &str, cfg: Config, max_new_tokens: usize, temperature: f32, top_k: usize) {
    cfg.validate();
    let tok_path = format!("{path}.tok");
    let tokenizer = if cfg.vocab == 258 { Tokenizer::new() } else { Tokenizer::load(&tok_path).unwrap_or_else(|e| panic!("failed to load tokenizer {tok_path}: {e}")) };
    let autograd_model = Model::new(cfg, 42); let parameters = autograd_model.parameters();
    if !std::path::Path::new(path).exists() { eprintln!("checkpoint not found: {path}"); std::process::exit(2); }
    checkpoint::load(path, &parameters).expect("failed to load checkpoint"); let runtime = RuntimeModel::from_parameters(cfg, &parameters);
    let mut cache = runtime.new_cache(); let mut tokens = tokenizer.encode(prompt); let _ = tokens.pop(); assert!(!tokens.is_empty() && tokens.len() <= cfg.context);
    let mut logits = runtime.logits(&runtime.prime(&tokens, &mut cache)); let mut rng_state = 0x9E37_79B9_7F4A_7C15u64;
    for _ in 0..max_new_tokens {
        let next = if temperature <= 1e-6 { argmax(&logits) } else { sample_token(&logits, temperature, top_k, &mut rng_state) };
        tokens.push(next); if next == tokenizer.eos() || tokens.len() >= cfg.context { break; }
        logits = runtime.logits(&runtime.next(next, &mut cache));
    }
    println!("{}", tokenizer.decode(&tokens));
}

fn bench(cfg: Config, prompt_tokens: usize, generated_tokens: usize) {
    assert!(prompt_tokens > 0 && generated_tokens > 0 && prompt_tokens + generated_tokens <= cfg.context);
    let model = Model::new(cfg, 42); let runtime = RuntimeModel::from_parameters(cfg, &model.parameters()); let tokens: Vec<usize> = (0..prompt_tokens).map(|i| 65 + (i % 26)).collect();
    let mut cache = runtime.new_cache(); let start = Instant::now(); let mut hidden = runtime.prime(&tokens, &mut cache); let prefill_seconds = start.elapsed().as_secs_f64(); let decode_start = Instant::now(); let mut token = argmax(&runtime.logits(&hidden));
    for _ in 0..generated_tokens { hidden = runtime.next(token, &mut cache); token = argmax(&runtime.logits(&hidden)); }
    let decode_seconds = decode_start.elapsed().as_secs_f64(); println!("benchmark: {} params | prompt {} | generated {}", cfg.params(), prompt_tokens, generated_tokens); println!("prefill: {:.3}s | {:.2} tok/s", prefill_seconds, prompt_tokens as f64 / prefill_seconds.max(f64::MIN_POSITIVE)); println!("decode: {:.3}s | {:.2} tok/s", decode_seconds, generated_tokens as f64 / decode_seconds.max(f64::MIN_POSITIVE));
}

fn print_usage() { println!("GemmaAgent Rust LLM\n\nCommands:\n  cargo test\n  cargo run --release -- train 300 [checkpoint] [--target] [--data FILE] [--checkpoint-every N] [--grad-accum N] [--targets-per-step N] [--lr LR]\n  cargo run --release -- infer [checkpoint] [prompt] [--target] [--tokens N] [--temperature T] [--top-k K]\n  cargo run --release -- bench [--target] [--prompt-tokens N] [--tokens N]"); }

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("train") => { let steps = args.get(2).and_then(|x| x.parse().ok()).unwrap_or(300); let path = args.get(3).map(String::as_str).unwrap_or("gemma-agent.ckpt"); train(steps, path, config_from_args(&args), parse_usize_arg(&args, "--checkpoint-every", 0), parse_usize_arg(&args, "--grad-accum", 1), parse_string_arg(&args, "--data"), parse_usize_arg(&args, "--targets-per-step", 128), Some(parse_float_arg(&args, "--lr", 0.0003))); }
        Some("infer") => { let path = args.get(2).map(String::as_str).unwrap_or("gemma-agent.ckpt"); let prompt = args.get(3).map(String::as_str).unwrap_or("Rust is"); infer(path, prompt, config_from_args(&args), parse_usize_arg(&args, "--tokens", 64), parse_float_arg(&args, "--temperature", 0.0), parse_usize_arg(&args, "--top-k", 0)); }
        Some("bench") => bench(config_from_args(&args), parse_usize_arg(&args, "--prompt-tokens", 32), parse_usize_arg(&args, "--tokens", 32)),
        _ => print_usage(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn cross_entropy_is_finite() { let cfg = Config::debug(); let tokenizer = Tokenizer::new(); let ids = tokenizer.encode("Rust"); let model = Model::new(cfg, 42); let mut rng = 7; let loss = cross_entropy(&model, &ids[..ids.len() - 1], b'!' as usize, 4, &mut rng); assert!(loss.data()[0].is_finite()); }
    #[test] fn sampler_respects_top_k() { let logits = [0., 1., 2., 3.]; let mut rng = 7; for _ in 0..64 { let token = sample_token(&logits, 1., 2, &mut rng); assert!(token == 2 || token == 3); } }
}
