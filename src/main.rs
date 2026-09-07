mod autograd;
mod checkpoint;
mod config;
mod model;
mod optim;
mod tokenizer;

use autograd::Value;
use config::Config;
use model::Model;
use optim::AdamW;
use tokenizer::Tokenizer;

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(index, _)| index)
        .unwrap_or(0)
}

fn cross_entropy(model: &Model, input: &[usize], target: usize) -> Value {
    let hidden = model.forward_hidden(input);
    let probabilities = model.logits(&hidden).softmax();
    probabilities.gather(target).log().neg()
}

fn config_from_args(args: &[String]) -> Config {
    if args.iter().any(|arg| arg == "--target") {
        Config::target()
    } else {
        Config::debug()
    }
}

fn train(steps: usize, path: &str, cfg: Config, checkpoint_every: usize) {
    cfg.validate();
    let tokenizer = Tokenizer::new();
    let encoded = tokenizer.encode(&tokenizer::tiny_corpus());
    assert!(
        encoded.len() > cfg.context + 1,
        "training corpus is shorter than the configured context"
    );

    println!(
        "GemmaAgent: {} params | context {} | {} heads | {:.2} MiB fp32",
        cfg.params(),
        cfg.context,
        cfg.heads,
        cfg.approx_parameter_memory_mb()
    );
    if cfg == Config::target() {
        println!("target profile selected; scalar CPU training is intentionally slow");
    }
    if checkpoint_every > 0 {
        println!("periodic checkpoints: every {checkpoint_every} steps -> {path}");
    }

    let model = Model::new(cfg, 42);
    let parameters = model.parameters();
    let mut optimizer = AdamW::new(if cfg == Config::target() {
        0.0005
    } else {
        0.002
    });

    let window_count = encoded.len() - cfg.context;
    for step in 1..=steps {
        let start = (step - 1) % window_count;
        let loss = cross_entropy(
            &model,
            &encoded[start..start + cfg.context],
            encoded[start + cfg.context],
        );
        let value = loss.data()[0];
        loss.backward();
        optimizer.step(&parameters);

        if step == 1 || step % 25 == 0 || step == steps {
            println!("step {step:4} loss {value:.5}");
        }
        if checkpoint_every > 0 && step < steps && step % checkpoint_every == 0 {
            checkpoint::save(path, &parameters).expect("failed to save periodic checkpoint");
            println!("checkpoint: {path} (step {step})");
        }
    }

    checkpoint::save(path, &parameters).expect("failed to save checkpoint");
    println!("checkpoint: {path}");
}

fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn uniform01(state: &mut u64) -> f32 {
    ((xorshift64(state) >> 40) as f32) / ((1u64 << 24) - 1) as f32
}

fn sample_token(logits: &[f32], temperature: f32, top_k: usize, rng_state: &mut u64) -> usize {
    assert!(temperature.is_finite() && temperature > 0.0);
    if temperature <= 1e-6 {
        return argmax(logits);
    }

    let mut indices: Vec<usize> = (0..logits.len()).collect();
    indices.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    if top_k > 0 && top_k < indices.len() {
        indices.truncate(top_k);
    }

    let max_logit = indices
        .iter()
        .map(|&i| logits[i] / temperature)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut weights = Vec::with_capacity(indices.len());
    let mut total = 0.0f32;
    for &index in &indices {
        let weight = (logits[index] / temperature - max_logit).exp();
        weights.push(weight);
        total += weight;
    }
    if !total.is_finite() || total <= 0.0 {
        return indices[0];
    }

    let threshold = uniform01(rng_state) * total;
    let mut cumulative = 0.0;
    for (index, weight) in indices.iter().zip(weights) {
        cumulative += weight;
        if cumulative >= threshold {
            return *index;
        }
    }
    *indices.last().unwrap()
}

fn parse_float_arg(args: &[String], name: &str, default: f32) -> f32 {
    args.iter()
        .position(|arg| arg == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|x| x.parse().ok())
        .unwrap_or(default)
}

fn parse_usize_arg(args: &[String], name: &str, default: usize) -> usize {
    args.iter()
        .position(|arg| arg == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|x| x.parse().ok())
        .unwrap_or(default)
}

fn infer(
    path: &str,
    prompt: &str,
    cfg: Config,
    max_new_tokens: usize,
    temperature: f32,
    top_k: usize,
) {
    cfg.validate();
    let tokenizer = Tokenizer::new();
    let model = Model::new(cfg, 42);
    let parameters = model.parameters();

    if !std::path::Path::new(path).exists() {
        eprintln!("checkpoint not found: {path}");
        eprintln!("train first, for example: cargo run --release -- train 300 {path}");
        std::process::exit(2);
    }
    checkpoint::load(path, &parameters).expect("failed to load checkpoint");

    let mut tokens = tokenizer.encode(prompt);
    let _ = tokens.pop();
    let mut rng_state = 0x9E37_79B9_7F4A_7C15u64;
    for _ in 0..max_new_tokens {
        let start = tokens.len().saturating_sub(cfg.context);
        let hidden = model.forward_hidden(&tokens[start..]);
        let logits = model.logits(&hidden).data();
        let next = if temperature <= 1e-6 {
            argmax(&logits)
        } else {
            sample_token(&logits, temperature, top_k, &mut rng_state)
        };
        tokens.push(next);
        if next == tokenizer::EOS {
            break;
        }
    }
    println!("{}", tokenizer.decode(&tokens));
}

fn print_usage() {
    println!("GemmaAgent Rust LLM");
    println!("\nCommands:");
    println!("  cargo test");
    println!("  cargo run --release -- train 300 [checkpoint] [--target] [--checkpoint-every N]");
    println!(
        "  cargo run --release -- infer [checkpoint] [prompt] [--target] [--tokens N] [--temperature T] [--top-k K]"
    );
    println!("\nDefault training profile is the small CPU-debug model.");
    println!("Use --target for the 19,275,776-parameter / context=1024 / 8-head profile.");
    println!("Periodic checkpointing is disabled by default; set --checkpoint-every 100 for long runs.");
    println!("Inference defaults to greedy decoding (temperature <= 0.000001). Set --temperature 0.8 for sampling.");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("train") => {
            let steps = args.get(2).and_then(|x| x.parse().ok()).unwrap_or(300);
            let path = args
                .get(3)
                .map(String::as_str)
                .unwrap_or("gemma-agent.ckpt");
            let checkpoint_every = parse_usize_arg(&args, "--checkpoint-every", 0);
            train(steps, path, config_from_args(&args), checkpoint_every);
        }
        Some("infer") => {
            let path = args
                .get(2)
                .map(String::as_str)
                .unwrap_or("gemma-agent.ckpt");
            let prompt = args.get(3).map(String::as_str).unwrap_or("Rust is");
            let max_new = parse_usize_arg(&args, "--tokens", 64);
            let temperature = parse_float_arg(&args, "--temperature", 0.0);
            let top_k = parse_usize_arg(&args, "--top-k", 0);
            infer(
                path,
                prompt,
                config_from_args(&args),
                max_new,
                temperature,
                top_k,
            );
        }
        _ => print_usage(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cross_entropy_is_finite() {
        let cfg = Config::debug();
        let tokenizer = Tokenizer::new();
        let ids = tokenizer.encode("Rust");
        let model = Model::new(cfg, 42);
        let loss = cross_entropy(&model, &ids[..ids.len() - 1], b'!' as usize);
        assert!(loss.data()[0].is_finite());
    }

    #[test]
    fn sampler_respects_top_k() {
        let logits = [0.0, 1.0, 2.0, 3.0];
        let mut rng = 7;
        for _ in 0..64 {
            let token = sample_token(&logits, 1.0, 2, &mut rng);
            assert!(token == 2 || token == 3);
        }
    }

    #[test]
    fn greedy_sampling_matches_argmax() {
        let logits = [0.0, -1.0, 4.0, 2.0];
        let mut rng = 1;
        assert_eq!(sample_token(&logits, 1e-7, 0, &mut rng), 2);
    }
}
