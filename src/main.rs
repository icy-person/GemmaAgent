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

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(index, _)| index)
        .unwrap_or(0)
}

fn cross_entropy_from_hidden(model: &Model, hidden: &Value, target: usize) -> Value {
    let probabilities = model.logits(hidden).softmax();
    probabilities.gather(target).log().neg()
}

fn cross_entropy(model: &Model, input: &[usize], target: usize, targets_per_step: usize) -> Value {
    assert!(input.len() >= 2, "training windows need at least two tokens");
    assert!(targets_per_step > 0, "targets per step must be at least 1");

    let hidden = model.forward_all_hidden(input);
    let usable = input.len() - 1;
    let count = targets_per_step.min(usable);

    let mut total = cross_entropy_from_hidden(model, &hidden[0], input[1]);
    let mut selected = 1usize;
    for i in 1..count {
        let pos = i * usable / count;
        total = total.add(&cross_entropy_from_hidden(model, &hidden[pos], input[pos + 1]));
        selected += 1;
    }

    // Keep the original next-token objective as well: the token immediately
    // after the context is still an important prediction target.
    total = total.add(&cross_entropy_from_hidden(
        model,
        hidden.last().expect("non-empty hidden sequence"),
        target,
    ));
    selected += 1;
    total.div_scalar(selected as f32)
}

fn config_from_args(args: &[String]) -> Config {
    if args.iter().any(|arg| arg == "--target") {
        Config::target()
    } else {
        Config::debug()
    }
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

fn parse_string_arg<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|arg| arg == name)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
}

fn train(
    steps: usize,
    path: &str,
    cfg: Config,
    checkpoint_every: usize,
    grad_accum: usize,
    data_path: Option<&str>,
    targets_per_step: usize,
) {
    cfg.validate();
    assert!(grad_accum > 0, "gradient accumulation must be at least 1");
    assert!(targets_per_step > 0, "targets per step must be at least 1");
    let tokenizer = Tokenizer::new();
    let corpus = match data_path {
        Some(path) => std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("failed to read training data '{path}': {e}")),
        None => tokenizer::tiny_corpus(),
    };
    assert!(!corpus.is_empty(), "training corpus must not be empty");
    let encoded = tokenizer.encode(&corpus);
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
    println!(
        "training corpus: {} bytes{}",
        corpus.len(),
        data_path
            .map(|p| format!(" from {p}"))
            .unwrap_or_default()
    );
    if cfg == Config::target() {
        println!("target profile selected; scalar CPU training is intentionally slow");
    }
    println!("gradient accumulation: {grad_accum}");
    println!("targets per window: {} + 1 next-context target", targets_per_step.min(cfg.context));
    if checkpoint_every > 0 {
        println!("periodic checkpoints: every {checkpoint_every} steps -> {path}");
    }

    let model = Model::new(cfg, 42);
    let parameters = model.parameters();
    let mut optimizer = AdamW::new(if cfg == Config::target() { 0.0005 } else { 0.0005 });

    let window_count = encoded.len() - cfg.context;
    let mut accumulated = 0usize;
    let mut loss_sum = 0.0f32;
    for step in 1..=steps {
        let start = (step - 1) % window_count;
        let loss = cross_entropy(
            &model,
            &encoded[start..start + cfg.context],
            encoded[start + cfg.context],
            targets_per_step,
        );
        let value = loss.data()[0];
        loss_sum += value;
        accumulated += 1;

        let divisor = if step == steps {
            accumulated
        } else {
            grad_accum
        };
        loss.div_scalar(divisor as f32).backward();

        if accumulated == grad_accum || step == steps {
            optimizer.step(&parameters);
            let mean_loss = loss_sum / accumulated as f32;
            let update = (step - 1) / grad_accum + 1;
            println!("update {update:4} (step {step:4}) mean_loss {mean_loss:.5}");
            accumulated = 0;
            loss_sum = 0.0;
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
    let autograd_model = Model::new(cfg, 42);
    let parameters = autograd_model.parameters();

    if !std::path::Path::new(path).exists() {
        eprintln!("checkpoint not found: {path}");
        eprintln!("train first, for example: cargo run --release -- train 300 {path}");
        std::process::exit(2);
    }
    checkpoint::load(path, &parameters).expect("failed to load checkpoint");

    let runtime = RuntimeModel::from_parameters(cfg, &parameters);
    let mut cache = runtime.new_cache();
    let mut tokens = tokenizer.encode(prompt);
    let _ = tokens.pop();
    assert!(!tokens.is_empty(), "prompt must contain at least one token");
    assert!(tokens.len() <= cfg.context, "prompt exceeds configured context");

    let mut logits = runtime.logits(&runtime.prime(&tokens, &mut cache));
    let mut rng_state = 0x9E37_79B9_7F4A_7C15u64;
    for _ in 0..max_new_tokens {
        let next = if temperature <= 1e-6 {
            argmax(&logits)
        } else {
            sample_token(&logits, temperature, top_k, &mut rng_state)
        };
        tokens.push(next);
        if next == tokenizer::EOS || tokens.len() >= cfg.context {
            break;
        }
        logits = runtime.logits(&runtime.next(next, &mut cache));
    }
    println!("{}", tokenizer.decode(&tokens));
}

fn bench(cfg: Config, prompt_tokens: usize, generated_tokens: usize) {
    assert!(prompt_tokens > 0 && generated_tokens > 0);
    assert!(prompt_tokens + generated_tokens <= cfg.context);
    let model = Model::new(cfg, 42);
    let runtime = RuntimeModel::from_parameters(cfg, &model.parameters());
    let tokens: Vec<usize> = (0..prompt_tokens).map(|i| 65 + (i % 26)).collect();
    let mut cache = runtime.new_cache();

    let start = Instant::now();
    let mut hidden = runtime.prime(&tokens, &mut cache);
    let prefill_seconds = start.elapsed().as_secs_f64();

    let decode_start = Instant::now();
    let mut token = argmax(&runtime.logits(&hidden));
    for _ in 0..generated_tokens {
        hidden = runtime.next(token, &mut cache);
        token = argmax(&runtime.logits(&hidden));
    }
    let decode_seconds = decode_start.elapsed().as_secs_f64();

    println!(
        "benchmark: {} params | prompt {} | generated {}",
        cfg.params(),
        prompt_tokens,
        generated_tokens
    );
    println!(
        "prefill: {:.3}s | {:.2} tok/s",
        prefill_seconds,
        prompt_tokens as f64 / prefill_seconds.max(f64::MIN_POSITIVE)
    );
    println!(
        "decode:  {:.3}s | {:.2} tok/s",
        decode_seconds,
        generated_tokens as f64 / decode_seconds.max(f64::MIN_POSITIVE)
    );
}

fn print_usage() {
    println!("GemmaAgent Rust LLM");
    println!("\nCommands:");
    println!("  cargo test");
    println!("  cargo run --release -- train 300 [checkpoint] [--target] [--data FILE] [--checkpoint-every N] [--grad-accum N] [--targets-per-step N]");
    println!("  cargo run --release -- infer [checkpoint] [prompt] [--target] [--tokens N] [--temperature T] [--top-k K]");
    println!("  cargo run --release -- bench [--target] [--prompt-tokens N] [--tokens N]");
    println!("\nDefault training profile is the small CPU-debug model.");
    println!("Use --target for the 19,275,776-parameter / context=1024 / 8-head profile.");
    println!("--data FILE trains on UTF-8 text from that file; without it the built-in tiny corpus is used.");
    println!("Gradient accumulation defaults to 1; larger values increase the effective batch without a larger graph.");
    println!("--targets-per-step controls how many causal positions are supervised in each context window; the final next-context target is always included.");
    println!("Inference uses a direct CPU runtime with KV cache; sampling defaults to greedy.");
    println!("Benchmark reports prefill and incremental decode throughput for the current CPU runtime.");
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
            let grad_accum = parse_usize_arg(&args, "--grad-accum", 1);
            let targets_per_step = parse_usize_arg(&args, "--targets-per-step", 8);
            let data_path = parse_string_arg(&args, "--data");
            train(
                steps,
                path,
                config_from_args(&args),
                checkpoint_every,
                grad_accum,
                data_path,
                targets_per_step,
            );
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
        Some("bench") => {
            let prompt_tokens = parse_usize_arg(&args, "--prompt-tokens", 32);
            let generated_tokens = parse_usize_arg(&args, "--tokens", 32);
            bench(
                config_from_args(&args),
                prompt_tokens,
                generated_tokens,
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
        let loss = cross_entropy(&model, &ids[..ids.len() - 1], b'!' as usize, 2);
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
