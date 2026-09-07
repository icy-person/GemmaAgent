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

fn train(steps: usize, path: &str, cfg: Config) {
    cfg.validate();
    let tokenizer = Tokenizer::new();
    let encoded = tokenizer.encode(&tokenizer::tiny_corpus());
    assert!(encoded.len() > cfg.context + 2, "training corpus is shorter than the configured context");

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

    let model = Model::new(cfg, 42);
    let parameters = model.parameters();
    let mut optimizer = AdamW::new(if cfg == Config::target() { 0.0005 } else { 0.002 });

    for step in 1..=steps {
        let max_start = encoded.len() - cfg.context - 1;
        let start = (step - 1) % max_start;
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
    }

    checkpoint::save(path, &parameters).expect("failed to save checkpoint");
    println!("checkpoint: {path}");
}

fn infer(path: &str, prompt: &str, cfg: Config, max_new_tokens: usize) {
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
    for _ in 0..max_new_tokens {
        let start = tokens.len().saturating_sub(cfg.context);
        let hidden = model.forward_hidden(&tokens[start..]);
        let next = argmax(&model.logits(&hidden).data());
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
    println!("  cargo run --release -- train 300 [checkpoint] [--target]");
    println!("  cargo run --release -- infer [checkpoint] [prompt] [--target] [--tokens N]");
    println!("\nDefault training profile is the small CPU-debug model.");
    println!("Use --target for the 19,275,776-parameter / context=1024 / 8-head profile.");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("train") => {
            let steps = args.get(2).and_then(|x| x.parse().ok()).unwrap_or(300);
            let path = args.get(3).map(String::as_str).unwrap_or("gemma-agent.ckpt");
            train(steps, path, config_from_args(&args));
        }
        Some("infer") => {
            let path = args.get(2).map(String::as_str).unwrap_or("gemma-agent.ckpt");
            let prompt = args.get(3).map(String::as_str).unwrap_or("Rust is");
            let max_new = args
                .iter()
                .position(|arg| arg == "--tokens")
                .and_then(|i| args.get(i + 1))
                .and_then(|x| x.parse().ok())
                .unwrap_or(64);
            infer(path, prompt, config_from_args(&args), max_new);
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
}
