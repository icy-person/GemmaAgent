#![recursion_limit = "256"]

#[path = "../autograd.rs"]
mod autograd;
#[path = "../checkpoint.rs"]
mod checkpoint;
#[path = "../config.rs"]
mod config;
#[path = "../model.rs"]
mod model;
#[path = "../optim.rs"]
mod optim;
#[path = "../tokenizer.rs"]
mod tokenizer;

use autograd::Value;
use config::Config;
use model::Model;
use optim::AdamW;
use tokenizer::Tokenizer;

fn parse_usize(args: &[String], name: &str, default: usize) -> usize {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(default)
}
fn parse_f32(args: &[String], name: &str, default: f32) -> f32 {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(default)
}
fn parse_string(args: &[String], name: &str, default: &str) -> String {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned().unwrap_or_else(|| default.to_owned())
}
fn xorshift64(state: &mut u64) -> u64 { let mut x = *state; x ^= x << 13; x ^= x >> 7; x ^= x << 17; *state = x; x }

fn cosine_lr(base: f32, min_ratio: f32, step: usize, warmup: usize, total: usize) -> f32 {
    if warmup > 0 && step < warmup {
        return base * (step + 1) as f32 / warmup as f32;
    }
    let decay_steps = total.saturating_sub(warmup).max(1);
    let progress = (step.saturating_sub(warmup) as f32 / decay_steps as f32).clamp(0.0, 1.0);
    let cosine = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
    base * (min_ratio + (1.0 - min_ratio) * cosine)
}

fn sample_unique_positions(usable: usize, count: usize, rng: &mut u64) -> Vec<usize> {
    let count = count.min(usable).max(1);
    let mut positions: Vec<usize> = (0..usable).collect();
    for i in 0..count {
        let remaining = usable - i;
        let j = i + (xorshift64(rng) as usize % remaining);
        positions.swap(i, j);
    }
    positions.truncate(count);
    positions
}

fn cross_entropy_from_hidden(model: &Model, hidden: &Value, target: usize) -> Value {
    model.logits(hidden).softmax().gather(target).log().neg()
}

fn cross_entropy(model: &Model, input: &[usize], target: usize, targets_per_step: usize, rng: &mut u64) -> Value {
    assert!(input.len() >= 2);
    let hidden = model.forward_all_hidden(input);
    let usable = input.len() - 1;
    let positions = sample_unique_positions(usable, targets_per_step, rng);
    let mut total = cross_entropy_from_hidden(model, &hidden[positions[0]], input[positions[0] + 1]);
    for &pos in positions.iter().skip(1) {
        total = total.add(&cross_entropy_from_hidden(model, &hidden[pos], input[pos + 1]));
    }
    total = total.add(&cross_entropy_from_hidden(model, hidden.last().expect("non-empty hidden sequence"), target));
    total.div_scalar(positions.len() as f32 + 1.0)
}

fn evaluate(model: &Model, tokenizer: &Tokenizer, path: &str, context: usize, targets_per_step: usize, samples: usize, seed: u64) -> f32 {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("failed to read validation data {path}: {e}"));
    let encoded = tokenizer.encode(&text);
    assert!(encoded.len() > context + 1, "validation corpus is shorter than context");
    let windows = encoded.len() - context;
    let mut rng = seed;
    let mut total = 0.0f32;
    for _ in 0..samples.max(1) {
        let start = (xorshift64(&mut rng) as usize) % windows;
        let loss = cross_entropy(model, &encoded[start..start + context], encoded[start + context], targets_per_step, &mut rng);
        total += loss.data()[0];
    }
    total / samples.max(1) as f32
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("GemmaAgent CPU trainer\n\nUsage:\n  cargo run --release --bin cpu-train -- --large --steps 500 --data data/train.txt --val-data data/val.txt --checkpoint checkpoints/model.ckpt --resume checkpoints/model.ckpt\n\n--steps is the number of optimizer updates in this invocation; optimizer state and total update count are resumed from the checkpoint.\n--targets-per-step samples unique causal target positions per context; use a larger value for more accurate gradients.\n--warmup N controls linear warmup updates (default max(20, steps/20)).\n--min-lr-ratio X sets the final cosine LR as a fraction of base LR (default 0.1).\n");
        return;
    }

    let cfg = if args.iter().any(|a| a == "--large") { Config::large() } else if args.iter().any(|a| a == "--target") { Config::target() } else { Config::debug() };
    cfg.validate();
    let steps = parse_usize(&args, "--steps", 500).max(1);
    let data_path = parse_string(&args, "--data", "data/train.txt");
    let val_path = parse_string(&args, "--val-data", "data/val.txt");
    let checkpoint_path = parse_string(&args, "--checkpoint", "checkpoints/gemma-agent.ckpt");
    let tokenizer_path = parse_string(&args, "--tokenizer", &format!("{checkpoint_path}.tok"));
    let resume_path = args.iter().position(|a| a == "--resume").and_then(|i| args.get(i + 1)).map(String::as_str);
    let grad_accum = parse_usize(&args, "--grad-accum", 8).max(1);
    let targets_per_step = parse_usize(&args, "--targets-per-step", 256).max(1);
    let eval_samples = parse_usize(&args, "--eval-samples", 16).max(1);
    let train_context = parse_usize(&args, "--train-context", cfg.context).clamp(2, cfg.context);
    let default_lr = if cfg == Config::large() { 0.0002 } else if cfg == Config::target() { 0.0003 } else { 0.001 };
    let lr = parse_f32(&args, "--lr", default_lr);
    let checkpoint_every = parse_usize(&args, "--checkpoint-every", 100);
    let warmup = parse_usize(&args, "--warmup", (steps / 20).max(20).min(steps));
    let min_lr_ratio = parse_f32(&args, "--min-lr-ratio", 0.1).clamp(0.0, 1.0);
    assert!(lr.is_finite() && lr > 0.0 && min_lr_ratio.is_finite());

    let corpus = std::fs::read_to_string(&data_path).unwrap_or_else(|e| panic!("failed to read {data_path}: {e}"));
    assert!(!corpus.is_empty(), "training corpus is empty");
    let tokenizer = if cfg.vocab == 258 {
        if std::path::Path::new(&tokenizer_path).exists() { Tokenizer::load(&tokenizer_path).unwrap_or_else(|e| panic!("failed to load tokenizer: {e}")) } else { Tokenizer::new() }
    } else if std::path::Path::new(&tokenizer_path).exists() {
        Tokenizer::load(&tokenizer_path).unwrap_or_else(|e| panic!("failed to load tokenizer {tokenizer_path}: {e}"))
    } else {
        let tok = Tokenizer::train(&corpus, cfg.vocab);
        tok.save(&tokenizer_path).unwrap_or_else(|e| panic!("failed to save tokenizer: {e}"));
        tok
    };
    assert_eq!(tokenizer.vocab_size(), cfg.vocab);
    let encoded = tokenizer.encode(&corpus);
    assert!(encoded.len() > train_context + 1, "corpus is shorter than configured training context");

    let model = Model::new(cfg, 42);
    let parameters = model.parameters();
    let mut optimizer = AdamW::new(lr);
    let mut completed_updates = 0usize;
    let mut rng_state = 0x00C0_FFEE_2026_0907_u64 ^ encoded.len() as u64;
    if let Some(path) = resume_path {
        let state = checkpoint::load_full(path, &parameters, &mut optimizer).unwrap_or_else(|e| panic!("failed to resume checkpoint {path}: {e}"));
        if let Some((update, rng)) = state { completed_updates = update; rng_state = rng; }
        optimizer.set_lr(lr);
        println!("resumed: completed_updates={completed_updates} optimizer_steps={}", optimizer.step_count());
    }

    let window_count = encoded.len() - train_context;
    let mut accumulated = 0usize;
    let mut loss_sum = 0.0f32;
    println!("cpu-train: params={} context={} train_context={} corpus_tokens={} steps={} grad_accum={} targets_per_step={} lr={} warmup={} min_lr_ratio={}", cfg.params(), cfg.context, train_context, encoded.len(), steps, grad_accum, targets_per_step, lr, warmup, min_lr_ratio);

    for local_step in 1..=steps {
        let absolute_step = completed_updates + local_step - 1;
        let current_lr = cosine_lr(lr, min_lr_ratio, absolute_step, warmup, completed_updates + steps);
        optimizer.set_lr(current_lr);
        let start = (xorshift64(&mut rng_state) as usize) % window_count;
        let loss = cross_entropy(&model, &encoded[start..start + train_context], encoded[start + train_context], targets_per_step, &mut rng_state);
        let value = loss.data()[0];
        assert!(value.is_finite(), "non-finite loss at local step {local_step}");
        loss_sum += value;
        accumulated += 1;
        let divisor = if local_step == steps { accumulated } else { grad_accum };
        loss.div_scalar(divisor as f32).backward();
        if accumulated == grad_accum || local_step == steps {
            optimizer.step(&parameters);
            completed_updates += 1;
            let mean_loss = loss_sum / accumulated as f32;
            println!("update={completed_updates} train_loss={mean_loss:.6} train_ppl={:.4} lr={current_lr:.8}", mean_loss.exp());
            accumulated = 0;
            loss_sum = 0.0;
        }
        if checkpoint_every > 0 && local_step % checkpoint_every == 0 {
            checkpoint::save_full(&checkpoint_path, &parameters, &optimizer, completed_updates, rng_state).unwrap_or_else(|e| panic!("failed to save checkpoint: {e}"));
            println!("checkpoint={checkpoint_path} update={completed_updates}");
        }
    }

    let val_loss = evaluate(&model, &tokenizer, &val_path, cfg.context.min(encoded.len() - 2), targets_per_step, eval_samples, rng_state ^ 0x51ED_2026);
    println!("validation_loss={val_loss:.6} validation_perplexity={:.4}", val_loss.exp());
    checkpoint::save_full(&checkpoint_path, &parameters, &optimizer, completed_updates, rng_state).unwrap_or_else(|e| panic!("failed to save final checkpoint: {e}"));
    println!("final_checkpoint={checkpoint_path} completed_updates={completed_updates}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scheduler_starts_below_base_and_reaches_floor() {
        assert!(cosine_lr(1.0, 0.1, 0, 10, 100) < 1.0);
        assert!((cosine_lr(1.0, 0.1, 99, 10, 100) - 0.1).abs() < 1e-6);
    }

    #[test]
    fn target_sampling_is_unique() {
        let mut rng = 123u64;
        let samples = sample_unique_positions(64, 32, &mut rng);
        let mut sorted = samples.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), samples.len());
        assert_eq!(samples.len(), 32);
    }
}
