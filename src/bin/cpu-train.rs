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
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn parse_f32(args: &[String], name: &str, default: f32) -> f32 {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn parse_string(args: &[String], name: &str, default: &str) -> String {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| default.to_owned())
}

fn cross_entropy_from_hidden(model: &Model, hidden: &Value, target: usize) -> Value {
    model.logits(hidden).softmax().gather(target).log().neg()
}

fn cross_entropy(model: &Model, input: &[usize], target: usize, targets_per_step: usize) -> Value {
    assert!(input.len() >= 2);
    let hidden = model.forward_all_hidden(input);
    let usable = input.len() - 1;
    let count = targets_per_step.min(usable).max(1);
    let mut total = cross_entropy_from_hidden(model, &hidden[0], input[1]);
    let mut selected = 1usize;
    for i in 1..count {
        let pos = i * usable / count;
        total = total.add(&cross_entropy_from_hidden(model, &hidden[pos], input[pos + 1]));
        selected += 1;
    }
    total = total.add(&cross_entropy_from_hidden(
        model,
        hidden.last().expect("non-empty hidden sequence"),
        target,
    ));
    selected += 1;
    total.div_scalar(selected as f32)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("GemmaAgent CPU trainer\n\nUsage:\n  cargo run --release --bin cpu-train -- --target --steps 500 --data data/train.txt --checkpoint checkpoints/model.ckpt --resume checkpoints/model.ckpt\n\nOptions:\n  --target                 19M parameter profile\n  --steps N                optimizer steps for this stage (default 500)\n  --data FILE              training corpus\n  --checkpoint FILE        output checkpoint\n  --resume FILE            load model weights before training\n  --grad-accum N           gradient accumulation (default 8)\n  --targets-per-step N     positions sampled from each context (default 64)\n  --lr X                   AdamW learning rate (default 0.0003)\n  --checkpoint-every N     save during stage (default 100)\n");
        return;
    }

    let cfg = if args.iter().any(|a| a == "--target") {
        Config::target()
    } else {
        Config::debug()
    };
    cfg.validate();

    let steps = parse_usize(&args, "--steps", 500);
    let data_path = parse_string(&args, "--data", "data/train.txt");
    let checkpoint_path = parse_string(&args, "--checkpoint", "checkpoints/gemma-agent.ckpt");
    let resume_path = args
        .iter()
        .position(|a| a == "--resume")
        .and_then(|i| args.get(i + 1))
        .map(String::as_str);
    let grad_accum = parse_usize(&args, "--grad-accum", 8).max(1);
    let targets_per_step = parse_usize(&args, "--targets-per-step", 64).max(1);
    let lr = parse_f32(&args, "--lr", if cfg == Config::target() { 0.0003 } else { 0.001 });
    let checkpoint_every = parse_usize(&args, "--checkpoint-every", 100);

    let corpus = std::fs::read_to_string(&data_path)
        .unwrap_or_else(|e| panic!("failed to read {data_path}: {e}"));
    assert!(!corpus.is_empty(), "training corpus is empty");
    let tokenizer = Tokenizer::new();
    let encoded = tokenizer.encode(&corpus);
    assert!(encoded.len() > cfg.context + 1, "corpus is shorter than context");

    let model = Model::new(cfg, 42);
    let parameters = model.parameters();
    if let Some(path) = resume_path {
        checkpoint::load(path, &parameters)
            .unwrap_or_else(|e| panic!("failed to resume checkpoint {path}: {e}"));
        println!("resumed weights: {path}");
    }

    let mut optimizer = AdamW::new(lr);
    let window_count = encoded.len() - cfg.context;
    let mut accumulated = 0usize;
    let mut loss_sum = 0.0f32;

    println!(
        "cpu-train: params={} context={} corpus_bytes={} steps={} grad_accum={} lr={}",
        cfg.params(), cfg.context, corpus.len(), steps, grad_accum, lr
    );

    for step in 1..=steps {
        let start = (step - 1) % window_count;
        let loss = cross_entropy(
            &model,
            &encoded[start..start + cfg.context],
            encoded[start + cfg.context],
            targets_per_step,
        );
        let value = loss.data()[0];
        assert!(value.is_finite(), "non-finite loss at step {step}");
        loss_sum += value;
        accumulated += 1;
        let divisor = if step == steps { accumulated } else { grad_accum };
        loss.div_scalar(divisor as f32).backward();

        if accumulated == grad_accum || step == steps {
            optimizer.step(&parameters);
            let mean_loss = loss_sum / accumulated as f32;
            println!("step={step} mean_loss={mean_loss:.6} perplexity={:.4}", mean_loss.exp());
            accumulated = 0;
            loss_sum = 0.0;
        }

        if checkpoint_every > 0 && step % checkpoint_every == 0 {
            checkpoint::save(&checkpoint_path, &parameters)
                .unwrap_or_else(|e| panic!("failed to save checkpoint: {e}"));
            println!("checkpoint={checkpoint_path} step={step}");
        }
    }

    checkpoint::save(&checkpoint_path, &parameters)
        .unwrap_or_else(|e| panic!("failed to save final checkpoint: {e}"));
    println!("final_checkpoint={checkpoint_path}");
}
