#[path = "../config.rs"]
mod config;
#[path = "../tokenizer.rs"]
mod tokenizer;
#[path = "../gpu.rs"]
mod gpu;

use std::process::ExitCode;

fn parse_usize(args: &[String], name: &str, default: usize) -> usize {
    args.iter()
        .position(|arg| arg == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn parse_f64(args: &[String], name: &str, default: f64) -> f64 {
    args.iter()
        .position(|arg| arg == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn parse_string(args: &[String], name: &str, default: &str) -> String {
    args.iter()
        .position(|arg| arg == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| default.to_owned())
}

fn print_usage() {
    println!(
        "GemmaAgent CUDA trainer\n\nUsage:\n  cargo run --release --features cuda --bin gpu-train -- --steps 5000 --data ./train.txt --checkpoint gemma-agent-gpu.safetensors\n\nOptions:\n  --steps N              optimization steps (default 5000)\n  --data FILE            UTF-8 training corpus (default ./train.txt)\n  --checkpoint FILE      safetensors checkpoint path\n  --batch-size N         windows per GPU batch (default 8)\n  --lr X                 AdamW learning rate (default 0.0003)\n  --checkpoint-every N   save every N steps (default 250)\n  --gpu N                CUDA device ordinal (default 0)\n\nThis trainer uses dense causal next-token loss over every position in each context window.\n"
    );
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_usage();
        return ExitCode::SUCCESS;
    }

    let steps = parse_usize(&args, "--steps", 5000);
    let data = parse_string(&args, "--data", "./train.txt");
    let checkpoint = parse_string(&args, "--checkpoint", "gemma-agent-gpu.safetensors");
    let batch_size = parse_usize(&args, "--batch-size", 8);
    let checkpoint_every = parse_usize(&args, "--checkpoint-every", 250);
    let gpu_index = parse_usize(&args, "--gpu", 0);
    let lr = parse_f64(&args, "--lr", 0.0003);

    match gpu::train(
        steps,
        &checkpoint,
        &data,
        batch_size,
        lr,
        checkpoint_every,
        gpu_index,
    ) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("GPU training failed: {err}");
            eprintln!("Check that the NVIDIA driver and CUDA toolkit are installed and that the requested GPU ordinal exists.");
            ExitCode::from(1)
        }
    }
}
