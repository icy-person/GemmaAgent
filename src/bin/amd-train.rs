#![recursion_limit = "256"]

#[cfg(feature = "amd-vulkan")]
#[path = "../amd.rs"]
mod amd;
#[cfg(feature = "amd-vulkan")]
#[path = "../amd_tokenizer.rs"]
mod amd_tokenizer;
#[cfg(feature = "amd-vulkan")]
#[path = "../config.rs"]
mod config;
#[cfg(feature = "amd-vulkan")]
#[path = "../tokenizer.rs"]
mod tokenizer;

#[cfg(feature = "amd-vulkan")]
fn parse_usize(args: &[String], name: &str, default: usize) -> usize {
    args.iter()
        .position(|arg| arg == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[cfg(feature = "amd-vulkan")]
fn parse_f64(args: &[String], name: &str, default: f64) -> f64 {
    args.iter()
        .position(|arg| arg == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[cfg(feature = "amd-vulkan")]
fn parse_string(args: &[String], name: &str, default: &str) -> String {
    args.iter()
        .position(|arg| arg == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| default.to_owned())
}

#[cfg(feature = "amd-vulkan")]
fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!(
            "GemmaAgent AMD Vulkan trainer\n\nUsage:\n  cargo run --release --features amd-vulkan --bin amd-train -- --target --steps 5000 --data ./train.txt --checkpoint gemma-agent-amd.bin\n\nOptions:\n  --target               use the 19M-parameter configuration\n  --steps N              optimizer updates (default 5000)\n  --data FILE            UTF-8 training corpus (default ./train.txt)\n  --checkpoint FILE      latest Burn binary model checkpoint\n  --best-checkpoint FILE best-validation checkpoint (default <checkpoint>.best)\n  --tokenizer FILE       target tokenizer (default <checkpoint>.tok)\n  --resume FILE          resume weights from an existing Burn checkpoint\n  --batch-size N         windows per GPU batch (default 2)\n  --grad-accum N         micro-batches per optimizer update (default 1)\n  --lr X                 base AdamW learning rate (default 0.0003)\n  --checkpoint-every N   save every N updates (default 250)\n  --eval-every N         validation interval (default 250)\n  --gpu-kind K           integrated, discrete, or best (default integrated)\n  --gpu N                GPU ordinal (default 0)\n"
        );
        return;
    }

    let cfg = if args.iter().any(|a| a == "--target") {
        config::Config::target()
    } else {
        config::Config::debug()
    };
    let steps = parse_usize(&args, "--steps", 5000);
    let data = parse_string(&args, "--data", "./train.txt");
    let checkpoint = parse_string(&args, "--checkpoint", "gemma-agent-amd.bin");
    let best_checkpoint = parse_string(&args, "--best-checkpoint", &format!("{checkpoint}.best"));
    let tokenizer = parse_string(&args, "--tokenizer", &format!("{checkpoint}.tok"));
    let resume = args
        .iter()
        .position(|arg| arg == "--resume")
        .and_then(|i| args.get(i + 1))
        .map(String::as_str);
    let batch_size = parse_usize(&args, "--batch-size", 2);
    let grad_accum = parse_usize(&args, "--grad-accum", 1);
    let checkpoint_every = parse_usize(&args, "--checkpoint-every", 250);
    let eval_every = parse_usize(&args, "--eval-every", 250);
    let gpu_index = parse_usize(&args, "--gpu", 0);
    let gpu_kind = parse_string(&args, "--gpu-kind", "integrated");
    let lr = parse_f64(&args, "--lr", 0.0003);

    amd::train(
        cfg,
        steps,
        &checkpoint,
        &best_checkpoint,
        &data,
        &tokenizer,
        resume,
        batch_size,
        grad_accum,
        lr,
        checkpoint_every,
        eval_every,
        gpu_index,
        &gpu_kind,
    );
}

#[cfg(not(feature = "amd-vulkan"))]
fn main() {
    eprintln!("amd-train requires the `amd-vulkan` feature.");
    eprintln!("Use: cargo run --release --features amd-vulkan --bin amd-train -- --help");
}
