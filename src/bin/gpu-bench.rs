#[cfg(feature = "cuda")]
#[path = "../config.rs"]
mod config;
#[cfg(feature = "cuda")]
#[path = "../tokenizer.rs"]
mod tokenizer;
#[cfg(feature = "cuda")]
#[path = "../gpu.rs"]
mod gpu;

#[cfg(feature = "cuda")]
fn parse_usize(args: &[String], name: &str, default: usize) -> usize {
    args.iter()
        .position(|arg| arg == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[cfg(feature = "cuda")]
fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("Usage: cargo run --release --features cuda --bin gpu-bench -- --batch-size 8 --context 128 --iterations 100 --gpu 0");
        return;
    }
    let batch_size = parse_usize(&args, "--batch-size", 8);
    let context = parse_usize(&args, "--context", 128);
    let iterations = parse_usize(&args, "--iterations", 100);
    let gpu = parse_usize(&args, "--gpu", 0);
    if let Err(err) = gpu::benchmark(gpu, batch_size, context, iterations) {
        eprintln!("GPU benchmark failed: {err}");
        std::process::exit(1);
    }
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("gpu-bench requires the `cuda` feature.");
    eprintln!("Use: cargo run --release --features cuda --bin gpu-bench -- --help");
}
