#[cfg(feature = "amd-vulkan")]
#[path = "../config.rs"]
mod config;
#[cfg(feature = "amd-vulkan")]
#[path = "../tokenizer.rs"]
mod tokenizer;
#[cfg(feature = "amd-vulkan")]
#[path = "../amd.rs"]
mod amd;

#[cfg(feature = "amd-vulkan")]
fn parse_usize(args: &[String], name: &str, default: usize) -> usize {
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
        println!("Usage: cargo run --release --features amd-vulkan --bin amd-bench -- [--target] --batch-size 4 --context 128 --iterations 100 --gpu-kind integrated --gpu 0");
        return;
    }
    let cfg = if args.iter().any(|a| a == "--target") {
        config::Config::target()
    } else {
        config::Config::debug()
    };
    let batch_size = parse_usize(&args, "--batch-size", 4);
    let context = parse_usize(&args, "--context", 128);
    let iterations = parse_usize(&args, "--iterations", 100);
    let gpu = parse_usize(&args, "--gpu", 0);
    let gpu_kind = parse_string(&args, "--gpu-kind", "integrated");
    amd::benchmark(cfg, gpu, &gpu_kind, batch_size, context, iterations);
}

#[cfg(not(feature = "amd-vulkan"))]
fn main() {
    eprintln!("amd-bench requires the `amd-vulkan` feature.");
    eprintln!("Use: cargo run --release --features amd-vulkan --bin amd-bench -- --help");
}
