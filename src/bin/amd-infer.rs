#![recursion_limit = "256"]

#[cfg(feature = "amd-vulkan")]
#[path = "../config.rs"]
mod config;
#[cfg(feature = "amd-vulkan")]
#[path = "../tokenizer.rs"]
mod tokenizer;
#[cfg(feature = "amd-vulkan")]
#[path = "../amd_tokenizer.rs"]
mod amd_tokenizer;
#[cfg(feature = "amd-vulkan")]
#[path = "../amd.rs"]
mod amd;

#[cfg(feature = "amd-vulkan")]
use burn::{module::Module, tensor::{Int, Tensor, TensorData}};

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
fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(index, _)| index)
        .unwrap_or(0)
}

#[cfg(feature = "amd-vulkan")]
fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("GemmaAgent AMD Vulkan inference\n\nUsage:\n  cargo run --release --features amd-vulkan --bin amd-infer -- --target --checkpoint gemma-agent-target-amd.bin --tokenizer gemma-agent-target-amd.bin.tok --prompt \"Rust is\" --tokens 64\n\nOptions:\n  --target               use the target configuration\n  --checkpoint FILE      Burn model checkpoint (default gemma-agent-amd.bin)\n  --tokenizer FILE       target tokenizer (default <checkpoint>.tok)\n  --prompt TEXT          prompt (default: Rust is)\n  --tokens N             maximum newly generated tokens (default 64)\n  --gpu-kind K           integrated, discrete, or best (default integrated)\n  --gpu N                GPU ordinal (default 0)\n");
        return;
    }

    let cfg = if args.iter().any(|a| a == "--target") {
        config::Config::target()
    } else {
        config::Config::debug()
    };
    let checkpoint = parse_string(&args, "--checkpoint", "gemma-agent-amd.bin");
    let tokenizer_path = parse_string(&args, "--tokenizer", &format!("{checkpoint}.tok"));
    let prompt = parse_string(&args, "--prompt", "Rust is");
    let max_new = parse_usize(&args, "--tokens", 64);
    let gpu_index = parse_usize(&args, "--gpu", 0);
    let gpu_kind = parse_string(&args, "--gpu-kind", "integrated");

    cfg.validate();
    let device = match gpu_kind.as_str() {
        "integrated" => burn_wgpu::WgpuDevice::IntegratedGpu(gpu_index),
        "discrete" => burn_wgpu::WgpuDevice::DiscreteGpu(gpu_index),
        "best" => burn_wgpu::WgpuDevice::BestAvailable,
        other => panic!("invalid --gpu-kind '{other}'"),
    };
    burn_wgpu::init_setup::<burn_wgpu::graphics::Vulkan>(&device, Default::default());

    let model_cfg = amd::AmdModelConfig::new(cfg);
    let recorder = burn::record::BinFileRecorder::<burn::record::FullPrecisionSettings>::default();
    let model: amd::AmdModel<amd::AmdBase> = model_cfg
        .init(&device)
        .load_file(&checkpoint, &recorder, &device)
        .unwrap_or_else(|e| panic!("failed to load checkpoint '{checkpoint}': {e}"));

    let target_tokenizer = cfg.vocab == config::Config::target().vocab;
    let mut tokens = if target_tokenizer {
        let tok = amd_tokenizer::AmdTokenizer::load(&tokenizer_path)
            .unwrap_or_else(|e| panic!("failed to load tokenizer '{tokenizer_path}': {e}"));
        tok.encode(&prompt)
    } else {
        tokenizer::Tokenizer::new().encode(&prompt)
    };
    let _ = tokens.pop();
    assert!(!tokens.is_empty(), "prompt must contain at least one token");
    assert!(tokens.len() <= cfg.context, "prompt exceeds configured context");

    for _ in 0..max_new {
        let seq = tokens.len();
        let input = Tensor::<amd::AmdBase, 2, Int>::from_data(
            TensorData::new(tokens.iter().map(|&id| id as i64).collect::<Vec<_>>(), [1, seq]),
            &device,
        );
        let logits = model.forward_logits(input);
        let last = logits
            .slice([0..1, seq - 1..seq, 0..cfg.vocab])
            .reshape([cfg.vocab]);
        let values = last
            .into_data()
            .to_vec::<f32>()
            .expect("failed to read logits from Vulkan device");
        let next = argmax(&values);
        tokens.push(next);
        let finished = if target_tokenizer {
            next == amd_tokenizer::EOS
        } else {
            next == tokenizer::EOS
        };
        if finished || tokens.len() >= cfg.context {
            break;
        }
    }

    if target_tokenizer {
        let tok = amd_tokenizer::AmdTokenizer::load(&tokenizer_path)
            .unwrap_or_else(|e| panic!("failed to load tokenizer '{tokenizer_path}': {e}"));
        println!("{}", tok.decode(&tokens));
    } else {
        println!("{}", tokenizer::Tokenizer::new().decode(&tokens));
    }
}

#[cfg(not(feature = "amd-vulkan"))]
fn main() {
    eprintln!("amd-infer requires the `amd-vulkan` feature.");
    eprintln!("Use: cargo run --release --features amd-vulkan --bin amd-infer -- --help");
}
