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
    args.iter().position(|arg| arg == name).and_then(|i| args.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(default)
}

#[cfg(feature = "amd-vulkan")]
fn parse_f32(args: &[String], name: &str, default: f32) -> f32 {
    args.iter().position(|arg| arg == name).and_then(|i| args.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(default)
}

#[cfg(feature = "amd-vulkan")]
fn parse_string(args: &[String], name: &str, default: &str) -> String {
    args.iter().position(|arg| arg == name).and_then(|i| args.get(i + 1)).cloned().unwrap_or_else(|| default.to_owned())
}

#[cfg(feature = "amd-vulkan")]
fn xorshift64(state: &mut u64) -> u64 { let mut x=*state; x^=x<<13; x^=x>>7; x^=x<<17; *state=x; x }

#[cfg(feature = "amd-vulkan")]
fn argmax(values: &[f32]) -> usize {
    values.iter().enumerate().max_by(|a,b| a.1.total_cmp(b.1)).map(|(i,_)| i).unwrap_or(0)
}

#[cfg(feature = "amd-vulkan")]
fn sample(values: &[f32], temperature: f32, top_k: usize, rng: &mut u64) -> usize {
    if temperature <= 1e-6 { return argmax(values); }
    let mut ids:(Vec<usize>)=(0..values.len()).collect();
    ids.sort_unstable_by(|&a,&b| values[b].total_cmp(&values[a]));
    if top_k>0 && top_k<ids.len(){ids.truncate(top_k);}
    let max_logit=ids.iter().map(|&i| values[i]/temperature).fold(f32::NEG_INFINITY,f32::max);
    let mut weights=Vec::with_capacity(ids.len()); let mut total=0.0f32;
    for &i in &ids { let w=(values[i]/temperature-max_logit).exp(); weights.push(w); total+=w; }
    if !total.is_finite() || total<=0.0 { return ids[0]; }
    let threshold=((xorshift64(rng)>>40) as f32/((1u64<<24)-1) as f32)*total;
    let mut acc=0.0; for (i,w) in ids.iter().zip(weights){acc+=w;if acc>=threshold{return *i;}}
    *ids.last().unwrap()
}

#[cfg(feature = "amd-vulkan")]
fn main() {
    let args:Vec<String>=std::env::args().collect();
    if args.iter().any(|a| a=="--help" || a=="-h") {
        println!("GemmaAgent AMD Vulkan inference\n\nUsage:\n  cargo run --release --features amd-vulkan --bin amd-infer -- --target --checkpoint gemma-agent-target-amd.bin --tokenizer gemma-agent-target-amd.bin.tok --prompt \"Rust is\" --tokens 64 --temperature 0.7 --top-k 40 --gpu-kind integrated --gpu 0\n\nOptions:\n  --target               target configuration\n  --checkpoint FILE      model checkpoint\n  --tokenizer FILE       target tokenizer\n  --prompt TEXT          prompt text\n  --tokens N             maximum generated tokens\n  --temperature T        sampling temperature; 0 means greedy\n  --top-k K              restrict sampling to top K logits; 0 disables\n  --seed N               RNG seed\n  --gpu-kind K           integrated, discrete, or best\n  --gpu N                GPU ordinal");
        return;
    }
    let cfg=if args.iter().any(|a|a=="--target"){config::Config::target()}else{config::Config::debug()};
    let checkpoint=parse_string(&args,"--checkpoint","gemma-agent-amd.bin");
    let tokenizer_path=parse_string(&args,"--tokenizer",&format!("{checkpoint}.tok"));
    let prompt=parse_string(&args,"--prompt","Rust is");
    let max_new=parse_usize(&args,"--tokens",64);
    let gpu_index=parse_usize(&args,"--gpu",0);
    let gpu_kind=parse_string(&args,"--gpu-kind","integrated");
    let top_k=parse_usize(&args,"--top-k",40);
    let temperature=parse_f32(&args,"--temperature",0.7);
    let mut rng=parse_usize(&args,"--seed",42) as u64 | 1;
    cfg.validate();
    let device=match gpu_kind.as_str(){"integrated"=>burn_wgpu::WgpuDevice::IntegratedGpu(gpu_index),"discrete"=>burn_wgpu::WgpuDevice::DiscreteGpu(gpu_index),"best"=>burn_wgpu::WgpuDevice::BestAvailable,other=>panic!("invalid --gpu-kind '{other}'")};
    burn_wgpu::init_setup::<burn_wgpu::graphics::Vulkan>(&device,Default::default());
    let model=amd::AmdModelConfig::new(cfg).init(&device).load_file(&checkpoint,&burn::record::BinFileRecorder::<burn::record::FullPrecisionSettings>::default(),&device).unwrap_or_else(|e|panic!("failed to load checkpoint '{checkpoint}': {e}"));

    let target=cfg.vocab==config::Config::target().vocab;
    let mut tokens=if target{amd_tokenizer::AmdTokenizer::load(&tokenizer_path).unwrap_or_else(|e|panic!("failed to load tokenizer: {e}")).encode(&prompt)}else{tokenizer::Tokenizer::new().encode(&prompt)};
    if tokens.last().copied()==Some(if target{amd_tokenizer::EOS}else{tokenizer::EOS}){tokens.pop();}
    assert!(!tokens.is_empty()); assert!(tokens.len()<=cfg.context);

    let (mut cache,mut logits)=amd::inference_prefill(&model,&tokens,&device);
    for _ in 0..max_new {
        let next=sample(&logits,temperature,top_k,&mut rng); tokens.push(next);
        let eos=if target{next==amd_tokenizer::EOS}else{next==tokenizer::EOS};
        if eos || tokens.len()>=cfg.context {break;}
        logits=amd::inference_step(&model,next,&mut cache,&device);
    }
    if target{let tok=amd_tokenizer::AmdTokenizer::load(&tokenizer_path).unwrap();println!("{}",tok.decode(&tokens));}
    else{println!("{}",tokenizer::Tokenizer::new().decode(&tokens));}
}

#[cfg(not(feature = "amd-vulkan"))]
fn main(){eprintln!("amd-infer requires the `amd-vulkan` feature.");}
