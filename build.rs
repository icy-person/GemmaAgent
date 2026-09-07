fn main() {
    let backends = [
        ("runner-cpu", std::env::var_os("CARGO_FEATURE_RUNNER_CPU").is_some()),
        ("amd-vulkan", std::env::var_os("CARGO_FEATURE_AMD_VULKAN").is_some()),
        ("android-vulkan", std::env::var_os("CARGO_FEATURE_ANDROID_VULKAN").is_some()),
    ];
    let selected = backends.iter().filter(|(_, enabled)| *enabled).count();
    if selected != 1 {
        let names = backends
            .iter()
            .filter(|(_, enabled)| *enabled)
            .map(|(name, _)| *name)
            .collect::<Vec<_>>();
        panic!(
            "exactly one GemmaAgent backend feature must be enabled; selected: {}. Use one of --features runner-cpu, --features amd-vulkan, or --features android-vulkan (with --no-default-features for non-CPU builds).",
            if names.is_empty() { "none".to_owned() } else { names.join(", ") }
        );
    }
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_RUNNER_CPU");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_AMD_VULKAN");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_ANDROID_VULKAN");
}
