# Backend architecture

Each runtime backend has its own directory and owns its implementation and entrypoints:

- `backend/cpu` — scalar CPU + custom reverse-mode autograd training
- `backend/amd` — desktop Radeon/AMD Vulkan training, benchmark and KV-cache inference
- `backend/android` — Android arm64 Vulkan inference using the shared Vulkan engine adapter

Backend-neutral code belongs only in `src/core`.

Cargo binaries are declared explicitly from `backend/*/bin` with `autobins=false`; there are no legacy binaries under `src/bin`.

Exactly one runtime feature is required per build: `runner-cpu`, `amd-vulkan`, or `android-vulkan`.
