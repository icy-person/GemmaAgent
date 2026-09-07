# Backend architecture

Each runtime backend has its own directory:

- `backend/cpu` — scalar CPU + autograd training
- `backend/amd` — desktop Vulkan training/benchmark/inference
- `backend/android` — Android arm64 Vulkan inference

Shared backend-neutral code belongs in `src/core`.

The public binaries under `src/bin` are thin compatibility launchers only.
