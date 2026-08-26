# GemmaAgent

A local, cross-platform autonomous agent built around Gemma 4 E4B IT.

- Kotlin + Compose on Android
- Kotlin/JVM desktop
- Rust native persistent experience memory and learning
- LiteRT-LM for local inference
- Model import only; model is not bundled in the APK
- 50-step agent loop
- Persistent success/failure experience memory
- Tool registry and safety boundaries

## Architecture

```text
UI -> Kotlin AgentRuntime -> LiteRT-LM/Gemma
                         -> ToolRegistry
                         -> Rust ExperienceMemory
```

Learning happens outside the model. The agent stores task trajectories, results, success/failure and scores, then retrieves similar experiences before future tasks. Gemma weights are never modified.

## Android model

Import the `.litertlm` deployment file for Gemma 4 E4B IT at runtime. Do not commit the multi-GB model to Git.

## Build

Prerequisites: Android Studio, JDK 17+, Android SDK 36, Rust, cargo-ndk and the Android NDK.

```bash
cargo install cargo-ndk
./rust-core/build_android.sh
./gradlew assembleDebug
```

The Android UI imports a local `.litertlm` file through the system document picker. The model is copied into app-private storage and is never packaged into the APK.

## Agent loop

```text
Task
 -> retrieve similar experiences
 -> Gemma
 -> tool call
 -> observe result
 -> Gemma
 -> ...
 -> final answer
 -> store experience
```

Failures are intentionally stored so the agent can avoid repeating bad strategies.

## Safety

The default Android filesystem tool is restricted to app-private storage. HTTP tools accept HTTP(S). There is no arbitrary shell tool in the default build.

## Cross-platform

The shared module contains the Kotlin agent contract and engine. Rust owns persistent experience storage and ranking. Android and desktop provide platform-specific model/tool integrations.
