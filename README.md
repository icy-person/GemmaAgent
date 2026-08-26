# GemmaAgent

A local, cross-platform autonomous agent built around Gemma 4 E4B IT.

- Kotlin + Compose on Android
- Kotlin/JVM desktop
- Rust native persistent experience memory and learning
- LiteRT-LM for local inference
- Model import only; model is not bundled in the APK
- 50-step agent loop by default, configurable up to 200
- Persistent success/failure experience memory
- Learned skills and memory facts
- Tool registry, permissions and SAFE/ASSISTED/AUTONOMOUS modes
- Multimodal model input plumbing for text/image/audio
- Full Android control dashboard

## Android UI

The Android app is organized into control panels:

- **Chat**: task input, agent mode, answer and live event stream.
- **Model**: model path/status, temperature, Top-K, Top-P, max iterations, prompt/context budget, memory Top-K, skill Top-K, reflection, failure learning, Apply & Reload, Reset defaults and Unload.
- **Memory**: persistent experience count and learning events.
- **Tools**: installed tools and their capabilities/permissions.
- **Learning**: experience-based learning explanation and event stream.
- **Settings**: persistent application settings.

Settings are stored locally and survive app restarts. Sampling settings are applied when the model is reloaded.

## Architecture

```text
Compose UI
   -> Kotlin AgentEngine
       -> LiteRT-LM / Gemma 4 E4B IT
       -> Tool Registry + Permissions
       -> Memory Retrieval
       -> Skill Retrieval
       -> Rust Experience Memory
```

Learning happens outside the model. The agent stores task trajectories, results, success/failure and scores, then retrieves similar experiences before future tasks. Gemma weights are never modified.

## Android model

Import the `.litertlm` deployment file for Gemma 4 E4B IT at runtime. Do not commit the multi-GB model to Git.

The app imports the model into app-private storage so LiteRT-LM can open it by filesystem path. The model is never packaged into the APK.

## Manual build

The GitHub Actions workflow is **manual only**. Pushes and pull requests do not build the project automatically.

Open **Actions → Build GemmaAgent → Run workflow** and choose:

- `debug` or `release` APK
- whether a desktop distribution should also be produced

Artifacts are uploaded to the workflow run.

Local prerequisites: JDK 21, Android SDK 36, Rust, cargo-ndk and Android NDK 29.

```bash
cargo install cargo-ndk
./rust-core/build_android.sh
gradle :app:assembleDebug
gradle :app:assembleRelease
gradle :desktop:packageDistributionForCurrentOS
```

## Agent loop

```text
Task
 -> retrieve similar experiences
 -> retrieve useful skills/facts
 -> Gemma
 -> tool call
 -> observe result
 -> reflect/verify
 -> Gemma
 -> ...
 -> final answer
 -> store experience
 -> optionally learn a reusable skill
```

Failures are intentionally stored so the agent can avoid repeating bad strategies.

## Safety

The default Android filesystem tool is restricted to app-private workspace storage. HTTP access is bounded. Dangerous tools can require approval depending on the configured mode. There is no arbitrary shell tool in the default Android build.

## Cross-platform

The shared module contains the Kotlin agent contract and engine. Rust owns persistent experience storage and ranking. Android and desktop provide platform-specific model/tool integrations.
