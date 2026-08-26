# GemmaAgent Runtime Architecture

## Runtime layers

```text
Gemma 4 E4B (.litertlm)
        |
   ModelRunner
        |
   AgentEngine
   /    |     \
Memory  RAG   ToolRegistry
  |      |        |
Memory  Index   Plugins
  |      |        |
  +------+--------+------------------------------+
         |        |        |        |             |
       Files    Web      GitHub   Build       Computer
       Archive  Browser   Git/CI  Cargo/...   Screenshot
```

## Learning without fine-tuning

The Agent never modifies Gemma weights. It stores:

- Experiences: task → actions → result → score
- Facts with confidence and source
- Learned skills as reusable tool sequences
- RAG documents/chunks with persistent retrieval
- Knowledge-graph entities and relationships
- Checkpoints for restart/resume metadata

## Permissions

Every plugin declares the permissions of its tools. Dangerous tools include:

- process/build execution
- Git write operations
- GitHub write operations
- mouse/keyboard control

In `SAFE` mode only READ permissions are allowed. `ASSISTED` uses the approval layer for dangerous operations. `AUTONOMOUS` allows dangerous tools.

## Web Research

The Web Research plugin can search, fetch and parse pages, while the Browser plugin can use installed Chromium to render JavaScript and capture PNG screenshots. Chromium is discovered from common executable names and is not bundled into the model/application distribution.

## GitHub connection

Public read operations use `https://api.github.com`. Authenticated write operations require the environment variable `GITHUB_TOKEN` and are still subject to Agent approval/mode.

Supported write operations include creating issues, creating pull requests, and dispatching GitHub Actions workflows.

Never commit the token to the repository, model prompt, memory store, or logs.

## Local build tools

The Development plugin can detect and run safe presets for Gradle, Cargo, Maven, npm and CMake. Commands have bounded execution time and output. The Arch workflow runs shared tests, desktop tests, builds the application image, and smoke-tests Chromium before packaging.

## Android

Android uses the shared AgentEngine and persistent Android memory/RAG storage. The model remains external to the APK and is imported at runtime. The manual Android workflow builds Rust for `arm64-v8a` and `x86_64`, runs shared JVM tests, then creates a debug or release APK.
