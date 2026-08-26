package com.example.gemmaagent.desktop.plugins

import com.example.gemmaagent.shared.AgentPlugin
import com.example.gemmaagent.shared.AgentTool
import com.example.gemmaagent.shared.Permission
import com.example.gemmaagent.shared.PluginManifest
import com.example.gemmaagent.shared.ToolDefinition
import com.example.gemmaagent.shared.ToolResult
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.io.File
import java.net.HttpURLConnection
import java.net.URI
import java.nio.charset.StandardCharsets
import java.util.concurrent.TimeUnit

class DevOpsPlugin(private val workspace: File) : AgentPlugin {
    override val manifest = PluginManifest(
        id = "builtin.devops",
        name = "GitHub & Build Tools",
        version = "1.0.0",
        description = "Git repository operations and local build/test/lint tools with bounded execution.",
        permissions = setOf(Permission.READ, Permission.WRITE, Permission.NETWORK, Permission.EXECUTE),
    )

    override fun tools(): List<AgentTool> = listOf(GitWorkspaceTool(workspace), BuildTool(workspace), GitHubTool())
}

private val devJson = Json { ignoreUnknownKeys = true }

private class GitWorkspaceTool(private val root: File) : AgentTool {
    override val definition = ToolDefinition(
        name = "git_workspace",
        description = "Manage a local git workspace. Actions: status, clone, pull, branches, checkout, log, diff. Input: {action:string, repo?:string, destination?:string, branch?:string}",
        category = "development",
        permissions = setOf(Permission.READ, Permission.WRITE, Permission.NETWORK, Permission.EXECUTE),
        dangerous = true,
    )

    override suspend fun execute(argumentsJson: String): ToolResult = runCatching {
        val o = devJson.parseToJsonElement(argumentsJson).jsonObject
        val action = o["action"]?.jsonPrimitive?.content ?: "status"
        when (action) {
            "status" -> run("git status --short --branch", 30)
            "branches" -> run("git branch -a", 30)
            "log" -> run("git log --oneline --decorate -20", 30)
            "diff" -> run("git diff --stat && git diff --no-ext-diff", 60)
            "pull" -> run("git pull --ff-only", 180)
            "checkout" -> {
                val branch = o["branch"]?.jsonPrimitive?.content ?: error("branch required")
                require(branch.matches(Regex("[A-Za-z0-9._/-]+"))) { "Invalid branch name" }
                run("git checkout $branch", 60)
            }
            "clone" -> {
                val repo = o["repo"]?.jsonPrimitive?.content ?: error("repo required")
                val destination = safe(o["destination"]?.jsonPrimitive?.content ?: error("destination required"))
                require(repo.startsWith("https://") || repo.startsWith("git@")) { "Only HTTPS or SSH git URLs are supported" }
                require(!destination.exists()) { "Destination already exists" }
                destination.parentFile?.mkdirs()
                runAtRoot("git clone --depth 1 ${shell(repo)} ${shell(destination.path)}", 600)
            }
            else -> ToolResult(false, "Unknown git action: $action")
        }
    }.getOrElse { ToolResult(false, "git error: ${it.message}") }

    private fun run(command: String, seconds: Long) = runAtRoot(command, seconds)
    private fun runAtRoot(command: String, seconds: Long): ToolResult {
        val p = ProcessBuilder("sh", "-lc", command).directory(root).redirectErrorStream(true).start()
        val finished = p.waitFor(seconds, TimeUnit.SECONDS)
        if (!finished) p.destroyForcibly()
        val out = p.inputStream.bufferedReader(StandardCharsets.UTF_8).use { it.readText().take(80_000) }
        val code = if (p.isAlive) -1 else p.exitValue()
        return ToolResult(finished && code == 0, "exit=$code\n$out", metadata = mapOf("exitCode" to code.toString()))
    }
    private fun shell(value: String) = "'" + value.replace("'", "'\\''") + "'"
    private fun safe(path: String): File {
        val base = root.canonicalFile
        val out = File(root, path).canonicalFile
        require(out.path == base.path || out.path.startsWith(base.path + File.separator)) { "Path escapes workspace" }
        return out
    }
}

private class BuildTool(private val root: File) : AgentTool {
    override val definition = ToolDefinition(
        name = "build",
        description = "Build/test/lint a project using a safe preset. Input: {action:'build'|'test'|'lint'|'clean'|'check', system?:'gradle'|'cargo'|'maven'|'npm'|'cmake'|'auto', target?:string, timeout_seconds?:number}",
        category = "development",
        permissions = setOf(Permission.EXECUTE, Permission.READ, Permission.WRITE),
        dangerous = true,
    )

    override suspend fun execute(argumentsJson: String): ToolResult = runCatching {
        val o = devJson.parseToJsonElement(argumentsJson).jsonObject
        val action = o["action"]?.jsonPrimitive?.content ?: "build"
        val system = o["system"]?.jsonPrimitive?.content ?: detectSystem(root)
        val target = o["target"]?.jsonPrimitive?.content?.trim().orEmpty()
        val timeout = o["timeout_seconds"]?.jsonPrimitive?.content?.toLongOrNull()?.coerceIn(10, 1800) ?: 600L
        val command = when (system) {
            "gradle" -> when (action) {
                "build" -> "./gradlew ${if (target.isBlank()) "build" else target}"
                "test" -> "./gradlew ${if (target.isBlank()) "test" else target}"
                "lint" -> "./gradlew ${if (target.isBlank()) "lint" else target}"
                "clean" -> "./gradlew clean"
                "check" -> "./gradlew check"
                else -> error("Unsupported gradle action")
            }
            "cargo" -> when (action) {
                "build" -> "cargo build ${target}"
                "test" -> "cargo test ${target}"
                "check" -> "cargo check ${target}"
                "clean" -> "cargo clean"
                "lint" -> "cargo clippy ${target} --all-targets --all-features -- -D warnings"
                else -> error("Unsupported cargo action")
            }
            "maven" -> when (action) {
                "build" -> "./mvnw package -DskipTests"
                "test" -> "./mvnw test"
                "clean" -> "./mvnw clean"
                "check" -> "./mvnw verify"
                else -> error("Unsupported maven action")
            }
            "npm" -> when (action) {
                "build" -> "npm run build"
                "test" -> "npm test"
                "lint" -> "npm run lint"
                else -> error("Unsupported npm action")
            }
            "cmake" -> when (action) {
                "build" -> "cmake --build build --parallel"
                "test" -> "ctest --test-dir build --output-on-failure"
                "clean" -> "cmake --build build --target clean"
                else -> error("Unsupported cmake action")
            }
            else -> error("Unknown build system: $system")
        }
        execute(command, timeout)
    }.getOrElse { ToolResult(false, "build error: ${it.message}") }

    private fun detectSystem(root: File): String = when {
        File(root, "gradlew").isFile || File(root, "build.gradle").exists() || File(root, "build.gradle.kts").exists() -> "gradle"
        File(root, "Cargo.toml").isFile -> "cargo"
        File(root, "mvnw").isFile || File(root, "pom.xml").isFile -> "maven"
        File(root, "package.json").isFile -> "npm"
        File(root, "CMakeLists.txt").isFile -> "cmake"
        else -> error("Unable to detect build system")
    }

    private fun execute(command: String, seconds: Long): ToolResult {
        val p = ProcessBuilder("sh", "-lc", command).directory(root).redirectErrorStream(true).start()
        val finished = p.waitFor(seconds, TimeUnit.SECONDS)
        if (!finished) p.destroyForcibly()
        val output = p.inputStream.bufferedReader(StandardCharsets.UTF_8).use { it.readText().take(120_000) }
        val code = if (p.isAlive) -1 else p.exitValue()
        return ToolResult(finished && code == 0, "exit=$code\n$output", metadata = mapOf("system" to detectSystem(root), "command" to command, "exitCode" to code.toString()))
    }
}

private class GitHubTool : AgentTool {
    override val definition = ToolDefinition(
        name = "github",
        description = "Read public GitHub repository metadata, files, issues and commit info. Input: {action:'repo'|'file'|'issue'|'commit', owner, repo, path?/number?/ref?}",
        category = "github",
        permissions = setOf(Permission.NETWORK, Permission.READ),
    )

    override suspend fun execute(argumentsJson: String): ToolResult = runCatching {
        val o = devJson.parseToJsonElement(argumentsJson).jsonObject
        val owner = o["owner"]?.jsonPrimitive?.content ?: error("owner required")
        val repo = o["repo"]?.jsonPrimitive?.content ?: error("repo required")
        require(owner.matches(Regex("[A-Za-z0-9_.-]+")) && repo.matches(Regex("[A-Za-z0-9_.-]+"))) { "Invalid repository" }
        val action = o["action"]?.jsonPrimitive?.content ?: "repo"
        val path = o["path"]?.jsonPrimitive?.content.orEmpty()
        val number = o["number"]?.jsonPrimitive?.content?.toIntOrNull()
        val ref = o["ref"]?.jsonPrimitive?.content?.let { "/$it" }.orEmpty()
        val url = when (action) {
            "repo" -> "https://api.github.com/repos/$owner/$repo"
            "file" -> "https://raw.githubusercontent.com/$owner/$repo${ref.ifBlank { "/HEAD" }}/$path"
            "issue" -> "https://api.github.com/repos/$owner/$repo/issues/${number ?: error("number required")}"
            "commit" -> "https://api.github.com/repos/$owner/$repo/commits/${o["sha"]?.jsonPrimitive?.content ?: error("sha required")}"
            else -> error("Unknown github action")
        }
        httpGet(url)
    }.getOrElse { ToolResult(false, "github error: ${it.message}") }

    private fun httpGet(url: String): ToolResult {
        val uri = URI(url)
        val c = uri.toURL().openConnection() as HttpURLConnection
        return try {
            c.connectTimeout = 15_000; c.readTimeout = 30_000; c.setRequestProperty("User-Agent", "GemmaAgent/1.0")
            val code = c.responseCode
            val body = (if (code in 200..399) c.inputStream else c.errorStream)?.bufferedReader(StandardCharsets.UTF_8)?.use { it.readText().take(100_000) }.orEmpty()
            ToolResult(code in 200..399, "HTTP $code\n$body", metadata = mapOf("url" to url))
        } finally { c.disconnect() }
    }
}
