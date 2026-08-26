package com.example.gemmaagent.desktop

import java.io.File
import java.util.concurrent.TimeUnit

/** Executes a local Chromium/Chrome binary in headless mode so JS-heavy pages can be rendered. */
class ChromiumRenderer(
    private val executableOverride: String? = null,
    private val timeoutSeconds: Long = 30,
) {
    data class RenderedPage(val url: String, val html: String, val browserPath: String)

    fun isAvailable(): Boolean = findExecutable() != null

    fun render(url: String): RenderedPage? {
        val executable = findExecutable() ?: return null
        require(url.startsWith("http://") || url.startsWith("https://")) { "Only HTTP(S) URLs are supported" }
        val temp = File.createTempFile("gemmaagent-render-", ".html")
        try {
            val command = listOf(
                executable,
                "--headless=new",
                "--disable-gpu",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-background-networking",
                "--virtual-time-budget=12000",
                "--run-all-compositor-stages-before-draw",
                "--dump-dom",
                url,
            )
            val process = ProcessBuilder(command)
                .redirectErrorStream(true)
                .redirectOutput(temp)
                .start()
            if (!process.waitFor(timeoutSeconds, TimeUnit.SECONDS)) {
                process.destroyForcibly()
                return null
            }
            if (process.exitValue() != 0 || !temp.isFile) return null
            val html = temp.readText(Charsets.UTF_8)
            if (html.isBlank()) return null
            return RenderedPage(url, html, executable)
        } finally {
            runCatching { temp.delete() }
        }
    }

    private fun findExecutable(): String? {
        val candidates = buildList {
            executableOverride?.takeIf { it.isNotBlank() }?.let(::add)
            addAll(listOf(
                "chromium",
                "chromium-browser",
                "google-chrome",
                "google-chrome-stable",
                "/usr/bin/chromium",
                "/usr/bin/google-chrome",
                "/usr/bin/google-chrome-stable",
            ))
        }
        return candidates.distinct().firstOrNull { candidate ->
            if (File(candidate).isAbsolute()) File(candidate).canExecute()
            else runCatching {
                ProcessBuilder("sh", "-lc", "command -v ${shellQuote(candidate)}")
                    .redirectErrorStream(true).start().let { p ->
                        p.waitFor(2, TimeUnit.SECONDS) && p.exitValue() == 0
                    }
            }.getOrDefault(false)
        }
    }

    private fun shellQuote(value: String): String = "'" + value.replace("'", "'\\''") + "'"
}
