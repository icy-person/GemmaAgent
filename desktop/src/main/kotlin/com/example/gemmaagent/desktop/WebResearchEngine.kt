package com.example.gemmaagent.desktop

import org.jsoup.Jsoup
import org.jsoup.nodes.Document
import java.net.HttpURLConnection
import java.net.URI
import java.nio.charset.StandardCharsets
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope

/** A local web research pipeline: search -> render -> parse -> rank -> deduplicate -> crawl. */
class WebResearchEngine(
    private val searchLimit: Int = 12,
    private val maxPages: Int = 10,
    private val maxPageChars: Int = 30_000,
    private val maxChunksPerPage: Int = 8,
    chromiumExecutable: String? = null,
) {
    private val chromium = ChromiumRenderer(chromiumExecutable)

    data class Config(
        val maxResults: Int = 8,
        val crawlDepth: Int = 1,
        val minScore: Double = 0.05,
        val maxTotalChars: Int = 70_000,
        val javascriptEnabled: Boolean = true,
        val renderTimeoutSeconds: Long = 30,
    )

    data class SearchResult(val url: String, val title: String, val snippet: String)
    data class Chunk(val text: String, val score: Double)
    data class Source(
        val url: String,
        val title: String,
        val text: String,
        val chunks: List<Chunk>,
        val links: List<String>,
        val renderedWithJavaScript: Boolean = false,
    )
    data class Report(
        val query: String,
        val sources: List<Source>,
        val focusedEvidence: List<String>,
        val javascriptAvailable: Boolean,
    )

    fun browserAvailable(): Boolean = chromium.isAvailable()

    suspend fun research(query: String, config: Config = Config()): Report {
        require(query.isNotBlank()) { "query is required" }
        val firstResults = search(query).take(config.maxResults)
        val firstSources = fetchSources(query, firstResults, config)
        val discovered = if (config.crawlDepth > 0) discoverAndFetch(query, firstSources, config) else emptyList()
        val all = deduplicateSources(firstSources + discovered).take(maxPages)
        val bounded = boundSources(all, config.maxTotalChars.coerceAtLeast(10_000))
        val evidence = bounded.flatMap { source ->
            source.chunks.map { "${source.title} (${source.url}): ${it.text}" }
        }.sortedByDescending { relevance(query, it) }.take(18)
        return Report(query, bounded, evidence, browserAvailable())
    }

    private fun boundSources(items: List<Source>, maxChars: Int): List<Source> {
        var used = 0
        val result = ArrayList<Source>()
        for (source in items) {
            if (used >= maxChars) break
            val remaining = maxChars - used
            val text = source.text.take(remaining)
            val chunks = source.chunks.takeWhile { chunk -> used + chunk.text.length <= maxChars }
            if (text.isBlank()) continue
            result += source.copy(
                text = text,
                chunks = if (chunks.isEmpty()) listOf(Chunk(text.take(4000), 0.0)) else chunks,
            )
            used += text.length
        }
        return result
    }

    private suspend fun fetchSources(query: String, results: List<SearchResult>, config: Config): List<Source> = coroutineScope {
        results.take(maxPages).map { result ->
            async(Dispatchers.IO) { runCatching { fetchSource(query, result.url, result.title, config) }.getOrNull() }
        }.awaitAll().filterNotNull()
    }

    private suspend fun discoverAndFetch(query: String, sources: List<Source>, config: Config): List<Source> = coroutineScope {
        val candidates = sources.flatMap { it.links }.distinct().take(10)
        candidates.map { url ->
            async(Dispatchers.IO) { runCatching { fetchSource(query, url, url, config) }.getOrNull() }
        }.awaitAll().filterNotNull()
    }

    private fun fetchSource(query: String, url: String, fallbackTitle: String, config: Config): Source {
        val rendered = if (config.javascriptEnabled) runCatching { chromium.render(url) }.getOrNull() else null
        val response = rendered?.html ?: httpGet(url)
        val doc = Jsoup.parse(response, url)
        val title = doc.title().trim().ifBlank { fallbackTitle }
        removeNoise(doc)
        val blocks = extractBlocks(doc)
        val ranked = blocks.map { it to relevance(query, it) }
            .filter { it.second >= config.minScore }
            .sortedByDescending { it.second }
        val selected = ranked.take(maxChunksPerPage)
        val text = selected.joinToString("\n\n") { it.first }.take(maxPageChars)
        val chunks = selected.map { (block, score) -> Chunk(block, score) }
        val links = extractRelevantLinks(doc, URI(url))
        return Source(
            url = url,
            title = title,
            text = text,
            chunks = chunks,
            links = links,
            renderedWithJavaScript = rendered != null,
        )
    }

    private fun removeNoise(doc: Document) {
        doc.select("script,style,noscript,template,svg,canvas,form,nav,footer,header,aside,dialog,[role=navigation],[role=banner],[role=contentinfo],[aria-hidden=true]").remove()
        doc.select(".cookie,.cookies,.consent,.gdpr,.advert,.advertisement,.ads,.social,.share,.newsletter,.popup,.modal,.sidebar,.menu,.nav,.footer,.header").remove()
    }

    private fun extractBlocks(doc: Document): List<String> {
        val root = doc.select("article,main,[role=main]").firstOrNull() ?: doc.body() ?: return emptyList()
        return root.select("h1,h2,h3,h4,p,li,blockquote,pre").mapNotNull { element ->
            val text = element.text().replace(Regex("\\s+"), " ").trim()
            val tag = element.tagName()
            val min = if (tag.startsWith("h")) 8 else 35
            text.takeIf { it.length >= min && it.length <= 5000 }
        }.distinct()
    }

    private fun extractRelevantLinks(doc: Document, base: URI): List<String> =
        doc.select("a[href]").mapNotNull { link ->
            val href = link.absUrl("href").ifBlank { link.attr("href") }
            runCatching {
                val uri = URI(href)
                require(uri.scheme == "http" || uri.scheme == "https")
                require(uri.host != null)
                uri.toString()
            }.getOrNull()
        }.filter { link ->
            val uri = runCatching { URI(link) }.getOrNull() ?: return@filter false
            uri.host == base.host || link.length <= 500
        }.distinct().take(20)

    private fun deduplicateSources(items: List<Source>): List<Source> {
        val byUrl = LinkedHashMap<String, Source>()
        for (item in items) {
            val canonical = canonicalize(item.url)
            val existing = byUrl[canonical]
            if (existing == null || item.text.length > existing.text.length) byUrl[canonical] = item
        }
        val byContent = LinkedHashMap<String, Source>()
        for (item in byUrl.values) {
            val fingerprint = fingerprint(item.text)
            if (!byContent.containsKey(fingerprint)) byContent[fingerprint] = item
        }
        return byContent.values.toList()
    }

    private fun canonicalize(url: String): String = runCatching {
        val uri = URI(url)
        buildString {
            append(uri.scheme.lowercase())
            append("://")
            append(uri.host.lowercase())
            if (uri.port != -1) append(':').append(uri.port)
            append(uri.path.ifBlank { "/" })
        }.removeSuffix("/")
    }.getOrDefault(url)

    private fun fingerprint(text: String): String = text.lowercase()
        .replace(Regex("[^\\p{L}\\p{N}]+"), " ")
        .trim()
        .split(' ')
        .take(220)
        .joinToString(" ")

    private fun relevance(query: String, text: String): Double {
        val q = query.lowercase().split(Regex("[^\\p{L}\\p{N}]+"))
            .filter { it.length >= 2 }.distinct()
        if (q.isEmpty()) return 0.0
        val lower = text.lowercase()
        val matched = q.count { token -> lower.contains(token) }
        val density = matched.toDouble() / q.size
        val lengthBoost = (text.length / 500.0).coerceIn(0.0, 1.0)
        return (density * 0.85 + lengthBoost * 0.15).coerceIn(0.0, 1.0)
    }

    private fun search(query: String): List<SearchResult> {
        val encoded = java.net.URLEncoder.encode(query, StandardCharsets.UTF_8)
        val html = httpGet("https://html.duckduckgo.com/html/?q=$encoded")
        val doc = Jsoup.parse(html)
        return doc.select("a.result__a").mapNotNull { anchor ->
            val url = anchor.absUrl("href")
            runCatching { URI(url); require(url.startsWith("http")) }.getOrNull() ?: return@mapNotNull null
            val snippet = anchor.parent()?.parent()?.selectFirst(".result__snippet")?.text().orEmpty()
            SearchResult(url, anchor.text().trim(), snippet)
        }.distinctBy { canonicalize(it.url) }.take(searchLimit)
    }

    private fun httpGet(url: String): String {
        val uri = URI(url)
        require(uri.scheme == "http" || uri.scheme == "https") { "Only HTTP(S) URLs are supported" }
        val connection = (uri.toURL().openConnection() as HttpURLConnection).apply {
            requestMethod = "GET"
            connectTimeout = 15_000
            readTimeout = 25_000
            instanceFollowRedirects = true
            setRequestProperty("User-Agent", "GemmaAgent/1.0 (+local research engine)")
            setRequestProperty("Accept", "text/html,application/xhtml+xml,text/plain;q=0.8")
            setRequestProperty("Accept-Language", "en-US,en;q=0.8")
        }
        return try {
            val code = connection.responseCode
            require(code in 200..399) { "HTTP $code" }
            connection.inputStream.bufferedReader(StandardCharsets.UTF_8).use { reader -> reader.readText().take(2_000_000) }
        } finally {
            connection.disconnect()
        }
    }
}
