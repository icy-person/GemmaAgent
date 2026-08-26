package com.example.gemmaagent.shared

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.longOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import kotlin.math.*

class ToolRegistry(initial: List<AgentTool> = emptyList()) {
    private val map = linkedMapOf<String, AgentTool>()
    init { initial.forEach { register(it) } }
    fun register(tool: AgentTool) { map[tool.definition.name] = tool }
    fun registerAll(tools: Iterable<AgentTool>) { tools.forEach(::register) }
    fun remove(name: String) { map.remove(name) }
    fun get(name: String): AgentTool? = map[name]
    fun all(): List<AgentTool> = map.values.toList()
}

class CalculatorTool : AgentTool {
    override val definition = ToolDefinition(
        name = "calculator",
        description = "Evaluate a basic arithmetic expression using numbers, + - * / % ^, parentheses and common math functions.",
        category = "data",
        permissions = setOf(Permission.READ),
    )
    override suspend fun execute(argumentsJson: String): ToolResult = runCatching {
        val expression = Json.parseToJsonElement(argumentsJson).jsonObject["expression"]?.jsonPrimitive?.content
            ?: return ToolResult(false, "Missing expression")
        val value = ExpressionParser(expression).parse()
        ToolResult(true, buildJsonObject { put("expression", expression); put("value", value) }.toString())
    }.getOrElse { ToolResult(false, "Calculator error: ${it.message}") }
}

class DateTimeTool : AgentTool {
    override val definition = ToolDefinition(
        name = "datetime",
        description = "Return the current epoch timestamp and a UTC ISO-8601-ish representation.",
        category = "system",
        permissions = setOf(Permission.READ),
    )
    override suspend fun execute(argumentsJson: String): ToolResult {
        val now = nowEpochMs()
        return ToolResult(true, "epochMs=$now")
    }
}

class EchoTool : AgentTool {
    override val definition = ToolDefinition(
        name = "echo",
        description = "Return supplied text unchanged; useful for testing tool calling.",
        category = "debug",
        permissions = setOf(Permission.READ),
    )
    override suspend fun execute(argumentsJson: String): ToolResult =
        ToolResult(true, argumentsJson)
}

private class ExpressionParser(private val source: String) {
    private var pos = 0
    fun parse(): Double {
        val result = expression()
        skipSpaces()
        require(pos == source.length) { "Unexpected input at $pos" }
        return result
    }
    private fun expression(): Double {
        var v = term()
        while (true) {
            skipSpaces()
            v = when {
                match('+') -> v + term()
                match('-') -> v - term()
                else -> return v
            }
        }
    }
    private fun term(): Double {
        var v = power()
        while (true) {
            skipSpaces()
            v = when {
                match('*') -> v * power()
                match('/') -> v / power()
                match('%') -> v % power()
                else -> return v
            }
        }
    }
    private fun power(): Double {
        var v = unary()
        skipSpaces()
        if (match('^')) v = v.pow(unary())
        return v
    }
    private fun unary(): Double {
        skipSpaces()
        if (match('+')) return unary()
        if (match('-')) return -unary()
        if (peekLetter()) {
            val name = identifier()
            skipSpaces()
            if (match('(')) {
                val arg = expression(); require(match(')')) { "Missing ')'" }
                return when (name.lowercase()) {
                    "sqrt" -> sqrt(arg)
                    "abs" -> abs(arg)
                    "sin" -> sin(arg)
                    "cos" -> cos(arg)
                    "tan" -> tan(arg)
                    "log" -> log10(arg)
                    "ln" -> ln(arg)
                    "exp" -> exp(arg)
                    "ceil" -> ceil(arg)
                    "floor" -> floor(arg)
                    else -> error("Unknown function $name")
                }
            }
            return when (name.lowercase()) {
                "pi" -> PI
                "e" -> E
                else -> error("Unknown identifier $name")
            }
        }
        if (match('(')) {
            val v = expression(); require(match(')')) { "Missing ')'" }; return v
        }
        return number()
    }
    private fun number(): Double {
        skipSpaces(); val start = pos
        while (pos < source.length && (source[pos].isDigit() || source[pos] == '.')) pos++
        require(pos > start) { "Number expected at $pos" }
        return source.substring(start, pos).toDouble()
    }
    private fun identifier(): String {
        val start = pos
        while (pos < source.length && (source[pos].isLetter() || source[pos] == '_')) pos++
        return source.substring(start, pos)
    }
    private fun peekLetter() = pos < source.length && source[pos].isLetter()
    private fun match(c: Char): Boolean { if (pos < source.length && source[pos] == c) { pos++; return true }; return false }
    private fun skipSpaces() { while (pos < source.length && source[pos].isWhitespace()) pos++ }
}
