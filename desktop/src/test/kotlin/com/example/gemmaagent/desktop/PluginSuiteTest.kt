package com.example.gemmaagent.desktop

import com.example.gemmaagent.desktop.plugins.installDefaultPlugins
import com.example.gemmaagent.shared.PluginRegistry
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertTrue

class PluginSuiteTest {
    @Test
    fun defaultPluginsInstallWithoutDuplicateTools() {
        val dir = Files.createTempDirectory("gemmaagent-test").toFile()
        val registry = PluginRegistry()
        kotlinx.coroutines.runBlocking {
            installDefaultPlugins(registry, dir)
            val plugins = registry.all()
            val tools = registry.allTools()
            assertTrue(plugins.size >= 8, "Expected the complete default plugin suite")
            assertTrue(tools.size >= 20, "Expected the complete default tool suite")
        }
        dir.deleteRecursively()
    }
}
