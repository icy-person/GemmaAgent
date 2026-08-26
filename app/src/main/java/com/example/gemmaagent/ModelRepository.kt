package com.example.gemmaagent

import android.content.Context
import android.net.Uri
import androidx.documentfile.provider.DocumentFile
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File

class ModelRepository(private val context: Context) {
    private val metadata = File(context.filesDir, "models.json")
    private val modelDir = File(context.filesDir, "models").apply { mkdirs() }

    suspend fun importModel(uri: Uri, targetName: String = "gemma-4-E4B-it.litertlm"): String = withContext(Dispatchers.IO) {
        require(targetName.endsWith(".litertlm", ignoreCase = true)) { "Select a .litertlm model" }
        copy(uri, targetName)
    }

    suspend fun importFolder(uri: Uri): String = withContext(Dispatchers.IO) {
        val root = DocumentFile.fromTreeUri(context, uri) ?: error("Cannot open folder")
        val model = findModel(root) ?: error("No .litertlm model found")
        copy(model.uri, model.name ?: "gemma-4-E4B-it.litertlm")
    }

    private fun copy(uri: Uri, sourceName: String): String {
        val safe = sourceName.replace(Regex("[^A-Za-z0-9._-]"), "_")
        require(safe.endsWith(".litertlm", ignoreCase = true)) { "Selected file is not a .litertlm model" }
        val target = File(modelDir, safe)
        val temp = File(modelDir, ".$safe.part-${System.currentTimeMillis()}")
        try {
            context.contentResolver.openInputStream(uri).use { input ->
                requireNotNull(input) { "Cannot open model" }
                temp.outputStream().use { output -> input.copyTo(output, 1024 * 1024) }
            }
            require(temp.isFile && temp.length() > 0L) { "Imported model is empty" }
            if (target.exists() && !target.delete()) error("Cannot replace existing model: ${target.name}")
            check(temp.renameTo(target)) { "Cannot finalize imported model" }
            metadata.writeText(
                JSONObject()
                    .put("name", target.name)
                    .put("path", target.absolutePath)
                    .put("runtime", "LiteRT-LM")
                    .put("importedAt", System.currentTimeMillis())
                    .toString(2)
            )
            return target.absolutePath
        } catch (t: Throwable) {
            runCatching { temp.delete() }
            throw t
        }
    }

    private fun findModel(dir: DocumentFile): DocumentFile? {
        dir.listFiles().forEach { child ->
            if (child.isFile && child.name?.endsWith(".litertlm", true) == true) return child
            if (child.isDirectory) findModel(child)?.let { return it }
        }
        return null
    }

    fun lastImportedPath(): String? = runCatching {
        if (!metadata.isFile) return null
        JSONObject(metadata.readText()).optString("path")
            .takeIf { it.isNotBlank() }
            ?.takeIf { File(it).isFile && File(it).length() > 0L }
    }.getOrNull()
}
