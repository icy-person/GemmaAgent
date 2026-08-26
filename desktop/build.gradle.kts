plugins {
    id("org.jetbrains.kotlin.jvm")
    id("org.jetbrains.kotlin.plugin.compose")
    id("org.jetbrains.compose")
    id("org.jetbrains.kotlin.plugin.serialization")
}

dependencies {
    implementation(project(":shared"))
    implementation(compose.desktop.currentOs)
    implementation(compose.material)
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.9.0")
    implementation("com.google.ai.edge.litertlm:litertlm-jvm:0.16.0")
    implementation("org.jsoup:jsoup:1.18.3")
}

compose.desktop {
    application {
        mainClass = "com.example.gemmaagent.desktop.MainKt"
        nativeDistributions {
            packageName = "GemmaAgent"
            packageVersion = "0.1.0"
            modules("java.net.http")
            targetFormats(
                org.jetbrains.compose.desktop.application.dsl.TargetFormat.Deb,
                org.jetbrains.compose.desktop.application.dsl.TargetFormat.Rpm,
            )
        }
    }
}