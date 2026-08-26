plugins {
    id("org.jetbrains.kotlin.jvm")
    id("org.jetbrains.kotlin.plugin.compose")
    id("org.jetbrains.compose")
}

dependencies {
    implementation(project(":shared"))
    implementation(compose.desktop.currentOs)
    implementation(compose.material)
    implementation("com.google.ai.edge.litertlm:litertlm-jvm:0.14.0")
}

compose.desktop {
    application {
        mainClass = "com.example.gemmaagent.desktop.MainKt"
        nativeDistributions {
            packageName = "GemmaAgent"
            packageVersion = "0.1.0"
            targetFormats(
                org.jetbrains.compose.desktop.application.dsl.TargetFormat.Deb,
                org.jetbrains.compose.desktop.application.dsl.TargetFormat.Rpm,
            )
        }
    }
}
