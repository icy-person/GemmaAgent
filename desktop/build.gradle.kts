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
    }
}
