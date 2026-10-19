buildscript {
    dependencies {
        // AGP 9.x uses built-in Kotlin. Director's LiteRT-LM runtime is compiled with
        // Kotlin 2.4 metadata, so explicitly upgrade the built-in Kotlin/KGP toolchain
        // rather than applying the legacy kotlin-android plugin to the app module.
        classpath("org.jetbrains.kotlin:kotlin-gradle-plugin:2.4.10")
    }
}

plugins {
    id("com.android.application") version "9.4.1" apply false
}
