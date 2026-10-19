buildscript {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
    dependencies {
        // AGP 9.x uses built-in Kotlin. LiteRT-LM 0.18.x is compiled with Kotlin 2.4
        // metadata, so raise AGP's built-in KGP runtime to the matching compiler line.
        // This does NOT apply the legacy org.jetbrains.kotlin.android plugin to the app.
        classpath("org.jetbrains.kotlin:kotlin-gradle-plugin:2.4.10")
    }
}

plugins {
    id("com.android.application") version "9.4.1" apply false
}
