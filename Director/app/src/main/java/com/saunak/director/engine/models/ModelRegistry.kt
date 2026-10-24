package com.saunak.director.engine.models

import java.util.concurrent.ConcurrentHashMap

/** Manifest registry only. It never silently invents URLs, checksums or licences. */
class ModelRegistry {
    private val manifests = ConcurrentHashMap<String, ModelManifest>()

    fun register(manifest: ModelManifest) {
        val key = key(manifest.id, manifest.version)
        val existing = manifests.putIfAbsent(key, manifest)
        require(existing == null || existing == manifest) {
            "Conflicting manifest already registered for ${manifest.id}:${manifest.version}"
        }
    }

    fun get(id: String, version: String): ModelManifest? = manifests[key(id, version)]

    fun all(): List<ModelManifest> = manifests.values.sortedWith(compareBy(ModelManifest::task, ModelManifest::id, ModelManifest::version))

    fun forTask(task: ModelTask): List<ModelManifest> = all().filter { it.task == task }

    fun commercialReleaseReady(): List<ModelManifest> = all().filter(ModelManifest::allowedForCommercialRelease)

    fun missingProvisioning(): List<ModelManifest> = all().filter { it.sourceUrl == null }

    private fun key(id: String, version: String) = "$id@$version"
}
