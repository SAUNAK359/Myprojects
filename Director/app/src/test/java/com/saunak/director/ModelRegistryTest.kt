package com.saunak.director

import com.saunak.director.engine.device.ComputeBackend
import com.saunak.director.engine.models.ModelInputSpec
import com.saunak.director.engine.models.ModelLicense
import com.saunak.director.engine.models.ModelManifest
import com.saunak.director.engine.models.ModelRegistry
import com.saunak.director.engine.models.ModelTask
import com.saunak.director.engine.models.ModelTier
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ModelRegistryTest {
    private fun manifest(commercial: Boolean, sourceUrl: String? = null) = ModelManifest(
        id = "test-model",
        version = "1.0",
        task = ModelTask.QUALITY,
        tier = ModelTier.CAPTURE,
        input = ModelInputSpec(224, 224, 3, false, "rgb", "0..1"),
        supportedBackends = setOf(ComputeBackend.CPU_XNNPACK),
        preferredBackends = listOf(ComputeBackend.CPU_XNNPACK),
        sha256 = "0".repeat(64),
        license = ModelLicense("Apache-2.0", if (commercial) "Apache-2.0" else "CC-BY-NC-4.0", commercial, commercial, false),
        nominalHalfLifeMs = 500,
        sourceUrl = sourceUrl
    )

    @Test fun releaseReadinessAndProvisioningStayExplicit() {
        val registry = ModelRegistry()
        registry.register(manifest(commercial = false))
        assertTrue(registry.commercialReleaseReady().isEmpty())
        assertEquals(1, registry.missingProvisioning().size)
        assertFalse(registry.all().single().allowedForCommercialRelease())
    }
}
