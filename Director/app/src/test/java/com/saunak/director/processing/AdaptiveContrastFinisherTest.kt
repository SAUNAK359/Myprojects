package com.saunak.director.processing

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class AdaptiveContrastFinisherTest {
    private fun quality(
        std: Double,
        p05: Double,
        p95: Double,
        darkClip: Double = 0.0,
        brightClip: Double = 0.0
    ) = TechnicalQuality(
        meanLuma = 120.0,
        lumaStdDev = std,
        darkClipFraction = darkClip,
        brightClipFraction = brightClip,
        edgeEnergy = 8.0,
        catastrophic = false,
        heuristicScore = 0.8,
        shadowP05 = p05,
        medianLuma = 118.0,
        highlightP95 = p95
    )

    @Test fun flatImageGetsMeasuredContrastBoost() {
        val gain = AdaptiveContrastFinisher.recommendedGain(quality(std = 23.0, p05 = 54.0, p95 = 171.0))
        assertTrue(gain > 1.05)
        assertTrue(gain <= 1.16)
    }

    @Test fun alreadyContrastyImageIsLeftAlone() {
        val gain = AdaptiveContrastFinisher.recommendedGain(quality(std = 51.0, p05 = 18.0, p95 = 228.0))
        assertEquals(1.0, gain, 1e-9)
    }

    @Test fun clippingSuppressesAggressiveContrast() {
        val clean = AdaptiveContrastFinisher.recommendedGain(quality(std = 21.0, p05 = 50.0, p95 = 170.0))
        val clipped = AdaptiveContrastFinisher.recommendedGain(
            quality(std = 21.0, p05 = 50.0, p95 = 170.0, darkClip = 0.07, brightClip = 0.05)
        )
        assertTrue(clipped < clean)
        assertTrue(clipped <= 1.03)
    }
}
