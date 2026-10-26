package com.saunak.director.engine.cognition

import com.saunak.director.processing.Look
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class SierraCognitionTest {
    private val policy = CognitionPolicy(
        schemaVersion = 1,
        staleAfterMs = 1_000,
        intentTtlMs = 1_500,
        minStyleReliability = 0.30,
        minThemeScore = 0.40,
        illuminationThresholds = IlluminationThresholds(170.0, 132.0, 92.0, 66.0, 42.0, 22.0),
        riskThresholds = RiskThresholds(226.0, 34.0, 0.70, 0.48, 52.0, 0.72),
        selectorWeights = SelectorWeights(0.30, 0.18, 0.16, 0.13, 0.10, 0.13)
    )

    @Test
    fun staleObservationIsRejected() {
        val plane = ReliabilityPlane(policy)
        val observation = ModelObservation(
            modelId = "depth",
            modelVersion = "1",
            stateId = 9,
            timestampNs = 1_000_000_000L,
            task = "depth",
            value = 1.4,
            confidence = 0.95,
            aleatoricUncertainty = 0.05,
            epistemicUncertainty = 0.05,
            oodScore = 0.02,
            expiresAtNs = 1_100_000_000L,
            provenance = ObservationProvenance(source = "test")
        )
        val result = plane.validate(observation, currentStateId = 10, nowNs = 1_050_000_000L)
        assertEquals(0.0, result.freshness, 0.0)
        assertTrue(ReliabilityFlag.STALE in result.flags)
        assertEquals(0.0, result.effectiveConfidence, 0.0)
    }

    @Test
    fun uncertaintyReducesArtisticAggressiveness() {
        val plane = ReliabilityPlane(policy)
        val reliable = ReliabilityEnvelope(0.95, 0.05, 1.0, 0.0)
        val uncertain = ReliabilityEnvelope(0.95, 0.75, 1.0, 0.0)
        assertTrue(plane.aggressiveness(reliable, 0.8) > plane.aggressiveness(uncertain, 0.8))
    }

    @Test
    fun themeCortexAbstainsWhenReliabilityIsWeak() {
        val theme = ThemeProfileAsset(
            id = "portrait",
            displayName = "Portrait",
            fallbackLook = "CLEAN_PORTRAIT",
            baseStrength = 0.5f,
            preferredBranches = setOf("portrait"),
            humanAffinity = 1.0,
            lowLightAffinity = 0.2,
            highlightProtection = 0.8,
            atmospherePriority = 0.6,
            skinNaturalness = 1.0,
            motionFreeze = 0.7,
            subjectRelightEv = 0.15,
            backgroundSuppressionEv = -0.04
        )
        val manifest = CognitionManifest(policy, listOf(theme))
        val world = world(scene = "portrait", illumination = IlluminationRegime.NORMAL, human = true)
        val selection = ThemeCortex(manifest).select(
            world,
            emptyForecast(),
            ReliabilityEnvelope(confidence = 0.4, uncertainty = 0.8, freshness = 1.0)
        )
        assertTrue(selection.abstained)
        assertEquals(Look.NATURAL, selection.fallbackLook)
    }

    @Test
    fun lowLightMovingPortraitRequestsFlashFusionCandidate() {
        val theme = ThemeProfileAsset(
            id = "night_portrait",
            displayName = "Night Portrait",
            fallbackLook = "CLEAN_PORTRAIT",
            baseStrength = 0.5f,
            preferredBranches = setOf("portrait"),
            preferredIllumination = setOf("low"),
            humanAffinity = 1.0,
            lowLightAffinity = 0.9,
            highlightProtection = 0.8,
            atmospherePriority = 0.7,
            skinNaturalness = 1.0,
            motionFreeze = 0.9,
            subjectRelightEv = 0.2,
            backgroundSuppressionEv = -0.05
        )
        val manifest = CognitionManifest(policy, listOf(theme))
        val plane = ReliabilityPlane(policy)
        val cortex = PhotographicCortex(manifest, plane, ThemeCortex(manifest))
        val world = world(scene = "portrait", illumination = IlluminationRegime.LOW, human = true, motion = 0.86)
        val reliability = ReliabilityEnvelope(0.95, 0.05, 1.0)
        val failures = FailureForecast(
            motionBlur = RiskEstimate(0.86, 0.95, "test"),
            subjectGhosting = RiskEstimate(0.82, 0.92, "test"),
            sensorNoise = RiskEstimate(0.78, 0.95, "test"),
            highlightClipping = RiskEstimate(0.25, 0.95, "test"),
            rollingShutter = RiskEstimate.unknown("test"),
            focusFailure = RiskEstimate.unknown("test"),
            flickerBanding = RiskEstimate.unknown("test"),
            reflection = RiskEstimate.unknown("test"),
            flare = RiskEstimate.unknown("test")
        )
        val selected = ThemeCortex(manifest).select(world, failures, reliability)
        val intent = cortex.compose(world, failures, reliability, selected, nowNs = 5_000_000_000L)
        assertEquals(CaptureFamily.FLASH_FUSION_CANDIDATE, intent.capture.family)
        assertEquals(IlluminationAction.CONSIDER_FLASH_FUSION, intent.capture.illuminationAction)
    }

    private fun world(
        scene: String,
        illumination: IlluminationRegime,
        human: Boolean,
        motion: Double = 0.1
    ) = PhotographicWorldState(
        stateId = 1,
        createdAtNs = 1_000_000_000L,
        sceneBranch = scene,
        sceneConfidence = 0.9,
        illumination = illumination,
        meanLuma = if (illumination == IlluminationRegime.LOW) 35.0 else 110.0,
        medianLuma = if (illumination == IlluminationRegime.LOW) 30.0 else 105.0,
        dynamicRangeStops = 8.0,
        highlightClipRisk = 0.2,
        shadowNoiseRisk = if (illumination == IlluminationRegime.LOW) 0.75 else 0.2,
        primaryHumanSubject = human,
        faceCount = if (human) 1 else 0,
        faceCoverage = if (human) 0.15 else 0.0,
        motionRisk = motion,
        cameraMotionUncertainty = 0.1,
        subjectMotionUncertainty = 0.1,
        rollingShutterRisk = 0.1,
        exposureTimeNs = 10_000_000L,
        sensitivityIso = 400,
        focusDistanceDiopters = 0.5f,
        zoomRatio = 1f,
        cameraId = "0",
        inputReliability = ReliabilityEnvelope(0.95, 0.05, 1.0)
    )

    private fun emptyForecast() = FailureForecast(
        motionBlur = RiskEstimate(0.1, 0.9, "test"),
        subjectGhosting = RiskEstimate(0.1, 0.9, "test"),
        sensorNoise = RiskEstimate(0.1, 0.9, "test"),
        highlightClipping = RiskEstimate(0.1, 0.9, "test"),
        rollingShutter = RiskEstimate.unknown("test"),
        focusFailure = RiskEstimate.unknown("test"),
        flickerBanding = RiskEstimate.unknown("test"),
        reflection = RiskEstimate.unknown("test"),
        flare = RiskEstimate.unknown("test")
    )
}
