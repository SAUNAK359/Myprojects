package com.saunak.director.engine.prif

import com.saunak.director.engine.capture.CaptureProgram
import com.saunak.director.engine.reconstruction.BayerPattern
import com.saunak.director.engine.reconstruction.CfaBlackLevel
import com.saunak.director.engine.reconstruction.NoiseModel
import com.saunak.director.engine.reconstruction.RawPlane

/** Device-specific physical camera model used by P-RIF reconstruction and forward verification. */
data class SensorDigitalTwin(
    val cameraId: String?,
    val bayerPattern: BayerPattern,
    val blackLevel: CfaBlackLevel,
    val whiteLevel: Float,
    val representativeNoise: NoiseModel,
    val rollingShutterSkewNs: Long?,
    val focalLengthMm: Float?,
    val aperture: Float?,
    val confidence: Double
) {
    init {
        require(whiteLevel > blackLevel.offsets.max())
        require(rollingShutterSkewNs == null || rollingShutterSkewNs >= 0)
        require(focalLengthMm == null || focalLengthMm > 0f)
        require(aperture == null || aperture > 0f)
        require(confidence in 0.0..1.0)
    }
}

enum class PrifAlignmentMode { CFA_TRANSLATION, SEMANTIC_LAYERED }
enum class PrifFusionMode { ROBUST_BAYESIAN, CONSERVATIVE_ANCHOR }

data class ReconstructionProgram(
    val captureProgram: CaptureProgram,
    val referenceFrameIndex: Int,
    val maxFrames: Int,
    val alignmentMode: PrifAlignmentMode,
    val fusionMode: PrifFusionMode,
    val alignmentSearchRadiusPx: Int,
    val robustDegreesOfFreedom: Double,
    val maxNormalizedResidual: Double,
    val subjectProtection: Double,
    val verifySensorConsistency: Boolean,
    val tileSize: Int
) {
    init {
        require(referenceFrameIndex in captureProgram.frames.indices)
        require(maxFrames in 1..captureProgram.frames.size)
        require(alignmentSearchRadiusPx in 0..64)
        require(robustDegreesOfFreedom in 1.0..30.0)
        require(maxNormalizedResidual in 1.0..12.0)
        require(subjectProtection in 0.0..1.0)
        require(tileSize in 128..1024 && tileSize % 2 == 0)
    }
}

/** Posterior mean + uncertainty in normalized linear sensor-radiance space. */
data class RadiancePosterior(
    val width: Int,
    val height: Int,
    val mean: FloatArray,
    val sensorVariance: FloatArray,
    val reconstructionVariance: FloatArray,
    val support: FloatArray,
    val dominantFrame: ByteArray,
    val supportCount: ByteArray,
    val pattern: BayerPattern
) {
    init {
        val n = width * height
        require(width > 1 && height > 1)
        require(mean.size == n)
        require(sensorVariance.size == n)
        require(reconstructionVariance.size == n)
        require(support.size == n)
        require(dominantFrame.size == n)
        require(supportCount.size == n)
        require(mean.all { it.isFinite() && it >= 0f })
        require(sensorVariance.all { it.isFinite() && it >= 0f })
        require(reconstructionVariance.all { it.isFinite() && it >= 0f })
        require(support.all { it.isFinite() && it in 0f..1f })
    }

    fun asNormalizedRaw(): RawPlane = RawPlane(
        width = width,
        height = height,
        values = mean.copyOf(),
        pattern = pattern,
        blackLevel = 0f,
        whiteLevel = 1f
    )

    val meanSupport: Double get() = support.map { it.toDouble() }.average()
    val meanUncertainty: Double get() = mean.indices.map {
        (sensorVariance[it] + reconstructionVariance[it]).toDouble()
    }.average()
}

data class FrameConsistency(
    val frameIndex: Int,
    val normalizedRmse: Double,
    val inlierFraction: Double,
    val saturationFraction: Double,
    val score: Double
) {
    init {
        require(frameIndex >= 0)
        require(normalizedRmse >= 0.0)
        require(inlierFraction in 0.0..1.0)
        require(saturationFraction in 0.0..1.0)
        require(score in 0.0..1.0)
    }
}

data class SensorConsistencyReport(
    val frames: List<FrameConsistency>,
    val aggregateScore: Double,
    val accepted: Boolean
) {
    init {
        require(frames.isNotEmpty())
        require(aggregateScore in 0.0..1.0)
    }
}

data class ProvenanceSummary(
    val averagePhysicalSupport: Double,
    val multiFrameFraction: Double,
    val uncertainFraction: Double,
    val singleFrameFraction: Double,
    val neuralFraction: Double = 0.0
) {
    init {
        listOf(averagePhysicalSupport, multiFrameFraction, uncertainFraction, singleFrameFraction, neuralFraction)
            .forEach { require(it in 0.0..1.0) }
    }
}

data class PhotoIntegrityReport(
    val score: Double,
    val sensorConsistency: Double,
    val physicalSupport: Double,
    val uncertaintyPenalty: Double,
    val provenance: ProvenanceSummary,
    val accepted: Boolean
) {
    init {
        require(score in 0.0..1.0)
        require(sensorConsistency in 0.0..1.0)
        require(physicalSupport in 0.0..1.0)
        require(uncertaintyPenalty in 0.0..1.0)
    }
}
