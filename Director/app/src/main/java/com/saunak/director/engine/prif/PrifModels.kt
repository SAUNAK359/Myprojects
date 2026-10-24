package com.saunak.director.engine.prif

import com.saunak.director.engine.capture.CaptureProgram
import com.saunak.director.engine.reconstruction.BayerPattern
import com.saunak.director.engine.reconstruction.CfaBlackLevel
import com.saunak.director.engine.reconstruction.NoiseModel
import com.saunak.director.engine.reconstruction.RawPlane

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
)

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
    fun asNormalizedRaw(): RawPlane = RawPlane(width, height, mean.copyOf(), pattern, 0f, 1f)
    val meanSupport: Double get() = support.map { it.toDouble() }.average()
    val meanUncertainty: Double get() = mean.indices.map { (sensorVariance[it] + reconstructionVariance[it]).toDouble() }.average()
}

data class FrameConsistency(val frameIndex: Int, val normalizedRmse: Double, val inlierFraction: Double, val saturationFraction: Double, val score: Double)
data class SensorConsistencyReport(val frames: List<FrameConsistency>, val aggregateScore: Double, val accepted: Boolean)
data class ProvenanceSummary(val averagePhysicalSupport: Double, val multiFrameFraction: Double, val uncertainFraction: Double, val singleFrameFraction: Double, val neuralFraction: Double = 0.0)
data class PhotoIntegrityReport(val score: Double, val sensorConsistency: Double, val physicalSupport: Double, val uncertaintyPenalty: Double, val provenance: ProvenanceSummary, val accepted: Boolean)
