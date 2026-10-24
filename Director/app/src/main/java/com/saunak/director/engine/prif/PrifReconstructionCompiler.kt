package com.saunak.director.engine.prif

import com.saunak.director.engine.capture.CaptureProgram
import com.saunak.director.engine.capture.MotionClass
import com.saunak.director.engine.motion.MotionSnapshot
import com.saunak.director.engine.subject.SubjectTrack

/**
 * Converts capture intent + temporal scene evidence into a deterministic reconstruction program.
 * This layer decides HOW to reconstruct; it never mutates pixels itself.
 */
object PrifReconstructionCompiler {
    fun compile(
        captureProgram: CaptureProgram,
        motion: MotionSnapshot?,
        subject: SubjectTrack?,
        thermalPressure: Double,
        memoryPressure: Double
    ): ReconstructionProgram {
        require(thermalPressure in 0.0..1.0)
        require(memoryPressure in 0.0..1.0)
        val pressure = maxOf(thermalPressure, memoryPressure)
        val subjectConfidence = subject?.let {
            (it.confidence * (1.0 - 0.65 * it.uncertainty)).coerceIn(0.0, 1.0)
        } ?: 0.0
        val subjectSpeed = subject?.kinematics?.speedPerSec ?: 0.0
        val motionRisk = motion?.conservativeRisk ?: 1.0
        val subjectProtection = (
            0.48 * motionRisk +
                0.34 * (subjectSpeed / 0.85).coerceIn(0.0, 1.0) +
                0.18 * subjectConfidence
            ).coerceIn(0.0, 1.0)

        val requested = captureProgram.frames.size
        val pressureCap = when {
            pressure >= 0.90 -> 2
            pressure >= 0.78 -> 3
            pressure >= 0.62 -> 4
            else -> requested
        }
        val motionCap = when {
            subjectProtection >= 0.82 -> 2
            subjectProtection >= 0.62 -> 3
            subjectProtection >= 0.42 -> 5
            else -> requested
        }
        val maxFrames = minOf(requested, pressureCap, motionCap).coerceAtLeast(1)

        val referenceFrame = captureProgram.frames.indices.minByOrNull { index ->
            kotlin.math.abs(captureProgram.frames[index].evOffset)
        } ?: 0

        val alignmentRadius = when {
            motionRisk >= 0.80 -> 32
            motionRisk >= 0.60 -> 24
            motionRisk >= 0.35 -> 16
            else -> 10
        }

        val layered = subject != null && subjectConfidence >= 0.45
        val fusionMode = if (subjectProtection >= 0.78) {
            PrifFusionMode.CONSERVATIVE_ANCHOR
        } else {
            PrifFusionMode.ROBUST_BAYESIAN
        }

        val robustDof = when {
            subjectProtection >= 0.75 -> 2.5
            subjectProtection >= 0.50 -> 3.5
            else -> 5.0
        }
        val residualLimit = when {
            subjectProtection >= 0.75 -> 2.5
            subjectProtection >= 0.50 -> 3.2
            else -> 4.5
        }

        val tileSize = when {
            pressure >= 0.78 -> 256
            pressure >= 0.55 -> 384
            else -> 512
        }

        return ReconstructionProgram(
            captureProgram = captureProgram,
            referenceFrameIndex = referenceFrame,
            maxFrames = maxFrames,
            alignmentMode = if (layered) PrifAlignmentMode.SEMANTIC_LAYERED else PrifAlignmentMode.CFA_TRANSLATION,
            fusionMode = fusionMode,
            alignmentSearchRadiusPx = alignmentRadius,
            robustDegreesOfFreedom = robustDof,
            maxNormalizedResidual = residualLimit,
            subjectProtection = subjectProtection,
            verifySensorConsistency = true,
            tileSize = tileSize
        )
    }

    /**
     * Chooses the RAW measurements to acquire before memory is committed. Reference exposure is
     * always retained; remaining frames are ranked by EV proximity and temporal distance.
     */
    fun selectCaptureIndices(program: ReconstructionProgram): List<Int> {
        val capture = program.captureProgram
        val reference = program.referenceFrameIndex
        if (program.maxFrames >= capture.frames.size) return capture.frames.indices.toList()
        return capture.frames.indices
            .sortedWith(
                compareBy<Int> { if (it == reference) 0 else 1 }
                    .thenBy { kotlin.math.abs(capture.frames[it].evOffset) }
                    .thenBy { kotlin.math.abs(it - reference) }
            )
            .take(program.maxFrames)
            .sorted()
    }

    fun motionClassForDiagnostics(motion: MotionSnapshot?): MotionClass =
        motion?.let { maxOf(it.camera.toMotionClass(), it.subject.toMotionClass()) } ?: MotionClass.HIGH
}
