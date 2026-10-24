package com.saunak.director.engine.prif

import com.saunak.director.engine.capture.CaptureFrameRole
import com.saunak.director.engine.capture.CaptureProgram
import com.saunak.director.engine.capture.MotionClass
import com.saunak.director.engine.motion.MotionSnapshot
import com.saunak.director.engine.subject.SubjectTrack
import kotlin.math.abs

object PrifReconstructionCompiler {
    fun compile(captureProgram: CaptureProgram, motion: MotionSnapshot?, subject: SubjectTrack?, thermalPressure: Double, memoryPressure: Double): ReconstructionProgram {
        require(thermalPressure in 0.0..1.0); require(memoryPressure in 0.0..1.0)
        val pressure = maxOf(thermalPressure, memoryPressure)
        val subjectConfidence = subject?.let { (it.confidence * (1.0 - 0.65 * it.uncertainty)).coerceIn(0.0, 1.0) } ?: 0.0
        val subjectSpeed = subject?.kinematics?.speedPerSec ?: 0.0
        val motionRisk = motion?.conservativeRisk ?: 1.0
        val subjectProtection = (0.48 * motionRisk + 0.34 * (subjectSpeed / 0.85).coerceIn(0.0, 1.0) + 0.18 * subjectConfidence).coerceIn(0.0, 1.0)
        val requested = captureProgram.frames.size
        val pressureCap = when { pressure >= 0.90 -> 2; pressure >= 0.78 -> 3; pressure >= 0.62 -> 4; else -> requested }
        val motionCap = when { subjectProtection >= 0.82 -> 2; subjectProtection >= 0.62 -> 3; subjectProtection >= 0.42 -> 5; else -> requested }
        val referenceFrame = captureProgram.frames.indices.minByOrNull { abs(captureProgram.frames[it].evOffset) } ?: 0
        return ReconstructionProgram(captureProgram, referenceFrame, minOf(requested, pressureCap, motionCap).coerceAtLeast(1), if (subject != null && subjectConfidence >= 0.45) PrifAlignmentMode.SEMANTIC_LAYERED else PrifAlignmentMode.CFA_TRANSLATION, if (subjectProtection >= 0.78) PrifFusionMode.CONSERVATIVE_ANCHOR else PrifFusionMode.ROBUST_BAYESIAN, when { motionRisk >= .80 -> 32; motionRisk >= .60 -> 24; motionRisk >= .35 -> 16; else -> 10 }, if (subjectProtection >= .75) 2.5 else if (subjectProtection >= .50) 3.5 else 5.0, if (subjectProtection >= .75) 2.5 else if (subjectProtection >= .50) 3.2 else 4.5, subjectProtection, true, if (pressure >= .78) 256 else if (pressure >= .55) 384 else 512)
    }
    fun selectCaptureIndices(program: ReconstructionProgram): List<Int> {
        val capture = program.captureProgram; val reference = program.referenceFrameIndex
        if (program.maxFrames >= capture.frames.size) return capture.frames.indices.toList()
        val selected = mutableListOf(reference); val remaining = capture.frames.indices.filter { it != reference }.toMutableSet()
        while (selected.size < program.maxFrames && remaining.isNotEmpty()) {
            val best = remaining.maxByOrNull { index -> informationGainProxy(index, selected, program) } ?: break
            selected += best; remaining -= best
        }
        return selected.sorted()
    }
    private fun informationGainProxy(index: Int, selected: List<Int>, program: ReconstructionProgram): Double {
        val capture = program.captureProgram; val frame = capture.frames[index]; val reference = program.referenceFrameIndex
        val temporalDistance = abs(index - reference).toDouble() / capture.frames.size.coerceAtLeast(1).toDouble()
        val nearestEv = selected.minOf { abs(frame.evOffset - capture.frames[it].evOffset) }
        val diversity = (nearestEv / 2.0).coerceIn(0.0, 1.0); val sameExposureSupport = (1.0 - abs(frame.evOffset) / 2.0).coerceIn(0.0, 1.0)
        val roleValue = when (frame.role) { CaptureFrameRole.REFERENCE -> 1.0; CaptureFrameRole.DETAIL -> .94; CaptureFrameRole.HIGHLIGHT -> if (capture.hdrFusion) .93 else .58; CaptureFrameRole.SHADOW -> if (capture.hdrFusion) .82 else .62 }
        val positiveExposureMotionPenalty = if (frame.evOffset > 0) program.subjectProtection * (frame.evOffset / 1.5).coerceIn(0.0, 1.0) * .52 else 0.0
        return roleValue + sameExposureSupport * (.28 + .30 * program.subjectProtection) + diversity * if (capture.hdrFusion) .34 else .10 - positiveExposureMotionPenalty - temporalDistance * (.12 + .34 * program.subjectProtection)
    }
    fun motionClassForDiagnostics(motion: MotionSnapshot?): MotionClass = motion?.let { maxOf(it.camera.toMotionClass(), it.subject.toMotionClass()) } ?: MotionClass.HIGH
}
