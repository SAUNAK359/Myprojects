package com.saunak.director.director

import android.graphics.Bitmap
import com.saunak.director.camera.CapturedStillTelemetry
import com.saunak.director.engine.SierraEngine
import com.saunak.director.processing.PhotoQuality
import java.util.Locale
import kotlin.math.max

object SierraRenderContextBuilder {
    fun build(
        source: Bitmap,
        telemetry: CapturedStillTelemetry,
        captureKind: String,
        reconstructionIntegrity: Double? = null
    ): String {
        val quality = PhotoQuality.analyze(source)
        val engine = SierraEngine.getOrNull()
        val subject = engine?.subjects?.latest()
        val motion = engine?.motion?.latest()
        val mask = subject?.mask
        val depth = subject?.depth
        val motionUncertainty = motion?.let { max(it.camera.uncertainty, it.subject.uncertainty) } ?: 1.0
        return buildString {
            appendLine("capture_kind=$captureKind")
            appendLine("capture_telemetry=${telemetry.compactDescription()}")
            reconstructionIntegrity?.let { appendLine("reconstruction_integrity=${fmt(it.coerceIn(0.0, 1.0))}") }
            appendLine("image=${source.width}x${source.height}")
            appendLine("mean_luma=${fmt(quality.meanLuma)};shadow_p05=${fmt(quality.shadowP05)};median_luma=${fmt(quality.medianLuma)};highlight_p95=${fmt(quality.highlightP95)}")
            appendLine("shadow_clip=${fmt(quality.darkClipFraction)};highlight_clip=${fmt(quality.brightClipFraction)};luma_std=${fmt(quality.lumaStdDev)};edge_energy=${fmt(quality.edgeEnergy)}")
            appendLine("subject_track_available=${subject != null};subject_confidence=${fmt(subject?.confidence ?: 0.0)}")
            appendLine("subject_mask_available=${mask != null};subject_mask_confidence=${fmt(mask?.confidence ?: 0.0)};subject_mask_coverage=${fmt(mask?.coverage ?: 0.0)}")
            appendLine("relative_depth_available=${depth != null};relative_depth_confidence=${fmt(depth?.confidence ?: 0.0)};foreground_separation=${fmt(depth?.foregroundSeparation ?: 0.0)}")
            appendLine("face_mask_available=false;skin_mask_available=false;sky_mask_available=false")
            appendLine("motion_available=${motion != null};motion_risk=${fmt(motion?.conservativeRisk ?: 0.0)};motion_uncertainty=${fmt(motionUncertainty)};motion_confidence=${fmt(motion?.fusionConfidence ?: 0.0)}")
            appendLine("executor_precision=linear_srgb_argb8888;working_long_edge_range=1024..4096")
            appendLine("executor_support=tone_curve,color_temperature_tint,saturation_vibrance,subject_mask_local_light,relative_depth_subject_separation,highlight_rolloff,shadow_anchor,texture_microcontrast,atmosphere_dehaze_glow_haze,optical_vignette,film_grain,final_output_contrast")
            appendLine("unsupported_semantic_masks=face,skin,sky")
        }
    }

    private fun fmt(v: Double): String = "%.4f".format(Locale.US, v)
}
