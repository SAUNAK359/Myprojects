package ai.auraforge.director.core

enum class CaptureProgram {
    STANDARD,
    PORTRAIT,
    NIGHT,
    HDR,
    ACTION,
    LANDSCAPE,
    MACRO,
    PANORAMA,
    LONG_EXPOSURE
}

enum class CompositionGuide {
    NONE,
    THIRDS,
    GOLDEN,
    CENTER,
    SYMMETRY,
    DIAGONAL
}

enum class ExtensionPreference {
    NONE,
    AUTO,
    BOKEH,
    HDR,
    NIGHT,
    FACE_RETOUCH
}

enum class FocusIntent {
    AUTO,
    FACE,
    SUBJECT,
    CENTER,
    MACRO,
    INFINITY
}

enum class FlashIntent {
    OFF,
    AUTO,
    ON,
    SCREEN_FILL
}

enum class WhiteBalanceIntent {
    AUTO,
    DAYLIGHT,
    CLOUDY,
    SHADE,
    TUNGSTEN,
    FLUORESCENT,
    WARM,
    COOL
}

enum class ColorStyle {
    NATURAL,
    CINEMATIC_WARM,
    CINEMATIC_COOL,
    FILM,
    VIVID,
    MUTED,
    MONO,
    HIGH_CONTRAST
}

enum class HardwareEffect {
    OFF,
    MONO,
    SEPIA,
    NEGATIVE,
    AQUA,
    POSTERIZE
}

enum class OutputIntent {
    JPEG,
    ULTRA_HDR,
    RAW_JPEG
}

/**
 * The VLM's photographic intent. The model chooses intent; CameraEngine maps it onto only the
 * controls the active Android camera actually exposes. Unsupported controls are ignored/fallbacked,
 * never guessed.
 */
data class PhotoPlan(
    val theme: String,
    val confidence: Float,
    val program: CaptureProgram = CaptureProgram.STANDARD,
    val compositionGuide: CompositionGuide = CompositionGuide.THIRDS,
    val extensionPreference: ExtensionPreference = ExtensionPreference.NONE,
    val focusIntent: FocusIntent = FocusIntent.AUTO,
    val flashIntent: FlashIntent = FlashIntent.OFF,
    val whiteBalanceIntent: WhiteBalanceIntent = WhiteBalanceIntent.AUTO,
    val colorStyle: ColorStyle = ColorStyle.NATURAL,
    val hardwareEffect: HardwareEffect = HardwareEffect.OFF,
    val outputIntent: OutputIntent = OutputIntent.JPEG,
    val zoomRatio: Float = 1f,
    val exposureEv: Float = 0f,
    val apertureFNumber: Float? = null,
    val focusX: Float? = null,
    val focusY: Float? = null,
    val contrast: Float = 0f,
    val saturation: Float = 0f,
    val temperature: Float = 0f,
    val vignette: Float = 0f,
    val retouchStrength: Float = 0f,
    val backgroundBlurStrength: Float = 0f,
    val facePriority: Boolean = false,
    val burstCount: Int = 1,
    val subjectGuidance: String? = null,
    val cameraGuidance: String? = null,
    val rationale: String? = null,
    val createdAtMs: Long = 0L
) {
    fun fingerprint(): String = buildString {
        append(program.name).append('|')
        append(compositionGuide.name).append('|')
        append(extensionPreference.name).append('|')
        append(focusIntent.name).append('|')
        append(flashIntent.name).append('|')
        append(whiteBalanceIntent.name).append('|')
        append(colorStyle.name).append('|')
        append(hardwareEffect.name).append('|')
        append(outputIntent.name).append('|')
        append((zoomRatio * 20f).toInt()).append('|')
        append((exposureEv * 10f).toInt()).append('|')
        append(apertureFNumber?.let { (it * 10f).toInt() } ?: -1)
    }

    fun blendedWith(newer: PhotoPlan, alpha: Float = 0.35f): PhotoPlan {
        val a = alpha.coerceIn(0f, 1f)
        fun mix(old: Float, next: Float) = old + (next - old) * a
        return newer.copy(
            zoomRatio = mix(zoomRatio, newer.zoomRatio),
            exposureEv = mix(exposureEv, newer.exposureEv),
            contrast = mix(contrast, newer.contrast),
            saturation = mix(saturation, newer.saturation),
            temperature = mix(temperature, newer.temperature),
            vignette = mix(vignette, newer.vignette),
            retouchStrength = mix(retouchStrength, newer.retouchStrength),
            backgroundBlurStrength = mix(backgroundBlurStrength, newer.backgroundBlurStrength)
        )
    }
}

data class PlanReadiness(
    val ready: Boolean,
    val score: Float,
    val reason: String
)
