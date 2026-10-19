package ai.auraforge.director.camera

import ai.auraforge.director.core.CaptureProgram
import ai.auraforge.director.core.ExtensionPreference
import ai.auraforge.director.core.FlashIntent
import ai.auraforge.director.core.FocusIntent
import ai.auraforge.director.core.HardwareEffect
import ai.auraforge.director.core.PhotoPlan
import ai.auraforge.director.core.WhiteBalanceIntent
import ai.auraforge.director.ml.FrameAnalyzer
import android.content.Context
import android.hardware.camera2.CaptureRequest
import androidx.camera.camera2.interop.Camera2CameraControl
import androidx.camera.camera2.interop.CaptureRequestOptions
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.FocusMeteringAction
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.extensions.ExtensionMode
import androidx.camera.extensions.ExtensionsManager
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.util.concurrent.ExecutorService
import java.util.concurrent.TimeUnit
import kotlin.math.roundToInt

@OptIn(ExperimentalCamera2Interop::class)
class CameraEngine(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner,
    private val previewView: PreviewView,
    private val analyzer: FrameAnalyzer,
    private val analysisExecutor: ExecutorService
) {
    private var provider: ProcessCameraProvider? = null
    private var extensionsManager: ExtensionsManager? = null
    private var camera: Camera? = null
    private var facing = CameraSelector.LENS_FACING_BACK
    private var activeExtensionMode: Int? = null
    private var lastAppliedPlanFingerprint: String? = null

    lateinit var imageCapture: ImageCapture
        private set

    var capabilities: CameraCapabilities? = null
        private set

    val isFrontCamera: Boolean get() = facing == CameraSelector.LENS_FACING_FRONT
    val isExtensionCaptureSession: Boolean get() = activeExtensionMode != null

    fun start(onReady: (CameraCapabilities) -> Unit, onError: (Throwable) -> Unit) {
        val future = ProcessCameraProvider.getInstance(context)
        future.addListener({
            runCatching {
                provider = future.get()
                bind()
                initializeExtensions(provider!!)
                capabilities?.let(onReady)
            }.onFailure(onError)
        }, ContextCompat.getMainExecutor(context))
    }

    private fun initializeExtensions(p: ProcessCameraProvider) {
        val future = ExtensionsManager.getInstanceAsync(context, p)
        future.addListener({
            extensionsManager = runCatching { future.get() }.getOrNull()
        }, ContextCompat.getMainExecutor(context))
    }

    fun availableExtensionNames(): Set<String> {
        val manager = extensionsManager ?: return emptySet()
        val selector = baseSelector()
        return buildSet {
            extensionModes().forEach { (name, mode) ->
                if (runCatching { manager.isExtensionAvailable(selector, mode) }.getOrDefault(false)) add(name)
            }
        }
    }

    /**
     * Switches to the OEM computational still-photo pipeline selected by Director. CameraX
     * extensions support Preview+ImageCapture but not ImageAnalysis, so this is intentionally a
     * capture-time session. Call restoreAnalysisSession() after saving the image.
     */
    fun prepareCaptureExtension(preference: ExtensionPreference): Boolean {
        val mode = mapExtension(preference) ?: return false
        val p = provider ?: return false
        val manager = extensionsManager ?: return false
        val base = baseSelector()
        if (!runCatching { manager.isExtensionAvailable(base, mode) }.getOrDefault(false)) return false

        return runCatching {
            p.unbindAll()
            val selector = manager.getExtensionEnabledCameraSelector(base, mode)
            val preview = Preview.Builder().build().also { it.setSurfaceProvider(previewView.surfaceProvider) }
            imageCapture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
                .build()
            camera = p.bindToLifecycle(lifecycleOwner, selector, preview, imageCapture)
            activeExtensionMode = mode
            lastAppliedPlanFingerprint = null
            true
        }.getOrElse {
            activeExtensionMode = null
            runCatching { bind() }
            false
        }
    }

    fun restoreAnalysisSession(): Boolean {
        if (activeExtensionMode == null) return true
        return runCatching {
            activeExtensionMode = null
            bind()
            true
        }.getOrElse { false }
    }

    fun flip(onReady: (CameraCapabilities) -> Unit = {}) {
        val p = provider ?: return
        val previous = facing
        val requested = if (facing == CameraSelector.LENS_FACING_BACK) CameraSelector.LENS_FACING_FRONT else CameraSelector.LENS_FACING_BACK
        val selector = CameraSelector.Builder().requireLensFacing(requested).build()
        if (!runCatching { p.hasCamera(selector) }.getOrDefault(false)) return

        facing = requested
        activeExtensionMode = null
        lastAppliedPlanFingerprint = null
        runCatching {
            bind()
            capabilities?.let(onReady)
        }.onFailure {
            facing = previous
            runCatching { bind() }
        }
    }

    private fun baseSelector(): CameraSelector = CameraSelector.Builder().requireLensFacing(facing).build()

    private fun bind() {
        val p = provider ?: return
        activeExtensionMode = null
        val selector = baseSelector()
        runCatching { bindUseCases(p, selector, preferHighResolution = true) }
            .recoverCatching {
                p.unbindAll()
                bindUseCases(p, selector, preferHighResolution = false)
            }
            .getOrThrow()
    }

    private fun bindUseCases(p: ProcessCameraProvider, selector: CameraSelector, preferHighResolution: Boolean) {
        p.unbindAll()
        val preview = Preview.Builder().build().also { it.setSurfaceProvider(previewView.surfaceProvider) }
        val captureBuilder = ImageCapture.Builder().setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
        if (preferHighResolution) {
            captureBuilder.setResolutionSelector(
                ResolutionSelector.Builder()
                    .setAllowedResolutionMode(ResolutionSelector.PREFER_HIGHER_RESOLUTION_OVER_CAPTURE_RATE)
                    .build()
            )
        }
        imageCapture = captureBuilder.build()
        val analysis = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
            .build()
            .also { it.setAnalyzer(analysisExecutor, analyzer) }

        camera = p.bindToLifecycle(lifecycleOwner, selector, preview, imageCapture, analysis)
        capabilities = CameraCapabilityInspector.inspect(camera!!.cameraInfo)
        lastAppliedPlanFingerprint = null
    }

    fun zoomRange(): ClosedFloatingPointRange<Float> {
        val z = camera?.cameraInfo?.zoomState?.value
        return if (z == null) 1f..1f else z.minZoomRatio..z.maxZoomRatio
    }

    fun currentZoom(): Float = camera?.cameraInfo?.zoomState?.value?.zoomRatio ?: 1f

    fun setZoom(ratio: Float) {
        val range = zoomRange()
        camera?.cameraControl?.setZoomRatio(ratio.coerceIn(range.start, range.endInclusive))
    }

    fun zoomBy(scaleFactor: Float) {
        val state = camera?.cameraInfo?.zoomState?.value ?: return
        setZoom(state.zoomRatio * scaleFactor)
    }

    fun toggleTorch(onChanged: (Boolean) -> Unit = {}) {
        val c = camera ?: return
        if (!c.cameraInfo.hasFlashUnit()) return
        val current = c.cameraInfo.torchState.value == androidx.camera.core.TorchState.ON
        c.cameraControl.enableTorch(!current).addListener({ onChanged(!current) }, ContextCompat.getMainExecutor(context))
    }

    fun focusAt(viewX: Float, viewY: Float) {
        val c = camera ?: return
        val point = previewView.meteringPointFactory.createPoint(viewX, viewY)
        val action = FocusMeteringAction.Builder(
            point,
            FocusMeteringAction.FLAG_AF or FocusMeteringAction.FLAG_AE or FocusMeteringAction.FLAG_AWB
        ).setAutoCancelDuration(3, TimeUnit.SECONDS).build()
        c.cameraControl.startFocusAndMetering(action)
    }

    fun focusNormalized(x: Float, y: Float, sticky: Boolean = true) {
        val c = camera ?: return
        if (previewView.width <= 0 || previewView.height <= 0) return
        val point = previewView.meteringPointFactory.createPoint(
            x.coerceIn(0f, 1f) * previewView.width,
            y.coerceIn(0f, 1f) * previewView.height
        )
        val builder = FocusMeteringAction.Builder(
            point,
            FocusMeteringAction.FLAG_AF or FocusMeteringAction.FLAG_AE or FocusMeteringAction.FLAG_AWB
        )
        if (!sticky) builder.setAutoCancelDuration(3, TimeUnit.SECONDS) else builder.disableAutoCancel()
        c.cameraControl.startFocusAndMetering(builder.build())
    }

    fun setExposureCompensation(index: Int) {
        val c = camera ?: return
        val range = c.cameraInfo.exposureState.exposureCompensationRange
        if (range.lower > range.upper) return
        c.cameraControl.setExposureCompensationIndex(index.coerceIn(range.lower, range.upper))
    }

    fun setExposureEv(ev: Float) {
        val caps = capabilities ?: return
        val range = caps.exposureCompensationRange ?: return
        val step = caps.exposureCompensationStep?.toDouble() ?: return
        if (step <= 0.0) return
        val index = (ev / step).roundToInt().coerceIn(range.lower, range.upper)
        setExposureCompensation(index)
    }

    fun exposureRange(): IntRange {
        val r = camera?.cameraInfo?.exposureState?.exposureCompensationRange ?: return 0..0
        return if (r.lower <= r.upper) r.lower..r.upper else 0..0
    }

    fun setAwbLock(locked: Boolean) {
        val c = camera ?: return
        val options = CaptureRequestOptions.Builder()
            .setCaptureRequestOption(CaptureRequest.CONTROL_AWB_LOCK, locked)
            .build()
        Camera2CameraControl.from(c.cameraControl).setCaptureRequestOptions(options)
    }

    fun applyPhotoPlan(plan: PhotoPlan, force: Boolean = false): AppliedCameraPlan {
        if (isExtensionCaptureSession) return AppliedCameraPlan(true, listOf("OEM extension session"), emptyList())
        val c = camera ?: return AppliedCameraPlan(false, emptyList(), listOf("camera unavailable"))
        val caps = capabilities ?: return AppliedCameraPlan(false, emptyList(), listOf("capabilities unavailable"))
        val fp = plan.fingerprint()
        if (!force && fp == lastAppliedPlanFingerprint) return AppliedCameraPlan(true, listOf("stable"), emptyList())

        val applied = mutableListOf<String>()
        val skipped = mutableListOf<String>()
        val options = CaptureRequestOptions.Builder()

        val zoom = plan.zoomRatio.coerceIn(zoomRange().start, zoomRange().endInclusive)
        setZoom(zoom)
        applied += "zoom ${"%.2f".format(zoom)}×"

        setExposureEv(plan.exposureEv.coerceIn(-3f, 3f))
        applied += "EV ${"%+.1f".format(plan.exposureEv.coerceIn(-3f, 3f))}"

        imageCapture.flashMode = when (plan.flashIntent) {
            FlashIntent.OFF, FlashIntent.SCREEN_FILL -> ImageCapture.FLASH_MODE_OFF
            FlashIntent.AUTO -> if (caps.flashAvailable) ImageCapture.FLASH_MODE_AUTO else ImageCapture.FLASH_MODE_OFF
            FlashIntent.ON -> if (caps.flashAvailable) ImageCapture.FLASH_MODE_ON else ImageCapture.FLASH_MODE_OFF
        }
        if (plan.flashIntent != FlashIntent.OFF && !caps.flashAvailable) skipped += "flash" else applied += "flash ${plan.flashIntent.name.lowercase()}"

        mapAwb(plan.whiteBalanceIntent)?.let { mode ->
            if (mode in caps.awbModes) {
                options.setCaptureRequestOption(CaptureRequest.CONTROL_AWB_MODE, mode)
                applied += "WB ${plan.whiteBalanceIntent.name.lowercase()}"
            } else skipped += "WB ${plan.whiteBalanceIntent.name.lowercase()}"
        }

        mapEffect(plan.hardwareEffect)?.let { mode ->
            if (mode in caps.effectModes) {
                options.setCaptureRequestOption(CaptureRequest.CONTROL_EFFECT_MODE, mode)
                applied += "effect ${plan.hardwareEffect.name.lowercase()}"
            } else skipped += "effect ${plan.hardwareEffect.name.lowercase()}"
        }

        val sceneMode = mapSceneMode(plan.program)
        if (sceneMode != null && sceneMode in caps.sceneModes) {
            options.setCaptureRequestOption(CaptureRequest.CONTROL_MODE, CaptureRequest.CONTROL_MODE_USE_SCENE_MODE)
            options.setCaptureRequestOption(CaptureRequest.CONTROL_SCENE_MODE, sceneMode)
            applied += "scene ${plan.program.name.lowercase()}"
        } else if (sceneMode != null) skipped += "scene ${plan.program.name.lowercase()}"

        val afMode = when (plan.focusIntent) {
            FocusIntent.MACRO -> CaptureRequest.CONTROL_AF_MODE_MACRO
            FocusIntent.INFINITY -> CaptureRequest.CONTROL_AF_MODE_OFF
            else -> CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE
        }
        if (afMode in caps.afModes) {
            options.setCaptureRequestOption(CaptureRequest.CONTROL_AF_MODE, afMode)
            if (plan.focusIntent == FocusIntent.INFINITY && caps.supportsManualFocus) {
                options.setCaptureRequestOption(CaptureRequest.LENS_FOCUS_DISTANCE, 0f)
            }
            applied += "focus ${plan.focusIntent.name.lowercase()}"
        }

        plan.apertureFNumber?.let { requested ->
            val aperture = caps.apertures.minByOrNull { kotlin.math.abs(it - requested) }
            if (aperture != null) {
                options.setCaptureRequestOption(CaptureRequest.LENS_APERTURE, aperture)
                applied += "f/${"%.1f".format(aperture)}"
            } else skipped += "aperture"
        }

        Camera2CameraControl.from(c.cameraControl).setCaptureRequestOptions(options.build())

        if (plan.focusIntent == FocusIntent.FACE || plan.focusIntent == FocusIntent.SUBJECT) {
            val x = plan.focusX ?: 0.5f
            val y = plan.focusY ?: 0.5f
            focusNormalized(if (isFrontCamera) 1f - x else x, y, sticky = true)
            applied += "subject metering"
        } else if (plan.focusIntent == FocusIntent.CENTER) focusNormalized(0.5f, 0.5f, sticky = true)

        lastAppliedPlanFingerprint = fp
        return AppliedCameraPlan(true, applied, skipped)
    }

    fun applyManualSensor(iso: Int, exposureTimeNs: Long, focusDistance: Float?): Boolean {
        val c = camera ?: return false
        val caps = capabilities ?: return false
        if (!caps.supportsManualSensor) return false
        val isoSafe = caps.sensitivityRange?.let { iso.coerceIn(it.lower, it.upper) } ?: iso
        val expSafe = caps.exposureTimeRange?.let { exposureTimeNs.coerceIn(it.lower, it.upper) } ?: exposureTimeNs
        val builder = CaptureRequestOptions.Builder()
            .setCaptureRequestOption(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_OFF)
            .setCaptureRequestOption(CaptureRequest.SENSOR_SENSITIVITY, isoSafe)
            .setCaptureRequestOption(CaptureRequest.SENSOR_EXPOSURE_TIME, expSafe)
        if (focusDistance != null && (caps.minFocusDistance ?: 0f) > 0f) {
            builder.setCaptureRequestOption(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_OFF)
            builder.setCaptureRequestOption(CaptureRequest.LENS_FOCUS_DISTANCE, focusDistance.coerceIn(0f, caps.minFocusDistance ?: focusDistance))
        }
        Camera2CameraControl.from(c.cameraControl).setCaptureRequestOptions(builder.build())
        lastAppliedPlanFingerprint = null
        return true
    }

    fun restoreAuto() {
        val c = camera ?: return
        if (isExtensionCaptureSession) return
        Camera2CameraControl.from(c.cameraControl).clearCaptureRequestOptions()
        lastAppliedPlanFingerprint = null
    }

    fun close() {
        runCatching { provider?.unbindAll() }
        provider = null
        extensionsManager = null
        camera = null
        activeExtensionMode = null
        lastAppliedPlanFingerprint = null
    }

    private fun mapAwb(intent: WhiteBalanceIntent): Int = when (intent) {
        WhiteBalanceIntent.AUTO -> CaptureRequest.CONTROL_AWB_MODE_AUTO
        WhiteBalanceIntent.DAYLIGHT -> CaptureRequest.CONTROL_AWB_MODE_DAYLIGHT
        WhiteBalanceIntent.CLOUDY -> CaptureRequest.CONTROL_AWB_MODE_CLOUDY_DAYLIGHT
        WhiteBalanceIntent.SHADE -> CaptureRequest.CONTROL_AWB_MODE_SHADE
        WhiteBalanceIntent.TUNGSTEN, WhiteBalanceIntent.WARM -> CaptureRequest.CONTROL_AWB_MODE_INCANDESCENT
        WhiteBalanceIntent.FLUORESCENT -> CaptureRequest.CONTROL_AWB_MODE_FLUORESCENT
        WhiteBalanceIntent.COOL -> CaptureRequest.CONTROL_AWB_MODE_DAYLIGHT
    }

    private fun mapEffect(effect: HardwareEffect): Int = when (effect) {
        HardwareEffect.OFF -> CaptureRequest.CONTROL_EFFECT_MODE_OFF
        HardwareEffect.MONO -> CaptureRequest.CONTROL_EFFECT_MODE_MONO
        HardwareEffect.SEPIA -> CaptureRequest.CONTROL_EFFECT_MODE_SEPIA
        HardwareEffect.NEGATIVE -> CaptureRequest.CONTROL_EFFECT_MODE_NEGATIVE
        HardwareEffect.AQUA -> CaptureRequest.CONTROL_EFFECT_MODE_AQUA
        HardwareEffect.POSTERIZE -> CaptureRequest.CONTROL_EFFECT_MODE_POSTERIZE
    }

    private fun mapSceneMode(program: CaptureProgram): Int? = when (program) {
        CaptureProgram.PORTRAIT -> CaptureRequest.CONTROL_SCENE_MODE_PORTRAIT
        CaptureProgram.NIGHT -> CaptureRequest.CONTROL_SCENE_MODE_NIGHT
        CaptureProgram.HDR -> CaptureRequest.CONTROL_SCENE_MODE_HDR
        CaptureProgram.ACTION -> CaptureRequest.CONTROL_SCENE_MODE_ACTION
        CaptureProgram.LANDSCAPE -> CaptureRequest.CONTROL_SCENE_MODE_LANDSCAPE
        else -> null
    }

    private fun mapExtension(preference: ExtensionPreference): Int? = when (preference) {
        ExtensionPreference.NONE -> null
        ExtensionPreference.AUTO -> ExtensionMode.AUTO
        ExtensionPreference.BOKEH -> ExtensionMode.BOKEH
        ExtensionPreference.HDR -> ExtensionMode.HDR
        ExtensionPreference.NIGHT -> ExtensionMode.NIGHT
        ExtensionPreference.FACE_RETOUCH -> ExtensionMode.FACE_RETOUCH
    }

    private fun extensionModes(): List<Pair<String, Int>> = listOf(
        ExtensionPreference.AUTO.name to ExtensionMode.AUTO,
        ExtensionPreference.BOKEH.name to ExtensionMode.BOKEH,
        ExtensionPreference.HDR.name to ExtensionMode.HDR,
        ExtensionPreference.NIGHT.name to ExtensionMode.NIGHT,
        ExtensionPreference.FACE_RETOUCH.name to ExtensionMode.FACE_RETOUCH
    )
}

data class AppliedCameraPlan(
    val success: Boolean,
    val applied: List<String>,
    val skipped: List<String>
)
