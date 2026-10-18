package ai.auraforge.director.camera

import ai.auraforge.director.ml.FrameAnalyzer
import android.content.Context
import android.hardware.camera2.CaptureRequest
import androidx.camera.camera2.interop.Camera2CameraControl
import androidx.camera.camera2.interop.CaptureRequestOptions
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.util.concurrent.ExecutorService
import java.util.concurrent.TimeUnit

@OptIn(ExperimentalCamera2Interop::class)
class CameraEngine(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner,
    private val previewView: PreviewView,
    private val analyzer: FrameAnalyzer,
    private val analysisExecutor: ExecutorService
) {
    private var provider: ProcessCameraProvider? = null
    private var camera: Camera? = null
    private var facing = CameraSelector.LENS_FACING_BACK

    lateinit var imageCapture: ImageCapture
        private set

    var capabilities: CameraCapabilities? = null
        private set

    val isFrontCamera: Boolean get() = facing == CameraSelector.LENS_FACING_FRONT

    fun start(onReady: (CameraCapabilities) -> Unit, onError: (Throwable) -> Unit) {
        val future = ProcessCameraProvider.getInstance(context)
        future.addListener({
            runCatching {
                provider = future.get()
                bind()
                capabilities?.let(onReady)
            }.onFailure(onError)
        }, ContextCompat.getMainExecutor(context))
    }

    fun flip(onReady: (CameraCapabilities) -> Unit = {}) {
        val p = provider ?: return
        val previous = facing
        val requested = if (facing == CameraSelector.LENS_FACING_BACK) {
            CameraSelector.LENS_FACING_FRONT
        } else {
            CameraSelector.LENS_FACING_BACK
        }
        val selector = CameraSelector.Builder().requireLensFacing(requested).build()
        val available = runCatching { p.hasCamera(selector) }.getOrDefault(false)
        if (!available) return

        facing = requested
        runCatching {
            bind()
            capabilities?.let(onReady)
        }.onFailure {
            facing = previous
            runCatching { bind() }
        }
    }

    private fun bind() {
        val p = provider ?: return
        val selector = CameraSelector.Builder().requireLensFacing(facing).build()

        runCatching {
            bindUseCases(p, selector, preferHighResolution = true)
        }.recoverCatching {
            p.unbindAll()
            bindUseCases(p, selector, preferHighResolution = false)
        }.getOrThrow()
    }

    private fun bindUseCases(
        p: ProcessCameraProvider,
        selector: CameraSelector,
        preferHighResolution: Boolean
    ) {
        p.unbindAll()

        val preview = Preview.Builder()
            .build()
            .also { it.setSurfaceProvider(previewView.surfaceProvider) }

        val captureBuilder = ImageCapture.Builder()
            .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)

        if (preferHighResolution) {
            val highResolutionSelector = ResolutionSelector.Builder()
                .setAllowedResolutionMode(ResolutionSelector.PREFER_HIGHER_RESOLUTION_OVER_CAPTURE_RATE)
                .build()
            captureBuilder.setResolutionSelector(highResolutionSelector)
        }

        imageCapture = captureBuilder.build()

        val analysis = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
            .build()
            .also { it.setAnalyzer(analysisExecutor, analyzer) }

        camera = p.bindToLifecycle(lifecycleOwner, selector, preview, imageCapture, analysis)
        capabilities = CameraCapabilityInspector.inspect(camera!!.cameraInfo)
    }

    fun zoomRange(): ClosedFloatingPointRange<Float> {
        val z = camera?.cameraInfo?.zoomState?.value
        return if (z == null) 1f..1f else z.minZoomRatio..z.maxZoomRatio
    }

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
        c.cameraControl.enableTorch(!current).addListener(
            { onChanged(!current) },
            ContextCompat.getMainExecutor(context)
        )
    }

    fun focusAt(viewX: Float, viewY: Float) {
        val c = camera ?: return
        val point = previewView.meteringPointFactory.createPoint(viewX, viewY)
        val action = androidx.camera.core.FocusMeteringAction.Builder(
            point,
            androidx.camera.core.FocusMeteringAction.FLAG_AF or
                androidx.camera.core.FocusMeteringAction.FLAG_AE or
                androidx.camera.core.FocusMeteringAction.FLAG_AWB
        ).setAutoCancelDuration(3, TimeUnit.SECONDS).build()
        c.cameraControl.startFocusAndMetering(action)
    }

    fun setExposureCompensation(index: Int) {
        val c = camera ?: return
        val range = c.cameraInfo.exposureState.exposureCompensationRange
        if (range.lower > range.upper) return
        c.cameraControl.setExposureCompensationIndex(index.coerceIn(range.lower, range.upper))
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
            builder.setCaptureRequestOption(
                CaptureRequest.LENS_FOCUS_DISTANCE,
                focusDistance.coerceIn(0f, caps.minFocusDistance ?: focusDistance)
            )
        }

        Camera2CameraControl.from(c.cameraControl).setCaptureRequestOptions(builder.build())
        return true
    }

    fun restoreAuto() {
        val c = camera ?: return
        Camera2CameraControl.from(c.cameraControl).clearCaptureRequestOptions()
    }

    fun close() {
        runCatching { provider?.unbindAll() }
        provider = null
        camera = null
    }
}
