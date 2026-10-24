package com.saunak.director.camera

import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.TotalCaptureResult
import android.util.Log
import android.view.Surface
import androidx.camera.camera2.interop.Camera2CameraControl
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.Camera2Interop
import androidx.camera.camera2.interop.CaptureRequestOptions
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ExperimentalImageCaptureOutputFormat
import androidx.camera.core.FocusMeteringAction
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.MeteringPointFactory
import androidx.camera.core.Preview
import androidx.camera.extensions.ExtensionMode
import androidx.camera.extensions.ExtensionsManager
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.google.common.util.concurrent.ListenableFuture
import kotlinx.coroutines.CancellableContinuation
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.delay
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withTimeoutOrNull
import java.io.File
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

@OptIn(ExperimentalCamera2Interop::class, ExperimentalImageCaptureOutputFormat::class)
class CameraEngine(private val context: Context) : AutoCloseable {
    private var provider: ProcessCameraProvider? = null
    private var camera: Camera? = null
    private var imageCapture: ImageCapture? = null
    private var imageAnalysis: ImageAnalysis? = null
    private var preview: Preview? = null
    private var boundView: PreviewView? = null
    private var owner: LifecycleOwner? = null
    private var extensionsManager: ExtensionsManager? = null
    private var lensFacing = CameraSelector.LENS_FACING_BACK
    private var currentExtension = DirectorExtension.NONE
    private val captureMutex = Mutex()

    private val analysisExecutor = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "director-analysis-frame").apply { priority = Thread.NORM_PRIORITY - 1 }
    }
    private val frameLock = Any()
    private var pendingFrame: CompletableDeferred<Bitmap>? = null
    private val rawMetadataByTimestamp = ConcurrentHashMap<Long, RawCaptureMetadata>()

    @Volatile private var lastAf: Int? = null
    @Volatile private var lastAe: Int? = null
    @Volatile private var lastAwb: Int? = null
    @Volatile private var lastExposureTimeNs: Long? = null
    @Volatile private var lastSensitivityIso: Int? = null
    @Volatile private var lastFrameDurationNs: Long? = null
    @Volatile private var lastSensorTimestampNs: Long? = null
    @Volatile private var lastRollingShutterSkewNs: Long? = null
    @Volatile private var lastFocalLengthMm: Float? = null
    @Volatile private var lastAperture: Float? = null
    @Volatile private var lastFocusDistanceDiopters: Float? = null

    private var userZoom = 1f
    private var userEvIndex = 0
    private var userTorch = false
    private var userFlash = StillFlashMode.OFF
    private var userExtension = DirectorExtension.NONE

    private val captureCallback = object : CameraCaptureSession.CaptureCallback() {
        override fun onCaptureCompleted(session: CameraCaptureSession, request: CaptureRequest, result: TotalCaptureResult) {
            lastAf = result.get(CaptureResult.CONTROL_AF_STATE)
            lastAe = result.get(CaptureResult.CONTROL_AE_STATE)
            lastAwb = result.get(CaptureResult.CONTROL_AWB_STATE)
            lastExposureTimeNs = result.get(CaptureResult.SENSOR_EXPOSURE_TIME)
            lastSensitivityIso = result.get(CaptureResult.SENSOR_SENSITIVITY)
            lastFrameDurationNs = result.get(CaptureResult.SENSOR_FRAME_DURATION)
            lastSensorTimestampNs = result.get(CaptureResult.SENSOR_TIMESTAMP)
            lastRollingShutterSkewNs = result.get(CaptureResult.SENSOR_ROLLING_SHUTTER_SKEW)
            lastFocalLengthMm = result.get(CaptureResult.LENS_FOCAL_LENGTH)
            lastAperture = result.get(CaptureResult.LENS_APERTURE)
            lastFocusDistanceDiopters = result.get(CaptureResult.LENS_FOCUS_DISTANCE)
            RawCaptureMetadata.fromResult(activeCameraId(), result)?.let { metadata ->
                rawMetadataByTimestamp[metadata.telemetry.sensorTimestampNs] = metadata
                trimRawMetadataHistory(metadata.telemetry.sensorTimestampNs)
            }
        }
    }

    suspend fun bind(owner: LifecycleOwner, view: PreviewView) {
        val p = ProcessCameraProvider.getInstance(context).await()
        provider = p
        this.owner = owner
        boundView = view
        extensionsManager = runCatching { ExtensionsManager.getInstanceAsync(context, p).await() }.getOrNull()
        bindInternal(currentExtension)
        rememberCurrentAsUserState()
    }

    private fun baseSelector(): CameraSelector = CameraSelector.Builder().requireLensFacing(lensFacing).build()

    private fun extensionMode(extension: DirectorExtension): Int = when (extension) {
        DirectorExtension.AUTO -> ExtensionMode.AUTO
        DirectorExtension.HDR -> ExtensionMode.HDR
        DirectorExtension.NIGHT -> ExtensionMode.NIGHT
        DirectorExtension.BOKEH -> ExtensionMode.BOKEH
        DirectorExtension.FACE_RETOUCH -> ExtensionMode.FACE_RETOUCH
        DirectorExtension.NONE -> ExtensionMode.NONE
    }

    private fun effectiveSelector(extension: DirectorExtension): CameraSelector {
        val base = baseSelector()
        if (extension == DirectorExtension.NONE) return base
        val manager = extensionsManager ?: error("Camera extensions unavailable")
        val mode = extensionMode(extension)
        check(manager.isExtensionAvailable(base, mode)) { "Extension ${extension.wireName} unsupported" }
        return manager.getExtensionEnabledCameraSelector(base, mode)
    }

    private fun buildPreview(view: PreviewView, interop: Boolean): Preview {
        val builder = Preview.Builder()
        if (interop) Camera2Interop.Extender(builder).setSessionCaptureCallback(captureCallback)
        return builder.build().also { it.surfaceProvider = view.surfaceProvider }
    }

    private fun buildImageCapture(view: PreviewView): ImageCapture = ImageCapture.Builder()
        .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
        .setTargetRotation(view.display?.rotation ?: Surface.ROTATION_0)
        .build()
        .also { it.flashMode = flashModeToCameraX(userFlash) }

    private fun buildRawJpegImageCapture(view: PreviewView): ImageCapture = ImageCapture.Builder()
        .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
        .setOutputFormat(ImageCapture.OUTPUT_FORMAT_RAW_JPEG)
        .setTargetRotation(view.display?.rotation ?: Surface.ROTATION_0)
        .build()
        .also { it.flashMode = ImageCapture.FLASH_MODE_OFF }

    private fun buildImageAnalysis(view: PreviewView): ImageAnalysis {
        val analysis = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
            .setTargetRotation(view.display?.rotation ?: Surface.ROTATION_0)
            .build()

        analysis.setAnalyzer(analysisExecutor) { image ->
            val waiter = synchronized(frameLock) {
                val current = pendingFrame
                if (current?.isActive == true) pendingFrame = null
                current
            }
            if (waiter == null || !waiter.isActive) {
                image.close()
                return@setAnalyzer
            }

            try {
                var bitmap = image.toBitmap()
                val rotation = image.imageInfo.rotationDegrees
                if (rotation != 0) {
                    val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
                    val rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
                    if (rotated !== bitmap && !bitmap.isRecycled) bitmap.recycle()
                    bitmap = rotated
                }
                waiter.complete(bitmap)
            } catch (t: Throwable) {
                waiter.completeExceptionally(t)
            } finally {
                image.close()
            }
        }
        return analysis
    }

    private fun clearAnalysisUseCase() {
        runCatching { imageAnalysis?.clearAnalyzer() }
        imageAnalysis = null
        synchronized(frameLock) {
            pendingFrame?.cancel()
            pendingFrame = null
        }
    }

    private fun resetCaptureTelemetry() {
        lastAf = null
        lastAe = null
        lastAwb = null
        lastExposureTimeNs = null
        lastSensitivityIso = null
        lastFrameDurationNs = null
        lastSensorTimestampNs = null
        lastRollingShutterSkewNs = null
        lastFocalLengthMm = null
        lastAperture = null
        lastFocusDistanceDiopters = null
        rawMetadataByTimestamp.clear()
    }

    private fun bindInternal(extension: DirectorExtension) {
        val p = checkNotNull(provider)
        val lifecycleOwner = checkNotNull(owner)
        val view = checkNotNull(boundView)
        clearAnalysisUseCase()
        p.unbindAll()
        resetCaptureTelemetry()

        preview = buildPreview(view, interop = extension == DirectorExtension.NONE)
        imageCapture = buildImageCapture(view)
        val selector = effectiveSelector(extension)

        try {
            if (extension == DirectorExtension.NONE) {
                val analysis = buildImageAnalysis(view)
                imageAnalysis = analysis
                camera = try {
                    p.bindToLifecycle(lifecycleOwner, selector, preview, imageCapture, analysis)
                } catch (analysisBindFailure: Throwable) {
                    Log.w("DirectorCamera", "Live analysis unavailable; using still fallback", analysisBindFailure)
                    analysis.clearAnalyzer()
                    imageAnalysis = null
                    p.unbindAll()
                    preview = buildPreview(view, interop = true)
                    imageCapture = buildImageCapture(view)
                    p.bindToLifecycle(lifecycleOwner, selector, preview, imageCapture)
                }
            } else {
                camera = p.bindToLifecycle(lifecycleOwner, selector, preview, imageCapture)
            }
            currentExtension = extension
        } catch (t: Throwable) {
            if (extension == DirectorExtension.NONE) throw t
            Log.w("DirectorCamera", "Extension bind failed; restoring normal camera", t)
            clearAnalysisUseCase()
            p.unbindAll()
            preview = buildPreview(view, interop = true)
            imageCapture = buildImageCapture(view)
            val analysis = buildImageAnalysis(view)
            imageAnalysis = analysis
            camera = try {
                p.bindToLifecycle(lifecycleOwner, baseSelector(), preview, imageCapture, analysis)
            } catch (analysisBindFailure: Throwable) {
                Log.w("DirectorCamera", "Fallback live analysis unavailable", analysisBindFailure)
                analysis.clearAnalyzer()
                imageAnalysis = null
                p.unbindAll()
                preview = buildPreview(view, interop = true)
                imageCapture = buildImageCapture(view)
                p.bindToLifecycle(lifecycleOwner, baseSelector(), preview, imageCapture)
            }
            currentExtension = DirectorExtension.NONE
        }
    }

    suspend fun captureAnalysisFrame(timeoutMs: Long = 800L): Bitmap? {
        if (imageAnalysis == null) return null
        val request = CompletableDeferred<Bitmap>()
        synchronized(frameLock) {
            pendingFrame?.cancel()
            pendingFrame = request
        }
        return try {
            withTimeoutOrNull(timeoutMs.coerceIn(150L, 1500L)) { request.await() }
        } catch (_: Throwable) {
            null
        } finally {
            synchronized(frameLock) {
                if (pendingFrame === request) pendingFrame = null
            }
        }
    }

    fun switchLens(): Boolean {
        val p = provider ?: return false
        val desired = if (lensFacing == CameraSelector.LENS_FACING_BACK) CameraSelector.LENS_FACING_FRONT else CameraSelector.LENS_FACING_BACK
        val selector = CameraSelector.Builder().requireLensFacing(desired).build()
        if (!p.hasCamera(selector)) return false
        lensFacing = desired
        userZoom = 1f
        userEvIndex = 0
        userTorch = false
        userFlash = StillFlashMode.OFF
        userExtension = DirectorExtension.NONE
        currentExtension = DirectorExtension.NONE
        bindInternal(DirectorExtension.NONE)
        rememberCurrentAsUserState()
        return true
    }

    fun tapToFocus(factory: MeteringPointFactory, x: Float, y: Float) {
        val point = factory.createPoint(x, y)
        val action = FocusMeteringAction.Builder(point, FocusMeteringAction.FLAG_AF or FocusMeteringAction.FLAG_AE or FocusMeteringAction.FLAG_AWB)
            .setAutoCancelDuration(4, TimeUnit.SECONDS)
            .build()
        camera?.cameraControl?.startFocusAndMetering(action)
    }

    suspend fun aiFocusNormalized(x: Float, y: Float) {
        val view = boundView ?: return
        if (view.width <= 0 || view.height <= 0) return
        val point = view.meteringPointFactory.createPoint(
            x.coerceIn(0f, 1f) * view.width,
            y.coerceIn(0f, 1f) * view.height
        )
        val action = FocusMeteringAction.Builder(point, FocusMeteringAction.FLAG_AF or FocusMeteringAction.FLAG_AE or FocusMeteringAction.FLAG_AWB)
            .setAutoCancelDuration(3, TimeUnit.SECONDS)
            .build()
        camera?.cameraControl?.startFocusAndMetereringSafe(action)
    }

    private suspend fun androidx.camera.core.CameraControl.startFocusAndMetereringSafe(action: FocusMeteringAction) {
        startFocusAndMetering(action).await()
    }

    fun userSetZoom(ratio: Float) {
        userZoom = clampZoom(ratio)
        camera?.cameraControl?.setZoomRatio(userZoom)
    }

    suspend fun aiSetZoom(ratio: Float) {
        camera?.cameraControl?.setZoomRatio(clampZoom(ratio))?.await()
    }

    private fun clampZoom(ratio: Float): Float {
        val state = camera?.cameraInfo?.zoomState?.value ?: return 1f
        return ratio.coerceIn(state.minZoomRatio, state.maxZoomRatio)
    }

    fun userSetExposureIndex(index: Int) {
        val e = camera?.cameraInfo?.exposureState ?: return
        if (!e.isExposureCompensationSupported) return
        userEvIndex = index.coerceIn(e.exposureCompensationRange.lower, e.exposureCompensationRange.upper)
        camera?.cameraControl?.setExposureCompensationIndex(userEvIndex)
    }

    suspend fun aiSetExposureEv(ev: Float) {
        val e = camera?.cameraInfo?.exposureState ?: return
        if (!e.isExposureCompensationSupported || e.exposureCompensationStep.toFloat() == 0f) return
        val index = (ev / e.exposureCompensationStep.toFloat()).toInt()
            .coerceIn(e.exposureCompensationRange.lower, e.exposureCompensationRange.upper)
        camera?.cameraControl?.setExposureCompensationIndex(index)?.await()
    }

    fun userSetTorch(enabled: Boolean) {
        val c = camera ?: return
        if (!c.cameraInfo.hasFlashUnit()) return
        userTorch = enabled
        c.cameraControl.enableTorch(enabled)
    }

    suspend fun aiSetTorch(enabled: Boolean) {
        val c = camera ?: return
        if (c.cameraInfo.hasFlashUnit()) c.cameraControl.enableTorch(enabled).await()
    }

    fun userSetFlash(mode: StillFlashMode) {
        userFlash = mode
        imageCapture?.flashMode = flashModeToCameraX(mode)
    }

    fun aiSetFlash(mode: StillFlashMode) {
        imageCapture?.flashMode = flashModeToCameraX(mode)
    }

    private fun flashModeToCameraX(mode: StillFlashMode): Int = when (mode) {
        StillFlashMode.OFF -> ImageCapture.FLASH_MODE_OFF
        StillFlashMode.AUTO -> ImageCapture.FLASH_MODE_AUTO
        StillFlashMode.ON -> ImageCapture.FLASH_MODE_ON
    }

    suspend fun aiSetExtension(extension: DirectorExtension) {
        if (extension != currentExtension) bindInternal(extension)
    }

    suspend fun restoreUserState() {
        if (currentExtension != userExtension) bindInternal(userExtension)
        aiSetZoom(userZoom)
        val e = camera?.cameraInfo?.exposureState
        if (e?.isExposureCompensationSupported == true) {
            camera?.cameraControl?.setExposureCompensationIndex(
                userEvIndex.coerceIn(e.exposureCompensationRange.lower, e.exposureCompensationRange.upper)
            )?.await()
        }
        aiSetTorch(userTorch)
        imageCapture?.flashMode = flashModeToCameraX(userFlash)
    }

    private fun rememberCurrentAsUserState() {
        val s = snapshot()
        userZoom = s.zoomRatio
        userEvIndex = s.exposureCompensationIndex
        userTorch = s.torchEnabled
        userFlash = s.flashMode
        userExtension = s.extension
    }

    suspend fun awaitSettledState(timeoutMs: Long = 900L) {
        if (currentExtension != DirectorExtension.NONE) {
            delay(280)
            return
        }
        val start = System.nanoTime()
        while ((System.nanoTime() - start) / 1_000_000L < timeoutMs) {
            if (threeASettled()) return
            delay(24)
        }
        delay(80)
    }

    private fun threeASettled(): Boolean {
        val af = lastAf
        val ae = lastAe
        val awb = lastAwb
        val afOk = af == null || af == CaptureResult.CONTROL_AF_STATE_INACTIVE ||
            af == CaptureResult.CONTROL_AF_STATE_FOCUSED_LOCKED ||
            af == CaptureResult.CONTROL_AF_STATE_NOT_FOCUSED_LOCKED ||
            af == CaptureResult.CONTROL_AF_STATE_PASSIVE_FOCUSED ||
            af == CaptureResult.CONTROL_AF_STATE_PASSIVE_UNFOCUSED
        val aeOk = ae == null || ae == CaptureResult.CONTROL_AE_STATE_INACTIVE ||
            ae == CaptureResult.CONTROL_AE_STATE_CONVERGED || ae == CaptureResult.CONTROL_AE_STATE_LOCKED ||
            ae == CaptureResult.CONTROL_AE_STATE_FLASH_REQUIRED
        val awbOk = awb == null || awb == CaptureResult.CONTROL_AWB_STATE_INACTIVE ||
            awb == CaptureResult.CONTROL_AWB_STATE_CONVERGED || awb == CaptureResult.CONTROL_AWB_STATE_LOCKED
        return afOk && aeOk && awbOk
    }

    fun capabilities(): CapabilityRegistry {
        val c = camera ?: return CapabilityRegistry(
            lensFacing, 1f, 1f, 0, 0, 0f, false, false, false, false, false,
            allowedToolIds = setOf(30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43)
        )
        val zoom = c.cameraInfo.zoomState.value
        val e = c.cameraInfo.exposureState
        val characteristics = activeCameraCharacteristics()
        val hwCaps = characteristics?.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES)?.toSet().orEmpty()
        val sensitivity = characteristics?.get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE)
        val exposureTimes = characteristics?.get(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE)
        val focalLengths = characteristics?.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.toList().orEmpty()
        val minFocus = characteristics?.get(CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE)
        val opticalModes = characteristics?.get(CameraCharacteristics.LENS_INFO_AVAILABLE_OPTICAL_STABILIZATION)?.toList().orEmpty()
        val videoModes = characteristics?.get(CameraCharacteristics.CONTROL_AVAILABLE_VIDEO_STABILIZATION_MODES)?.toList().orEmpty()
        val ext = supportedExtensions()
        val focusSupported = focusMeteringSupported(c)
        val ids = mutableSetOf(30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43)
        if (focusSupported) ids += 1
        if (zoom != null && zoom.maxZoomRatio > zoom.minZoomRatio) ids += 2
        if (e.isExposureCompensationSupported) ids += 3
        if (c.cameraInfo.hasFlashUnit()) ids += setOf(4, 5)
        if (ext.isNotEmpty()) ids += 6
        return CapabilityRegistry(
            lensFacing = lensFacing,
            minZoom = zoom?.minZoomRatio ?: 1f,
            maxZoom = zoom?.maxZoomRatio ?: 1f,
            exposureMinIndex = if (e.isExposureCompensationSupported) e.exposureCompensationRange.lower else 0,
            exposureMaxIndex = if (e.isExposureCompensationSupported) e.exposureCompensationRange.upper else 0,
            exposureStepEv = if (e.isExposureCompensationSupported) e.exposureCompensationStep.toFloat() else 0f,
            hasFlash = c.cameraInfo.hasFlashUnit(),
            torchSupported = c.cameraInfo.hasFlashUnit(),
            focusMeteringSupported = focusSupported,
            manualSensorSupported = CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_MANUAL_SENSOR in hwCaps,
            rawSupported = CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_RAW in hwCaps,
            isoMin = sensitivity?.lower,
            isoMax = sensitivity?.upper,
            exposureTimeMinNs = exposureTimes?.lower,
            exposureTimeMaxNs = exposureTimes?.upper,
            minimumFocusDistance = minFocus,
            focalLengthsMm = focalLengths,
            opticalStabilizationModes = opticalModes,
            videoStabilizationModes = videoModes,
            extensions = ext,
            allowedToolIds = ids
        )
    }

    private fun focusMeteringSupported(c: Camera): Boolean {
        val view = boundView ?: return false
        if (view.width <= 0 || view.height <= 0) return false
        return runCatching {
            val point = view.meteringPointFactory.createPoint(view.width / 2f, view.height / 2f)
            val action = FocusMeteringAction.Builder(point, FocusMeteringAction.FLAG_AF or FocusMeteringAction.FLAG_AE).build()
            c.cameraInfo.isFocusMeteringSupported(action)
        }.getOrDefault(false)
    }

    fun activeCameraId(): String? = camera?.let { current ->
        runCatching { Camera2CameraInfo.from(current.cameraInfo).cameraId }.getOrNull()
    }

    private fun activeCameraCharacteristics(): CameraCharacteristics? {
        val id = activeCameraId() ?: return null
        return runCatching {
            context.getSystemService(CameraManager::class.java).getCameraCharacteristics(id)
        }.getOrNull()
    }

    fun sensorTelemetry(): CameraSensorTelemetry? {
        val exposure = lastExposureTimeNs ?: return null
        val iso = lastSensitivityIso ?: return null
        val timestamp = lastSensorTimestampNs ?: return null
        val frameDuration = (lastFrameDurationNs ?: exposure).coerceAtLeast(exposure)
        return CameraSensorTelemetry(
            cameraId = activeCameraId(),
            sensorTimestampNs = timestamp,
            exposureTimeNs = exposure,
            sensitivityIso = iso,
            frameDurationNs = frameDuration,
            rollingShutterSkewNs = lastRollingShutterSkewNs,
            focalLengthMm = lastFocalLengthMm,
            aperture = lastAperture,
            focusDistanceDiopters = lastFocusDistanceDiopters,
            afState = lastAf,
            aeState = lastAe,
            awbState = lastAwb
        )
    }

    private fun supportedExtensions(): Set<DirectorExtension> {
        val manager = extensionsManager ?: return emptySet()
        val base = baseSelector()
        return DirectorExtension.entries.filter {
            it != DirectorExtension.NONE &&
                runCatching { manager.isExtensionAvailable(base, extensionMode(it)) }.getOrDefault(false)
        }.toSet()
    }

    fun snapshot(): CameraSnapshot {
        val c = camera
        val flash = when (imageCapture?.flashMode) {
            ImageCapture.FLASH_MODE_AUTO -> StillFlashMode.AUTO
            ImageCapture.FLASH_MODE_ON -> StillFlashMode.ON
            else -> StillFlashMode.OFF
        }
        return CameraSnapshot(
            lensFacing,
            c?.cameraInfo?.zoomState?.value?.zoomRatio ?: 1f,
            c?.cameraInfo?.exposureState?.exposureCompensationIndex ?: 0,
            c?.cameraInfo?.torchState?.value == 1,
            flash,
            currentExtension,
            surfaceRotationToDegrees(boundView?.display?.rotation ?: Surface.ROTATION_0)
        )
    }

    fun uiState(): CameraUiState {
        val caps = capabilities()
        val s = snapshot()
        return CameraUiState(
            caps.minZoom, caps.maxZoom, s.zoomRatio,
            caps.exposureStepEv > 0f, caps.exposureMinIndex, caps.exposureMaxIndex,
            s.exposureCompensationIndex, caps.exposureStepEv,
            caps.torchSupported, s.torchEnabled, s.flashMode, lensFacing, caps.extensions
        )
    }

    @OptIn(ExperimentalImageCaptureOutputFormat::class)
    fun supportedImageCaptureFormats(): Set<Int> = camera?.let { current ->
        runCatching { ImageCapture.getImageCaptureCapabilities(current.cameraInfo).supportedOutputFormats.toSet() }
            .getOrDefault(setOf(ImageCapture.OUTPUT_FORMAT_JPEG))
    } ?: emptySet()

    @OptIn(ExperimentalImageCaptureOutputFormat::class)
    fun rawJpegCaptureSupported(): Boolean {
        val caps = capabilities()
        return currentExtension == DirectorExtension.NONE &&
            caps.rawSupported && caps.manualSensorSupported &&
            ImageCapture.OUTPUT_FORMAT_RAW_JPEG in supportedImageCaptureFormats() &&
            activeCameraCharacteristics()?.let(RawSensorExtractor::calibration) != null
    }

    suspend fun captureRawJpegBurst(specs: List<ManualRawFrameSpec>): List<RawJpegCaptureFrame> = captureMutex.withLock {
        require(specs.size in 1..MAX_RAW_BURST_FRAMES) { "RAW burst must contain 1..$MAX_RAW_BURST_FRAMES frames" }
        check(currentExtension == DirectorExtension.NONE) { "RAW capture is mutually exclusive with OEM camera extensions" }
        val caps = capabilities()
        check(caps.rawSupported && caps.manualSensorSupported) { "RAW + MANUAL_SENSOR capability required" }
        check(ImageCapture.OUTPUT_FORMAT_RAW_JPEG in supportedImageCaptureFormats()) { "RAW+JPEG output unsupported by active CameraX camera" }
        specs.forEach { validateManualSpec(it, caps) }

        val characteristics = checkNotNull(activeCameraCharacteristics()) { "Camera characteristics unavailable" }
        val calibration = checkNotNull(RawSensorExtractor.calibration(characteristics)) { "Unsupported RAW CFA/calibration" }
        val p = checkNotNull(provider)
        val lifecycleOwner = checkNotNull(owner)
        val view = checkNotNull(boundView)
        val results = ArrayList<RawJpegCaptureFrame>(specs.size)
        var rawControl: Camera2CameraControl? = null

        try {
            clearAnalysisUseCase()
            p.unbindAll()
            resetCaptureTelemetry()
            preview = buildPreview(view, interop = true)
            val rawCapture = buildRawJpegImageCapture(view)
            imageCapture = rawCapture
            camera = p.bindToLifecycle(lifecycleOwner, baseSelector(), preview, rawCapture)
            currentExtension = DirectorExtension.NONE
            rawControl = Camera2CameraControl.from(checkNotNull(camera).cameraControl)

            for (spec in specs) {
                val options = CaptureRequestOptions.Builder()
                    .setCaptureRequestOption(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_OFF)
                    .setCaptureRequestOption(CaptureRequest.SENSOR_EXPOSURE_TIME, spec.exposureTimeNs)
                    .setCaptureRequestOption(CaptureRequest.SENSOR_SENSITIVITY, spec.sensitivityIso)
                    .setCaptureRequestOption(CaptureRequest.SENSOR_FRAME_DURATION, spec.frameDurationNs.coerceAtLeast(spec.exposureTimeNs))
                    .build()
                rawControl.setCaptureRequestOptions(options).await()
                val pending = captureRawJpegInMemory(rawCapture, calibration)
                val metadata = awaitRawMetadata(pending.raw.timestampNs)
                    ?: error("No timestamp-matched Camera2 metadata for RAW frame ${pending.raw.timestampNs}")
                val dynamicBlack = metadata.dynamicBlackLevel
                    ?.takeIf { it.size == 4 }
                    ?.let { com.saunak.director.engine.reconstruction.CfaBlackLevel(it.copyOf()) }
                    ?: pending.raw.raw.blackLevelPattern
                val calibratedRaw = pending.raw.copy(
                    raw = pending.raw.raw.copy(
                        blackLevel = dynamicBlack.offsets.min(),
                        blackLevelPattern = dynamicBlack
                    ),
                    metadata = metadata
                )
                results += RawJpegCaptureFrame(
                    raw = calibratedRaw,
                    jpegBytes = pending.jpegBytes,
                    jpegRotationDegrees = pending.jpegRotationDegrees,
                    metadata = metadata
                )
            }
            results.toList()
        } finally {
            runCatching { rawControl?.clearCaptureRequestOptions()?.await() }
            runCatching { bindInternal(userExtension) }
            runCatching { restoreUserState() }
        }
    }

    suspend fun capturePrivate(file: File): File = captureMutex.withLock {
        file.parentFile?.mkdirs()
        captureToFile(file)
    }

    private suspend fun captureRawJpegInMemory(
        capture: ImageCapture,
        calibration: RawSensorCalibration
    ): PendingRawJpegFrame = suspendCancellableCoroutine { cont ->
        val guard = Any()
        var raw: RawSensorFrame? = null
        var jpeg: ByteArray? = null
        var jpegRotation = 0
        var terminal = false

        fun completeIfReady() {
            val rawFrame = raw
            val jpegBytes = jpeg
            if (!terminal && rawFrame != null && jpegBytes != null && cont.isActive) {
                terminal = true
                cont.resume(PendingRawJpegFrame(rawFrame, jpegBytes, jpegRotation))
            }
        }

        capture.takePicture(
            ContextCompat.getMainExecutor(context),
            object : ImageCapture.OnImageCapturedCallback() {
                override fun onCaptureSuccess(image: ImageProxy) {
                    try {
                        synchronized(guard) {
                            if (terminal || !cont.isActive) return
                            when (image.format) {
                                ImageFormat.RAW_SENSOR -> raw = RawSensorExtractor.extract(image, calibration)
                                ImageFormat.JPEG -> {
                                    jpeg = copyCompressedImage(image)
                                    jpegRotation = image.imageInfo.rotationDegrees
                                }
                                else -> error("Unexpected RAW+JPEG callback format ${image.format}")
                            }
                            completeIfReady()
                        }
                    } catch (t: Throwable) {
                        synchronized(guard) {
                            if (!terminal && cont.isActive) {
                                terminal = true
                                cont.resumeWithException(t)
                            }
                        }
                    } finally {
                        image.close()
                    }
                }

                override fun onError(exception: ImageCaptureException) {
                    synchronized(guard) {
                        if (!terminal && cont.isActive) {
                            terminal = true
                            cont.resumeWithException(exception)
                        }
                    }
                }
            }
        )
    }

    private fun copyCompressedImage(image: ImageProxy): ByteArray {
        require(image.planes.isNotEmpty())
        val buffer = image.planes[0].buffer.duplicate()
        val bytes = ByteArray(buffer.remaining())
        buffer.get(bytes)
        require(bytes.isNotEmpty()) { "Camera returned empty JPEG payload" }
        return bytes
    }

    private suspend fun awaitRawMetadata(timestampNs: Long): RawCaptureMetadata? {
        rawMetadataByTimestamp[timestampNs]?.let { return it }
        repeat(12) {
            delay(10)
            rawMetadataByTimestamp[timestampNs]?.let { return it }
        }
        return rawMetadataByTimestamp.entries
            .map { it.key to it.value }
            .filter { kotlin.math.abs(it.first - timestampNs) <= METADATA_TIMESTAMP_TOLERANCE_NS }
            .minByOrNull { kotlin.math.abs(it.first - timestampNs) }
            ?.second
    }

    private fun validateManualSpec(spec: ManualRawFrameSpec, caps: CapabilityRegistry) {
        require(spec.exposureTimeNs > 0 && spec.sensitivityIso > 0)
        caps.exposureTimeMinNs?.let { require(spec.exposureTimeNs >= it) }
        caps.exposureTimeMaxNs?.let { require(spec.exposureTimeNs <= it) }
        caps.isoMin?.let { require(spec.sensitivityIso >= it) }
        caps.isoMax?.let { require(spec.sensitivityIso <= it) }
        require(spec.frameDurationNs >= spec.exposureTimeNs)
    }

    private fun trimRawMetadataHistory(latestTimestampNs: Long) {
        if (rawMetadataByTimestamp.size <= MAX_METADATA_HISTORY) return
        rawMetadataByTimestamp.keys.removeIf { timestamp -> latestTimestampNs - timestamp > METADATA_RETENTION_NS }
        if (rawMetadataByTimestamp.size > MAX_METADATA_HISTORY) {
            rawMetadataByTimestamp.keys.sorted().take(rawMetadataByTimestamp.size - MAX_METADATA_HISTORY)
                .forEach(rawMetadataByTimestamp::remove)
        }
    }

    private suspend fun captureToFile(file: File): File = suspendCancellableCoroutine { cont ->
        val capture = imageCapture ?: return@suspendCancellableCoroutine cont.resumeWithException(
            IllegalStateException("Camera not ready")
        )
        capture.targetRotation = boundView?.display?.rotation ?: Surface.ROTATION_0
        val options = ImageCapture.OutputFileOptions.Builder(file).build()
        capture.takePicture(
            options,
            ContextCompat.getMainExecutor(context),
            object : ImageCapture.OnImageSavedCallback {
                override fun onImageSaved(outputFileResults: ImageCapture.OutputFileResults) {
                    if (cont.isActive) cont.resume(file) else file.delete()
                }

                override fun onError(exception: ImageCaptureException) {
                    if (cont.isActive) cont.resumeWithException(exception) else file.delete()
                }
            }
        )
        cont.invokeOnCancellation { if (file.exists() && file.length() == 0L) file.delete() }
    }

    override fun close() {
        clearAnalysisUseCase()
        analysisExecutor.shutdownNow()
        rawMetadataByTimestamp.clear()
        provider?.unbindAll()
    }

    companion object {
        private const val MAX_RAW_BURST_FRAMES = 8
        private const val MAX_METADATA_HISTORY = 32
        private const val METADATA_RETENTION_NS = 2_000_000_000L
        private const val METADATA_TIMESTAMP_TOLERANCE_NS = 1_000_000L

        internal fun surfaceRotationToDegrees(rotation: Int): Int = when (rotation) {
            Surface.ROTATION_0 -> 0
            Surface.ROTATION_90 -> 90
            Surface.ROTATION_180 -> 180
            Surface.ROTATION_270 -> 270
            else -> 0
        }
    }
}

data class ManualRawFrameSpec(
    val exposureTimeNs: Long,
    val sensitivityIso: Int,
    val frameDurationNs: Long = exposureTimeNs
) {
    init {
        require(exposureTimeNs > 0)
        require(sensitivityIso > 0)
        require(frameDurationNs >= exposureTimeNs)
    }
}

data class RawJpegCaptureFrame(
    val raw: RawSensorFrame,
    val jpegBytes: ByteArray,
    val jpegRotationDegrees: Int,
    val metadata: RawCaptureMetadata
) {
    init {
        require(jpegBytes.isNotEmpty())
        require(jpegRotationDegrees in setOf(0, 90, 180, 270))
        require(raw.timestampNs == metadata.telemetry.sensorTimestampNs)
    }
}

private data class PendingRawJpegFrame(
    val raw: RawSensorFrame,
    val jpegBytes: ByteArray,
    val jpegRotationDegrees: Int
)

data class CameraSensorTelemetry(
    val cameraId: String?,
    val sensorTimestampNs: Long,
    val exposureTimeNs: Long,
    val sensitivityIso: Int,
    val frameDurationNs: Long,
    val rollingShutterSkewNs: Long?,
    val focalLengthMm: Float?,
    val aperture: Float?,
    val focusDistanceDiopters: Float?,
    val afState: Int?,
    val aeState: Int?,
    val awbState: Int?
) {
    init {
        require(sensorTimestampNs > 0)
        require(exposureTimeNs > 0)
        require(sensitivityIso > 0)
        require(frameDurationNs >= exposureTimeNs)
        require(rollingShutterSkewNs == null || rollingShutterSkewNs >= 0)
    }
}

private suspend fun <T> ListenableFuture<T>.await(): T = suspendCancellableCoroutine { cont: CancellableContinuation<T> ->
    addListener(
        {
            try {
                if (cont.isActive) cont.resume(get())
            } catch (t: Throwable) {
                if (cont.isActive) cont.resumeWithException(t)
            }
        },
        Runnable::run
    )
    cont.invokeOnCancellation { cancel(true) }
}
