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
import java.io.BufferedOutputStream
import java.io.DataOutputStream
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlinx.coroutines.CancellableContinuation
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.delay
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withTimeoutOrNull

@OptIn(ExperimentalCamera2Interop::class)
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

    private fun buildImageCapture(
        view: PreviewView,
        outputFormat: Int = ImageCapture.OUTPUT_FORMAT_JPEG,
        captureMode: Int = ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY
    ): ImageCapture = ImageCapture.Builder()
        .setCaptureMode(captureMode)
        .setOutputFormat(outputFormat)
        .setTargetRotation(view.display?.rotation ?: Surface.ROTATION_0)
        .build()
        .also { it.flashMode = if (outputFormat == ImageCapture.OUTPUT_FORMAT_RAW_JPEG) ImageCapture.FLASH_MODE_OFF else flashModeToCameraX(userFlash) }

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

    private fun bindRawJpegMode() {
        val p = checkNotNull(provider)
        val lifecycleOwner = checkNotNull(owner)
        val view = checkNotNull(boundView)
        check(currentExtension == DirectorExtension.NONE) { "RAW capture cannot replace an active OEM extension" }
        clearAnalysisUseCase()
        p.unbindAll()
        resetCaptureTelemetry()

        preview = buildPreview(view, interop = true)
        imageCapture = buildImageCapture(
            view = view,
            outputFormat = ImageCapture.OUTPUT_FORMAT_RAW_JPEG,
            captureMode = ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY
        )
        camera = p.bindToLifecycle(lifecycleOwner, baseSelector(), preview, imageCapture)
        currentExtension = DirectorExtension.NONE
    }

    /** Returns one fresh preview frame. No full-resolution still is captured or written to disk. */
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

    fun supportedImageCaptureFormats(): Set<Int> = camera?.let { c ->
        runCatching { ImageCapture.getImageCaptureCapabilities(c.cameraInfo).supportedOutputFormats.toSet() }.getOrDefault(emptySet())
    } ?: emptySet()

    fun supportsComputationalRawBurst(): Boolean {
        val caps = capabilities()
        return currentExtension == DirectorExtension.NONE &&
            !userTorch && userFlash == StillFlashMode.OFF &&
            caps.rawSupported && caps.manualSensorSupported &&
            ImageCapture.OUTPUT_FORMAT_RAW_JPEG in supportedImageCaptureFormats()
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

    suspend fun capturePrivate(file: File): File = captureMutex.withLock {
        file.parentFile?.mkdirs()
        captureToFile(file)
    }

    suspend fun captureRawJpegBurst(specs: List<ManualRawFrameSpec>): RawJpegBurstCapture = captureMutex.withLock {
        require(specs.isNotEmpty() && specs.size <= MAX_RAW_BURST_FRAMES) { "RAW burst must contain 1..$MAX_RAW_BURST_FRAMES frames" }
        check(supportsComputationalRawBurst()) { "Current camera/session is not certified for Sierra RAW+JPEG" }
        val caps = capabilities()
        specs.forEach { spec ->
            caps.exposureTimeMinNs?.let { require(spec.exposureTimeNs >= it) }
            caps.exposureTimeMaxNs?.let { require(spec.exposureTimeNs <= it) }
            caps.isoMin?.let { require(spec.sensitivityIso >= it) }
            caps.isoMax?.let { require(spec.sensitivityIso <= it) }
        }

        val workingDir = File(context.cacheDir, "sierra_raw_v4/${System.currentTimeMillis()}_${System.nanoTime()}").apply { mkdirs() }
        val frames = ArrayList<RawJpegCapturedFrame>(specs.size)
        var restored = false
        try {
            bindRawJpegMode()
            aiSetZoom(userZoom)
            awaitSettledState(900)

            specs.forEachIndexed { index, spec ->
                applyManualSensor(spec)
                val rawFile = File(workingDir, "frame_${index.toString().padStart(2, '0')}.sraw")
                val jpegFile = File(workingDir, "frame_${index.toString().padStart(2, '0')}.jpg")
                captureRawJpegMemoryPair(rawFile, jpegFile, spec)
                frames += RawJpegCapturedFrame(
                    index = index,
                    rawFile = rawFile,
                    jpegFile = jpegFile,
                    requested = spec,
                    observed = sensorTelemetry()
                )
            }
            RawJpegBurstCapture(workingDir, frames)
        } catch (t: Throwable) {
            runCatching { workingDir.deleteRecursively() }
            throw t
        } finally {
            runCatching { clearManualSensorInterop() }
            runCatching {
                bindInternal(userExtension)
                restored = true
            }
            if (restored) runCatching { restoreUserState() }
        }
    }

    private suspend fun applyManualSensor(spec: ManualRawFrameSpec) {
        val c = checkNotNull(camera) { "Camera not ready" }
        val options = CaptureRequestOptions.Builder()
            .setCaptureRequestOption(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_OFF)
            .setCaptureRequestOption(CaptureRequest.SENSOR_EXPOSURE_TIME, spec.exposureTimeNs)
            .setCaptureRequestOption(CaptureRequest.SENSOR_SENSITIVITY, spec.sensitivityIso)
            .build()
        Camera2CameraControl.from(c.cameraControl).setCaptureRequestOptions(options).await()
    }

    private suspend fun clearManualSensorInterop() {
        val c = camera ?: return
        Camera2CameraControl.from(c.cameraControl).clearCaptureRequestOptions().await()
    }

    private suspend fun captureRawJpegMemoryPair(rawFile: File, jpegFile: File, spec: ManualRawFrameSpec) =
        suspendCancellableCoroutine<Unit> { cont ->
            val capture = imageCapture ?: return@suspendCancellableCoroutine cont.resumeWithException(
                IllegalStateException("RAW camera not ready")
            )
            capture.targetRotation = boundView?.display?.rotation ?: Surface.ROTATION_0
            val gate = Any()
            var rawSaved = false
            var jpegSaved = false
            var failed = false

            fun finishIfComplete() {
                synchronized(gate) {
                    if (!failed && rawSaved && jpegSaved && cont.isActive) cont.resume(Unit)
                }
            }

            capture.takePicture(
                ContextCompat.getMainExecutor(context),
                object : ImageCapture.OnImageCapturedCallback() {
                    override fun onCaptureSuccess(image: ImageProxy) {
                        try {
                            when (image.format) {
                                ImageFormat.RAW_SENSOR -> {
                                    persistRawSensor(image, rawFile, spec)
                                    synchronized(gate) { rawSaved = true }
                                }
                                ImageFormat.JPEG, ImageFormat.JPEG_R -> {
                                    persistCompressedImage(image, jpegFile)
                                    synchronized(gate) { jpegSaved = true }
                                }
                                else -> error("Unexpected RAW+JPEG callback format ${image.format}")
                            }
                            finishIfComplete()
                        } catch (t: Throwable) {
                            synchronized(gate) { failed = true }
                            runCatching { rawFile.delete() }
                            runCatching { jpegFile.delete() }
                            if (cont.isActive) cont.resumeWithException(t)
                        } finally {
                            image.close()
                        }
                    }

                    override fun onError(exception: ImageCaptureException) {
                        synchronized(gate) { failed = true }
                        runCatching { rawFile.delete() }
                        runCatching { jpegFile.delete() }
                        if (cont.isActive) cont.resumeWithException(exception)
                    }
                }
            )
            cont.invokeOnCancellation {
                runCatching { rawFile.delete() }
                runCatching { jpegFile.delete() }
            }
        }

    private fun persistRawSensor(image: ImageProxy, file: File, spec: ManualRawFrameSpec) {
        require(image.format == ImageFormat.RAW_SENSOR) { "Expected RAW_SENSOR" }
        val plane = image.planes.singleOrNull() ?: error("RAW_SENSOR must expose one plane")
        val source = plane.buffer.duplicate()
        file.parentFile?.mkdirs()
        DataOutputStream(BufferedOutputStream(FileOutputStream(file), IO_BUFFER_BYTES)).use { out ->
            out.writeUTF(SRAW_MAGIC)
            out.writeInt(SRAW_VERSION)
            out.writeInt(image.width)
            out.writeInt(image.height)
            out.writeInt(plane.rowStride)
            out.writeInt(plane.pixelStride)
            out.writeInt(image.imageInfo.rotationDegrees)
            out.writeLong(spec.exposureTimeNs)
            out.writeInt(spec.sensitivityIso)
            out.writeLong(lastSensorTimestampNs ?: 0L)
            out.writeInt(source.remaining())
            val scratch = ByteArray(minOf(IO_BUFFER_BYTES, maxOf(1, source.remaining())))
            while (source.hasRemaining()) {
                val count = minOf(source.remaining(), scratch.size)
                source.get(scratch, 0, count)
                out.write(scratch, 0, count)
            }
        }
        check(file.isFile && file.length() > 32L) { "RAW sensor artifact was not persisted" }
    }

    private fun persistCompressedImage(image: ImageProxy, file: File) {
        val plane = image.planes.singleOrNull() ?: error("Compressed image must expose one plane")
        val source = plane.buffer.duplicate()
        file.parentFile?.mkdirs()
        BufferedOutputStream(FileOutputStream(file), IO_BUFFER_BYTES).use { out ->
            val scratch = ByteArray(minOf(IO_BUFFER_BYTES, maxOf(1, source.remaining())))
            while (source.hasRemaining()) {
                val count = minOf(source.remaining(), scratch.size)
                source.get(scratch, 0, count)
                out.write(scratch, 0, count)
            }
        }
        check(file.isFile && file.length() > 0L) { "JPEG safety image was not persisted" }
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
        provider?.unbindAll()
    }

    companion object {
        const val SRAW_MAGIC = "SIERRA_RAW_SENSOR"
        const val SRAW_VERSION = 1
        private const val IO_BUFFER_BYTES = 1024 * 1024
        private const val MAX_RAW_BURST_FRAMES = 8

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
    val role: String,
    val evOffset: Double
) {
    init {
        require(exposureTimeNs > 0)
        require(sensitivityIso > 0)
        require(role.isNotBlank())
        require(evOffset.isFinite())
    }
}

data class RawJpegCapturedFrame(
    val index: Int,
    val rawFile: File,
    val jpegFile: File,
    val requested: ManualRawFrameSpec,
    val observed: CameraSensorTelemetry?
) {
    init {
        require(index >= 0)
        require(rawFile.isFile && rawFile.length() > 0)
        require(jpegFile.isFile && jpegFile.length() > 0)
    }
}

data class RawJpegBurstCapture(
    val workingDirectory: File,
    val frames: List<RawJpegCapturedFrame>
) {
    init {
        require(workingDirectory.isDirectory)
        require(frames.isNotEmpty())
    }

    fun cleanup() { runCatching { workingDirectory.deleteRecursively() } }
}

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
