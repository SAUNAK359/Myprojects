package com.saunak.director.camera

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.net.Uri
import android.provider.MediaStore
import android.util.Log
import android.view.Surface
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.FocusMeteringAction
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.MeteringPointFactory
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.google.common.util.concurrent.ListenableFuture
import kotlinx.coroutines.CancellableContinuation
import kotlinx.coroutines.delay
import kotlinx.coroutines.suspendCancellableCoroutine
import java.io.File
import java.text.SimpleDateFormat
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

class CameraEngine(private val context: Context) : AutoCloseable {
    private var provider: ProcessCameraProvider? = null
    private var camera: Camera? = null
    private var imageCapture: ImageCapture? = null
    private var imageAnalysis: ImageAnalysis? = null
    private var preview: Preview? = null
    private var boundView: PreviewView? = null
    private var lensFacing = CameraSelector.LENS_FACING_BACK
    private val analysisExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    @Volatile private var frameListener: ((Bitmap, Long) -> Unit)? = null
    @Volatile private var analysisAvailable = false

    suspend fun bind(owner: LifecycleOwner, view: PreviewView) {
        provider = ProcessCameraProvider.getInstance(context).await(); boundView = view; bindInternal(owner, view)
    }
    private fun bindInternal(owner: LifecycleOwner, view: PreviewView) {
        val p = checkNotNull(provider); p.unbindAll()
        preview = Preview.Builder().build().also { it.surfaceProvider = view.surfaceProvider }
        imageCapture = ImageCapture.Builder().setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY).setTargetRotation(view.display?.rotation ?: Surface.ROTATION_0).build()
        imageAnalysis = ImageAnalysis.Builder().setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST).setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888).setTargetRotation(view.display?.rotation ?: Surface.ROTATION_0).build().also { it.setAnalyzer(analysisExecutor) { proxy -> analyze(proxy) } }
        val selector = CameraSelector.Builder().requireLensFacing(lensFacing).build()
        try { camera = p.bindToLifecycle(owner, selector, preview, imageCapture, imageAnalysis); analysisAvailable = true }
        catch (t: Throwable) {
            Log.w("DirectorCamera", "Preview+capture+analysis unsupported; falling back to normal camera", t)
            p.unbindAll(); imageAnalysis?.clearAnalyzer(); imageAnalysis = null; camera = p.bindToLifecycle(owner, selector, preview, imageCapture); analysisAvailable = false
        }
    }
    private fun analyze(proxy: ImageProxy) {
        try {
            val listener = frameListener ?: return
            var bitmap = proxy.toBitmap(); val degrees = proxy.imageInfo.rotationDegrees
            if (degrees != 0) { val matrix = Matrix().apply { postRotate(degrees.toFloat()) }; val rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true); if (rotated !== bitmap) bitmap.recycle(); bitmap = rotated }
            listener(bitmap, System.nanoTime())
        } catch (t: Throwable) { Log.w("DirectorCamera", "Analysis frame dropped", t) } finally { proxy.close() }
    }
    fun setFrameListener(listener: ((Bitmap, Long) -> Unit)?) { frameListener = listener }
    fun isAnalysisAvailable(): Boolean = analysisAvailable
    fun switchLens(owner: LifecycleOwner, view: PreviewView): Boolean {
        val desired = if (lensFacing == CameraSelector.LENS_FACING_BACK) CameraSelector.LENS_FACING_FRONT else CameraSelector.LENS_FACING_BACK
        val p = provider ?: return false; val selector = CameraSelector.Builder().requireLensFacing(desired).build(); if (!p.hasCamera(selector)) return false
        lensFacing = desired; bindInternal(owner, view); return true
    }
    fun tapToFocus(factory: MeteringPointFactory, x: Float, y: Float) {
        val point = factory.createPoint(x, y)
        val action = FocusMeteringAction.Builder(point, FocusMeteringAction.FLAG_AF or FocusMeteringAction.FLAG_AE or FocusMeteringAction.FLAG_AWB).setAutoCancelDuration(4, TimeUnit.SECONDS).build()
        camera?.cameraControl?.startFocusAndMetering(action)
    }
    fun setZoom(ratio: Float) { val c = camera ?: return; val state = c.cameraInfo.zoomState.value ?: return; c.cameraControl.setZoomRatio(ratio.coerceIn(state.minZoomRatio, state.maxZoomRatio)) }
    suspend fun setZoomAwait(ratio: Float) { val c = camera ?: return; val state = c.cameraInfo.zoomState.value ?: return; c.cameraControl.setZoomRatio(ratio.coerceIn(state.minZoomRatio, state.maxZoomRatio)).await() }
    fun setExposureIndex(index: Int) { val c = camera ?: return; val e = c.cameraInfo.exposureState; if (!e.isExposureCompensationSupported) return; c.cameraControl.setExposureCompensationIndex(index.coerceIn(e.exposureCompensationRange.lower, e.exposureCompensationRange.upper)) }
    suspend fun setExposureIndexAwait(index: Int) { val c = camera ?: return; val e = c.cameraInfo.exposureState; if (!e.isExposureCompensationSupported) return; c.cameraControl.setExposureCompensationIndex(index.coerceIn(e.exposureCompensationRange.lower, e.exposureCompensationRange.upper)).await() }
    fun setTorch(enabled: Boolean) { val c = camera ?: return; if (c.cameraInfo.hasFlashUnit()) c.cameraControl.enableTorch(enabled) }
    suspend fun setTorchAwait(enabled: Boolean) { val c = camera ?: return; if (c.cameraInfo.hasFlashUnit()) c.cameraControl.enableTorch(enabled).await() }
    suspend fun awaitSettledState() { delay(220) }
    fun capabilities(): CapabilityRegistry {
        val c = camera ?: return CapabilityRegistry(lensFacing,1f,1f,0,0,0f,false,false,allowedToolIds=setOf(100))
        val zoom=c.cameraInfo.zoomState.value; val e=c.cameraInfo.exposureState; val range=e.exposureCompensationRange; val ids=mutableSetOf(100,1)
        if (zoom!=null && zoom.maxZoomRatio>zoom.minZoomRatio) ids+=2; if(e.isExposureCompensationSupported) ids+=3; if(c.cameraInfo.hasFlashUnit()) ids+=4; ids+=setOf(30,31,32,33,34,35,36,37)
        return CapabilityRegistry(lensFacing,zoom?.minZoomRatio?:1f,zoom?.maxZoomRatio?:1f,if(e.isExposureCompensationSupported) range.lower else 0,if(e.isExposureCompensationSupported) range.upper else 0,if(e.isExposureCompensationSupported) e.exposureCompensationStep.toFloat() else 0f,c.cameraInfo.hasFlashUnit(),c.cameraInfo.hasFlashUnit(),allowedToolIds=ids)
    }
    fun snapshot(): CameraSnapshot { val c=camera; return CameraSnapshot(lensFacing,c?.cameraInfo?.zoomState?.value?.zoomRatio?:1f,c?.cameraInfo?.exposureState?.exposureCompensationIndex?:0,c?.cameraInfo?.torchState?.value==1,boundView?.display?.rotation?:0) }
    suspend fun captureTemp(): File = suspendCancellableCoroutine { cont ->
        val capture=imageCapture?:return@suspendCancellableCoroutine cont.resumeWithException(IllegalStateException("Camera not ready")); capture.targetRotation=boundView?.display?.rotation?:Surface.ROTATION_0
        val file=File(context.cacheDir,"director_attempt_${System.currentTimeMillis()}_${System.nanoTime()}.jpg"); val options=ImageCapture.OutputFileOptions.Builder(file).build()
        capture.takePicture(options,ContextCompat.getMainExecutor(context),object:ImageCapture.OnImageSavedCallback{override fun onImageSaved(r:ImageCapture.OutputFileResults){if(cont.isActive)cont.resume(file)else file.delete()};override fun onError(e:ImageCaptureException){if(cont.isActive)cont.resumeWithException(e)else file.delete()}});cont.invokeOnCancellation{file.delete()}
    }
    suspend fun captureToGallery(): Uri = suspendCancellableCoroutine { cont ->
        val capture=imageCapture?:return@suspendCancellableCoroutine cont.resumeWithException(IllegalStateException("Camera not ready")); capture.targetRotation=boundView?.display?.rotation?:Surface.ROTATION_0
        val name="DIRECTOR_${SimpleDateFormat("yyyyMMdd_HHmmss_SSS",Locale.US).format(System.currentTimeMillis())}.jpg"; val values=ContentValues().apply{put(MediaStore.Images.Media.DISPLAY_NAME,name);put(MediaStore.Images.Media.MIME_TYPE,"image/jpeg");put(MediaStore.Images.Media.RELATIVE_PATH,"Pictures/Director")}
        val options=ImageCapture.OutputFileOptions.Builder(context.contentResolver,MediaStore.Images.Media.EXTERNAL_CONTENT_URI,values).build()
        capture.takePicture(options,ContextCompat.getMainExecutor(context),object:ImageCapture.OnImageSavedCallback{override fun onImageSaved(r:ImageCapture.OutputFileResults){val u=r.savedUri;if(u!=null&&cont.isActive)cont.resume(u)else if(cont.isActive)cont.resumeWithException(IllegalStateException("MediaStore did not return a URI"))};override fun onError(e:ImageCaptureException){if(cont.isActive)cont.resumeWithException(e)}})
    }
    override fun close(){frameListener=null;imageAnalysis?.clearAnalyzer();provider?.unbindAll();analysisExecutor.shutdownNow()}
}
private suspend fun <T> ListenableFuture<T>.await():T=suspendCancellableCoroutine{cont:CancellableContinuation<T>->addListener({try{if(cont.isActive)cont.resume(get())}catch(t:Throwable){if(cont.isActive)cont.resumeWithException(t)}},Runnable::run);cont.invokeOnCancellation{cancel(true)}}
