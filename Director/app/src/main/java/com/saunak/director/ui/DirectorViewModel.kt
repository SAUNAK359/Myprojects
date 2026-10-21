package com.saunak.director.ui

import android.app.Application
import android.graphics.Bitmap
import androidx.camera.view.PreviewView
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.viewModelScope
import com.saunak.director.ai.LlamaCppVlmRuntime
import com.saunak.director.ai.ModelManager
import com.saunak.director.camera.CameraEngine
import com.saunak.director.director.DirectorOrchestrator
import com.saunak.director.director.DirectorStage
import com.saunak.director.director.DirectorUiState
import com.saunak.director.platform.ThermalMonitor
import com.saunak.director.skills.SkillEngine
import com.saunak.director.storage.PhotoRepository
import com.saunak.director.tools.ToolExecutor
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

class DirectorViewModel(app: Application) : AndroidViewModel(app) {
    private val camera = CameraEngine(app)
    private val modelManager = ModelManager(app)
    private val runtime = LlamaCppVlmRuntime()
    private val orchestrator = DirectorOrchestrator(camera, runtime, SkillEngine(app), ToolExecutor(camera), PhotoRepository(app), ThermalMonitor(app)) { _ui.value = it }
    private val _ui = MutableStateFlow(DirectorUiState())
    val ui: StateFlow<DirectorUiState> = _ui.asStateFlow()
    private var previewView: PreviewView? = null
    private var owner: LifecycleOwner? = null
    @Volatile private var aiPreparationStarted = false
    private var aiJob: Job? = null

    fun bind(owner: LifecycleOwner, view: PreviewView) {
        if (previewView === view && this.owner === owner) return
        previewView = view; this.owner = owner
        viewModelScope.launch {
            runCatching { camera.bind(owner, view) }.onSuccess {
                camera.setFrameListener { bitmap, ts -> onFrame(bitmap, ts) }
                _ui.value = _ui.value.copy(stage = DirectorStage.CAMERA_ONLY, status = "Camera ready")
                if (modelManager.isInstalled()) prepareAi(false) else _ui.value = _ui.value.copy(stage = DirectorStage.AI_DOWNLOAD_REQUIRED, status = "Camera ready", requestModelDownload = true)
            }.onFailure { _ui.value = DirectorUiState(DirectorStage.ERROR, false, "Camera unavailable", error = it.message) }
        }
    }

    fun installDirectorAi() = prepareAi(true)
    fun dismissModelDownload() { _ui.value = _ui.value.copy(stage = DirectorStage.CAMERA_ONLY, status = "Camera ready", requestModelDownload = false, modelProgress = null) }

    private fun prepareAi(downloadAllowed: Boolean) {
        if (runtime.isReady() || aiPreparationStarted) return
        if (!modelManager.isInstalled() && !downloadAllowed) return
        aiPreparationStarted = true
        aiJob = viewModelScope.launch(Dispatchers.IO) {
            _ui.value = _ui.value.copy(stage = DirectorStage.LOADING_AI, status = if (modelManager.isInstalled()) "Starting Director…" else "Downloading Director AI…", requestModelDownload = false, modelProgress = 0f)
            val files = modelManager.ensureInstalled { p -> _ui.value = _ui.value.copy(modelProgress = p) }.getOrElse {
                aiPreparationStarted = false; _ui.value = DirectorUiState(DirectorStage.CAMERA_ONLY, false, "Camera ready • AI unavailable", error = it.message); return@launch
            }
            runtime.load(files.model.absolutePath, files.projector.absolutePath).onSuccess {
                aiPreparationStarted = false
                _ui.value = if (camera.isAnalysisAvailable()) DirectorUiState(DirectorStage.OBSERVING, true, "Director ready") else DirectorUiState(DirectorStage.CAMERA_ONLY, false, "Camera ready • AI analysis unsupported")
            }.onFailure { aiPreparationStarted = false; _ui.value = DirectorUiState(DirectorStage.CAMERA_ONLY, false, "Camera ready • AI unavailable", error = it.message) }
        }
    }

    fun onAppBackgrounded() {
        orchestrator.invalidateAutonomy()
        if (aiPreparationStarted && !runtime.isReady()) { aiJob?.cancel(); aiPreparationStarted = false; _ui.value = _ui.value.copy(stage = DirectorStage.CAMERA_ONLY, status = "Camera ready", modelProgress = null) }
    }
    fun onAppForegrounded() { if (!runtime.isReady() && modelManager.isInstalled()) prepareAi(false) }

    private fun onFrame(bitmap: Bitmap, ts: Long) {
        orchestrator.observeScene(bitmap, ts)
        if (orchestrator.isBusy()) { bitmap.recycle(); return }
        viewModelScope.launch(Dispatchers.Default) { orchestrator.onPreviewFrame(bitmap, ts) }
    }

    fun focus(x: Float, y: Float) { orchestrator.invalidateAutonomy(); previewView?.let { camera.tapToFocus(it.meteringPointFactory, x, y) } }
    fun zoom(value: Float) { orchestrator.invalidateAutonomy(); camera.setZoom(value) }
    fun exposure(index: Int) { orchestrator.invalidateAutonomy(); camera.setExposureIndex(index) }
    fun torch(enabled: Boolean) { orchestrator.invalidateAutonomy(); camera.setTorch(enabled) }
    fun switchCamera() {
        orchestrator.invalidateAutonomy(); val o = owner; val v = previewView
        if (o != null && v != null) runCatching { camera.switchLens(o, v) }.onFailure { _ui.value = _ui.value.copy(error = it.message) }
    }
    fun manualCapture() {
        orchestrator.invalidateAutonomy()
        viewModelScope.launch {
            _ui.value = _ui.value.copy(stage = DirectorStage.CAPTURING, status = "Capturing…")
            runCatching { camera.captureToGallery() }.onSuccess {
                _ui.value = _ui.value.copy(stage = if (runtime.isReady() && camera.isAnalysisAvailable()) DirectorStage.OBSERVING else DirectorStage.CAMERA_ONLY, status = "Saved", error = null)
            }.onFailure { _ui.value = _ui.value.copy(stage = DirectorStage.ERROR, status = "Capture failed", error = it.message) }
        }
    }
    override fun onCleared() { orchestrator.cancel(); runtime.close(); camera.close(); super.onCleared() }
}
