package com.saunak.director.ui

import android.app.Application
import android.content.ComponentCallbacks2
import androidx.camera.view.PreviewView
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.viewModelScope
import com.saunak.director.ai.LlamaCppVlmRuntime
import com.saunak.director.ai.ModelManager
import com.saunak.director.camera.CameraEngine
import com.saunak.director.camera.StillFlashMode
import com.saunak.director.director.DirectorOrchestrator
import com.saunak.director.director.DirectorStage
import com.saunak.director.director.DirectorUiState
import com.saunak.director.platform.MemoryPressureBus
import com.saunak.director.platform.ThermalMonitor
import com.saunak.director.skills.SkillEngine
import com.saunak.director.storage.PhotoRepository
import com.saunak.director.tools.ToolExecutor
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

class DirectorViewModel(app: Application) : AndroidViewModel(app) {
    private val camera = CameraEngine(app)
    private val modelManager = ModelManager(app)
    private val runtime = LlamaCppVlmRuntime()
    private val repository = PhotoRepository(app)
    private val orchestrator = DirectorOrchestrator(camera, runtime, SkillEngine(app), ToolExecutor(camera), repository, ThermalMonitor(app)) { state ->
        _ui.value = state.copy(camera = runCatching { camera.uiState() }.getOrDefault(state.camera))
    }
    private val _ui = MutableStateFlow(DirectorUiState())
    val ui: StateFlow<DirectorUiState> = _ui.asStateFlow()
    private var previewView: PreviewView? = null
    private var owner: LifecycleOwner? = null
    @Volatile private var aiPreparationStarted = false
    private var aiJob: Job? = null
    private var cameraBound = false

    private val memoryListener: (Int) -> Unit = { level ->
        if (level >= ComponentCallbacks2.TRIM_MEMORY_RUNNING_CRITICAL) {
            orchestrator.cancelForUserAction()
            viewModelScope.launch(Dispatchers.IO) {
                runtime.unload()
                _ui.value = _ui.value.copy(stage = DirectorStage.CAMERA_ONLY, aiAvailable = false, status = "Camera ready • AI released for memory", modelProgress = null)
            }
        }
    }

    init {
        MemoryPressureBus.add(memoryListener)
        viewModelScope.launch(Dispatchers.IO) {
            val recovered = orchestrator.recoverPendingReferences()
            if (recovered.isNotEmpty()) _ui.value = _ui.value.copy(status = "Recovered ${recovered.size} protected photo${if (recovered.size == 1) "" else "s"}", lastSavedUri = recovered.last().toString())
        }
    }

    fun bind(owner: LifecycleOwner, view: PreviewView) {
        if (cameraBound && previewView === view && this.owner === owner) return
        previewView = view; this.owner = owner
        viewModelScope.launch {
            runCatching { camera.bind(owner, view) }.onSuccess {
                cameraBound = true
                _ui.value = _ui.value.copy(stage = if (runtime.isReady()) DirectorStage.READY else DirectorStage.CAMERA_ONLY, aiAvailable = runtime.isReady(), status = if (runtime.isReady()) "Director ready" else "Camera ready", camera = camera.uiState())
                if (modelManager.isInstalled()) prepareAi(false) else _ui.value = _ui.value.copy(stage = DirectorStage.AI_DOWNLOAD_REQUIRED, status = "Camera ready", requestModelDownload = true)
            }.onFailure { _ui.value = _ui.value.copy(stage = DirectorStage.ERROR, status = "Camera unavailable", error = it.message) }
        }
    }

    fun installDirectorAi() = prepareAi(true)
    fun dismissModelDownload() { _ui.value = _ui.value.copy(stage = DirectorStage.CAMERA_ONLY, status = "Camera ready", requestModelDownload = false, modelProgress = null) }

    private fun prepareAi(downloadAllowed: Boolean) {
        if (runtime.isReady() || aiPreparationStarted) return
        if (!modelManager.isInstalled() && !downloadAllowed) return
        aiPreparationStarted = true
        aiJob = viewModelScope.launch(Dispatchers.IO) {
            _ui.value = _ui.value.copy(stage = DirectorStage.LOADING_AI, status = if (modelManager.isInstalled()) "Starting Director…" else "Downloading Director AI…", requestModelDownload = false, modelProgress = if (modelManager.isInstalled()) null else 0f)
            val files = modelManager.ensureInstalled { p -> _ui.value = _ui.value.copy(modelProgress = p) }.getOrElse {
                aiPreparationStarted = false
                _ui.value = _ui.value.copy(stage = DirectorStage.CAMERA_ONLY, aiAvailable = false, status = "Camera ready • AI unavailable", modelProgress = null, error = it.message)
                return@launch
            }
            runtime.load(files.model.absolutePath, files.projector.absolutePath).onSuccess {
                aiPreparationStarted = false
                _ui.value = _ui.value.copy(stage = DirectorStage.READY, aiAvailable = true, status = "Director ready", modelProgress = null, error = null, camera = runCatching { camera.uiState() }.getOrDefault(_ui.value.camera))
            }.onFailure {
                modelManager.invalidateVerification()
                aiPreparationStarted = false
                _ui.value = _ui.value.copy(stage = DirectorStage.CAMERA_ONLY, aiAvailable = false, status = "Camera ready • AI verification needed", modelProgress = null, error = it.message)
            }
        }
    }

    fun shutterCapture() { viewModelScope.launch { orchestrator.onShutterPressed() } }

    fun focus(x: Float, y: Float) {
        orchestrator.cancelForUserAction()
        previewView?.let { camera.tapToFocus(it.meteringPointFactory, x, y) }
        refreshCameraSoon()
    }
    fun zoom(value: Float) { orchestrator.cancelForUserAction(); camera.userSetZoom(value); refreshCameraSoon() }
    fun exposure(index: Int) { orchestrator.cancelForUserAction(); camera.userSetExposureIndex(index); refreshCameraSoon() }
    fun torch(enabled: Boolean) { orchestrator.cancelForUserAction(); camera.userSetTorch(enabled); refreshCameraSoon() }
    fun cycleFlash() {
        orchestrator.cancelForUserAction()
        val next = when (_ui.value.camera.flashMode) { StillFlashMode.OFF -> StillFlashMode.AUTO; StillFlashMode.AUTO -> StillFlashMode.ON; StillFlashMode.ON -> StillFlashMode.OFF }
        camera.userSetFlash(next); refreshCameraSoon()
    }
    fun switchCamera() {
        orchestrator.cancelForUserAction()
        runCatching { camera.switchLens() }
            .onSuccess { _ui.value = _ui.value.copy(camera = camera.uiState(), status = if (it) "Camera switched" else "Other camera unavailable") }
            .onFailure { _ui.value = _ui.value.copy(error = it.message, status = "Camera switch failed") }
    }

    private fun refreshCameraSoon() {
        viewModelScope.launch {
            delay(80)
            _ui.value = _ui.value.copy(camera = runCatching { camera.uiState() }.getOrDefault(_ui.value.camera))
        }
    }

    fun onAppBackgrounded() {
        orchestrator.cancelForUserAction()
        if (aiPreparationStarted && !runtime.isReady()) {
            aiJob?.cancel(); aiPreparationStarted = false
            _ui.value = _ui.value.copy(stage = DirectorStage.CAMERA_ONLY, status = "Camera ready", modelProgress = null)
        }
    }
    fun onAppForegrounded() { if (!runtime.isReady() && modelManager.isInstalled()) prepareAi(false) }

    override fun onCleared() {
        MemoryPressureBus.remove(memoryListener)
        orchestrator.cancelForUserAction()
        runtime.close(); camera.close()
        super.onCleared()
    }
}
