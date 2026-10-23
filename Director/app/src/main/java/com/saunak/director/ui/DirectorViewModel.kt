package com.saunak.director.ui

import android.app.Application
import androidx.camera.view.PreviewView
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.viewModelScope
import com.saunak.director.ai.GeminiVlmRuntime
import com.saunak.director.ai.SecureApiKeyStore
import com.saunak.director.camera.CameraEngine
import com.saunak.director.camera.StillFlashMode
import com.saunak.director.director.DirectorOrchestrator
import com.saunak.director.director.DirectorStage
import com.saunak.director.director.DirectorUiState
import com.saunak.director.platform.ThermalMonitor
import com.saunak.director.skills.SkillEngine
import com.saunak.director.storage.PhotoRepository
import com.saunak.director.storage.RecipeMemory
import com.saunak.director.tools.ToolExecutor
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

class DirectorViewModel(app: Application) : AndroidViewModel(app) {
    private val camera = CameraEngine(app)
    private val keyStore = SecureApiKeyStore(app)
    private val recipeMemory = RecipeMemory(app)
    private val runtime = GeminiVlmRuntime(app, keyStore = keyStore, recipeMemory = recipeMemory)
    private val repository = PhotoRepository(app)
    private val orchestrator = DirectorOrchestrator(
        camera = camera,
        vlm = runtime,
        skills = SkillEngine(app),
        tools = ToolExecutor(camera),
        storage = repository,
        recipeMemory = recipeMemory,
        thermal = ThermalMonitor(app)
    ) { state ->
        _ui.value = state.copy(
            camera = runCatching { camera.uiState() }.getOrDefault(state.camera),
            hasApiKey = keyStore.hasKey(),
            recipeCount = recipeMemory.count(),
            showSettings = _ui.value.showSettings,
            showTools = _ui.value.showTools,
            gridEnabled = _ui.value.gridEnabled,
            levelEnabled = _ui.value.levelEnabled
        )
    }

    private val _ui = MutableStateFlow(
        DirectorUiState(
            hasApiKey = keyStore.hasKey(),
            aiAvailable = keyStore.hasKey(),
            recipeCount = recipeMemory.count(),
            status = if (keyStore.hasKey()) "Camera ready • Director online" else "Camera ready • add Gemini key"
        )
    )
    val ui: StateFlow<DirectorUiState> = _ui.asStateFlow()

    private var previewView: PreviewView? = null
    private var owner: LifecycleOwner? = null
    private var cameraBound = false

    init {
        viewModelScope.launch {
            val recovered = orchestrator.recoverPendingReferences()
            if (recovered.isNotEmpty()) {
                _ui.value = _ui.value.copy(
                    status = "Recovered ${recovered.size} protected photo${if (recovered.size == 1) "" else "s"}",
                    lastSavedUri = recovered.last().toString()
                )
            }
        }
    }

    fun bind(owner: LifecycleOwner, view: PreviewView) {
        if (cameraBound && previewView === view && this.owner === owner) return
        previewView = view
        this.owner = owner
        viewModelScope.launch {
            runCatching { camera.bind(owner, view) }
                .onSuccess {
                    cameraBound = true
                    refreshCamera("Camera ready${if (keyStore.hasKey()) " • Director online" else " • add Gemini key"}")
                }
                .onFailure {
                    _ui.value = _ui.value.copy(stage = DirectorStage.ERROR, status = "Camera unavailable", error = it.message)
                }
        }
    }

    fun primaryCapture() {
        viewModelScope.launch { orchestrator.onPrimaryPressed() }
    }

    fun focus(x: Float, y: Float) {
        orchestrator.cancelForUserAction()
        previewView?.let { camera.tapToFocus(it.meteringPointFactory, x, y) }
        refreshCameraSoon("Focus adjusted")
    }

    fun zoom(value: Float) {
        orchestrator.cancelForUserAction()
        camera.userSetZoom(value)
        refreshCameraSoon("Zoom ${"%.1f".format(value)}×")
    }

    fun exposure(index: Int) {
        orchestrator.cancelForUserAction()
        camera.userSetExposureIndex(index)
        refreshCameraSoon("Exposure adjusted")
    }

    fun torch(enabled: Boolean) {
        orchestrator.cancelForUserAction()
        camera.userSetTorch(enabled)
        refreshCameraSoon(if (enabled) "Torch on" else "Torch off")
    }

    fun cycleFlash() {
        orchestrator.cancelForUserAction()
        val next = when (_ui.value.camera.flashMode) {
            StillFlashMode.OFF -> StillFlashMode.AUTO
            StillFlashMode.AUTO -> StillFlashMode.ON
            StillFlashMode.ON -> StillFlashMode.OFF
        }
        camera.userSetFlash(next)
        refreshCameraSoon("Flash ${next.wireName}")
    }

    fun switchCamera() {
        orchestrator.cancelForUserAction()
        runCatching { camera.switchLens() }
            .onSuccess { refreshCamera(if (it) "Camera switched" else "Other camera unavailable") }
            .onFailure { _ui.value = _ui.value.copy(error = it.message, status = "Camera switch failed") }
    }

    fun toggleSettings() { _ui.value = _ui.value.copy(showSettings = !_ui.value.showSettings, showTools = false) }
    fun dismissSettings() { _ui.value = _ui.value.copy(showSettings = false) }
    fun toggleTools() { _ui.value = _ui.value.copy(showTools = !_ui.value.showTools, showSettings = false) }
    fun toggleGrid() { _ui.value = _ui.value.copy(gridEnabled = !_ui.value.gridEnabled) }
    fun toggleLevel() { _ui.value = _ui.value.copy(levelEnabled = !_ui.value.levelEnabled) }

    fun saveApiKey(value: String): String? = runCatching {
        keyStore.save(value)
        _ui.value = _ui.value.copy(
            hasApiKey = true,
            aiAvailable = true,
            showSettings = false,
            status = "Gemini connected • Director ready",
            error = null
        )
        null
    }.getOrElse { it.message ?: "Could not save key" }

    fun clearApiKey() {
        orchestrator.cancelForUserAction()
        keyStore.clear()
        _ui.value = _ui.value.copy(
            hasApiKey = false,
            aiAvailable = false,
            showSettings = false,
            stage = DirectorStage.CAMERA_ONLY,
            status = "Camera ready • add Gemini key"
        )
    }

    private fun refreshCamera(status: String = _ui.value.status) {
        _ui.value = _ui.value.copy(
            camera = runCatching { camera.uiState() }.getOrDefault(_ui.value.camera),
            status = status,
            hasApiKey = keyStore.hasKey(),
            aiAvailable = keyStore.hasKey(),
            recipeCount = recipeMemory.count(),
            stage = if (keyStore.hasKey()) DirectorStage.READY else DirectorStage.CAMERA_ONLY
        )
    }

    private fun refreshCameraSoon(status: String) {
        viewModelScope.launch {
            delay(80)
            refreshCamera(status)
        }
    }

    fun onAppBackgrounded() { orchestrator.cancelForUserAction() }
    fun onAppForegrounded() { refreshCamera() }

    override fun onCleared() {
        orchestrator.cancelForUserAction()
        runtime.close()
        camera.close()
        super.onCleared()
    }
}
