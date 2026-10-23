package com.saunak.director.ui

import android.app.Application
import androidx.camera.view.PreviewView
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.viewModelScope
import com.saunak.director.ai.AiProviderPreset
import com.saunak.director.ai.AiProviderSettings
import com.saunak.director.ai.DeviceConfigSkill
import com.saunak.director.ai.MultiProviderVlmRuntime
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
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.Locale

class DirectorViewModel(app: Application) : AndroidViewModel(app) {
    private val camera = CameraEngine(app)
    private val keyStore = SecureApiKeyStore(app)
    private val providerSettings = AiProviderSettings(app)
    private val recipeMemory = RecipeMemory(app)
    private val deviceConfigSkill = DeviceConfigSkill(app)
    private val runtime = MultiProviderVlmRuntime(
        app,
        keyStore = keyStore,
        settings = providerSettings,
        deviceSkill = deviceConfigSkill,
        recipeMemory = recipeMemory
    )
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
            hasApiKey = runtime.isReady(),
            recipeCount = recipeMemory.count(),
            showSettings = _ui.value.showSettings,
            showTools = _ui.value.showTools,
            gridEnabled = _ui.value.gridEnabled,
            levelEnabled = _ui.value.levelEnabled
        )
    }

    private val _ui = MutableStateFlow(
        DirectorUiState(
            hasApiKey = runtime.isReady(),
            aiAvailable = runtime.isReady(),
            recipeCount = recipeMemory.count(),
            status = if (runtime.isReady()) "Camera ready • ${runtime.selectedPreset().label}" else "Camera ready • configure AI provider"
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
                    withContext(Dispatchers.IO) {
                        runCatching { deviceConfigSkill.refresh(camera.capabilities(), camera.snapshot()) }
                    }
                    refreshCamera("Camera ready${if (runtime.isReady()) " • ${runtime.selectedPreset().label}" else " • configure AI"}")
                }
                .onFailure {
                    _ui.value = _ui.value.copy(stage = DirectorStage.ERROR, status = "Camera unavailable", error = it.message)
                }
        }
    }

    fun primaryCapture() { viewModelScope.launch { orchestrator.onPrimaryPressed() } }

    fun focus(x: Float, y: Float) {
        orchestrator.cancelForUserAction()
        previewView?.let { camera.tapToFocus(it.meteringPointFactory, x, y) }
        refreshCameraSoon("Focus adjusted")
    }

    fun zoom(value: Float) {
        orchestrator.cancelForUserAction()
        camera.userSetZoom(value)
        refreshCameraSoon("Zoom ${String.format(Locale.US, "%.1f", value)}×")
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
            .onSuccess { switched ->
                if (switched) {
                    viewModelScope.launch(Dispatchers.IO) {
                        runCatching { deviceConfigSkill.refresh(camera.capabilities(), camera.snapshot()) }
                    }
                }
                refreshCamera(if (switched) "Camera switched" else "Other camera unavailable")
            }
            .onFailure { _ui.value = _ui.value.copy(error = it.message, status = "Camera switch failed") }
    }

    fun toggleSettings() { _ui.value = _ui.value.copy(showSettings = !_ui.value.showSettings, showTools = false) }
    fun dismissSettings() { _ui.value = _ui.value.copy(showSettings = false) }
    fun toggleTools() { _ui.value = _ui.value.copy(showTools = !_ui.value.showTools, showSettings = false) }
    fun toggleGrid() { _ui.value = _ui.value.copy(gridEnabled = !_ui.value.gridEnabled) }
    fun toggleLevel() { _ui.value = _ui.value.copy(levelEnabled = !_ui.value.levelEnabled) }

    fun providerPresets(): List<AiProviderPreset> = runtime.presets()
    fun selectedProvider(): AiProviderPreset = runtime.selectedPreset()
    fun providerHasCredential(preset: AiProviderPreset): Boolean = runtime.hasCredential(preset)
    fun openRouterCustomModel(): String = runtime.openRouterCustomModel()
    fun customEndpoint(): String = runtime.customEndpoint()
    fun customModel(): String = runtime.customModel()

    fun selectProvider(preset: AiProviderPreset) {
        orchestrator.cancelForUserAction()
        runtime.selectPreset(preset)
        val ready = runtime.isReady()
        _ui.value = _ui.value.copy(
            hasApiKey = ready,
            aiAvailable = ready,
            stage = if (ready) DirectorStage.READY else DirectorStage.CAMERA_ONLY,
            status = if (ready) "Selected • ${preset.label}" else if (preset == AiProviderPreset.AUTO) "Auto • add at least one provider key" else "${preset.label} • add API key",
            error = null
        )
    }

    fun saveProvider(
        preset: AiProviderPreset,
        apiKey: String,
        endpoint: String = "",
        model: String = ""
    ): String? = runCatching {
        require(preset != AiProviderPreset.AUTO) { "Choose a provider before adding a key" }
        if (preset == AiProviderPreset.CUSTOM_OPENAI) {
            require(endpoint.trim().startsWith("https://")) { "Custom endpoint must use HTTPS" }
            require(model.isNotBlank()) { "Custom model ID is required" }
            runtime.saveCustomRoute(endpoint, model)
        }
        if (preset == AiProviderPreset.OPENROUTER_CUSTOM) {
            require(model.isNotBlank()) { "OpenRouter model ID is required" }
            runtime.saveOpenRouterCustomModel(model)
        }
        if (apiKey.isNotBlank()) runtime.saveCredential(preset, apiKey)
        require(runtime.hasCredential(preset)) { "API key is required" }
        runtime.selectPreset(preset)
        _ui.value = _ui.value.copy(
            hasApiKey = runtime.isReady(),
            aiAvailable = false,
            status = "Testing ${preset.label}…",
            error = null
        )
        viewModelScope.launch {
            runtime.validateConnection(preset)
                .onSuccess { label ->
                    _ui.value = _ui.value.copy(
                        hasApiKey = true,
                        aiAvailable = true,
                        stage = DirectorStage.READY,
                        status = "$label connected • Director ready",
                        error = null
                    )
                }
                .onFailure { failure ->
                    _ui.value = _ui.value.copy(
                        hasApiKey = runtime.isReady(),
                        aiAvailable = false,
                        stage = DirectorStage.ERROR,
                        status = providerFailureMessage(failure),
                        error = failure.message
                    )
                }
        }
        null
    }.getOrElse { it.message ?: "Could not save provider" }

    fun clearProvider(preset: AiProviderPreset) {
        orchestrator.cancelForUserAction()
        runtime.clearCredential(preset)
        val ready = runtime.isReady()
        _ui.value = _ui.value.copy(
            hasApiKey = ready,
            aiAvailable = ready,
            stage = if (ready) DirectorStage.READY else DirectorStage.CAMERA_ONLY,
            status = if (ready) "Provider key removed • another route is available" else "Camera ready • configure AI provider",
            error = null
        )
    }

    /** Backward-compatible hook for older Settings UI. */
    fun saveApiKey(value: String): String? {
        val preset = selectedProvider().takeUnless { it == AiProviderPreset.AUTO } ?: AiProviderPreset.NVIDIA_NEMOTRON
        return saveProvider(preset, value)
    }

    fun clearApiKey() {
        val preset = selectedProvider().takeUnless { it == AiProviderPreset.AUTO } ?: AiProviderPreset.NVIDIA_NEMOTRON
        clearProvider(preset)
    }

    private fun providerFailureMessage(t: Throwable): String {
        val message = t.message.orEmpty().trim()
        val lower = message.lowercase()
        return when {
            "401" in lower || "invalid" in lower && "key" in lower -> "API key invalid • update it in Settings"
            "402" in lower || "credits" in lower || "billing" in lower -> "Provider credits/billing required"
            "403" in lower || "blocked" in lower -> "Provider access blocked • check key/account"
            "404" in lower || "unavailable" in lower -> "Selected model/endpoint unavailable"
            "429" in lower || "rate limit" in lower -> "Provider rate limit reached • switch route or retry"
            "timeout" in lower || "timed out" in lower -> "AI provider timed out • switch route or retry"
            message.isNotBlank() -> "AI provider error • ${message.take(96)}"
            else -> "AI provider connection failed"
        }
    }

    private fun refreshCamera(status: String = _ui.value.status) {
        val ready = runtime.isReady()
        _ui.value = _ui.value.copy(
            camera = runCatching { camera.uiState() }.getOrDefault(_ui.value.camera),
            status = status,
            hasApiKey = ready,
            aiAvailable = ready,
            recipeCount = recipeMemory.count(),
            stage = if (ready) DirectorStage.READY else DirectorStage.CAMERA_ONLY
        )
    }

    private fun refreshCameraSoon(status: String) {
        viewModelScope.launch { delay(80); refreshCamera(status) }
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
