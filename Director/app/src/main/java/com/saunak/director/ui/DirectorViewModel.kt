package com.saunak.director.ui

import android.app.Application
import androidx.camera.view.PreviewView
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.viewModelScope
import com.saunak.director.BuildConfig
import com.saunak.director.ai.AiProviderPreset
import com.saunak.director.ai.AiProviderSettings
import com.saunak.director.ai.DeviceConfigSkill
import com.saunak.director.ai.MultiProviderVlmRuntime
import com.saunak.director.ai.SecureApiKeyStore
import com.saunak.director.ai.VlmRuntime
import com.saunak.director.backend.SierraBackendClient
import com.saunak.director.backend.SierraServerVlmRuntime
import com.saunak.director.backend.SierraSignInRequest
import com.saunak.director.backend.SierraSignUpRequest
import com.saunak.director.backend.SierraUserView
import com.saunak.director.camera.CameraEngine
import com.saunak.director.camera.StillFlashMode
import com.saunak.director.director.DirectorOrchestrator
import com.saunak.director.director.DirectorStage
import com.saunak.director.director.DirectorUiState
import com.saunak.director.director.SierraMode
import com.saunak.director.platform.ThermalMonitor
import com.saunak.director.skills.SkillEngine
import com.saunak.director.storage.PhotoRepository
import com.saunak.director.storage.RecipeMemory
import com.saunak.director.storage.SierraCropPolicy
import com.saunak.director.tools.ToolExecutor
import kotlinx.coroutines.CancellationException
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
    private val directRuntime = MultiProviderVlmRuntime(
        app,
        keyStore = keyStore,
        settings = providerSettings,
        deviceSkill = deviceConfigSkill,
        recipeMemory = recipeMemory
    )
    private val backend = SierraBackendClient(app, BuildConfig.SIERRA_SERVER_BASE_URL)
    private val serverRuntime = SierraServerVlmRuntime(app, backend)
    private val runtime: VlmRuntime = if (backend.configured) serverRuntime else directRuntime
    private val repository = PhotoRepository(app)
    private val preferences = app.getSharedPreferences("sierra_vision_preferences", Application.MODE_PRIVATE)
    private val initialPrivateMode = preferences.getBoolean("private_mode", false)
    private val initialAspectRatioOverride = preferences.getString("aspect_ratio_override", null)
        ?.takeIf(SierraCropPolicy::isSupported)
    private val initialKeepOriginal = preferences.getBoolean("keep_original", false)
    private val cachedAccount = backend.cachedUser()

    private val orchestrator = DirectorOrchestrator(
        camera = camera,
        vlm = runtime,
        skills = SkillEngine(app),
        tools = ToolExecutor(camera),
        storage = repository,
        recipeMemory = recipeMemory,
        thermal = ThermalMonitor(app)
    ) { state ->
        val current = _ui.value
        _ui.value = state.copy(
            camera = runCatching { camera.uiState() }.getOrDefault(state.camera),
            hasApiKey = runtime.isReady(),
            recipeCount = recipeMemory.count(),
            showSettings = current.showSettings,
            showTools = current.showTools,
            showGallery = current.showGallery,
            showAccount = current.showAccount,
            galleryItems = current.galleryItems,
            gridEnabled = current.gridEnabled,
            levelEnabled = current.levelEnabled,
            privateMode = current.privateMode,
            aspectRatioOverride = current.aspectRatioOverride,
            keepOriginal = current.keepOriginal,
            serverConfigured = current.serverConfigured,
            signedIn = current.signedIn,
            accountBusy = current.accountBusy,
            accountError = current.accountError,
            userDisplayName = current.userDisplayName,
            userEmail = current.userEmail,
            creditsAvailable = current.creditsAvailable,
            creditsReserved = current.creditsReserved
        )
        if (state.lastSavedUri != null && state.stage == DirectorStage.READY) {
            refreshGallery()
            if (backend.configured && backend.authenticated) refreshAccount()
        }
    }

    private val _ui = MutableStateFlow(
        DirectorUiState(
            hasApiKey = runtime.isReady(),
            aiAvailable = runtime.isReady() && !initialPrivateMode,
            recipeCount = recipeMemory.count(),
            status = initialStatus(initialPrivateMode),
            sierraMode = if (initialPrivateMode || !runtime.isReady()) SierraMode.BETA else SierraMode.AI,
            privateMode = initialPrivateMode,
            aspectRatioOverride = initialAspectRatioOverride,
            aspectRatio = initialAspectRatioOverride ?: "4:3",
            keepOriginal = initialKeepOriginal,
            serverConfigured = backend.configured,
            signedIn = backend.authenticated,
            userDisplayName = cachedAccount?.displayName,
            userEmail = cachedAccount?.email,
            creditsAvailable = cachedAccount?.availableCredits,
            creditsReserved = cachedAccount?.creditReserved ?: 0
        )
    )
    val ui: StateFlow<DirectorUiState> = _ui.asStateFlow()

    private var previewView: PreviewView? = null
    private var owner: LifecycleOwner? = null
    private var cameraBound = false

    init {
        orchestrator.setCloudEnabled(!initialPrivateMode)
        orchestrator.setAspectRatioOverride(initialAspectRatioOverride)
        orchestrator.setKeepOriginal(initialKeepOriginal)
        refreshGallery()
        viewModelScope.launch {
            if (backend.configured && backend.authenticated) {
                serverRuntime.retryPendingCompletion()
                refreshAccountInternal()
            }
            val recovered = orchestrator.recoverPendingReferences()
            if (recovered.isNotEmpty()) {
                _ui.value = _ui.value.copy(
                    status = "Recovered ${recovered.size} protected photo${if (recovered.size == 1) "" else "s"}",
                    lastSavedUri = recovered.last().toString()
                )
                refreshGallery()
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
                    refreshCamera()
                }
                .onFailure {
                    _ui.value = _ui.value.copy(
                        stage = DirectorStage.ERROR,
                        status = "Camera unavailable",
                        error = it.message
                    )
                }
        }
    }

    fun primaryCapture() {
        viewModelScope.launch {
            try {
                orchestrator.onPrimaryPressed()
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (oom: OutOfMemoryError) {
                runtime.cancel()
                _ui.value = _ui.value.copy(
                    stage = DirectorStage.ERROR,
                    aiAvailable = runtime.isReady() && !_ui.value.privateMode,
                    status = "Sierra memory pressure • try again",
                    error = "Analysis used too much memory"
                )
            } catch (t: Throwable) {
                runtime.cancel()
                _ui.value = _ui.value.copy(
                    stage = DirectorStage.ERROR,
                    aiAvailable = runtime.isReady() && !_ui.value.privateMode,
                    status = "Sierra could not prepare the shot",
                    error = t.message ?: t::class.java.simpleName
                )
            }
        }
    }

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

    fun togglePrivateMode() {
        val enabled = !_ui.value.privateMode
        orchestrator.cancelForUserAction()
        orchestrator.setCloudEnabled(!enabled)
        preferences.edit().putBoolean("private_mode", enabled).apply()
        _ui.value = _ui.value.copy(
            privateMode = enabled,
            aiAvailable = !enabled && runtime.isReady(),
            sierraMode = if (!enabled && runtime.isReady()) SierraMode.AI else SierraMode.BETA,
            stage = DirectorStage.READY,
            status = if (enabled) "Sierra β Private • on-device only" else readyStatus(),
            error = null
        )
    }

    fun setAspectRatioOverride(value: String?) {
        val safe = value?.takeIf(SierraCropPolicy::isSupported)
        orchestrator.cancelForUserAction()
        orchestrator.setAspectRatioOverride(safe)
        preferences.edit().apply {
            if (safe == null) remove("aspect_ratio_override") else putString("aspect_ratio_override", safe)
        }.apply()
        _ui.value = _ui.value.copy(
            aspectRatioOverride = safe,
            aspectRatio = safe ?: _ui.value.aspectRatio,
            stage = DirectorStage.READY,
            status = if (safe == null) "Aspect ratio • Sierra Auto" else "Aspect ratio • $safe",
            error = null
        )
    }

    fun toggleKeepOriginal() {
        val enabled = !_ui.value.keepOriginal
        orchestrator.setKeepOriginal(enabled)
        preferences.edit().putBoolean("keep_original", enabled).apply()
        _ui.value = _ui.value.copy(
            keepOriginal = enabled,
            status = if (enabled) "Originals will also be saved" else "Saving Sierra final only"
        )
    }

    fun openGallery() {
        refreshGallery()
        _ui.value = _ui.value.copy(
            showGallery = true,
            showTools = false,
            showSettings = false,
            showAccount = false
        )
    }

    fun closeGallery() { _ui.value = _ui.value.copy(showGallery = false) }

    fun refreshGallery() {
        viewModelScope.launch(Dispatchers.IO) {
            val items = runCatching { repository.recentFinals() }
                .getOrDefault(emptyList())
                .map { it.toString() }
            withContext(Dispatchers.Main) { _ui.value = _ui.value.copy(galleryItems = items) }
        }
    }

    fun openAccount() {
        _ui.value = _ui.value.copy(showAccount = true, showTools = false, showGallery = false)
        if (backend.configured && backend.authenticated) refreshAccount()
    }

    fun closeAccount() { _ui.value = _ui.value.copy(showAccount = false, accountError = null) }

    fun signIn(email: String, password: String) {
        if (!backend.configured) {
            _ui.value = _ui.value.copy(accountError = "This APK was not built with a Sierra server URL")
            return
        }
        viewModelScope.launch {
            setAccountBusy(true)
            backend.signIn(SierraSignInRequest(email.trim(), password))
                .onSuccess { pair -> applyAccount(pair.user, null) }
                .onFailure { applyAccountFailure(it) }
            setAccountBusy(false)
            refreshCamera()
        }
    }

    fun signUp(name: String, email: String, phone: String, dob: String, password: String) {
        if (!backend.configured) {
            _ui.value = _ui.value.copy(accountError = "This APK was not built with a Sierra server URL")
            return
        }
        viewModelScope.launch {
            setAccountBusy(true)
            backend.signUp(
                SierraSignUpRequest(
                    displayName = name.trim(),
                    email = email.trim(),
                    phone = phone.trim(),
                    dob = dob.trim(),
                    password = password
                )
            ).onSuccess { pair -> applyAccount(pair.user, null) }
                .onFailure { applyAccountFailure(it) }
            setAccountBusy(false)
            refreshCamera()
        }
    }

    fun signOut() {
        orchestrator.cancelForUserAction()
        viewModelScope.launch {
            setAccountBusy(true)
            backend.signOut()
            _ui.value = _ui.value.copy(
                signedIn = false,
                userDisplayName = null,
                userEmail = null,
                creditsAvailable = null,
                creditsReserved = 0,
                accountError = null,
                sierraMode = SierraMode.BETA,
                aiAvailable = false
            )
            setAccountBusy(false)
            refreshCamera("Signed out • Sierra β ready")
        }
    }

    fun refreshAccount() {
        if (!backend.configured || !backend.authenticated) return
        viewModelScope.launch { refreshAccountInternal() }
    }

    private suspend fun refreshAccountInternal() {
        backend.session()
            .onSuccess { applyAccount(it, null) }
            .onFailure { throwable ->
                _ui.value = _ui.value.copy(
                    signedIn = backend.authenticated,
                    accountError = throwable.message ?: "Could not reach Sierra server"
                )
            }
    }

    private fun applyAccount(user: SierraUserView, error: String?) {
        val current = _ui.value
        _ui.value = current.copy(
            signedIn = true,
            accountError = error,
            userDisplayName = user.displayName,
            userEmail = user.email,
            creditsAvailable = user.availableCredits,
            creditsReserved = user.creditReserved,
            aiAvailable = !current.privateMode && runtime.isReady(),
            sierraMode = if (!current.privateMode && runtime.isReady()) SierraMode.AI else SierraMode.BETA
        )
    }

    private fun applyAccountFailure(t: Throwable) {
        _ui.value = _ui.value.copy(
            signedIn = backend.authenticated,
            accountError = t.message ?: "Sierra account request failed"
        )
    }

    private fun setAccountBusy(value: Boolean) {
        _ui.value = _ui.value.copy(accountBusy = value, accountError = if (value) null else _ui.value.accountError)
    }

    fun toggleSettings() {
        _ui.value = _ui.value.copy(
            showSettings = !_ui.value.showSettings,
            showTools = false,
            showAccount = false
        )
    }
    fun dismissSettings() { _ui.value = _ui.value.copy(showSettings = false) }
    fun toggleTools() {
        _ui.value = _ui.value.copy(
            showTools = !_ui.value.showTools,
            showSettings = false,
            showAccount = false
        )
    }
    fun dismissTools() { _ui.value = _ui.value.copy(showTools = false) }
    fun toggleGrid() { _ui.value = _ui.value.copy(gridEnabled = !_ui.value.gridEnabled) }
    fun toggleLevel() { _ui.value = _ui.value.copy(levelEnabled = !_ui.value.levelEnabled) }

    // Development-only direct-provider controls are intentionally disconnected from customer
    // Sierra builds whenever SIERRA_SERVER_BASE_URL is configured.
    fun providerPresets(): List<AiProviderPreset> = directRuntime.presets()
    fun selectedProvider(): AiProviderPreset = directRuntime.selectedPreset()
    fun providerHasCredential(preset: AiProviderPreset): Boolean = directRuntime.hasCredential(preset)
    fun openRouterCustomModel(): String = directRuntime.openRouterCustomModel()
    fun customEndpoint(): String = directRuntime.customEndpoint()
    fun customModel(): String = directRuntime.customModel()

    fun selectProvider(preset: AiProviderPreset) {
        if (backend.configured) return
        orchestrator.cancelForUserAction()
        directRuntime.selectPreset(preset)
        refreshCamera()
    }

    fun saveProvider(
        preset: AiProviderPreset,
        apiKey: String,
        endpoint: String = "",
        model: String = ""
    ): String? = runCatching {
        require(!backend.configured) { "Provider routing is managed by the Sierra laptop server" }
        require(preset != AiProviderPreset.AUTO) { "Choose a provider before adding a key" }
        if (preset == AiProviderPreset.CUSTOM_OPENAI) {
            require(endpoint.trim().startsWith("https://")) { "Custom endpoint must use HTTPS" }
            require(model.isNotBlank()) { "Custom model ID is required" }
            directRuntime.saveCustomRoute(endpoint, model)
        }
        if (preset == AiProviderPreset.OPENROUTER_CUSTOM) {
            require(model.isNotBlank()) { "OpenRouter model ID is required" }
            directRuntime.saveOpenRouterCustomModel(model)
        }
        if (apiKey.isNotBlank()) directRuntime.saveCredential(preset, apiKey)
        require(directRuntime.hasCredential(preset)) { "API key is required" }
        directRuntime.selectPreset(preset)
        viewModelScope.launch {
            directRuntime.validateConnection(preset)
                .onSuccess { refreshCamera("Sierra AI route connected") }
                .onFailure { refreshCamera("Sierra β ready • cloud route unavailable") }
        }
        null
    }.getOrElse { it.message ?: "Could not save provider" }

    fun clearProvider(preset: AiProviderPreset) {
        if (backend.configured) return
        directRuntime.clearCredential(preset)
        refreshCamera()
    }
    fun saveApiKey(value: String): String? = saveProvider(
        selectedProvider().takeUnless { it == AiProviderPreset.AUTO } ?: AiProviderPreset.NVIDIA_NEMOTRON,
        value
    )
    fun clearApiKey() = clearProvider(
        selectedProvider().takeUnless { it == AiProviderPreset.AUTO } ?: AiProviderPreset.NVIDIA_NEMOTRON
    )

    private fun refreshCamera(status: String? = null) {
        val current = _ui.value
        val privateMode = current.privateMode
        val aiReady = runtime.isReady() && !privateMode
        _ui.value = current.copy(
            camera = runCatching { camera.uiState() }.getOrDefault(current.camera),
            status = status ?: if (privateMode) "Sierra β Private • on-device only" else readyStatus(),
            hasApiKey = runtime.isReady(),
            aiAvailable = aiReady,
            recipeCount = recipeMemory.count(),
            sierraMode = if (aiReady) SierraMode.AI else SierraMode.BETA,
            stage = DirectorStage.READY,
            serverConfigured = backend.configured,
            signedIn = if (backend.configured) backend.authenticated else current.signedIn
        )
    }

    private fun readyStatus(): String = when {
        backend.configured && !backend.authenticated -> "Sierra β ready • sign in for Sierra AI"
        runtime.isReady() -> "Sierra AI + β ready"
        else -> "Sierra β ready"
    }

    private fun initialStatus(privateMode: Boolean): String = when {
        privateMode -> "Sierra β Private • on-device only"
        backend.configured && !backend.authenticated -> "Sierra β ready • sign in for Sierra AI"
        runtime.isReady() -> "Sierra AI + β ready"
        else -> "Sierra β ready"
    }

    private fun refreshCameraSoon(status: String) {
        viewModelScope.launch { delay(80); refreshCamera(status) }
    }

    fun onAppBackgrounded() { orchestrator.cancelForUserAction() }
    fun onAppForegrounded() {
        refreshCamera()
        refreshGallery()
        if (backend.configured && backend.authenticated) refreshAccount()
    }

    override fun onCleared() {
        orchestrator.cancelForUserAction()
        runtime.close()
        if (runtime !== serverRuntime) backend.close()
        camera.close()
        super.onCleared()
    }
}
