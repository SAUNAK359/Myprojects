package com.saunak.director.ai

import android.content.Context

class AiProviderSettings(context: Context) {
    private val prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)

    fun selected(): AiProviderPreset = AiProviderPreset.fromId(prefs.getString(KEY_SELECTED, null))

    fun select(preset: AiProviderPreset) {
        prefs.edit().putString(KEY_SELECTED, preset.id).apply()
    }

    fun customEndpoint(): String = prefs.getString(KEY_CUSTOM_ENDPOINT, "").orEmpty().trim()
    fun customModel(): String = prefs.getString(KEY_CUSTOM_MODEL, "").orEmpty().trim()
    fun openRouterCustomModel(): String = prefs.getString(KEY_OPENROUTER_MODEL, "").orEmpty().trim()

    fun saveCustom(endpoint: String, model: String) {
        prefs.edit()
            .putString(KEY_CUSTOM_ENDPOINT, endpoint.trim())
            .putString(KEY_CUSTOM_MODEL, model.trim())
            .apply()
    }

    fun saveOpenRouterModel(model: String) {
        prefs.edit().putString(KEY_OPENROUTER_MODEL, model.trim()).apply()
    }

    companion object {
        private const val PREFS = "director_ai_provider_settings"
        private const val KEY_SELECTED = "selected_provider"
        private const val KEY_CUSTOM_ENDPOINT = "custom_endpoint"
        private const val KEY_CUSTOM_MODEL = "custom_model"
        private const val KEY_OPENROUTER_MODEL = "openrouter_custom_model"
    }
}
