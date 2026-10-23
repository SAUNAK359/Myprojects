package com.saunak.director.ai

enum class AiProviderPreset(
    val id: String,
    val label: String,
    val credentialId: String?,
    val endpoint: String?,
    val model: String?,
    val freeRoute: Boolean = false,
    val includeInAuto: Boolean = true
) {
    AUTO(
        id = "auto",
        label = "Auto • best available",
        credentialId = null,
        endpoint = null,
        model = null
    ),
    GROQ_QWEN38(
        id = "groq_qwen38",
        label = "Groq • Qwen 3.8 27B",
        credentialId = "groq",
        endpoint = "https://api.groq.com/openai/v1/chat/completions",
        model = "qwen/qwen3.8-27b",
        freeRoute = true
    ),
    OPENROUTER_NEMOTRON_FREE(
        id = "openrouter_nemotron_free",
        label = "OpenRouter • Nemotron 3.5 VL Free",
        credentialId = "openrouter",
        endpoint = "https://openrouter.ai/api/v1/chat/completions",
        model = "nvidia/nemotron-3.5-super-vl:free",
        freeRoute = true
    ),
    OPENROUTER_QWEN38_FREE(
        id = "openrouter_qwen38_free",
        label = "OpenRouter • Qwen 3.8 27B Free",
        credentialId = "openrouter",
        endpoint = "https://openrouter.ai/api/v1/chat/completions",
        model = "qwen/qwen3.8-27b:free",
        freeRoute = true
    ),
    OPENROUTER_GEMMA4_31B_FREE(
        id = "openrouter_gemma4_31b_free",
        label = "OpenRouter • Gemma 4 31B Free",
        credentialId = "openrouter",
        endpoint = "https://openrouter.ai/api/v1/chat/completions",
        model = "google/gemma-4-31b-it:free",
        freeRoute = true
    ),
    NVIDIA_NEMOTRON(
        id = "nvidia_nemotron",
        label = "NVIDIA Direct • Nemotron 3.5 Super VL",
        credentialId = "nvidia",
        endpoint = "https://integrate.api.nvidia.com/v1/chat/completions",
        model = "nvidia/nemotron-3.5-super-vl-120b-a12b",
        freeRoute = true
    ),
    GEMINI_38_FLASH(
        id = "gemini_38_flash",
        label = "Google • Gemini 3.8 Flash",
        credentialId = "gemini",
        endpoint = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        model = "gemini-3.8-flash"
    ),
    OPENROUTER_NEMOTRON(
        id = "openrouter_nemotron",
        label = "OpenRouter • Nemotron 3.5 Super VL",
        credentialId = "openrouter",
        endpoint = "https://openrouter.ai/api/v1/chat/completions",
        model = "nvidia/nemotron-3.5-super-vl",
        includeInAuto = false
    ),
    OPENROUTER_QWEN38(
        id = "openrouter_qwen38",
        label = "OpenRouter • Qwen 3.8 27B",
        credentialId = "openrouter",
        endpoint = "https://openrouter.ai/api/v1/chat/completions",
        model = "qwen/qwen3.8-27b",
        includeInAuto = false
    ),
    OPENROUTER_FREE_ROUTER(
        id = "openrouter_free_router",
        label = "OpenRouter • Automatic Free Model",
        credentialId = "openrouter",
        endpoint = "https://openrouter.ai/api/v1/chat/completions",
        model = "openrouter/free",
        freeRoute = true,
        includeInAuto = false
    ),
    CUSTOM_OPENAI(
        id = "custom_openai",
        label = "Custom • OpenAI-compatible",
        credentialId = "custom",
        endpoint = null,
        model = null,
        includeInAuto = true
    );

    companion object {
        fun fromId(id: String?): AiProviderPreset = entries.firstOrNull { it.id == id } ?: AUTO

        val selectable: List<AiProviderPreset> = entries.toList()

        val autoPriority: List<AiProviderPreset> = listOf(
            GROQ_QWEN38,
            OPENROUTER_NEMOTRON_FREE,
            OPENROUTER_QWEN38_FREE,
            OPENROUTER_GEMMA4_31B_FREE,
            NVIDIA_NEMOTRON,
            GEMINI_38_FLASH,
            CUSTOM_OPENAI
        )
    }
}

data class ResolvedAiRoute(
    val preset: AiProviderPreset,
    val endpoint: String,
    val model: String,
    val apiKey: String
)
