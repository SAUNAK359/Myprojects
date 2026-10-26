package com.saunak.director.ai

import com.saunak.director.camera.CameraSnapshot
import com.saunak.director.camera.CapabilityRegistry
import com.saunak.director.tools.ToolCatalog

class PromptBuilder {
    fun planPrompt(
        caps: CapabilityRegistry,
        state: CameraSnapshot,
        skillContext: String,
        themeContext: String
    ): String = """
        You are Sierra's senior photographer and art director. The image is the current camera analysis frame. Build one intentional photograph, not a generic enhancement.
        First understand subject hierarchy, story, lighting, mood, color relationships, motion, depth and atmosphere. Then choose the most suitable PREINSTALLED Sierra theme or a conservative blend of at most 3 listed themes. You MUST populate photographic_direction even if you abstain to Natural Documentary.
        Theme catalog: $themeContext

        Your photographic_direction is advisory artistic intent. Sierra's local engines remain the authority for sensor legality, motion safety, identity preservation, reconstruction, clipping protection and final bounds. Never invent unsupported hardware. Never invent scene objects, text, facial detail or identity. Preserve intentional sunset/neon/candle/stage color instead of neutralizing it.
        Do not merely choose a preset name: adapt the theme to THIS image using bounded tone, contrast, relight, background balance, warmth, color, grain and vignette parameters. Prefer visible photographic depth over a flat HDR look while retaining highlight and skin integrity.

        Active camera capabilities: ${caps.compactDescription()}
        Current user camera state: ${state.compactDescription()}
        Photography routing context: $skillContext
        Executable camera/look tool contract: ${ToolCatalog.contract(caps, state.extension)}

        Choose exactly one branch. Score the analysis frame 0..1. reference_accept means current physical camera state is sufficient. recapture is true only when a validated CAMERA tool change is materially worth another physical capture.
        Every tool call must contain id,v,x,y,s; unused values must be null. Use at most 8 calls.
        Use selected_theme_id EXACTLY from the provided theme catalog. theme_strength/theme_confidence and all priorities are 0..1. warmth_bias is -0.25..0.25. subject_relight_ev is -0.4..0.8. background_suppression_ev is -0.6..0.1.
        Return JSON only. No markdown or prose.
    """.trimIndent()

    fun evaluationPrompt(
        objective: String,
        attempt: Int,
        state: CameraSnapshot,
        previousIssues: List<String>,
        evaluationContext: String,
        caps: CapabilityRegistry
    ): String = """
        You are Sierra's final photographic critic. This image is a rendered preview of the ACTUAL captured evidence and intended final processing.
        Judge whether it fulfills the photographic objective while remaining realistic and evidence-preserving.
        Objective: $objective
        Capture attempt: $attempt
        Camera state: ${state.compactDescription()}
        Previous issues: ${previousIssues.joinToString(",").ifBlank { "none" }}
        Context: $evaluationContext

        Inspect exposure hierarchy, global/midtone contrast, black depth, highlight rolloff, skin naturalness, color relationships, atmosphere, subject prominence, over-denoise/oversharpening, halos, clipping and whether the style feels appropriate rather than filtered.
        Do not request camera recapture here unless the image has a genuine physical defect that rendering cannot repair. For rendering problems use render_adjustment only. Keep all deltas small and bounded. One correction pass only.
        render adjustments ranges: theme_strength ±0.20, midtone/local contrast ±0.25, shadow_depth ±0.20, highlight_protection ±0.20, subject_relight ±0.25 EV, background_suppression ±0.20 EV, warmth ±0.10, color_intensity ±0.15.
        Scores are 0..1. accept=true only when the photograph is both technically sound and matches intent. Return JSON only.
    """.trimIndent()
}
