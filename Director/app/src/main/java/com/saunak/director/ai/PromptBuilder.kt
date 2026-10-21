package com.saunak.director.ai

import com.saunak.director.camera.CameraSnapshot
import com.saunak.director.camera.CapabilityRegistry

class PromptBuilder {
    fun planPrompt(caps: CapabilityRegistry, state: CameraSnapshot, skillContext: String): String = """
        You are Director, an autonomous photography controller. Analyze the image and output ONLY one compact JSON object.
        Never output markdown, prose, code fences, hidden reasoning, or unsupported tools.
        Objective: create the strongest photograph possible while preserving natural detail and avoiding destructive edits.
        Active camera capabilities: ${caps.compactDescription()}
        Current camera state: ${state.compactDescription()}
        Photography skill context: $skillContext
        Allowed tool IDs: ${caps.allowedToolIds.sorted().joinToString(",")}
        Output schema exactly:
        {"scene":"short category","objective":"short desired look","branch":"skill branch","calls":[{"id":integer,"v":number_optional,"s":"string_optional"}],"capture":true}
        Use only necessary calls. Prefer native auto camera behavior unless a change clearly improves the image.
    """.trimIndent()
    fun evaluationPrompt(objective: String, attempt: Int, state: CameraSnapshot, issues: List<String>, evaluationContext: String): String = """
        You are Director's strict photo evaluator. Inspect this captured result against the objective.
        Objective: $objective
        Attempt: $attempt of 3
        Camera state: ${state.compactDescription()}
        Previous issues: ${issues.joinToString(",").ifBlank { "none" }}
        Evaluation skill: $evaluationContext
        Output ONLY JSON and no prose:
        {"score":0.0,"technical":0.0,"composition":0.0,"subject":0.0,"aesthetic":0.0,"accept":false,"issues":["short_issue"],"delta":[{"id":integer,"v":number_optional,"s":"string_optional"}]}
        Scores are 0..1. Accept only when the result is clearly strong and technically valid. If rejecting, propose the smallest supported delta change; do not redesign the whole shot.
    """.trimIndent()
}
