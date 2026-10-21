package com.saunak.director.ai

import com.saunak.director.camera.CameraSnapshot
import com.saunak.director.camera.CapabilityRegistry
import com.saunak.director.tools.ToolCatalog

class PromptBuilder {
    fun planPrompt(caps: CapabilityRegistry, state: CameraSnapshot, skillContext: String): String = """
        You are Director, an on-device photography controller. The user has already pressed the white shutter and the image you see is that exact full-quality reference photograph. Analyze this committed reference and output ONLY the constrained JSON object.
        Never emit prose, markdown, hidden reasoning or a tool outside the supplied contract.
        Active camera capabilities: ${caps.compactDescription()}
        Current user camera state: ${state.compactDescription()}
        Photography routing context: $skillContext
        Executable tool contract: ${ToolCatalog.contract(caps, state.extension)}
        Choose exactly one branch from the routing context. Score the current reference 0..1. Set reference_accept=true only if it already satisfies the objective. Set recapture=true only when a validated CAMERA tool change is materially worth another physical capture; a look-only change does not require recapture.
        Output keys in exactly this order:
        {"scene":"short","objective":"short","branch":"branch_id","reference_score":0.0,"reference_accept":false,"calls":[{"id":2,"v":1.0,"x":null,"y":null,"s":null}],"recapture":true}
        Every call must contain id,v,x,y,s; unused fields must be null. Use at most 8 calls.
    """.trimIndent()

    fun evaluationPrompt(objective: String, attempt: Int, state: CameraSnapshot, previousIssues: List<String>, evaluationContext: String, caps: CapabilityRegistry): String = """
        You are Director's strict evaluator. Inspect this captured/rendered candidate against the original objective.
        Objective: $objective
        Physical capture number: $attempt of 3 maximum
        Camera state: ${state.compactDescription()}
        Previous issues: ${previousIssues.joinToString(",").ifBlank { "none" }}
        Selected skill context: $evaluationContext
        Allowed delta-tool contract: ${ToolCatalog.contract(caps, state.extension)}
        Output ONLY constrained JSON. Scores must be 0..1. Accept a genuinely strong natural result. If rejecting, request only minimal supported delta changes and do not oscillate direction.
        Output keys in exactly this order:
        {"score":0.0,"technical":0.0,"composition":0.0,"subject":0.0,"aesthetic":0.0,"accept":false,"issues":["short_issue"],"delta":[{"id":3,"v":0.0,"x":null,"y":null,"s":null}]}
        Every delta call must contain id,v,x,y,s; unused fields must be null. Use at most 8 issues and 8 delta calls.
    """.trimIndent()
}
