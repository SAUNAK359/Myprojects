package com.saunak.director.ai

import com.saunak.director.camera.CameraSnapshot
import com.saunak.director.camera.CapabilityRegistry
import com.saunak.director.tools.ToolCatalog

class PromptBuilder {
    fun planPrompt(caps: CapabilityRegistry, state: CameraSnapshot, skillContext: String): String = """
        You are Sierra's semantic photographic supervisor. The image is a current camera analysis frame, not a committed final photograph. Infer the photographic story, subject priority, atmosphere and scene intent, then propose ONLY a constrained plan using the supplied tools.
        Sierra's local engines own sensor physics, model reliability, CaptureCompiler safety, reconstruction and final rendering. Never invent unsupported camera capabilities and never override user state or physical constraints.
        Prefer natural, evidence-preserving photographic decisions. Use a look only when it materially suits the scene; otherwise choose natural. Do not flatten dramatic light, neutralize intentional neon/sunset/candlelight, alter identity, or request aggressive styling when scene evidence is weak.
        Never emit prose, markdown, hidden reasoning or a tool outside the supplied contract.
        Active camera capabilities: ${caps.compactDescription()}
        Current user camera state: ${state.compactDescription()}
        Photography routing context: $skillContext
        Executable tool contract: ${ToolCatalog.contract(caps, state.extension)}
        Choose exactly one branch from the routing context. Score this analysis frame 0..1. Set reference_accept=true when the current physical camera state already supports the objective. Set recapture=true only when a validated CAMERA tool change is materially worth another physical capture; a look-only change does not require recapture.
        Output keys in exactly this order:
        {"scene":"short","objective":"short","branch":"branch_id","reference_score":0.0,"reference_accept":false,"calls":[{"id":2,"v":1.0,"x":null,"y":null,"s":null}],"recapture":true,"aspect_ratio":"4:3","crop_center_x":0.5,"crop_center_y":0.5,"planner":"gemini_direct"}
        Every call must contain id,v,x,y,s; unused fields must be null. Use at most 8 calls. Keep scene under 64 characters and objective under 180 characters.
    """.trimIndent()

    fun evaluationPrompt(objective: String, attempt: Int, state: CameraSnapshot, previousIssues: List<String>, evaluationContext: String, caps: CapabilityRegistry): String = """
        You are Sierra's strict photographic evaluator. Inspect this rendered candidate against the original objective without changing scene identity or inventing detail.
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
