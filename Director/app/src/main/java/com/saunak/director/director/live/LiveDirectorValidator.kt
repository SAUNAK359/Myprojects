package com.saunak.director.director.live

import com.saunak.director.ai.DirectorSchemaCompiler

/** Fail-closed validation for Gemini-authored executable Director graphs. */
object LiveDirectorValidator {
    fun validate(spec: SierraDirectorSpec, contract: DirectorSchemaCompiler.Contract): SierraDirectorSpec {
        require(spec.schemaVersion == contract.version) { "Director schema version mismatch" }
        require(spec.schemaHash == contract.hash) { "Director schema capability hash mismatch" }
        require(spec.goalSummary.length <= 240) { "Director goal summary too long" }
        require(spec.goals.size in 1..18) { "Director goal count out of bounds" }

        val ids = spec.goals.map { it.id }
        require(ids.distinct().size == ids.size) { "Director graph contains duplicate goal ids" }
        val byId = spec.goals.associateBy { it.id }
        spec.goals.forEach { node ->
            require(node.id.length <= 64) { "Director goal id too long" }
            require(node.type in contract.allowedGoalTypes) { "Unsupported Director goal type ${node.type}" }
            require(node.entityId == null || node.entityId == spec.primaryEntity.entityId) {
                "Director goal references unknown entity ${node.entityId}"
            }
            require(node.dependsOn.distinct().size == node.dependsOn.size) { "Duplicate Director dependencies" }
            require(node.invalidates.distinct().size == node.invalidates.size) { "Duplicate Director invalidations" }
            require(node.dependsOn.none { it == node.id } && node.invalidates.none { it == node.id }) { "Director goal cannot reference itself" }
            require(node.dependsOn.all(byId::containsKey)) { "Director goal has unknown dependency" }
            require(node.invalidates.all(byId::containsKey)) { "Director goal has unknown invalidation target" }
            require(node.acceptableTolerance >= node.idealTolerance) { "Director ideal tolerance exceeds acceptable tolerance" }
            require(node.instructionLow.length <= 120 && node.instructionHigh.length <= 120 && node.instructionHold.length <= 120) {
                "Director coaching instruction too long"
            }
            if (node.type == DirectorGoalTypes.POSE_KEYPOINT_ALIGNMENT) {
                require(node.targetKeypoints.isNotEmpty()) { "Pose goal requires target keypoints" }
                require(node.targetKeypoints.map { it.index }.distinct().size == node.targetKeypoints.size) { "Duplicate pose keypoint targets" }
            } else {
                require(node.targetKeypoints.isEmpty()) { "Non-pose Director goal cannot carry pose keypoints" }
            }
            when (node.type) {
                DirectorGoalTypes.ENTITY_PLACEMENT,
                DirectorGoalTypes.SUBJECT_SCALE,
                DirectorGoalTypes.ENTITY_VISIBILITY,
                DirectorGoalTypes.POSE_KEYPOINT_ALIGNMENT,
                DirectorGoalTypes.FACE_PLACEMENT,
                DirectorGoalTypes.BACKGROUND_SEPARATION -> require(node.entityId == spec.primaryEntity.entityId) {
                    "Entity-specific Director goal must bind to primary entity"
                }
            }
            if (node.required) require(node.priority != GoalPriority.PREFERENCE) {
                "Required Director goal cannot be preference-only"
            }
        }

        require(spec.goals.any { it.type == DirectorGoalTypes.ENTITY_PLACEMENT && it.entityId == spec.primaryEntity.entityId }) {
            "Director graph must contain executable primary-entity placement"
        }
        require(spec.goals.any { it.priority == GoalPriority.HARD || it.required }) {
            "Director graph must contain at least one hard/required gate"
        }
        require(spec.readiness.exitFactor > spec.readiness.enterFactor) { "Director hysteresis requires exit_factor > enter_factor" }
        require(spec.readiness.maxStabilityMs >= spec.readiness.baseStabilityMs) { "Invalid Director stability policy" }

        detectDependencyCycle(spec.goals)
        validateInvalidationSafety(spec.goals)
        return spec
    }

    private fun detectDependencyCycle(nodes: List<DirectorGoalNode>) {
        val graph = nodes.associate { it.id to it.dependsOn }
        val visiting = mutableSetOf<String>()
        val visited = mutableSetOf<String>()
        fun visit(id: String) {
            if (id in visited) return
            require(id !in visiting) { "Director graph contains a dependency cycle" }
            visiting += id
            graph[id].orEmpty().forEach(::visit)
            visiting -= id
            visited += id
        }
        graph.keys.forEach(::visit)
    }

    private fun validateInvalidationSafety(nodes: List<DirectorGoalNode>) {
        val byId = nodes.associateBy { it.id }
        nodes.forEach { node ->
            node.invalidates.forEach { targetId ->
                val target = byId.getValue(targetId)
                // Lower-priority refinements must not invalidate hard safety gates. This avoids a
                // coaching plan where cosmetic optimization repeatedly destroys technical readiness.
                require(!(node.priority == GoalPriority.PREFERENCE && target.priority == GoalPriority.HARD)) {
                    "Preference goal cannot invalidate a hard Director gate"
                }
            }
        }
    }
}
