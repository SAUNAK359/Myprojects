package com.saunak.director.director.live

import com.saunak.director.ai.DirectorSchemaCompiler

/** Fail-closed validation for Gemini-authored executable Director graphs. */
object LiveDirectorValidator {
    fun validate(spec: SierraDirectorSpec, contract: DirectorSchemaCompiler.Contract): SierraDirectorSpec {
        require(spec.schemaVersion == contract.version) { "Director schema version mismatch" }
        require(spec.schemaHash == contract.hash) { "Director schema capability hash mismatch" }
        require(spec.goalSummary.length <= 240) { "Director goal summary too long" }
        require(spec.goals.size in 1..DirectorSchemaCompiler.MAX_GOALS) { "Director goal count out of bounds" }

        val ids = spec.goals.map { it.id }
        require(ids.distinct().size == ids.size) { "Director graph contains duplicate goal ids" }
        val byId = spec.goals.associateBy { it.id }
        spec.goals.forEach { node ->
            require(node.id.length <= 64) { "Director goal id too long" }
            require(node.type in contract.allowedGoalTypes) { "Unsupported Director goal type ${node.type}" }
            require(node.entityId == null || node.entityId == spec.primaryEntity.entityId) { "Director goal references unknown entity ${node.entityId}" }
            require(node.dependsOn.size <= DirectorSchemaCompiler.MAX_GRAPH_EDGES && node.invalidates.size <= DirectorSchemaCompiler.MAX_GRAPH_EDGES) { "Director graph edge count out of bounds" }
            require(node.dependsOn.distinct().size == node.dependsOn.size) { "Duplicate Director dependencies" }
            require(node.invalidates.distinct().size == node.invalidates.size) { "Duplicate Director invalidations" }
            require(node.dependsOn.none { it == node.id } && node.invalidates.none { it == node.id }) { "Director goal cannot reference itself" }
            require(node.dependsOn.all(byId::containsKey)) { "Director goal has unknown dependency" }
            require(node.invalidates.all(byId::containsKey)) { "Director goal has unknown invalidation target" }
            require(node.acceptableTolerance >= node.idealTolerance) { "Director ideal tolerance exceeds acceptable tolerance" }
            require(node.instructionLow.length <= 120 && node.instructionHigh.length <= 120 && node.instructionHold.length <= 120) { "Director coaching instruction too long" }
            validateGoalSemantics(node, spec)
            if (node.required) require(node.priority != GoalPriority.PREFERENCE) { "Required Director goal cannot be preference-only" }
        }

        require(spec.goals.any { it.type == DirectorGoalTypes.ENTITY_PLACEMENT && it.entityId == spec.primaryEntity.entityId }) {
            "Director graph must contain executable primary-entity placement"
        }
        require(spec.goals.any { it.priority == GoalPriority.HARD || it.required }) { "Director graph must contain at least one hard/required gate" }
        require(spec.readiness.exitFactor > spec.readiness.enterFactor) { "Director hysteresis requires exit_factor > enter_factor" }
        require(spec.readiness.maxStabilityMs >= spec.readiness.baseStabilityMs) { "Invalid Director stability policy" }
        require(spec.goals.filter { it.priority != GoalPriority.PREFERENCE }.all { it.stabilityMs <= spec.readiness.maxStabilityMs }) { "Director goal stability exceeds readiness maximum" }

        detectDependencyCycle(spec.goals)
        validateInvalidationSafety(spec.goals)
        return spec
    }

    private fun validateGoalSemantics(node: DirectorGoalNode, spec: SierraDirectorSpec) {
        fun requireEntity() = require(node.entityId == spec.primaryEntity.entityId) { "Entity-specific Director goal must bind to primary entity" }
        fun range(min: Double, max: Double) {
            require(node.targetMin == null || node.targetMin in min..max) { "Director ${node.type} target_min outside measurable range" }
            require(node.targetMax == null || node.targetMax in min..max) { "Director ${node.type} target_max outside measurable range" }
        }
        fun requireTarget() = require(node.targetMin != null || node.targetMax != null) { "Director ${node.type} requires an executable target interval" }

        when (node.type) {
            DirectorGoalTypes.ENTITY_PLACEMENT -> {
                requireEntity()
                require(node.targetMin == null && node.targetMax == null) { "Entity placement uses primary_entity geometry, not scalar targets" }
                require(node.idealTolerance in 0.0..0.75 && node.acceptableTolerance in 0.75..1.50) { "Entity placement tolerance must be a bounded multiplier of primary geometry tolerance" }
            }
            DirectorGoalTypes.SUBJECT_SCALE -> { requireEntity(); requireTarget(); range(0.0, 1.0) }
            DirectorGoalTypes.ENTITY_VISIBILITY -> {
                requireEntity(); requireTarget(); range(0.0, 1.0)
                node.targetMin?.let { require(it + 1e-9 >= spec.primaryEntity.visibilityMin) { "Visibility target cannot weaken primary-entity visibility requirement" } }
            }
            DirectorGoalTypes.SUBJECT_LUMINANCE -> { requireTarget(); range(0.0, 1.0) }
            DirectorGoalTypes.LIGHT_BALANCE_X -> { requireTarget(); range(-1.0, 1.0) }
            DirectorGoalTypes.MOTION_STABILITY -> {
                require(node.targetMax != null) { "Motion stability requires a maximum motion target" }
                range(0.0, 1.0)
            }
            DirectorGoalTypes.TECHNICAL_SHARPNESS -> { requireTarget(); range(0.0, 1.0) }
            DirectorGoalTypes.BACKGROUND_SEPARATION -> { requireEntity(); requireTarget(); range(-1.0, 1.0) }
            DirectorGoalTypes.POSE_KEYPOINT_ALIGNMENT -> {
                requireEntity()
                require(node.targetKeypoints.isNotEmpty()) { "Pose goal requires target keypoints" }
                require(node.targetKeypoints.size <= DirectorSchemaCompiler.MAX_POSE_KEYPOINTS) { "Too many Director pose keypoints" }
                require(node.targetKeypoints.map { it.index }.distinct().size == node.targetKeypoints.size) { "Duplicate pose keypoint targets" }
                require(node.targetMin == null) { "Pose alignment does not use target_min" }
                require(node.targetMax != null && node.targetMax in 0.005..0.30) { "Pose alignment requires normalized maximum landmark error" }
            }
            DirectorGoalTypes.FACE_PLACEMENT -> {
                requireEntity()
                require(node.targetMin == null) { "Face placement does not use target_min" }
                require(node.targetMax != null && node.targetMax in 0.005..0.30) { "Face placement requires normalized maximum center error" }
            }
            else -> error("Unsupported Director goal type ${node.type}")
        }
        if (node.type != DirectorGoalTypes.POSE_KEYPOINT_ALIGNMENT) require(node.targetKeypoints.isEmpty()) { "Non-pose Director goal cannot carry pose keypoints" }
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
                require(!(node.priority == GoalPriority.PREFERENCE && target.priority == GoalPriority.HARD)) { "Preference goal cannot invalidate a hard Director gate" }
            }
        }
    }
}
