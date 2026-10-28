package com.saunak.director

import com.saunak.director.ai.DirectorSchemaCompiler
import com.saunak.director.director.DirectorStrategy
import com.saunak.director.director.InterventionPolicy
import com.saunak.director.director.live.DirectorActor
import com.saunak.director.director.live.DirectorGoalNode
import com.saunak.director.director.live.DirectorGoalTypes
import com.saunak.director.director.live.DirectorReadinessPolicy
import com.saunak.director.director.live.DirectorRect
import com.saunak.director.director.live.GoalPriority
import com.saunak.director.director.live.LiveDirectorValidator
import com.saunak.director.director.live.PrimaryEntityTarget
import com.saunak.director.director.live.SierraDirectorSpec
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class LiveDirectorStrategyTest {
    private val contract = DirectorSchemaCompiler.Contract(
        version = "director.v2",
        hash = "test-hash",
        allowedGoalTypes = setOf(
            DirectorGoalTypes.ENTITY_PLACEMENT,
            DirectorGoalTypes.POSE_KEYPOINT_ALIGNMENT,
            DirectorGoalTypes.SUBJECT_BACKGROUND_EV,
            DirectorGoalTypes.BACKGROUND_DISTRACTION
        ),
        jsonSchema = JSONObject(),
        promptContract = "test"
    )

    private fun basePlacement(priority: GoalPriority = GoalPriority.HARD, required: Boolean = true, actor: DirectorActor = DirectorActor.PHOTOGRAPHER) = DirectorGoalNode(
        id = "placement",
        type = DirectorGoalTypes.ENTITY_PLACEMENT,
        priority = priority,
        required = required,
        weight = 1.0,
        entityId = "subject",
        idealTolerance = 0.35,
        acceptableTolerance = 1.0,
        actor = actor,
        stabilityMs = 400
    )

    private fun baseSpec(strategy: DirectorStrategy, intervention: InterventionPolicy, goals: List<DirectorGoalNode>) = SierraDirectorSpec(
        schemaVersion = contract.version,
        schemaHash = contract.hash,
        goalSummary = "test photograph",
        strategy = strategy,
        interventionPolicy = intervention,
        primaryEntity = PrimaryEntityTarget(
            entityId = "subject",
            semanticLabel = "person",
            entityKind = "person",
            currentBox = DirectorRect(.25, .15, .65, .90),
            targetBox = DirectorRect(.28, .14, .68, .89),
            movableActor = DirectorActor.PHOTOGRAPHER
        ),
        goals = goals,
        readiness = DirectorReadinessPolicy(maxStabilityMs = 900)
    )

    @Test
    fun candidMinimalAllowsPhotographerFramingWithoutSubjectIntervention() {
        val spec = baseSpec(DirectorStrategy.CANDID, InterventionPolicy.MINIMAL, listOf(basePlacement()))
        val validated = LiveDirectorValidator.validate(spec, contract)
        assertEquals(DirectorStrategy.CANDID, validated.strategy)
        assertEquals(InterventionPolicy.MINIMAL, validated.interventionPolicy)
    }

    @Test
    fun candidMinimalRejectsBlockingSubjectReposition() {
        val spec = baseSpec(
            DirectorStrategy.CANDID,
            InterventionPolicy.MINIMAL,
            listOf(basePlacement(actor = DirectorActor.SUBJECT))
        )
        assertThrows(IllegalArgumentException::class.java) { LiveDirectorValidator.validate(spec, contract) }
    }

    @Test
    fun observeRejectsAnyBlockingSubjectInstruction() {
        val placement = basePlacement(actor = DirectorActor.PHOTOGRAPHER)
        val light = DirectorGoalNode(
            id = "light",
            type = DirectorGoalTypes.SUBJECT_BACKGROUND_EV,
            priority = GoalPriority.IMPORTANT,
            required = false,
            weight = .8,
            entityId = "subject",
            targetMin = .2,
            targetMax = 1.0,
            actor = DirectorActor.SUBJECT,
            idealTolerance = .2,
            acceptableTolerance = 1.0,
            stabilityMs = 400
        )
        val spec = baseSpec(DirectorStrategy.CANDID, InterventionPolicy.OBSERVE, listOf(placement, light))
        assertThrows(IllegalArgumentException::class.java) { LiveDirectorValidator.validate(spec, contract) }
    }

    @Test
    fun backgroundDistractionRequiresUpperBound() {
        val placement = basePlacement()
        val distraction = DirectorGoalNode(
            id = "background",
            type = DirectorGoalTypes.BACKGROUND_DISTRACTION,
            priority = GoalPriority.IMPORTANT,
            required = false,
            weight = .6,
            targetMin = 0.0,
            targetMax = null,
            actor = DirectorActor.PHOTOGRAPHER,
            idealTolerance = .2,
            acceptableTolerance = 1.0,
            stabilityMs = 400
        )
        val spec = baseSpec(DirectorStrategy.PORTRAIT, InterventionPolicy.DIRECTED, listOf(placement, distraction))
        assertThrows(IllegalArgumentException::class.java) { LiveDirectorValidator.validate(spec, contract) }
    }
}
