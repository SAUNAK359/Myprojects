package com.saunak.director.director.live

import android.graphics.Bitmap
import com.saunak.director.director.InterventionPolicy
import com.saunak.director.director.SierraPerceptionFabric
import com.saunak.director.engine.subject.NormalizedRect
import com.saunak.director.processing.TechnicalQuality
import java.util.ArrayDeque
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.sqrt

/** Local deterministic execution layer for Gemini's photographic goal graph. */
class SierraLiveDirectorRuntime {
    private data class ReadinessSample(val timestampNs: Long, val pass: Boolean, val score: Double)

    private val anchor = VisualEntityAnchor()
    private val hysteresisState = mutableMapOf<String, Boolean>()
    private val readinessHistory = ArrayDeque<ReadinessSample>()
    private var spec: SierraDirectorSpec? = null
    private var previousEntityBounds: NormalizedRect? = null
    private var previousObservedAtNs: Long? = null
    private var lastState = LiveDirectorState()

    fun initialize(spec: SierraDirectorSpec, planningFrame: Bitmap): LiveDirectorState {
        this.spec = spec
        anchor.reset()
        val anchored = anchor.initialize(planningFrame, spec.primaryEntity.currentBox)
        previousEntityBounds = anchored.bounds
        previousObservedAtNs = System.nanoTime()
        hysteresisState.clear()
        readinessHistory.clear()
        lastState = LiveDirectorState(
            running = true,
            entityLocked = true,
            overlay = DirectorOverlayState(
                targetBox = spec.primaryEntity.targetBox,
                currentBox = DirectorRect.from(anchored.bounds),
                sourceWidth = planningFrame.width,
                sourceHeight = planningFrame.height,
                confidence = anchored.confidence,
                activeGoalType = DirectorGoalTypes.ENTITY_PLACEMENT,
                showTarget = true
            ),
            status = "Sierra Director target locked • ${spec.strategy.name.lowercase().replace('_', ' ')}"
        )
        return lastState
    }

    fun stop(): LiveDirectorState {
        spec = null
        anchor.reset()
        hysteresisState.clear()
        readinessHistory.clear()
        previousEntityBounds = null
        previousObservedAtNs = null
        lastState = LiveDirectorState()
        return lastState
    }

    fun latest(): LiveDirectorState = lastState

    fun observe(
        frame: Bitmap,
        perception: SierraPerceptionFabric.Snapshot?,
        quality: TechnicalQuality,
        timestampNs: Long = System.nanoTime()
    ): LiveDirectorState {
        val graph = spec ?: return LiveDirectorState(status = "Director target unavailable")
        val anchorObs = anchor.track(frame)
        val measuredEntity = chooseEntityBounds(graph, anchorObs, perception)
        val entityBounds = measuredEntity?.first
        val entityConfidence = measuredEntity?.second ?: 0.0
        val entityVisibility = anchorObs?.visibility ?: entityBounds?.let(::boxVisibility) ?: 0.0
        val entityLocked = entityBounds != null && entityConfidence >= MIN_ENTITY_LOCK_CONFIDENCE
        val motion = entityMotion(entityBounds, timestampNs)

        val measurements = graph.goals.associate { node ->
            node.id to measure(node, graph, frame, perception, quality, entityBounds, entityConfidence, entityVisibility, motion, timestampNs)
        }
        val evaluations = graph.goals.map { node -> evaluate(node, measurements.getValue(node.id), graph.readiness) }
        val hardPass = evaluations.filter { it.node.priority == GoalPriority.HARD || it.node.required }
            .all { it.measurement.state != GoalState.UNKNOWN && it.satisfaction >= graph.readiness.requiredNodeFloor }
        val important = evaluations.filter { it.node.priority == GoalPriority.IMPORTANT && it.measurement.state != GoalState.UNKNOWN }
        val importantScore = weightedScore(important)
        val requiredImportantPass = evaluations.filter { it.node.required && it.node.priority == GoalPriority.IMPORTANT }
            .all { it.measurement.state != GoalState.UNKNOWN && it.satisfaction >= graph.readiness.requiredNodeFloor }
        val dependencyPass = evaluations.filter { it.node.priority != GoalPriority.PREFERENCE }.all { evaluation ->
            evaluation.node.dependsOn.all { depId ->
                evaluations.firstOrNull { it.node.id == depId }?.measurement?.state in setOf(GoalState.ACCEPTABLE, GoalState.IDEAL)
            }
        }
        val candidateReady = entityLocked && hardPass && requiredImportantPass && dependencyPass && importantScore >= graph.readiness.importantScoreMin
        val adaptiveWindowMs = adaptiveStabilityMs(graph, evaluations, motion)
        val stable = updateStability(timestampNs, candidateReady, importantScore, graph.readiness, adaptiveWindowMs)
        val captureReady = candidateReady && stable

        val instruction = if (captureReady) CoachingInstruction("capture_ready", "Hold there", DirectorActor.NONE, 0.0, achieved = true)
            else selectInstruction(graph, evaluations, entityBounds)
        val active = instruction?.nodeId?.let { id -> graph.goals.firstOrNull { it.id == id } }
        val currentPose = perception?.subject?.pose?.keypoints?.sortedBy { it.index }?.map { it.position }.orEmpty()
        val overlay = DirectorOverlayState(
            targetBox = graph.primaryEntity.targetBox,
            currentBox = entityBounds?.let(DirectorRect::from),
            targetKeypoints = active?.targetKeypoints.orEmpty(),
            currentKeypoints = if (active?.type == DirectorGoalTypes.POSE_KEYPOINT_ALIGNMENT) currentPose else emptyList(),
            sourceWidth = frame.width,
            sourceHeight = frame.height,
            confidence = entityConfidence,
            activeGoalType = active?.type ?: if (captureReady) "capture_ready" else DirectorGoalTypes.ENTITY_PLACEMENT,
            showTarget = entityConfidence >= OVERLAY_MIN_CONFIDENCE || entityBounds == null
        )

        lastState = LiveDirectorState(
            running = true,
            entityLocked = entityLocked,
            readinessScore = importantScore,
            hardPass = hardPass,
            stable = stable,
            captureReady = captureReady,
            stableSinceNs = stableWindowStart(timestampNs, adaptiveWindowMs),
            instruction = instruction,
            overlay = overlay,
            evaluations = evaluations,
            status = when {
                !entityLocked -> "Find ${graph.primaryEntity.semanticLabel} in the guide"
                captureReady -> "Ready • hold there"
                candidateReady -> "Good • hold steady"
                instruction != null -> instruction.text
                else -> "Sierra checking the photograph"
            }
        )
        return lastState
    }

    private fun chooseEntityBounds(
        graph: SierraDirectorSpec,
        anchorObs: VisualEntityAnchor.Observation?,
        perception: SierraPerceptionFabric.Snapshot?
    ): Pair<NormalizedRect, Double>? {
        val anchorBounds = anchorObs?.bounds
        val anchorConfidence = anchorObs?.confidence ?: 0.0
        val subject = perception?.subject
        if (subject != null && isHumanKind(graph.primaryEntity.entityKind)) {
            val compatible = anchorBounds == null || subject.bounds.iou(anchorBounds) >= 0.08 || subject.center.distanceTo(anchorBounds.center) <= 0.18
            if (compatible && subject.confidence >= 0.42) {
                val fused = if (anchorBounds != null && anchorConfidence >= 0.35) blendRects(anchorBounds, subject.bounds, 0.42, 0.58) else subject.bounds
                return fused to max(subject.confidence, anchorConfidence * 0.85)
            }
        }
        return anchorBounds?.let { it to anchorConfidence }
    }

    private fun measure(
        node: DirectorGoalNode,
        graph: SierraDirectorSpec,
        frame: Bitmap,
        perception: SierraPerceptionFabric.Snapshot?,
        quality: TechnicalQuality,
        entity: NormalizedRect?,
        entityConfidence: Double,
        entityVisibility: Double,
        entityMotion: Double?,
        timestampNs: Long
    ): GoalMeasurement {
        fun unknown(source: String) = GoalMeasurement(node.id, null, 0.0, timestampNs, source)
        return when (node.type) {
            DirectorGoalTypes.ENTITY_PLACEMENT -> {
                if (entity == null) unknown("entity_unobserved") else {
                    val target = graph.primaryEntity.targetBox.normalizedRect()
                    val centerError = entity.center.distanceTo(target.center)
                    val scaleError = max(abs(entity.width - target.width) / max(target.width, 1e-6), abs(entity.height - target.height) / max(target.height, 1e-6))
                    val iouPenalty = 1.0 - entity.iou(target)
                    val normalized = maxOf(centerError / graph.primaryEntity.centerTolerance, scaleError / graph.primaryEntity.scaleTolerance, iouPenalty / max(1.0 - graph.primaryEntity.minIou, 1e-3))
                    GoalMeasurement(node.id, normalized, entityConfidence, timestampNs, "primary_entity_geometry")
                }
            }
            DirectorGoalTypes.SUBJECT_SCALE -> entity?.let { GoalMeasurement(node.id, sqrt(it.area), entityConfidence, timestampNs, "entity_box_scale") } ?: unknown("entity_unobserved")
            DirectorGoalTypes.ENTITY_VISIBILITY -> entity?.let { GoalMeasurement(node.id, entityVisibility, entityConfidence, timestampNs, "entity_frame_visibility") } ?: unknown("entity_unobserved")
            DirectorGoalTypes.SUBJECT_LUMINANCE -> entity?.let { GoalMeasurement(node.id, regionMeanLuma(frame, it) / 255.0, entityConfidence, timestampNs, "entity_luminance") } ?: unknown("entity_unobserved")
            DirectorGoalTypes.LIGHT_BALANCE_X -> entity?.let { GoalMeasurement(node.id, horizontalLightBalance(frame, it), entityConfidence, timestampNs, "entity_light_balance") } ?: unknown("entity_unobserved")
            DirectorGoalTypes.SUBJECT_BACKGROUND_EV -> entity?.let {
                GoalMeasurement(node.id, subjectBackgroundEv(frame, it), entityConfidence, timestampNs, "subject_background_luminance_ratio")
            } ?: unknown("entity_unobserved")
            DirectorGoalTypes.BACKGROUND_DISTRACTION -> entity?.let {
                GoalMeasurement(node.id, backgroundDistraction(frame, it), entityConfidence, timestampNs, "background_saliency_grid")
            } ?: unknown("entity_unobserved")
            DirectorGoalTypes.TECHNICAL_SHARPNESS -> GoalMeasurement(node.id, (quality.edgeEnergy / 18.0).coerceIn(0.0, 1.0), if (quality.catastrophic) 0.2 else 0.9, timestampNs, "technical_quality")
            DirectorGoalTypes.MOTION_STABILITY -> entityMotion?.let { GoalMeasurement(node.id, it, entityConfidence, timestampNs, "entity_temporal_motion") } ?: unknown("motion_unobserved")
            DirectorGoalTypes.BACKGROUND_SEPARATION -> perception?.subject?.depth?.let { GoalMeasurement(node.id, it.foregroundSeparation, it.confidence, timestampNs, it.source) } ?: unknown("relative_depth_unavailable")
            DirectorGoalTypes.POSE_KEYPOINT_ALIGNMENT -> {
                val pose = perception?.subject?.pose ?: return unknown("pose_unavailable")
                val actual = pose.keypoints.associateBy { it.index }
                var weightedError = 0.0
                var weight = 0.0
                node.targetKeypoints.forEach { target ->
                    val point = actual[target.index] ?: return@forEach
                    if (point.confidence < target.confidenceMin) return@forEach
                    weightedError += hypot(point.position.x - target.x, point.position.y - target.y) * point.confidence
                    weight += point.confidence
                }
                if (weight <= 1e-6) unknown("pose_keypoints_unobserved") else GoalMeasurement(node.id, weightedError / weight, pose.confidence, timestampNs, pose.source)
            }
            DirectorGoalTypes.FACE_PLACEMENT -> {
                val face = perception?.faces?.maxByOrNull { it.area * it.score } ?: return unknown("face_unavailable")
                val rect = NormalizedRect.fromPixels(face.x.toDouble(), face.y.toDouble(), face.width.toDouble(), face.height.toDouble(), frame.width, frame.height)
                GoalMeasurement(node.id, rect.center.distanceTo(graph.primaryEntity.targetBox.normalizedRect().center), face.score.toDouble(), timestampNs, "yunet_face")
            }
            else -> unknown("unsupported_measurement")
        }
    }

    private fun evaluate(node: DirectorGoalNode, raw: GoalMeasurement, policy: DirectorReadinessPolicy): GoalEvaluation {
        if (raw.value == null || raw.confidence < node.minConfidence) {
            hysteresisState[node.id] = false
            return GoalEvaluation(node, raw.copy(state = GoalState.UNKNOWN), 0.0, false)
        }
        val error = when (node.type) {
            DirectorGoalTypes.ENTITY_PLACEMENT -> raw.value
            DirectorGoalTypes.POSE_KEYPOINT_ALIGNMENT, DirectorGoalTypes.FACE_PLACEMENT -> abs(raw.value) / (node.targetMax ?: node.acceptableTolerance.coerceAtLeast(1e-4))
            else -> intervalErrorRatio(raw.value, node.targetMin, node.targetMax)
        }
        val wasPass = hysteresisState[node.id] == true
        val boundary = if (wasPass) node.acceptableTolerance * policy.exitFactor else node.acceptableTolerance * policy.enterFactor
        val pass = error <= boundary
        hysteresisState[node.id] = pass
        val state = when { !pass -> GoalState.FAIL; error <= node.idealTolerance -> GoalState.IDEAL; else -> GoalState.ACCEPTABLE }
        val satisfaction = when (state) {
            GoalState.IDEAL -> 1.0
            GoalState.ACCEPTABLE -> (1.0 - 0.35 * (error / max(node.acceptableTolerance, 1e-6))).coerceIn(0.60, 1.0)
            GoalState.FAIL -> (1.0 - error / max(node.acceptableTolerance * policy.exitFactor * 2.0, 1e-6)).coerceIn(0.0, 0.59)
            GoalState.UNKNOWN -> 0.0
        }
        return GoalEvaluation(node, raw.copy(errorRatio = error, state = state), satisfaction, pass)
    }

    private fun selectInstruction(graph: SierraDirectorSpec, evaluations: List<GoalEvaluation>, entity: NormalizedRect?): CoachingInstruction? {
        val byId = evaluations.associateBy { it.node.id }
        val candidates = evaluations.filter { e ->
            e.measurement.state in setOf(GoalState.FAIL, GoalState.UNKNOWN) &&
                e.node.priority != GoalPriority.PREFERENCE &&
                interventionAllows(graph, e.node) &&
                e.node.dependsOn.all { dep -> byId[dep]?.measurement?.state in setOf(GoalState.ACCEPTABLE, GoalState.IDEAL) }
        }
        val selected = candidates.maxByOrNull { e ->
            val importance = when (e.node.priority) { GoalPriority.HARD -> 1.5; GoalPriority.IMPORTANT -> 1.0; GoalPriority.PREFERENCE -> 0.35 }
            importance * e.node.weight * (1.0 - e.satisfaction) * max(e.measurement.confidence, 0.30)
        } ?: return null

        if (selected.measurement.state == GoalState.UNKNOWN) {
            val text = when (selected.node.type) {
                DirectorGoalTypes.ENTITY_PLACEMENT, DirectorGoalTypes.ENTITY_VISIBILITY -> "Place ${graph.primaryEntity.semanticLabel} inside the guide"
                DirectorGoalTypes.POSE_KEYPOINT_ALIGNMENT -> "Keep the subject visible for pose guidance"
                else -> "Hold briefly while Sierra measures ${friendlyType(selected.node.type)}"
            }
            return CoachingInstruction(selected.node.id, text, selected.node.actor, 0.85)
        }

        if (selected.node.type == DirectorGoalTypes.ENTITY_PLACEMENT && entity != null) {
            val target = graph.primaryEntity.targetBox.normalizedRect()
            val dxImage = target.center.x - entity.center.x
            val dyImage = target.center.y - entity.center.y
            val actor = if (selected.node.actor == DirectorActor.NONE) graph.primaryEntity.movableActor else selected.node.actor
            val actorDx = if (actor == DirectorActor.PHOTOGRAPHER) -dxImage else dxImage
            val actorDy = if (actor == DirectorActor.PHOTOGRAPHER) -dyImage else dyImage
            val dominantHorizontal = abs(actorDx) >= abs(actorDy)
            val text = when {
                dominantHorizontal && actorDx < -0.012 -> movementPhrase(actor, "left")
                dominantHorizontal && actorDx > 0.012 -> movementPhrase(actor, "right")
                !dominantHorizontal && actorDy < -0.012 -> movementPhrase(actor, "up")
                !dominantHorizontal && actorDy > 0.012 -> movementPhrase(actor, "down")
                sqrt(entity.area) < sqrt(target.area) * (1.0 - graph.primaryEntity.scaleTolerance) -> if (actor == DirectorActor.PHOTOGRAPHER) "Come slightly closer" else "Move ${entityNoun(actor)} slightly closer"
                sqrt(entity.area) > sqrt(target.area) * (1.0 + graph.primaryEntity.scaleTolerance) -> if (actor == DirectorActor.PHOTOGRAPHER) "Step back slightly" else "Move ${entityNoun(actor)} slightly farther"
                else -> selected.node.instructionHold.ifBlank { "Fine-tune the framing" }
            }
            return CoachingInstruction(selected.node.id, text, actor, (1.0 - selected.satisfaction).coerceIn(0.2,1.0), actorDx, actorDy)
        }

        val value = selected.measurement.value
        val text = when (selected.node.type) {
            DirectorGoalTypes.SUBJECT_BACKGROUND_EV -> when {
                value == null -> "Hold while Sierra reads the light"
                selected.node.targetMin != null && value < selected.node.targetMin -> selected.node.instructionLow.ifBlank { "Move slightly toward the light" }
                selected.node.targetMax != null && value > selected.node.targetMax -> selected.node.instructionHigh.ifBlank { "Reduce subject-to-background light contrast" }
                else -> selected.node.instructionHold
            }
            DirectorGoalTypes.BACKGROUND_DISTRACTION -> selected.node.instructionHigh.ifBlank { "Shift position to simplify the background" }
            else -> when {
                value == null -> selected.node.instructionHold
                selected.node.targetMin != null && value < selected.node.targetMin -> selected.node.instructionLow
                selected.node.targetMax != null && value > selected.node.targetMax -> selected.node.instructionHigh
                else -> selected.node.instructionHold
            }
        }.ifBlank { "Adjust ${friendlyType(selected.node.type)}" }
        return CoachingInstruction(selected.node.id, text, selected.node.actor, (1.0 - selected.satisfaction).coerceIn(0.2,1.0))
    }

    private fun interventionAllows(graph: SierraDirectorSpec, node: DirectorGoalNode): Boolean = when (graph.interventionPolicy) {
        InterventionPolicy.HIGH_DIRECTION, InterventionPolicy.DIRECTED -> true
        InterventionPolicy.MINIMAL -> node.actor != DirectorActor.SUBJECT && node.type != DirectorGoalTypes.POSE_KEYPOINT_ALIGNMENT
        InterventionPolicy.OBSERVE -> node.actor !in setOf(DirectorActor.SUBJECT, DirectorActor.OBJECT) && node.type != DirectorGoalTypes.POSE_KEYPOINT_ALIGNMENT
    }

    private fun updateStability(nowNs: Long, pass: Boolean, score: Double, policy: DirectorReadinessPolicy, requiredWindowMs: Long): Boolean {
        readinessHistory.addLast(ReadinessSample(nowNs, pass, score))
        val retainNs = policy.maxStabilityMs * 1_000_000L
        while (readinessHistory.isNotEmpty() && nowNs - readinessHistory.first.timestampNs > retainNs) readinessHistory.removeFirst()
        val windowNs = requiredWindowMs * 1_000_000L
        val relevant = readinessHistory.filter { nowNs - it.timestampNs <= windowNs }
        if (relevant.size < MIN_STABILITY_SAMPLES) return false
        val span = relevant.last().timestampNs - relevant.first().timestampNs
        if (span < windowNs * 0.82) return false
        val validFraction = relevant.count { it.pass }.toDouble() / relevant.size
        return relevant.last().pass && validFraction >= policy.validFractionMin
    }

    private fun stableWindowStart(nowNs: Long, requiredWindowMs: Long): Long? {
        if (readinessHistory.isEmpty()) return null
        val cutoff = nowNs - requiredWindowMs * 1_000_000L
        return readinessHistory.firstOrNull { it.timestampNs >= cutoff && it.pass }?.timestampNs
    }

    private fun adaptiveStabilityMs(graph: SierraDirectorSpec, evaluations: List<GoalEvaluation>, motion: Double?): Long {
        val unknownRequired = evaluations.count { (it.node.required || it.node.priority == GoalPriority.HARD) && it.measurement.state == GoalState.UNKNOWN }
        val confidenceValues = evaluations.filter { it.measurement.state != GoalState.UNKNOWN }.map { it.measurement.confidence }
        val confidence = if (confidenceValues.isEmpty()) 0.0 else confidenceValues.average()
        val uncertainty = (1.0 - confidence + unknownRequired * 0.12).coerceIn(0.0, 1.0)
        val motionFactor = (motion ?: 0.12).coerceIn(0.0, 1.0)
        val nodeMinimum = evaluations.filter { it.node.priority != GoalPriority.PREFERENCE }.maxOfOrNull { it.node.stabilityMs } ?: graph.readiness.baseStabilityMs
        val proposed = max(nodeMinimum, graph.readiness.baseStabilityMs + (320.0 * uncertainty + 260.0 * motionFactor).toLong())
        return proposed.coerceIn(graph.readiness.baseStabilityMs, graph.readiness.maxStabilityMs)
    }

    private fun weightedScore(values: List<GoalEvaluation>): Double {
        if (values.isEmpty()) return 1.0
        val denominator = values.sumOf { it.node.weight * it.measurement.confidence }.coerceAtLeast(1e-6)
        return (values.sumOf { it.node.weight * it.measurement.confidence * it.satisfaction } / denominator).coerceIn(0.0, 1.0)
    }

    private fun intervalErrorRatio(value: Double, minTarget: Double?, maxTarget: Double?): Double {
        if (minTarget == null && maxTarget == null) return abs(value)
        if (minTarget != null && maxTarget != null && value in minTarget..maxTarget) return 0.0
        val width = if (minTarget != null && maxTarget != null) max(maxTarget - minTarget, 0.05) else max(abs(minTarget ?: maxTarget ?: 1.0), 0.05)
        return when {
            minTarget != null && value < minTarget -> (minTarget - value) / width
            maxTarget != null && value > maxTarget -> (value - maxTarget) / width
            else -> 0.0
        }
    }

    private fun entityMotion(current: NormalizedRect?, nowNs: Long): Double? {
        val previous = previousEntityBounds
        val previousTime = previousObservedAtNs
        previousEntityBounds = current ?: previousEntityBounds
        previousObservedAtNs = if (current != null) nowNs else previousObservedAtNs
        if (current == null || previous == null || previousTime == null || nowNs <= previousTime) return null
        val dt = (nowNs - previousTime) / 1_000_000_000.0
        return (previous.center.distanceTo(current.center) / max(dt, 1e-3) / 1.2).coerceIn(0.0, 1.0)
    }

    private fun regionMeanLuma(frame: Bitmap, rect: NormalizedRect): Double {
        val samplesX = 12
        val samplesY = 12
        var sum = 0.0
        var count = 0
        for (y in 0 until samplesY) for (x in 0 until samplesX) {
            val nx = rect.left + (x + 0.5) / samplesX * rect.width
            val ny = rect.top + (y + 0.5) / samplesY * rect.height
            val px = (nx * (frame.width - 1)).toInt().coerceIn(0, frame.width - 1)
            val py = (ny * (frame.height - 1)).toInt().coerceIn(0, frame.height - 1)
            val p = frame.getPixel(px, py)
            sum += (((77 * ((p shr 16) and 255) + 150 * ((p shr 8) and 255) + 29 * (p and 255)) shr 8)).toDouble()
            count++
        }
        return if (count == 0) 0.0 else sum / count
    }

    private fun horizontalLightBalance(frame: Bitmap, rect: NormalizedRect): Double {
        val mid = rect.center.x
        if (mid <= rect.left + 1e-6 || mid >= rect.right - 1e-6) return 0.0
        val l = regionMeanLuma(frame, NormalizedRect(rect.left, rect.top, mid, rect.bottom))
        val r = regionMeanLuma(frame, NormalizedRect(mid, rect.top, rect.right, rect.bottom))
        return ((r - l) / max(max(l, r), 24.0)).coerceIn(-1.0, 1.0)
    }

    private fun subjectBackgroundEv(frame: Bitmap, subject: NormalizedRect): Double {
        val subjectLuma = regionMeanLuma(frame, subject).coerceAtLeast(3.0)
        val expanded = NormalizedRect(
            (subject.left - subject.width * 0.45).coerceAtLeast(0.0),
            (subject.top - subject.height * 0.30).coerceAtLeast(0.0),
            (subject.right + subject.width * 0.45).coerceAtMost(1.0),
            (subject.bottom + subject.height * 0.30).coerceAtMost(1.0)
        )
        val outer = regionMeanLuma(frame, expanded).coerceAtLeast(3.0)
        val backgroundApprox = ((outer * expanded.area - subjectLuma * subject.area) / max(expanded.area - subject.area, 1e-4)).coerceAtLeast(3.0)
        return (ln(subjectLuma / backgroundApprox) / ln(2.0)).coerceIn(-3.0, 3.0)
    }

    private fun backgroundDistraction(frame: Bitmap, subject: NormalizedRect): Double {
        val global = regionMeanLuma(frame, NormalizedRect(0.0, 0.0, 1.0, 1.0))
        var weighted = 0.0
        var count = 0
        val grid = 6
        for (gy in 0 until grid) for (gx in 0 until grid) {
            val left = gx.toDouble() / grid
            val top = gy.toDouble() / grid
            val right = (gx + 1.0) / grid
            val bottom = (gy + 1.0) / grid
            val tile = NormalizedRect(left, top, right, bottom)
            if (tile.iou(subject) > 0.08 || tile.center.distanceTo(subject.center) < max(subject.width, subject.height) * 0.45) continue
            val luma = regionMeanLuma(frame, tile)
            val contrast = abs(luma - global) / 255.0
            val edge = tileEdgeEnergy(frame, tile)
            weighted += (0.60 * contrast + 0.40 * edge).coerceIn(0.0, 1.0)
            count++
        }
        return if (count == 0) 0.0 else (weighted / count).coerceIn(0.0, 1.0)
    }

    private fun tileEdgeEnergy(frame: Bitmap, rect: NormalizedRect): Double {
        val n = 8
        var sum = 0.0
        var count = 0
        var previousRow = DoubleArray(n)
        for (y in 0 until n) {
            val row = DoubleArray(n)
            for (x in 0 until n) {
                val nx = rect.left + (x + 0.5) / n * rect.width
                val ny = rect.top + (y + 0.5) / n * rect.height
                val px = (nx * (frame.width - 1)).toInt().coerceIn(0, frame.width - 1)
                val py = (ny * (frame.height - 1)).toInt().coerceIn(0, frame.height - 1)
                val p = frame.getPixel(px, py)
                row[x] = ((77 * ((p shr 16) and 255) + 150 * ((p shr 8) and 255) + 29 * (p and 255)) shr 8).toDouble()
                if (x > 0) { sum += abs(row[x] - row[x - 1]); count++ }
                if (y > 0) { sum += abs(row[x] - previousRow[x]); count++ }
            }
            previousRow = row
        }
        return if (count == 0) 0.0 else (sum / count / 96.0).coerceIn(0.0, 1.0)
    }

    private fun boxVisibility(rect: NormalizedRect): Double {
        val safe = 0.018
        val loss = listOf(rect.left, rect.top, 1.0 - rect.right, 1.0 - rect.bottom).sumOf { if (it >= safe) 0.0 else (safe - it) / safe }
        return (1.0 - loss * 0.18).coerceIn(0.0,1.0)
    }

    private fun blendRects(a: NormalizedRect, b: NormalizedRect, wa: Double, wb: Double): NormalizedRect {
        val total = wa + wb
        return NormalizedRect((a.left*wa+b.left*wb)/total, (a.top*wa+b.top*wb)/total, (a.right*wa+b.right*wb)/total, (a.bottom*wa+b.bottom*wb)/total).clamped()
    }

    private fun isHumanKind(kind: String): Boolean = kind.lowercase() in setOf("person","human","portrait","people","child","adult")
    private fun friendlyType(type: String): String = type.replace('_',' ')
    private fun entityNoun(actor: DirectorActor): String = when (actor) { DirectorActor.SUBJECT -> "the subject"; DirectorActor.OBJECT -> "the object"; else -> "it" }
    private fun movementPhrase(actor: DirectorActor, direction: String): String = when (actor) {
        DirectorActor.PHOTOGRAPHER -> "Move $direction"
        DirectorActor.SUBJECT -> "Ask the subject to move $direction"
        DirectorActor.OBJECT -> "Move the object $direction"
        DirectorActor.ANY -> "Shift $direction"
        DirectorActor.NONE -> "Adjust $direction"
    }

    companion object {
        private const val MIN_ENTITY_LOCK_CONFIDENCE = 0.42
        private const val OVERLAY_MIN_CONFIDENCE = 0.28
        private const val MIN_STABILITY_SAMPLES = 3
    }
}
