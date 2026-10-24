package com.saunak.director

import com.saunak.director.engine.quality.QualityAxis
import com.saunak.director.engine.quality.QualityCouncil
import com.saunak.director.engine.quality.QualityDisposition
import com.saunak.director.engine.quality.QualitySignal
import com.saunak.director.engine.quality.RepairStage
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class QualityCouncilTest {
    @Test fun unsupportedEvidenceRejectsImage() {
        val verdict = QualityCouncil().evaluate(listOf(
            QualitySignal(QualityAxis.EVIDENCE, 0.18, 0.95, "evidence-ledger", issue = "face texture unsupported"),
            QualitySignal(QualityAxis.TECHNICAL, 0.9, 0.9, "technical")
        ))
        assertEquals(QualityDisposition.REJECT, verdict.disposition)
        assertTrue(verdict.criticalReasons.isNotEmpty())
    }

    @Test fun localNoiseFailureProducesTargetedDenoiseRepair() {
        val verdict = QualityCouncil().evaluate(listOf(
            QualitySignal(QualityAxis.TECHNICAL, 0.31, 0.9, "noise-estimator", issue = "shadow noise"),
            QualitySignal(QualityAxis.EVIDENCE, 0.9, 0.95, "evidence-ledger"),
            QualitySignal(QualityAxis.PERCEPTUAL, 0.8, 0.8, "iqa")
        ))
        assertEquals(QualityDisposition.LOCAL_REPAIR, verdict.disposition)
        assertEquals(RepairStage.DENOISE, verdict.repairs.first().stage)
    }
}
