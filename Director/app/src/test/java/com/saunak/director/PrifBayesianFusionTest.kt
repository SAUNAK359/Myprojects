package com.saunak.director

import com.saunak.director.engine.prif.PrifBayesianFusion
import com.saunak.director.engine.prif.SensorConsistencyVerifier
import com.saunak.director.engine.reconstruction.BayerPattern
import com.saunak.director.engine.reconstruction.NoiseModel
import com.saunak.director.engine.reconstruction.RawObservation
import com.saunak.director.engine.reconstruction.RawPlane
import org.junit.Assert.assertTrue
import org.junit.Test

class PrifBayesianFusionTest {
    private fun observation(value: Float, exposureNs: Long = 10_000_000L): RawObservation {
        val raw = RawPlane(
            width = 8,
            height = 8,
            values = FloatArray(64) { value },
            pattern = BayerPattern.RGGB,
            blackLevel = 0f,
            whiteLevel = 1f
        )
        return RawObservation(
            raw = raw,
            exposureTimeNs = exposureNs,
            sensitivityIso = 100,
            noise = NoiseModel(0.002f, 0.0001f)
        )
    }

    @Test fun consistentBurstReducesPosteriorSensorVariance() {
        val single = PrifBayesianFusion.fuse(listOf(observation(0.40f)), 4.0, 4.0, false)
        val burst = PrifBayesianFusion.fuse(
            listOf(observation(0.40f), observation(0.402f), observation(0.398f)),
            4.0, 4.0, false
        )
        assertTrue(burst.sensorVariance.average() < single.sensorVariance.average())
        assertTrue(burst.meanSupport > single.meanSupport)
    }

    @Test fun largeOutlierDoesNotPullRadianceFarFromAnchor() {
        val posterior = PrifBayesianFusion.fuse(
            listOf(observation(0.40f), observation(0.405f), observation(0.95f)),
            3.0, 3.0, false
        )
        assertTrue(posterior.mean.average() in 0.38..0.44)
    }

    @Test fun forwardConsistencyRewardsSensorSupportedRadiance() {
        val observations = listOf(observation(0.35f), observation(0.352f), observation(0.348f))
        val posterior = PrifBayesianFusion.fuse(observations, 4.0, 4.0, false)
        val report = SensorConsistencyVerifier.verify(posterior, observations, acceptanceThreshold = 0.45)
        assertTrue(report.accepted)
        assertTrue(report.aggregateScore > 0.45)
    }
}
