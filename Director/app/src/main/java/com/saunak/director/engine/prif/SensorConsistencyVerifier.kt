package com.saunak.director.engine.prif

import com.saunak.director.engine.reconstruction.RawImageFormation
import com.saunak.director.engine.reconstruction.RawObservation
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.sqrt

object SensorConsistencyVerifier {
    fun verify(posterior: RadiancePosterior, observations: List<RawObservation>, acceptanceThreshold: Double = 0.62): SensorConsistencyReport {
        require(observations.isNotEmpty())
        val referenceEnergy = observations.first().exposureTimeNs.toDouble() * observations.first().sensitivityIso.toDouble()
        val frameReports = observations.mapIndexed { frameIndex, obs ->
            val actual = RawImageFormation.normalize(obs.raw); val energy = obs.exposureTimeNs.toDouble()*obs.sensitivityIso.toDouble(); val forwardScale=(energy/referenceEnergy).coerceIn(.0625,16.0)
            var squared=0.0;var inliers=0;var compared=0;var saturated=0;var i=0
            val stride=4
            while(i<actual.size){val predicted=(posterior.mean[i].toDouble()*forwardScale).coerceIn(0.0,1.0);val measured=actual[i].toDouble();if(measured>=.995)saturated++;val variance=max(obs.noise.variance(measured.toFloat()).toDouble(),1e-8);val z=abs(measured-predicted)/sqrt(variance);squared+=z*z;if(z<=3.0)inliers++;compared++;i+=stride}
            val rmse=if(compared>0)sqrt(squared/compared) else Double.POSITIVE_INFINITY;val inlier=if(compared>0)inliers.toDouble()/compared else 0.0;val sat=if(compared>0)saturated.toDouble()/compared else 1.0;val score=(.62*exp(-.18*rmse*rmse)+.30*inlier+.08*(1.0-sat)).coerceIn(0.0,1.0)
            FrameConsistency(frameIndex,rmse,inlier,sat,score)
        }
        val aggregate=frameReports.map{it.score}.average().coerceIn(0.0,1.0);return SensorConsistencyReport(frameReports,aggregate,aggregate>=acceptanceThreshold)
    }
}
object PhotoIntegrityEvaluator {
    fun evaluate(posterior: RadiancePosterior, consistency: SensorConsistencyReport, uncertaintyThreshold: Double=.02, acceptanceThreshold: Double=.58): PhotoIntegrityReport {
        var multi=0;var single=0;var uncertain=0;val n=posterior.mean.size.coerceAtLeast(1)
        for(i in posterior.mean.indices){val count=posterior.supportCount[i].toInt() and 0xff;if(count>=2)multi++ else single++;if(posterior.sensorVariance[i]+posterior.reconstructionVariance[i]>uncertaintyThreshold)uncertain++}
        val p=ProvenanceSummary(posterior.meanSupport.coerceIn(0.0,1.0),multi.toDouble()/n,uncertain.toDouble()/n,single.toDouble()/n,0.0);val penalty=(p.uncertainFraction*.72+posterior.meanUncertainty.coerceIn(0.0,1.0)*.28).coerceIn(0.0,1.0);val score=(.46*consistency.aggregateScore+.40*p.averagePhysicalSupport+.14*p.multiFrameFraction-.22*penalty).coerceIn(0.0,1.0);return PhotoIntegrityReport(score,consistency.aggregateScore,p.averagePhysicalSupport,penalty,p,consistency.accepted&&score>=acceptanceThreshold)
    }
}
