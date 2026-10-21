package com.saunak.director.director

import android.graphics.Bitmap
import com.saunak.director.ai.VlmRuntime
import com.saunak.director.camera.CameraEngine
import com.saunak.director.platform.ThermalMonitor
import com.saunak.director.processing.Look
import com.saunak.director.processing.LookProcessor
import com.saunak.director.skills.SkillEngine
import com.saunak.director.storage.PhotoRepository
import com.saunak.director.tools.ToolExecutor
import kotlinx.coroutines.CancellationException
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

class DirectorOrchestrator(private val camera: CameraEngine, private val vlm: VlmRuntime, private val skills: SkillEngine, private val tools: ToolExecutor, private val storage: PhotoRepository, private val thermal: ThermalMonitor, private val onState: (DirectorUiState) -> Unit) {
    private val busy=AtomicBoolean(false); private val controlEpoch=AtomicLong(0L); private val sampler=FrameSampler(); private val sceneMonitor=SceneChangeMonitor(); private val acceptanceThreshold=0.84; private val lastCompletedSceneVersion=AtomicLong(-1L)
    fun isBusy():Boolean=busy.get()
    fun observeScene(bitmap:Bitmap,timestampNanos:Long){if(sceneMonitor.observe(bitmap,timestampNanos)&&busy.get())invalidateAutonomy()}
    fun invalidateAutonomy(){controlEpoch.incrementAndGet();vlm.cancel()}
    suspend fun onPreviewFrame(bitmap:Bitmap,timestampNanos:Long){
        val frame=sampler.accept(bitmap,timestampNanos,busy.get())?:return;val currentSceneVersion=sceneMonitor.currentVersion();if(currentSceneVersion==lastCompletedSceneVersion.get()){frame.recycle();return}
        val thermalPolicy=thermal.policy();if(!camera.isAnalysisAvailable()||!vlm.isReady()||!thermalPolicy.allowAi){frame.recycle();return};vlm.setMaxImageDimension(thermalPolicy.maxAiDimension);if(!busy.compareAndSet(false,true)){frame.recycle();return}
        val epoch=controlEpoch.get();val sceneVersion=sceneMonitor.currentVersion()
        try{direct(frame,epoch,sceneVersion)}catch(_:StaleDirectorWork){}catch(_:CancellationException){}catch(t:Throwable){onState(DirectorUiState(DirectorStage.ERROR,true,"Director skipped this shot",error=t.message))}finally{if(!frame.isRecycled)frame.recycle();busy.set(false);if(vlm.isReady())onState(DirectorUiState(DirectorStage.OBSERVING,true,"Director ready"))}
    }
    private suspend fun direct(frame:Bitmap,epoch:Long,sceneVersion:Long){
        assertFresh(epoch,sceneVersion);onState(DirectorUiState(DirectorStage.PLANNING,true,"Optimizing shot…"));val caps=camera.capabilities();val initialState=camera.snapshot()
        val plan=vlm.plan(frame,caps,initialState,skills.planningContext()).getOrElse{onState(DirectorUiState(DirectorStage.ERROR,true,"Director skipped this frame",error=it.message));return};assertFresh(epoch,sceneVersion)
        onState(DirectorUiState(DirectorStage.APPLYING,true,"Setting the shot…"));var look=tools.apply(plan.toolCalls,caps).look;assertFresh(epoch,sceneVersion);if(!plan.capture)return
        val attempts=mutableListOf<Attempt>();val tempFiles=mutableListOf<File>();var issues=emptyList<String>();val maxAttempts=if(thermal.policy().allowRetry)3 else 1
        try{for(attemptIndex in 1..maxAttempts){assertFresh(epoch,sceneVersion);if(!thermal.policy().allowAi)break;camera.awaitSettledState();assertFresh(epoch,sceneVersion);onState(DirectorUiState(DirectorStage.CAPTURING,true,"Capturing…",attempt=attemptIndex));val file=camera.captureTemp();tempFiles+=file;assertFresh(epoch,sceneVersion)
            val decoded=storage.decode(file);val rendered=LookProcessor.apply(decoded,look)
            try{onState(DirectorUiState(DirectorStage.CHECKING,true,"Checking…",attempt=attemptIndex));val eval=vlm.evaluate(rendered,plan.objective,attemptIndex,camera.snapshot(),issues,skills.evaluationContext()).getOrElse{EvaluationOutput(0.0,accept=false,issues=listOf("evaluation_failed"))};assertFresh(epoch,sceneVersion);val technicallyValid=rendered.width>0&&rendered.height>0&&file.length()>16_384;attempts+=Attempt(file,eval.score.coerceIn(0.0,1.0),technicallyValid,look);if(technicallyValid&&eval.accept&&eval.score>=acceptanceThreshold){saveFinal(file,look,attempts,sceneVersion);return};issues=eval.issues.take(8);if(attemptIndex<maxAttempts&&thermal.policy().allowRetry){onState(DirectorUiState(DirectorStage.CORRECTING,true,"Refining…",attempt=attemptIndex,lastScore=eval.score));look=tools.apply(eval.deltaCalls,caps,look).look}}
            finally{if(rendered!==decoded&&!rendered.isRecycled)rendered.recycle();if(!decoded.isRecycled)decoded.recycle()}}
            assertFresh(epoch,sceneVersion);val best=attempts.filter{it.valid}.maxByOrNull{it.score}?:attempts.maxByOrNull{it.score};if(best!=null)saveFinal(best.file,best.look,attempts,sceneVersion)
        }finally{storage.cleanup(tempFiles)}
    }
    private suspend fun saveFinal(file:File,look:Look,attempts:List<Attempt>,sceneVersion:Long){onState(DirectorUiState(DirectorStage.SAVING,true,"Saving…"));val source=storage.decode(file);val finalBitmap=LookProcessor.apply(source,look);var saved=false;try{storage.saveProcessed(finalBitmap);saved=true;lastCompletedSceneVersion.set(sceneVersion)}finally{if(finalBitmap!==source&&!finalBitmap.isRecycled)finalBitmap.recycle();if(!source.isRecycled)source.recycle();if(saved)storage.cleanup(attempts.map{it.file})}}
    private fun assertFresh(epoch:Long,sceneVersion:Long){if(controlEpoch.get()!=epoch||sceneMonitor.currentVersion()!=sceneVersion)throw StaleDirectorWork()}
    fun cancel()=invalidateAutonomy()
    private class StaleDirectorWork:RuntimeException()
    private data class Attempt(val file:File,val score:Double,val valid:Boolean,val look:Look)
}
