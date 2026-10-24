package com.saunak.director.storage

import android.graphics.Bitmap
import android.net.Uri
import com.saunak.director.processing.LookProcessor
import com.saunak.director.processing.LookSpec
import java.io.File
import kotlin.math.abs
import kotlin.math.roundToInt

suspend fun PhotoRepository.savePrifReconstruction(
    bitmap: Bitmap,
    sourceJpeg: File,
    look: LookSpec,
    aspectRatio: String,
    cropCenterX: Double,
    cropCenterY: Double
): Uri {
    require(!bitmap.isRecycled)
    val ratio = SierraCropPolicy.ratioOrNull(aspectRatio)
    val cropped = if (ratio == null) bitmap else cropPrifBitmap(bitmap, ratio.first, ratio.second, cropCenterX, cropCenterY)
    val rendered = LookProcessor.apply(cropped, look)
    return try { saveProcessed(rendered, sourceJpeg) } finally {
        if (rendered !== cropped && !rendered.isRecycled) rendered.recycle()
        if (cropped !== bitmap && !cropped.isRecycled) cropped.recycle()
    }
}

private fun cropPrifBitmap(source: Bitmap, ratioWidth: Int, ratioHeight: Int, centerX: Double, centerY: Double): Bitmap {
    val target = ratioWidth.toDouble() / ratioHeight.toDouble(); val current = source.width.toDouble() / source.height.toDouble()
    if (abs(target-current) < .015) return source
    val cropWidth:Int; val cropHeight:Int
    if(current>target){cropHeight=source.height;cropWidth=(cropHeight*target).roundToInt().coerceIn(1,source.width)}else{cropWidth=source.width;cropHeight=(cropWidth/target).roundToInt().coerceIn(1,source.height)}
    val left=(centerX.coerceIn(0.0,1.0)*source.width-cropWidth/2.0).roundToInt().coerceIn(0,source.width-cropWidth);val top=(centerY.coerceIn(0.0,1.0)*source.height-cropHeight/2.0).roundToInt().coerceIn(0,source.height-cropHeight)
    return Bitmap.createBitmap(source,left,top,cropWidth,cropHeight)
}
