package com.saunak.director.camera

import android.graphics.ImageFormat
import androidx.camera.core.ImageProxy
import com.saunak.director.engine.reconstruction.CfaBlackLevel
import com.saunak.director.engine.reconstruction.StoredRawFrame
import com.saunak.director.engine.reconstruction.StoredRawWriter
import java.io.File
import java.io.RandomAccessFile

data class RawCaptureTarget(
    val rawFile: File,
    val jpegFile: File
) {
    init {
        require(rawFile.absolutePath != jpegFile.absolutePath) { "RAW and JPEG targets must be different files" }
    }

    fun prepare() {
        rawFile.parentFile?.mkdirs()
        jpegFile.parentFile?.mkdirs()
        require(rawFile.parentFile?.isDirectory != false) { "RAW target directory unavailable" }
        require(jpegFile.parentFile?.isDirectory != false) { "JPEG target directory unavailable" }
        rawFile.delete()
        jpegFile.delete()
    }

    fun cleanup() {
        runCatching { rawFile.delete() }
        runCatching { jpegFile.delete() }
    }
}

data class StoredRawJpegCaptureFrame(
    val raw: StoredRawFrame,
    val jpegFile: File,
    val jpegRotationDegrees: Int,
    val metadata: RawCaptureMetadata
) {
    init {
        require(jpegFile.isFile && jpegFile.length() > 0L)
        require(jpegRotationDegrees in setOf(0, 90, 180, 270))
        require(raw.timestampNs == metadata.telemetry.sensorTimestampNs)
        require(raw.metadata?.telemetry?.sensorTimestampNs == raw.timestampNs)
    }
}

class StoredRawBurstCaptureException(
    val completed: List<StoredRawJpegCaptureFrame>,
    cause: Throwable
) : IllegalStateException(
    "RAW burst stopped after ${completed.size} completed frame(s): ${cause.message ?: cause::class.java.simpleName}",
    cause
)

/**
 * Copies camera callback payloads directly to disk. No full-resolution RAW FloatArray or JPEG byte
 * array survives the callback. This is the production capture sink used before tiled reconstruction.
 */
object RawCaptureFileSink {
    fun writeRaw(
        image: ImageProxy,
        calibration: RawSensorCalibration,
        target: RawCaptureTarget
    ): StoredRawFrame {
        require(image.format == ImageFormat.RAW_SENSOR) { "Expected RAW_SENSOR callback, got ${image.format}" }
        return StoredRawWriter.write(
            image = image,
            calibration = calibration,
            output = target.rawFile,
            metadata = null
        )
    }

    fun writeJpeg(image: ImageProxy, target: RawCaptureTarget): Int {
        require(image.format == ImageFormat.JPEG) { "Expected JPEG callback, got ${image.format}" }
        require(image.planes.isNotEmpty())
        target.jpegFile.parentFile?.mkdirs()
        val buffer = image.planes[0].buffer.duplicate()
        require(buffer.hasRemaining()) { "Camera returned empty JPEG payload" }
        RandomAccessFile(target.jpegFile, "rw").use { raf ->
            raf.setLength(0L)
            val channel = raf.channel
            while (buffer.hasRemaining()) {
                check(channel.write(buffer) > 0) { "Failed to persist JPEG safety frame" }
            }
            channel.force(false)
        }
        check(target.jpegFile.isFile && target.jpegFile.length() > 0L) { "JPEG safety frame was not persisted" }
        return image.imageInfo.rotationDegrees
    }

    fun attachMetadata(frame: StoredRawFrame, metadata: RawCaptureMetadata): StoredRawFrame {
        require(frame.timestampNs == metadata.telemetry.sensorTimestampNs) {
            "RAW metadata timestamp does not match the sensor frame"
        }
        val dynamicBlack = metadata.dynamicBlackLevel
            ?.takeIf { it.size == 4 && it.all { value -> value.isFinite() && value >= 0f } }
            ?.let { CfaBlackLevel(it.copyOf()) }
            ?: frame.blackLevelPattern
        return frame.copy(
            blackLevelPattern = dynamicBlack,
            metadata = metadata
        )
    }
}
