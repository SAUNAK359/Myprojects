package com.saunak.director.engine.reconstruction

import android.graphics.ImageFormat
import androidx.camera.core.ImageProxy
import com.saunak.director.camera.RawCaptureMetadata
import com.saunak.director.camera.RawSensorCalibration
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

data class StoredRawFrame(
    val file: File,
    val width: Int,
    val height: Int,
    val pattern: BayerPattern,
    val blackLevelPattern: CfaBlackLevel,
    val whiteLevel: Float,
    val timestampNs: Long,
    val rotationDegrees: Int,
    val metadata: RawCaptureMetadata?
) {
    init {
        require(file.isFile && file.length() == width.toLong() * height.toLong() * 2L)
        require(width > 1 && height > 1)
        require(whiteLevel > blackLevelPattern.offsets.max())
        require(timestampNs > 0)
        require(rotationDegrees in setOf(0, 90, 180, 270))
    }

    fun openReader(): StoredRawReader = StoredRawReader(this)
}

/** Writes RAW_SENSOR into a canonical little-endian, padding-free unsigned-16 file. */
object StoredRawWriter {
    fun write(
        image: ImageProxy,
        calibration: RawSensorCalibration,
        output: File,
        metadata: RawCaptureMetadata? = null
    ): StoredRawFrame {
        require(image.format == ImageFormat.RAW_SENSOR)
        require(image.planes.size == 1)
        val plane = image.planes[0]
        require(plane.pixelStride >= 2)
        require(plane.rowStride >= image.width * plane.pixelStride)
        output.parentFile?.mkdirs()
        val dynamicBlack = metadata?.dynamicBlackLevel
            ?.takeIf { it.size == 4 && it.all(Float::isFinite) }
            ?.let { CfaBlackLevel(it.copyOf()) }
            ?: calibration.staticBlackLevel

        val input = plane.buffer.duplicate().order(ByteOrder.nativeOrder())
        RandomAccessFile(output, "rw").use { raf ->
            raf.setLength(image.width.toLong() * image.height.toLong() * 2L)
            val channel = raf.channel
            val row = ByteBuffer.allocateDirect(image.width * 2).order(ByteOrder.LITTLE_ENDIAN)
            for (y in 0 until image.height) {
                row.clear()
                val rowBase = y * plane.rowStride
                for (x in 0 until image.width) {
                    val offset = rowBase + x * plane.pixelStride
                    require(offset + 1 < input.limit())
                    row.putShort(input.getShort(offset))
                }
                row.flip()
                while (row.hasRemaining()) channel.write(row)
            }
            channel.force(false)
        }
        return StoredRawFrame(
            file = output,
            width = image.width,
            height = image.height,
            pattern = calibration.pattern,
            blackLevelPattern = dynamicBlack,
            whiteLevel = calibration.whiteLevel,
            timestampNs = image.imageInfo.timestamp,
            rotationDegrees = image.imageInfo.rotationDegrees,
            metadata = metadata
        )
    }
}

/** Random-access tile reader; tile origins are kept on the 2x2 CFA lattice. */
class StoredRawReader internal constructor(
    private val frame: StoredRawFrame
) : AutoCloseable {
    private val raf = RandomAccessFile(frame.file, "r")
    private val channel: FileChannel = raf.channel

    fun readTile(x: Int, y: Int, width: Int, height: Int): RawPlane {
        require(x >= 0 && y >= 0 && width > 1 && height > 1)
        require((x and 1) == 0 && (y and 1) == 0) { "Tile origin must preserve CFA phase" }
        require(x + width <= frame.width && y + height <= frame.height)
        val values = FloatArray(width * height)
        val rowBytes = width * 2
        val rowBuffer = ByteBuffer.allocateDirect(rowBytes).order(ByteOrder.LITTLE_ENDIAN)
        for (row in 0 until height) {
            rowBuffer.clear()
            var fileOffset = ((y + row).toLong() * frame.width.toLong() + x.toLong()) * 2L
            var read = 0
            while (read < rowBytes) {
                val count = channel.read(rowBuffer, fileOffset + read)
                check(count > 0) { "Unexpected EOF in stored RAW frame" }
                read += count
            }
            rowBuffer.flip()
            val base = row * width
            for (col in 0 until width) values[base + col] = (rowBuffer.short.toInt() and 0xffff).toFloat()
        }
        return RawPlane(
            width = width,
            height = height,
            values = values,
            pattern = frame.pattern,
            blackLevel = frame.blackLevelPattern.offsets.min(),
            whiteLevel = frame.whiteLevel,
            blackLevelPattern = frame.blackLevelPattern
        )
    }

    /** Sparse same-CFA samples for low-memory global alignment. */
    fun sampleGrid(step: Int = 16): SparseCfaGrid {
        require(step >= 2 && step % 2 == 0)
        val xs = ((frame.width - 1) / step) + 1
        val ys = ((frame.height - 1) / step) + 1
        val values = FloatArray(xs * ys)
        val sample = ByteBuffer.allocateDirect(2).order(ByteOrder.LITTLE_ENDIAN)
        var index = 0
        for (gy in 0 until ys) {
            val py = (gy * step).coerceAtMost(frame.height - 1) and -2
            for (gx in 0 until xs) {
                val px = (gx * step).coerceAtMost(frame.width - 1) and -2
                sample.clear()
                val offset = (py.toLong() * frame.width + px.toLong()) * 2L
                check(channel.read(sample, offset) == 2)
                sample.flip()
                val raw = (sample.short.toInt() and 0xffff).toFloat()
                val black = frame.blackLevelPattern[px, py]
                values[index++] = ((raw - black) / (frame.whiteLevel - black)).coerceIn(0f, 1f)
            }
        }
        return SparseCfaGrid(xs, ys, step, values)
    }

    override fun close() {
        channel.close()
        raf.close()
    }
}

data class SparseCfaGrid(
    val width: Int,
    val height: Int,
    val pixelStep: Int,
    val values: FloatArray
) {
    init {
        require(width > 0 && height > 0)
        require(pixelStep >= 2 && pixelStep % 2 == 0)
        require(values.size == width * height)
    }
}
