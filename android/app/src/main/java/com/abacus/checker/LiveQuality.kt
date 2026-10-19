package com.abacus.checker

import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy

/** What the camera preview looks like right now; drives the hint under the preview. */
enum class LiveHint(val message: String, val good: Boolean) {
    DARK("Too dark – move to a brighter place", false),
    GLARE("Light is reflecting on the page – tilt the phone", false),
    BLURRY("Hold the phone steady", false),
    GOOD("Looks good – fit the whole table in the frame and tap", true),
}

/**
 * Cheap checks on each preview frame (about 640 px wide), using the brightness channel only.
 * Thresholds come from the sample photos at preview size (prototype/degrade_test.py data):
 * good photos have brightness 141+, sharpness 616+, no blown-out pixels; clearly too-dark ones are
 * 35–47, badly blurred ones 84–444, strong glare 2% blown out. These are hints only - the full
 * photo-quality check runs after the photo is taken.
 */
class LiveQuality(private val onHint: (LiveHint) -> Unit) : ImageAnalysis.Analyzer {
    private var last: LiveHint? = null
    private var smoothSharp = 0.0

    override fun analyze(image: ImageProxy) {
        image.use {
            val plane = it.planes[0]
            val w = it.width
            val h = it.height
            val stride = plane.rowStride
            val buf = plane.buffer
            val y = ByteArray(buf.remaining()).also { b -> buf.get(b) }
            val step = maxOf(1, w / 640)  // work at about 640 px width
            val hist = IntArray(256)
            var blown = 0
            var n = 0
            var lapSum = 0.0
            var lapSq = 0.0
            var lapN = 0
            var row = step
            while (row < h - step) {
                var col = step
                while (col < w - step) {
                    val c = y[row * stride + col].toInt() and 0xFF
                    hist[c]++
                    if (c >= 250) blown++
                    n++
                    val lap = (y[(row - step) * stride + col].toInt() and 0xFF) +
                        (y[(row + step) * stride + col].toInt() and 0xFF) +
                        (y[row * stride + col - step].toInt() and 0xFF) +
                        (y[row * stride + col + step].toInt() and 0xFF) - 4 * c
                    lapSum += lap
                    lapSq += lap.toDouble() * lap
                    lapN++
                    col += step
                }
                row += step
            }
            if (n == 0) return
            var acc = 0
            var median = 0
            for (v in 0..255) {
                acc += hist[v]
                if (acc >= n / 2) { median = v; break }
            }
            val mean = lapSum / lapN
            val sharp = lapSq / lapN - mean * mean
            smoothSharp = if (smoothSharp == 0.0) sharp else 0.7 * smoothSharp + 0.3 * sharp  // calm the flicker
            val hint = when {
                median < 50 -> LiveHint.DARK
                blown.toDouble() / n > 0.01 -> LiveHint.GLARE  // lenient: a white page in bright light is not glare
                smoothSharp < 150 -> LiveHint.BLURRY
                else -> LiveHint.GOOD
            }
            if (hint != last) {
                last = hint
                onHint(hint)
            }
        }
    }
}
