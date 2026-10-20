package com.abacus.checker

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Matrix
import android.graphics.RectF
import android.util.AttributeSet
import android.view.GestureDetector
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import androidx.appcompat.widget.AppCompatImageView

/** Image that can be pinched to zoom, dragged when zoomed, and double-tapped to zoom in / back out. */
class ZoomImageView @JvmOverloads constructor(context: Context, attrs: AttributeSet? = null) :
    AppCompatImageView(context, attrs) {

    private val m = Matrix()
    private var fitScale = 1f
    private val maxZoom = 6f

    private val scaler = ScaleGestureDetector(context, object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
        override fun onScale(d: ScaleGestureDetector): Boolean {
            zoomBy(d.scaleFactor, d.focusX, d.focusY)
            return true
        }
    })

    private val gestures = GestureDetector(context, object : GestureDetector.SimpleOnGestureListener() {
        override fun onScroll(e1: MotionEvent?, e2: MotionEvent, dx: Float, dy: Float): Boolean {
            m.postTranslate(-dx, -dy)
            clampAndApply()
            return true
        }

        override fun onDoubleTap(e: MotionEvent): Boolean {
            if (currentScale() > fitScale * 1.05f) fit() else zoomBy(2.5f, e.x, e.y)
            return true
        }
    })

    init {
        scaleType = ScaleType.MATRIX
    }

    private fun currentScale(): Float = FloatArray(9).also { m.getValues(it) }[Matrix.MSCALE_X]

    /** Whole image visible, as wide as the view, centred. */
    private fun fit() {
        val d = drawable ?: return
        if (width == 0 || height == 0) return
        val w = d.intrinsicWidth.toFloat()
        val h = d.intrinsicHeight.toFloat()
        fitScale = minOf(width / w, height / h)
        m.setScale(fitScale, fitScale)
        m.postTranslate((width - w * fitScale) / 2, (height - h * fitScale) / 2)
        imageMatrix = m
    }

    private fun zoomBy(factor: Float, fx: Float, fy: Float) {
        val target = (currentScale() * factor).coerceIn(fitScale, fitScale * maxZoom)
        val f = target / currentScale()
        m.postScale(f, f, fx, fy)
        clampAndApply()
    }

    /** Don't let the image drift away: centre it when smaller than the view, otherwise keep edges in view. */
    private fun clampAndApply() {
        val d = drawable ?: return
        val r = RectF(0f, 0f, d.intrinsicWidth.toFloat(), d.intrinsicHeight.toFloat())
        m.mapRect(r)
        val dx = when {
            r.width() <= width -> (width - r.width()) / 2 - r.left
            r.left > 0 -> -r.left
            r.right < width -> width - r.right
            else -> 0f
        }
        val dy = when {
            r.height() <= height -> (height - r.height()) / 2 - r.top
            r.top > 0 -> -r.top
            r.bottom < height -> height - r.bottom
            else -> 0f
        }
        m.postTranslate(dx, dy)
        imageMatrix = m
    }

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)
        fit()
    }

    override fun setImageBitmap(bm: android.graphics.Bitmap?) {
        super.setImageBitmap(bm)
        post { fit() }
    }

    @SuppressLint("ClickableViewAccessibility")
    override fun onTouchEvent(event: MotionEvent): Boolean {
        scaler.onTouchEvent(event)
        if (!scaler.isInProgress) gestures.onTouchEvent(event)
        parent?.requestDisallowInterceptTouchEvent(currentScale() > fitScale * 1.01f)
        return true
    }
}
