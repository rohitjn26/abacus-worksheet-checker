package com.abacus.checker

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.content.res.ColorStateList
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import java.io.File
import java.util.concurrent.Executors

/** Camera screen: live preview, a hint about the picture, and the shutter button. */
class MainActivity : AppCompatActivity() {
    private lateinit var preview: PreviewView
    private lateinit var hint: TextView
    private lateinit var shutter: Button
    private var capture: ImageCapture? = null
    private val analysisExecutor = Executors.newSingleThreadExecutor()

    private val askCamera = registerForActivityResult(ActivityResultContracts.RequestPermission()) { ok ->
        if (ok) startCamera() else hint.text = getString(R.string.need_camera)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        preview = findViewById(R.id.preview)
        hint = findViewById(R.id.hint)
        shutter = findViewById(R.id.shutter)
        shutter.setOnClickListener { takePhoto() }

        OnnxBridge.init(this)
        if (!Python.isStarted()) Python.start(AndroidPlatform(this))
        // Debug builds only: check a photo file passed in by a test (adb) instead of using the camera.
        val testPhoto = intent.getStringExtra("photo")
        if (BuildConfig.DEBUG && testPhoto != null) {
            startActivity(Intent(this, ResultActivity::class.java).putExtra("photo", testPhoto)
                .putExtra("profile", intent.getBooleanExtra("profile", false)))
        }
        // Load the checker and its models in the background, so the first photo isn't slow.
        Thread { runCatching { Python.getInstance().getModule("android_api").callAttr("warm_up") } }.start()

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            askCamera.launch(Manifest.permission.CAMERA)
        }
    }

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val provider = providerFuture.get()
            val previewUseCase = Preview.Builder().build().also { it.surfaceProvider = preview.surfaceProvider }
            capture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
                .build()
            @Suppress("DEPRECATION")
            val analysis = ImageAnalysis.Builder()
                .setTargetResolution(android.util.Size(640, 480))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also { it.setAnalyzer(analysisExecutor, LiveQuality { h -> runOnUiThread { showHint(h) } }) }
            provider.unbindAll()
            provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, previewUseCase, capture, analysis)
        }, ContextCompat.getMainExecutor(this))
    }

    private fun showHint(h: LiveHint) {
        hint.text = h.message
        val color = ContextCompat.getColor(this, if (h.good) R.color.good else R.color.warn)
        hint.setTextColor(color)
        shutter.backgroundTintList = ColorStateList.valueOf(color)
    }

    private fun takePhoto() {
        val cap = capture ?: return
        val file = File(cacheDir, "sheet_${System.currentTimeMillis()}.jpg")
        shutter.isEnabled = false
        cap.takePicture(
            ImageCapture.OutputFileOptions.Builder(file).build(),
            ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageSavedCallback {
                override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                    shutter.isEnabled = true
                    startActivity(Intent(this@MainActivity, ResultActivity::class.java).putExtra("photo", file.path))
                }

                override fun onError(e: ImageCaptureException) {
                    shutter.isEnabled = true
                    Toast.makeText(this@MainActivity, getString(R.string.photo_failed), Toast.LENGTH_LONG).show()
                }
            },
        )
    }

    override fun onDestroy() {
        super.onDestroy()
        analysisExecutor.shutdown()
    }
}
