package com.abacus.checker

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.content.res.ColorStateList
import android.net.Uri
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.PickVisualMediaRequest
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
import java.io.File
import java.util.concurrent.Executors

/**
 * Camera screen: live preview with a hint about the picture, the shutter, a gallery button and the
 * results list. Every photo is checked in the background, so the parent can take the next one at once.
 */
class MainActivity : AppCompatActivity() {
    private lateinit var preview: PreviewView
    private lateinit var hint: TextView
    private lateinit var shutter: Button
    private lateinit var results: Button
    private var capture: ImageCapture? = null
    private val analysisExecutor = Executors.newSingleThreadExecutor()
    private val onSheetsChanged: () -> Unit = { updateResultsButton() }

    private val askCamera = registerForActivityResult(ActivityResultContracts.RequestPermission()) { ok ->
        if (ok) startCamera() else hint.text = getString(R.string.need_camera)
    }

    private val pickPhotos = registerForActivityResult(ActivityResultContracts.PickMultipleVisualMedia(20)) { uris ->
        if (uris.isNotEmpty()) {
            uris.forEach { addFromGallery(it) }
            openResults()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        findViewById<android.view.View>(R.id.root).padForSystemBars()
        preview = findViewById(R.id.preview)
        hint = findViewById(R.id.hint)
        shutter = findViewById(R.id.shutter)
        results = findViewById(R.id.results)
        shutter.setOnClickListener { takePhoto() }
        results.setOnClickListener { openResults() }
        findViewById<Button>(R.id.gallery).setOnClickListener {
            pickPhotos.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
        }
        Sheets.listen(onSheetsChanged)
        updateResultsButton()

        // Debug builds only: check a photo file passed in by a test (adb) instead of using the camera.
        val testPhoto = intent.getStringExtra("photo")
        if (BuildConfig.DEBUG && testPhoto != null) {
            Sheets.add(File(testPhoto))
            openResults()
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            askCamera.launch(Manifest.permission.CAMERA)
        }
    }

    override fun onNewIntent(intent: Intent) {  // debug test hook again, when the camera screen is already open
        super.onNewIntent(intent)
        val testPhoto = intent.getStringExtra("photo")
        if (BuildConfig.DEBUG && testPhoto != null) {
            Sheets.add(File(testPhoto))
            openResults()
        }
    }

    private fun openResults() = startActivity(Intent(this, SheetsActivity::class.java))

    private fun updateResultsButton() {
        val n = Sheets.all.size
        val checking = Sheets.all.count { it.state == Sheet.State.CHECKING }
        results.isEnabled = n > 0
        results.text = when {
            n == 0 -> getString(R.string.results)
            checking > 0 -> getString(R.string.results_checking, n, checking)
            else -> getString(R.string.results_n, n)
        }
    }

    private fun addFromGallery(uri: Uri) {
        val file = File(cacheDir, "sheet_${System.nanoTime()}.jpg")
        contentResolver.openInputStream(uri)?.use { input -> file.outputStream().use { input.copyTo(it) } }
        Sheets.add(file)
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
        val file = File(cacheDir, "sheet_${System.nanoTime()}.jpg")
        shutter.isEnabled = false
        cap.takePicture(
            ImageCapture.OutputFileOptions.Builder(file).build(),
            ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageSavedCallback {
                override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                    shutter.isEnabled = true
                    val sheet = Sheets.add(file)
                    Toast.makeText(this@MainActivity, getString(R.string.sheet_added, sheet.title), Toast.LENGTH_SHORT).show()
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
        Sheets.unlisten(onSheetsChanged)
        analysisExecutor.shutdown()
    }
}
