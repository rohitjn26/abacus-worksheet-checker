package com.abacus.checker

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.BitmapFactory
import android.os.Build
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat

/** One checked sheet: zoomable marked image, summary, and Share / Save image / Save PDF. */
class SheetActivity : AppCompatActivity() {
    private lateinit var sheet: Sheet
    private var pendingSave: (() -> Unit)? = null

    // Only Android 9 and older need a permission to save into Pictures / Downloads.
    private val askStorage = registerForActivityResult(ActivityResultContracts.RequestPermission()) { ok ->
        if (ok) pendingSave?.invoke() else toast(getString(R.string.need_storage))
        pendingSave = null
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_sheet)
        findViewById<View>(R.id.root).padForSystemBars()
        sheet = Sheets.get(intent.getIntExtra("id", -1)) ?: return finish()
        title = sheet.title
        findViewById<TextView>(R.id.summary).text = buildString {
            append(sheet.title).append("   ").append(Export.summaryLine(sheet))
            append("\n").append(getString(R.string.legend))
            if (sheet.accepted) append("\n").append(getString(R.string.low_quality_note))
        }
        sheet.marked?.let { findViewById<ZoomImageView>(R.id.sheet).setImageBitmap(BitmapFactory.decodeFile(it.path)) }
        findViewById<Button>(R.id.share).setOnClickListener { Export.shareImage(this, sheet) }
        findViewById<Button>(R.id.save_image).setOnClickListener { save { Export.saveImage(this, sheet) } }
        findViewById<Button>(R.id.save_pdf).setOnClickListener { save { Export.savePdf(this, sheet) } }
    }

    private fun save(action: () -> String) {
        val run = {
            try {
                toast(getString(R.string.saved_to, action()))
            } catch (e: Exception) {
                toast(getString(R.string.save_failed))
            }
        }
        if (Build.VERSION.SDK_INT < 29 &&
            ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED
        ) {
            pendingSave = { run() }
            askStorage.launch(Manifest.permission.WRITE_EXTERNAL_STORAGE)
        } else {
            run()
        }
    }

    private fun toast(msg: String) = Toast.makeText(this, msg, Toast.LENGTH_LONG).show()
}
