package com.abacus.checker

import android.graphics.BitmapFactory
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.ProgressBar
import android.widget.TextView
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import com.chaquo.python.Python
import org.json.JSONObject
import java.io.File

/**
 * Runs the checker on the photo (in the background), then either asks the parent to retake it
 * (with the reason, and a "Use anyway" option) or shows the marked sheet and a summary.
 */
class ResultActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_result)
        val photo = intent.getStringExtra("photo") ?: return finish()
        findViewById<Button>(R.id.again).setOnClickListener { finish() }

        val out = File(cacheDir, "checked_${System.currentTimeMillis()}.jpg")
        Thread {
            val json = runCatching {
                val fn = if (BuildConfig.DEBUG && intent.getBooleanExtra("profile", false)) "profile_check" else "check"
                val text = Python.getInstance().getModule("android_api").callAttr(fn, photo, out.path).toString()
                if (fn == "profile_check") {
                    text.lines().forEach { android.util.Log.i("AbacusProfile", it) }
                }
                text.lines().first()
            }.getOrElse { JSONObject().put("error", it.message ?: "Something went wrong.").toString() }
            android.util.Log.i("AbacusResult", "$photo -> $json")
            runOnUiThread { show(JSONObject(json), out) }
        }.start()
    }

    private fun show(result: JSONObject, sheet: File) {
        findViewById<ProgressBar>(R.id.progress).visibility = View.GONE
        val status = findViewById<TextView>(R.id.status)
        if (result.has("error")) {
            status.text = result.getString("error")
            return
        }
        val retake = result.getJSONArray("retake")
        if (retake.length() > 0) {
            val reasons = (0 until retake.length()).joinToString("\n\n") { retake.getString(it) }
            AlertDialog.Builder(this)
                .setTitle(R.string.retake_title)
                .setMessage(reasons)
                .setCancelable(false)
                .setPositiveButton(R.string.retake) { _, _ -> finish() }
                .setNegativeButton(R.string.use_anyway) { _, _ -> showSheet(result, sheet, lowQuality = true) }
                .show()
        } else {
            showSheet(result, sheet, lowQuality = false)
        }
    }

    private fun showSheet(result: JSONObject, sheet: File, lowQuality: Boolean) {
        val status = findViewById<TextView>(R.id.status)
        val c = result.getJSONObject("counts")
        val checked = c.getInt("right") + c.getInt("wrong") + c.getInt("review")
        status.text = if (checked == 0) {
            getString(R.string.nothing_found)
        } else {
            getString(R.string.summary, c.getInt("right"), c.getInt("wrong"), c.getInt("review")) +
                (if (lowQuality) "\n" + getString(R.string.low_quality_note) else "")
        }
        if (sheet.exists()) {
            findViewById<ImageView>(R.id.sheet).setImageBitmap(BitmapFactory.decodeFile(sheet.path))
        }
    }
}
