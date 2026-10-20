package com.abacus.checker

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import java.io.File

/** All sheets of this session: checking / done (tap to open) / needs a retake (Retake or Use anyway). */
class SheetsActivity : AppCompatActivity() {
    private val adapter = Adapter()
    private val onChanged: () -> Unit = { adapter.notifyDataSetChanged() }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_sheets)
        findViewById<View>(R.id.root).padForSystemBars()
        findViewById<RecyclerView>(R.id.list).apply {
            layoutManager = LinearLayoutManager(this@SheetsActivity)
            adapter = this@SheetsActivity.adapter
        }
        findViewById<Button>(R.id.more).setOnClickListener { finish() }  // back to the camera
        Sheets.listen(onChanged)
    }

    override fun onResume() {
        super.onResume()
        adapter.notifyDataSetChanged()
    }

    override fun onDestroy() {
        super.onDestroy()
        Sheets.unlisten(onChanged)
    }

    private class Holder(v: View) : RecyclerView.ViewHolder(v) {
        val thumb: ImageView = v.findViewById(R.id.thumb)
        val title: TextView = v.findViewById(R.id.title)
        val status: TextView = v.findViewById(R.id.status)
        val retake: Button = v.findViewById(R.id.retake)
        val useAnyway: Button = v.findViewById(R.id.use_anyway)
        var shown: File? = null
    }

    private inner class Adapter : RecyclerView.Adapter<Holder>() {
        override fun getItemCount() = Sheets.all.size

        override fun onCreateViewHolder(parent: ViewGroup, viewType: Int) =
            Holder(LayoutInflater.from(parent.context).inflate(R.layout.item_sheet, parent, false))

        override fun onBindViewHolder(h: Holder, position: Int) {
            val sheet = Sheets.all[Sheets.all.size - 1 - position]  // newest first
            h.title.text = sheet.title
            val image = sheet.marked ?: sheet.photo
            if (h.shown != image) {
                h.thumb.setImageBitmap(thumbnail(image))
                h.shown = image
            }
            val needsRetake = sheet.state == Sheet.State.RETAKE && !sheet.accepted
            h.status.text = when {
                sheet.state == Sheet.State.CHECKING -> getString(R.string.checking)
                sheet.state == Sheet.State.ERROR -> sheet.error
                needsRetake -> getString(R.string.retake_title) + ":\n" + sheet.retakeReasons.joinToString("\n")
                else -> Export.summaryLine(sheet) + if (sheet.accepted) "\n" + getString(R.string.low_quality_note) else ""
            }
            h.retake.visibility = if (needsRetake || sheet.state == Sheet.State.ERROR) View.VISIBLE else View.GONE
            h.useAnyway.visibility = if (needsRetake) View.VISIBLE else View.GONE
            h.retake.setOnClickListener {
                Sheets.remove(sheet)
                finish()  // back to the camera to take it again
            }
            h.useAnyway.setOnClickListener {
                sheet.accepted = true
                notifyDataSetChanged()
                open(sheet)
            }
            h.itemView.setOnClickListener {
                if (sheet.state == Sheet.State.DONE || (sheet.state == Sheet.State.RETAKE && sheet.accepted)) open(sheet)
            }
        }
    }

    private fun open(sheet: Sheet) = startActivity(Intent(this, SheetActivity::class.java).putExtra("id", sheet.id))

    private fun thumbnail(file: File): Bitmap? {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(file.path, bounds)
        var sample = 1
        while (bounds.outWidth / sample > 600) sample *= 2
        return BitmapFactory.decodeFile(file.path, BitmapFactory.Options().apply { inSampleSize = sample })
    }
}
