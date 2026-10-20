package com.abacus.checker

import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.pdf.PdfDocument
import android.media.MediaScannerConnection
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import androidx.core.content.FileProvider
import java.io.File
import java.io.OutputStream

/** Share / save a checked sheet as a picture or a PDF. */
object Export {
    private const val FOLDER = "Abacus Checker"

    private fun stamp() = java.text.SimpleDateFormat("yyyy-MM-dd_HH-mm", java.util.Locale.US).format(java.util.Date())

    /** Android's share menu (WhatsApp, Gmail, Telegram...) with the marked sheet as an image. */
    fun shareImage(ctx: Context, sheet: Sheet) {
        val file = sheet.marked ?: return
        val uri = FileProvider.getUriForFile(ctx, "${ctx.packageName}.files", file)
        val send = Intent(Intent.ACTION_SEND).apply {
            type = "image/jpeg"
            putExtra(Intent.EXTRA_STREAM, uri)
            putExtra(Intent.EXTRA_TEXT, "${sheet.title}: ${summaryLine(sheet)}")
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        ctx.startActivity(Intent.createChooser(send, ctx.getString(R.string.share_with)))
    }

    /** Save the marked sheet to the phone's gallery (Pictures/Abacus Checker). Returns where it went. */
    fun saveImage(ctx: Context, sheet: Sheet): String {
        val file = sheet.marked ?: error("nothing to save")
        val name = "abacus_${sheet.id}_${stamp()}.jpg"
        return save(ctx, name, "image/jpeg", Environment.DIRECTORY_PICTURES,
            MediaStore.Images.Media.EXTERNAL_CONTENT_URI) { out -> file.inputStream().use { it.copyTo(out) } }
    }

    /** Save the marked sheet as a PDF to Downloads/Abacus Checker. Returns where it went. */
    fun savePdf(ctx: Context, sheet: Sheet): String {
        val name = "abacus_${sheet.id}_${stamp()}.pdf"
        val downloads = if (Build.VERSION.SDK_INT >= 29) MediaStore.Downloads.EXTERNAL_CONTENT_URI else null
        return save(ctx, name, "application/pdf", Environment.DIRECTORY_DOWNLOADS, downloads) { out -> writePdf(ctx, sheet, out) }
    }

    private fun save(ctx: Context, name: String, mime: String, dir: String, collection: Uri?, write: (OutputStream) -> Unit): String {
        if (Build.VERSION.SDK_INT >= 29 && collection != null) {
            val values = ContentValues().apply {
                put(MediaStore.MediaColumns.DISPLAY_NAME, name)
                put(MediaStore.MediaColumns.MIME_TYPE, mime)
                put(MediaStore.MediaColumns.RELATIVE_PATH, "$dir/$FOLDER")
            }
            val uri = ctx.contentResolver.insert(collection, values) ?: error("could not create the file")
            ctx.contentResolver.openOutputStream(uri)!!.use(write)
        } else {  // Android 9 and older: needs the storage permission (asked for by the screen)
            @Suppress("DEPRECATION")
            val folder = File(Environment.getExternalStoragePublicDirectory(dir), FOLDER).apply { mkdirs() }
            val file = File(folder, name)
            file.outputStream().use(write)
            MediaScannerConnection.scanFile(ctx, arrayOf(file.path), arrayOf(mime), null)
        }
        return "$dir/$FOLDER/$name"
    }

    fun summaryLine(sheet: Sheet): String {
        val c = sheet.result?.optJSONObject("counts") ?: return ""
        return "✅ ${c.optInt("right")} right   ❌ ${c.optInt("wrong")} wrong   🟠 ${c.optInt("review")} please check"
    }

    /** A4 PDF: title, summary and key, then the marked sheet at page width (split over pages if tall). */
    private fun writePdf(ctx: Context, sheet: Sheet, out: OutputStream) {
        val bmp: Bitmap = BitmapFactory.decodeFile(sheet.marked!!.path)
        val pageW = 595
        val pageH = 842
        val margin = 36
        val usableW = pageW - 2 * margin
        val scale = usableW.toFloat() / bmp.width
        val doc = PdfDocument()
        val text = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.BLACK; textSize = 13f }
        val title = Paint(text).apply { textSize = 18f; isFakeBoldText = true }
        var srcTop = 0
        var pageNo = 1
        while (srcTop < bmp.height) {
            val page = doc.startPage(PdfDocument.PageInfo.Builder(pageW, pageH, pageNo).create())
            val c: Canvas = page.canvas
            var y = margin.toFloat()
            if (pageNo == 1) {
                y += 18f
                c.drawText("${ctx.getString(R.string.app_name)} – ${sheet.title}", margin.toFloat(), y, title)
                y += 22f
                c.drawText(summaryLine(sheet), margin.toFloat(), y, text)
                y += 18f
                c.drawText(ctx.getString(R.string.legend), margin.toFloat(), y, text)
                if (sheet.accepted) {
                    y += 18f
                    c.drawText(ctx.getString(R.string.low_quality_note), margin.toFloat(), y, text)
                }
                y += 16f
            }
            val room = ((pageH - margin - y) / scale).toInt()
            val srcBottom = minOf(bmp.height, srcTop + room)
            val dstH = (srcBottom - srcTop) * scale
            c.drawBitmap(bmp, Rect(0, srcTop, bmp.width, srcBottom),
                android.graphics.RectF(margin.toFloat(), y, (margin + usableW).toFloat(), y + dstH), null)
            doc.finishPage(page)
            srcTop = srcBottom
            pageNo++
        }
        doc.writeTo(out)
        doc.close()
    }
}
