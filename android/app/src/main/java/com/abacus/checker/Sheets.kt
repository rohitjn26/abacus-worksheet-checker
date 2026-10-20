package com.abacus.checker

import android.os.Handler
import android.os.Looper
import com.chaquo.python.Python
import org.json.JSONObject
import java.io.File
import java.util.concurrent.Executors

/** One photographed or picked worksheet and what the checker made of it. */
class Sheet(val id: Int, val photo: File) {
    enum class State { CHECKING, DONE, RETAKE, ERROR }

    @Volatile var state = State.CHECKING
    var result: JSONObject? = null
    var marked: File? = null
    var accepted = false   // parent chose "Use anyway" on a photo that needed a retake
    var error: String? = null

    val title get() = "Sheet $id"
    val retakeReasons: List<String>
        get() = result?.optJSONArray("retake")?.let { a -> (0 until a.length()).map { a.getString(it) } } ?: emptyList()
}

/**
 * The sheets of this session, checked one after another on a background thread, so the parent can keep
 * taking photos while earlier ones are being checked. Screens listen for changes.
 */
object Sheets {
    val all = mutableListOf<Sheet>()
    private var nextId = 1
    private val worker = Executors.newSingleThreadExecutor()
    private val main = Handler(Looper.getMainLooper())
    private val listeners = mutableSetOf<() -> Unit>()

    fun listen(l: () -> Unit) { listeners += l }
    fun unlisten(l: () -> Unit) { listeners -= l }
    private fun changed() = main.post { listeners.toList().forEach { it() } }

    fun get(id: Int) = all.firstOrNull { it.id == id }

    fun remove(sheet: Sheet) {
        all.remove(sheet)
        changed()
    }

    /** Add a photo and queue it for checking. */
    fun add(photo: File): Sheet {
        val sheet = Sheet(nextId++, photo)
        all += sheet
        changed()
        worker.execute { check(sheet) }
        return sheet
    }

    private fun check(sheet: Sheet) {
        val out = File(sheet.photo.parentFile, "checked_${sheet.id}_${System.currentTimeMillis()}.jpg")
        try {
            val json = Python.getInstance().getModule("android_api").callAttr("check", sheet.photo.path, out.path).toString()
            android.util.Log.i("AbacusResult", "${sheet.photo} -> $json")
            val r = JSONObject(json)
            if (r.has("error")) {
                sheet.error = r.getString("error")
                sheet.state = Sheet.State.ERROR
            } else {
                sheet.result = r
                sheet.marked = out.takeIf { it.exists() }
                sheet.state = if (sheet.retakeReasons.isEmpty()) Sheet.State.DONE else Sheet.State.RETAKE
            }
        } catch (e: Exception) {
            sheet.error = e.message ?: "Something went wrong."
            sheet.state = Sheet.State.ERROR
        }
        changed()
    }
}
