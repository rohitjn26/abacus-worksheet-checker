package com.abacus.checker

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Result of one model call, handed back to Python: float32 data (native byte order) and its shape. */
class OnnxResult(val data: ByteArray, val shape: LongArray)

/**
 * Runs the checker's ONNX models for the Python code (android_api.JavaBackend).
 * Models are loaded from the app's assets on first use and kept for later photos.
 */
object OnnxBridge {
    private lateinit var env: OrtEnvironment
    private lateinit var appContext: Context
    private val sessions = HashMap<String, OrtSession>()

    fun init(context: Context) {
        appContext = context.applicationContext
        env = OrtEnvironment.getEnvironment()
    }

    private fun session(model: String): OrtSession = synchronized(sessions) {
        sessions.getOrPut(model) {
            val bytes = appContext.assets.open(model).use { it.readBytes() }
            env.createSession(bytes, OrtSession.SessionOptions())
        }
    }

    @JvmStatic
    fun run(model: String, input: String, data: ByteArray, shape: IntArray): OnnxResult {
        val floats = ByteBuffer.wrap(data).order(ByteOrder.nativeOrder()).asFloatBuffer()
        OnnxTensor.createTensor(env, floats, shape.map { it.toLong() }.toLongArray()).use { tensor ->
            session(model).run(mapOf(input to tensor)).use { outputs ->
                val out = outputs[0] as OnnxTensor
                val values = out.floatBuffer
                val bytes = ByteBuffer.allocate(values.remaining() * 4).order(ByteOrder.nativeOrder())
                bytes.asFloatBuffer().put(values)
                return OnnxResult(bytes.array(), out.info.shape)
            }
        }
    }
}
