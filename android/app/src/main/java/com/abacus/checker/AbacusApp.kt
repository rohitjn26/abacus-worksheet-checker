package com.abacus.checker

import android.app.Application
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform

class AbacusApp : Application() {
    override fun onCreate() {
        super.onCreate()
        OnnxBridge.init(this)
        if (!Python.isStarted()) Python.start(AndroidPlatform(this))
        // Load the checker and its models in the background, so the first photo isn't slow.
        Thread { runCatching { Python.getInstance().getModule("android_api").callAttr("warm_up") } }.start()
    }
}
