# Python (android_api.py) calls these by name through Chaquopy - keep them as they are.
-keep class com.abacus.checker.OnnxBridge { *; }
-keep class com.abacus.checker.OnnxResult { *; }
# Libraries whose Java classes are reached from native code.
-keep class ai.onnxruntime.** { *; }
-keep class com.chaquo.python.** { *; }
