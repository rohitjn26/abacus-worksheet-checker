# Abacus Checker – Android app (v0.1)

For parents at home: photograph the worksheet, get every answer marked right / wrong / please check.
Runs fully offline; nothing leaves the phone.

## How it works

- **Camera screen** (Kotlin, CameraX): live preview and a hint under it, from cheap checks on each preview
  frame (`LiveQuality.kt`): too dark, light reflecting, hold steady, or "looks good" (green shutter).
- **After the photo**: the tested Python checker from `../prototype` runs on the phone through
  [Chaquopy](https://chaquo.com/chaquopy/) (Python 3.10, numpy, OpenCV 4.5). The models run in Kotlin with
  ONNX Runtime for Android (`OnnxBridge.kt`), called from Python (`app/src/main/python/android_api.py`).
- **If the photo isn't good enough** (`quality.retake_reasons`): a dialog with the reason, **Retake** and
  **Use anyway** (results then carry a "photo quality was low" note).
- **Result screen**: the marked sheet (OK / X n / ? = n) and a summary.

The checker's `.py` files and the three `.onnx` models are copied from `../prototype` at build time
(`syncEnginePython`, `syncEngineModels` in `app/build.gradle.kts`), so the app ships exactly the tested code.

## Build

Needs JDK 17 and the Android SDK (platform 35, build-tools 35), plus Python 3.10 on the build machine
(Chaquopy uses it; path set in `app/build.gradle.kts`).

```bash
export JAVA_HOME=/opt/homebrew/opt/openjdk@17 ANDROID_HOME=$HOME/Library/Android/sdk
./gradlew clean assembleDebug  # -> app/build/outputs/apk/debug/app-debug.apk (~71 MB, arm64 phones)
```

## Testing without a camera (debug builds only)

Copy a photo into the app and pass it in; the result is logged under the tag `AbacusResult`
(add `--ez profile true` for a timing profile under `AbacusProfile`):

```bash
adb push sheet.jpg /data/local/tmp/t.jpg
adb shell run-as com.abacus.checker sh -c 'mkdir -p cache && cp /data/local/tmp/t.jpg cache/t.jpg'
adb shell am start -n com.abacus.checker/.MainActivity --es photo /data/user/0/com.abacus.checker/cache/t.jpg
adb logcat -s AbacusResult
```

On the Android emulator (Pixel 6, Android 15, on an Apple-silicon Mac) a photo takes about 7–9 s to check
and gives the same verdicts as the Mac. A real phone may be slower; not yet measured.

## Known limits (v0.1)

- English only; arm64 phones only (nearly all phones from the last ~6 years).
- Live-hint thresholds were set from WhatsApp photos, not live camera frames; tune them on real phones.
- ~71 MB: Python + numpy + OpenCV (~30 MB), ONNX Runtime (~17 MB), models (13 MB: printed-number reader
  in 8-bit, PaddleOCR weights stored 16-bit - both give the same results on all 25 photos), app code
  (~4 MB, shrunk with R8). A native Kotlin port of the checker (no Python) would bring it to ~35 MB.
- Always build with `clean`: incremental debug builds leave unused space in the APK (90 MB instead of 71).
