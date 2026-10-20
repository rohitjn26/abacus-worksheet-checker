#!/bin/bash
# Try the app on this Mac: opens a virtual Android phone (emulator) in a window, uses the Mac's webcam as
# the phone's camera, installs the app and starts it.
#
#   ./run_on_mac.sh                       # newest AbacusChecker-*.apk next to this folder
#   ./run_on_mac.sh path/to/app.apk
#   PHOTOS="../19022026" ./run_on_mac.sh  # also put these photos in the phone's gallery (for "Gallery")
#   CAMERA=emulated ./run_on_mac.sh       # the emulator's made-up camera scene instead of the webcam
#
# Close the phone window (or run: adb emu kill) to stop it.
set -e
SDK=${ANDROID_HOME:-$HOME/Library/Android/sdk}
ADB="$SDK/platform-tools/adb"
HERE="$(cd "$(dirname "$0")" && pwd)"
APK=${1:-$(ls -t "$HERE"/../AbacusChecker-*.apk | head -1)}
CAMERA=${CAMERA:-webcam0}

if ! "$ADB" devices | grep -q "emulator-"; then
    echo "Starting the virtual phone..."
    "$SDK/emulator/emulator" -avd abacus_test -camera-back "$CAMERA" -no-snapshot-save > /tmp/abacus_emulator.log 2>&1 &
fi
"$ADB" wait-for-device
until [ "$("$ADB" shell getprop sys.boot_completed 2>/dev/null | tr -d '\r')" = "1" ]; do sleep 2; done

echo "Installing $(basename "$APK")..."
"$ADB" install -r -g "$APK" > /dev/null

if [ -n "$PHOTOS" ]; then
    echo "Adding photos to the phone's gallery..."
    for f in "$PHOTOS"/*.jp*g; do
        name="$(basename "$f" | tr ' ()' '___')"
        "$ADB" push "$f" "/sdcard/Pictures/$name" > /dev/null
        "$ADB" shell am broadcast -a android.intent.action.MEDIA_SCANNER_SCAN_FILE -d "file:///sdcard/Pictures/$name" > /dev/null
    done
fi

"$ADB" shell am start -n com.abacus.checker/.MainActivity > /dev/null
echo "Ready. Saved images / PDFs can be copied back with:  adb pull \"/sdcard/Download/Abacus Checker\" ."
