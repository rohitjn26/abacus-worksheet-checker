plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("com.chaquo.python")
}

// The checker's Python code and ONNX models live in ../prototype (single source of truth);
// they are copied into the build here, so the app always ships exactly what was tested.
val prototype = rootProject.file("../prototype")
val enginePy = layout.buildDirectory.dir("engine/python")
val engineAssets = layout.buildDirectory.dir("engine/assets")
val syncEnginePython by tasks.registering(Sync::class) {
    from(prototype) { include("check.py", "cells.py", "grid.py", "norm.py", "quality.py", "policy.py", "runtime.py", "ctc.py") }
    from("src/main/python")
    into(enginePy)
}
val syncEngineModels by tasks.registering(Sync::class) {
    from(prototype) { include("reader_v4_int8.onnx", "hand_head_v2.onnx", "ppocrv5_mobile_frozen_fp16w.onnx") }
    into(engineAssets)
}

android {
    namespace = "com.abacus.checker"
    compileSdk = 35
    defaultConfig {
        applicationId = "com.abacus.checker"
        minSdk = 24
        targetSdk = 35
        versionCode = 1
        versionName = "0.1"
        ndk { abiFilters += listOf("arm64-v8a") }
    }
    buildTypes {
        // Shrink unused Android / library code (R8). Classes Python reaches by name are kept in proguard-rules.pro.
        debug {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    buildFeatures { buildConfig = true }
    sourceSets["main"].assets.srcDir(engineAssets)
    androidResources { noCompress += "onnx" }
}

chaquopy {
    defaultConfig {
        version = "3.10"
        buildPython("/opt/homebrew/bin/python3.10")
        pip {
            install("numpy==1.26.2")
            install("opencv-python==4.5.1.48")
        }
    }
    sourceSets {
        getByName("main") {
            setSrcDirs(listOf(enginePy.get().asFile.path))
        }
    }
}

tasks.named("preBuild") { dependsOn(syncEnginePython, syncEngineModels) }
// Every task that reads the copied engine files must run after the copy.
tasks.matching { it.name.contains("Python") && it.name != "syncEnginePython" }.configureEach { dependsOn(syncEnginePython) }
tasks.matching { it.name.contains("Assets") }.configureEach { dependsOn(syncEngineModels) }

dependencies {
    val camerax = "1.4.1"
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.constraintlayout:constraintlayout:2.2.0")
    implementation("androidx.camera:camera-core:$camerax")
    implementation("androidx.camera:camera-camera2:$camerax")
    implementation("androidx.camera:camera-lifecycle:$camerax")
    implementation("androidx.camera:camera-view:$camerax")
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.20.0")
}
