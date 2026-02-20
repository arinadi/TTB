# Android Whisper Integration Plan

## Goal
Running local, offline speech-to-text on Android using the Whisper model.

## Analysis: "Can `faster-whisper` run on Android?"
**Directly? No.**
`faster-whisper` relies on **CTranslate2** (Python/C++), which is optimized for desktop/server GPUs (CUDA) and CPUs. While technically possible via Termux or Chaquopy, it is heavy, unoptimized for mobile batteries, and hard to integrate with a native Android UI.

**The Solution: `whisper.cpp`**
To achieve "faster" performance on Android, the standard industry solution is **`whisper.cpp`**.
-   **Port**: High-performance C++ port of OpenAI's Whisper.
-   **Optimized**: Uses ARM NEON instructions (core of Android CPUs) for extreme speed.
-   **Size**: No Python runtime needed. Just a small shared library (`.so`) + the model file.
-   **Integration**: Connects to Kotlin via JNI (Java Native Interface).

---

## Project Specifications

### 1. App Architecture (aligned with [android.md](file:///d:/TTB/android.md))
-   **Language**: Kotlin
-   **UI**: Jetpack Compose (Material3)
-   **Architecture**: MVVM (Model-View-ViewModel) + Clean Architecture
-   **DI**: Koin
-   **Concurrency**: Coroutines & Flow

### 2. Whisper Implementation Details
We will use a pre-built Android wrapper for `whisper.cpp` or build the JNI bindings ourselves.
-   **Library**: `com.github.ggerganov:whisper.cpp` (or a community wrapper like `KmsCreek/whisper-mobile-android` if active, otherwise build from source).
-   **Model Management**:
    -   Models (`tiny`, `base`, `small`) are downloaded on demand so the initial APK size remains small.
    -   Models stored in `Context.filesDir`.
-   **Audio Capture**:
    -   `AudioRecord` API (Native Android) for raw PCM 16-bit 16kHz data (required by Whisper).

### 3. Feature Breakdown
1.  **Model Downloader**: Screen to download `tiny.en` (75MB) or `base.en` (142MB).
2.  **Recorder UI**: Button to hold-to-record or tap-to-toggle.
3.  **Real-time Transcriber**:
    -   *Streaming Mode*: Process audio in 30s chunks (Whisper's window).
    -   *VAD (Voice Activity Detection)*: Optional future optimization to stop recording silence.
4.  **Result Display**: Live text updates in a limit-less lazy list.

### 4. Required Libraries (Gradle)

```kotlin
dependencies {
    // Core
    implementation(platform("androidx.compose:compose-bom:2024.02.00"))
    implementation("androidx.compose.material3:material3")
    
    // Architecture
    implementation("io.insert-koin:koin-androidx-compose:3.5.3")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.7.0")
    
    // Whisper (JNI) - Example placeholder, standard practice is to include the .aar
    // implementation("com.github.ggerganov:whisper.cpp:android-lib:1.5.0") 
    // OR manual JNI compilation integration.
    
    // Networking (for downloading models)
    implementation("io.ktor:ktor-client-android:2.3.8")
    
    // Permissions (Mic)
    implementation("com.google.accompanist:accompanist-permissions:0.34.0")
}
```

## Step-by-Step Implementation Plan

### Phase 1: JNI & Native Setup
1.  Clone `whisper.cpp`.
2.  Compile `.so` files for `arm64-v8a` and `armeabi-v7a`.
3.  Create a Kotlin class `WhisperLib` with `external fun` methods.

### Phase 2: Audio Engine
1.  Create `AudioRecorder` class using `AudioRecord`.
2.  Configure it to capture **16kHz, Mono, PCM 16-bit** (Must match Whisper specs exactly).

### Phase 3: UI & Logic
1.  Build `TranscriptionViewModel`.
2.  Create `RecordScreen` with Compose.
3.  Connect the loop: Audio -> Buffer -> Whisper JNI -> String -> UI.

## Alternative: Server-Side Processing
If on-device accuracy (`tiny`/`base` models) is too low, we can fallback to the existing **Telegram Bot API**:
1.  Record Audio on Android.
2.  Upload to `main.py` (your Python bot).
3.  Let the server GPU run `large-v2` (faster-whisper).
4.  Receive text back.
*This requires internet, but guarantees best accuracy.*
