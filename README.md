# 🤖 TTB (Telegram Transcription Bot)

**TTB** is an advanced Telegram bot that utilizes **OpenAI Whisper** for high-precision audio/video transcription and **Google Gemini** for automatic summarization in Indonesian (default).

Specifically designed to run on **Google Colab** (Free GPU) using the "Vibe Coding" method, where this repository acts as the *source of truth* pulled by Colab at *runtime*.

## ⚡ Limits & Compatibility

| Component | Type | Limit |
| :--- | :--- | :--- |
| **Whisper (Transcription)** | **Fully Local** (Colab GPU) | **Unlimited**. No duration or file count limits. Runs 100% offline once model is loaded. |
| **Hugging Face** | Model Download | **Rate Limit Only**. Adding `HF_TOKEN` prevents temporary download blocks from HF servers. |
| **Google Gemini** | Cloud API | **Free Tier Quota**. Subject to your Google API Key limits (~15 RPM, 1,500/day). |

## ✨ Key Features

-   **Accurate Transcription**: Uses **faster-whisper** (`large-v2` default for SEA languages) with optimized beam search.
-   **Smart Summarization**: Integrates Google Gemini 2.5 Flash to summarize transcripts into key points (Indonesian).
-   **Large File Support**: Handles audio/video files up to Telegram's limit, and supports **Multi-part ZIP archives** (e.g., `file.zip.001`) for very large files.
-   **GPU Acceleration**: Optimized for fast performance on GPU (CUDA), with FP16/INT8 dynamic loading.
-   **Clean Formatting**: Text output is formatted as **clean paragraphs** separated by double newlines, with timestamps removed for better readability.
-   **Context-Aware**: Uses VAD (Voice Activity Detection) and Repetition Penalties to reduce hallucinations.

## 🚀 How to Run (Google Colab)

The easiest and recommended way is to use Google Colab.

1.  **Setup Secrets**:
    In Google Colab, open the **Secrets** tab (key icon 🔑 on the left sidebar) and add:
    -   `TELEGRAM_BOT_TOKEN`: Bot token from BotFather.
    -   `TELEGRAM_CHAT_ID`: Your Telegram chat ID (for security, the bot only responds to this ID).
    -   `GEMINI_API_KEY`: API Key from Google AI Studio (Optional, for summarization features).
    -   `GITHUB_TOKEN`: GitHub Personal Access Token (Optional, if this repo is Private).
    -   `HF_TOKEN`: Hugging Face Token (Optional, prevents model download rate limits).

2.  **Enable GPU**:
    Ensure the Runtime type is set to **T4 GPU** (Menu: *Runtime > Change runtime type*).

3.  **Run**:
    Copy the code block below into a single cell in your Colab notebook and run it. This script will automatically clone/update the repository, install dependencies, and start the bot.

    ```python
    # @title 🚀 Setup & Run TTB
    import os
    import sys
    import time
    from google.colab import userdata

    # --- 1. Initialize & Load Secrets ---
    os.environ['INIT_START'] = str(int(time.time()))
    
    try:
        # Secrets
        os.environ['TELEGRAM_BOT_TOKEN'] = userdata.get('TELEGRAM_BOT_TOKEN')
        os.environ['TELEGRAM_CHAT_ID'] = userdata.get('TELEGRAM_CHAT_ID')
        gemini_key = userdata.get('GEMINI_API_KEY')
        if gemini_key:
            os.environ['GEMINI_API_KEY'] = gemini_key
            
        # Optional: HuggingFace Token (to avoid rate limits/warnings)
        hf_token = userdata.get('HF_TOKEN')
        if hf_token:
            os.environ['HF_TOKEN'] = hf_token
            
        # Optional: HuggingFace Token (to avoid rate limits/warnings)
        hf_token = userdata.get('HF_TOKEN')
        if hf_token:
            os.environ['HF_TOKEN'] = hf_token
            
        print("✅ Loaded Keys and Timer")
        
    except Exception as e:
        print(f"⚠️ Warning: Could not load secrets from userdata: {e}")

    # --- CONFIGURATION ---
    # ⚠️ REPLACE THIS WITH YOUR REPOSITORY URL IF NEEDED
    REPO_URL = "https://github.com/arinadi/TTB.git" 
    REPO_NAME = "TTB"
    # ---------------------

    print("🔄 Checking environment...")

    try:
        # Try to use GITHUB_TOKEN if available for private repos
        token = userdata.get('GITHUB_TOKEN')
        if token and "github.com" in REPO_URL:
            REPO_URL = REPO_URL.replace("https://", f"https://{token}@")
    except Exception:
        pass

    # 2. Clone or Update Repository
    if not os.path.exists(REPO_NAME):
        print(f"⏳ Cloning {REPO_NAME}...")
        !git clone --depth 1 {REPO_URL}
        %cd {REPO_NAME}
    else:
        print(f"⏳ Updating {REPO_NAME}...")
        %cd {REPO_NAME}
        !git fetch --depth 1 origin
        !git reset --hard origin/main
        
    print(f"✅ Code ready ({int(time.time()) - int(os.environ['INIT_START'])}s)")

    # 3. Install Dependencies (using uv for speed)
    print("⏳ Installing dependencies with uv...")
    !bash setup_uv.sh
    print(f"✅ Dependencies installed ({int(time.time()) - int(os.environ['INIT_START'])}s.")

    # 4. Run the Bot
    print("🚀 Starting TTB...")
    !python main.py
    ```

## 🧠 Vibe Coding Tips

-   **Structure**: Logic is in `.py` files (`main.py`, `utils.py`) for easier Git handling and AI agent interaction. Colab is just the *runner*.

## 💻 How to Run (Local)

If you have your own GPU (NVIDIA) or want to run on CPU (slower):

1.  **Clone Repo**:
    ```bash
    git clone https://github.com/arinadi/TTB.git
    cd TTB
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements_local.txt
    ```
    *Note: Ensure `ffmpeg` is installed on your system.*

3.  **Setup Environment Variables**:
    Set the following in `.env` or system variables:
    -   `TELEGRAM_BOT_TOKEN`
    -   `TELEGRAM_CHAT_ID`
    -   `GEMINI_API_KEY`

4.  **Run Bot**:
    ```bash
    python main.py
    ```

## 📂 File Structure

-   `main.py`: Main entry point. Contains Telegram bot logic, queue system, and model initialization.
-   `config.py`: Centralized configuration and secrets management.
-   `utils.py`: Helper functions for text formatting, logging, and Gemini API wrapper.
-   `gradio_handler.py`: Optional Gradio web interface for large file uploads.
-   `requirements.txt`: Optimized list for Colab.
-   `requirements_local.txt`: Full list for local dev.

## 🛠 Advanced Configuration

All settings are managed in `config.py` and can be overridden via Environment Variables:

-   **Model**: `WHISPER_MODEL` (default: `large-v2`).
-   **Precision**: `WHISPER_PRECISION` (`auto`, `float16`, `int8`).
-   **VAD**: `VAD_FILTER` (True/False) to reduce hallucinations.
-   **Decoding**: 
    - `WHISPER_PATIENCE` (Default: 2.0)
    - `REPETITION_PENALTY` (Default: 1.1)
-   **Idle Monitor**: `ENABLE_IDLE_MONITOR` (Colab saver).

## 🔄 Network Resilience

This bot has robust error handling for connection issues in Colab:

-   **Auto-Retry**: Transient network errors are auto-retried (max 2x) without shutdown.
-   **Extended Timeouts**: Optimized specifically for Colab's network environment.
-   **Connection Pool**: Pool size 8 for stable connections.
