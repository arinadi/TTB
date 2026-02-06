# 🤖 TTB (Telegram Transcription Bot)

**TTB** is an advanced Telegram bot that utilizes **OpenAI Whisper** for high-precision audio/video transcription and **Google Gemini** for automatic summarization in Indonesian (default).

Specifically designed to run on **Google Colab** (Free GPU) using the "Vibe Coding" method, where this repository acts as the *source of truth* pulled by Colab at *runtime*.

## ✨ Key Features

-   **Accurate Transcription**: Uses Whisper models (default: `large-v3`) for best results.
-   **Smart Summarization**: Integrates Google Gemini 2.5 Flash to summarize transcripts into key points (Indonesian).
-   **Large File Support**: Handles audio/video files up to Telegram's limit, and supports **Multi-part ZIP archives** (e.g., `file.zip.001`) for very large files.
-   **GPU Acceleration**: Optimized for fast performance on GPU (CUDA), with CPU fallback.
-   **Smart Formatting**: Text output with smart timestamps based on speech pauses.
-   **Task Management**: Queue system to handle multiple requests sequentially.

## 🚀 How to Run (Google Colab)

The easiest and recommended way is to use Google Colab.

1.  **Setup Secrets**:
    In Google Colab, open the **Secrets** tab (key icon 🔑 on the left sidebar) and add:
    -   `TELEGRAM_BOT_TOKEN`: Bot token from BotFather.
    -   `TELEGRAM_CHAT_ID`: Your Telegram chat ID (for security, the bot only responds to this ID).
    -   `GEMINI_API_KEY`: API Key from Google AI Studio (Optional, for summarization features).
    -   `GITHUB_TOKEN`: GitHub Personal Access Token (Optional, if this repo is Private).

2.  **Enable GPU**:
    Ensure the Runtime type is set to **T4 GPU** (Menu: *Runtime > Change runtime type*).

3.  **Run**:
    Copy the code block below into a single cell in your Colab notebook and run it. This script will automatically clone/update the repository, install dependencies, and start the bot.

    ```python
    # @title 🚀 Setup & Run TTB
    import os
    import sys
    from google.colab import userdata

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

    # 1. Clone or Update Repository
    if not os.path.exists(REPO_NAME):
        print(f"⏳ Cloning {REPO_NAME}...")
        !git clone {REPO_URL}
        %cd {REPO_NAME}
    else:
        print(f"⏳ Updating {REPO_NAME}...")
        %cd {REPO_NAME}
        !git pull

    # 2. Install Dependencies
    print("⏳ Installing dependencies...")
    # Show requirements.txt content
    !cat requirements.txt
    !pip install -q -r requirements.txt
    print("✅ Dependencies installed.")

    # 3. Inject Secrets & Configuration
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
            
        # Optimization: T4 Runtime Saving
        # Adjust these to shutdown faster when idle
        os.environ['IDLE_SHUTDOWN_MINUTES'] = "10" # Default: 3
        os.environ['IDLE_WARNING_MINUTES'] = "5"  # Default: 2
        os.environ['IDLE_NOTIFY_MINUTES'] = "1"   # Default: 1
        
    except Exception as e:
        print(f"⚠️ Warning: Could not load secrets from userdata: {e}")

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
    pip install -r requirements.txt
    ```
    *Note: Ensure `ffmpeg` is installed on your system.*

3.  **Setup Environment Variables**:
    Set the following variables in your terminal or a `.env` file:
    -   `TELEGRAM_BOT_TOKEN`
    -   `TELEGRAM_CHAT_ID`
    -   `GEMINI_API_KEY`

4.  **Run Bot**:
    ```bash
    python main.py
    ```

## 📂 File Structure

-   `main.py`: Main entry point. Contains Telegram bot logic, queue system, and model initialization.
-   `utils.py`: Helper functions for text formatting, duration, and Gemini API wrapper.
-   `requirements.txt`: List of required Python libraries.
-   `TTBv1.py`: (Legacy) Old monolithic version, kept as reference.

## 🛠 Advanced Configuration

You can change default settings inside `main.py` (`Config` class):
-   `MODEL_SIZE`: `large-v3`, `medium`, `small`, etc.
-   `MAX_AUDIO_DURATION_MINUTES`: Audio duration limit (default: 90 minutes).
-   `ENABLE_IDLE_MONITOR`: Automatically turn off Colab runtime if idle (saves resources).
