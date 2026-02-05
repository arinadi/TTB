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

1.  **Open Google Colab File**:
    Open the `colab_runner.ipynb` file in this repository, or upload it to your Google Colab.

2.  **Setup Secrets**:
    In Google Colab, open the **Secrets** tab (key icon 🔑 on the left sidebar) and add:
    -   `TELEGRAM_BOT_TOKEN`: Bot token from BotFather.
    -   `TELEGRAM_CHAT_ID`: Your Telegram chat ID (for security, the bot only responds to this ID).
    -   `GEMINI_API_KEY`: API Key from Google AI Studio (Optional, for summarization features).
    -   `GITHUB_TOKEN`: GitHub Personal Access Token (Optional, if this repo is Private).

3.  **Enable GPU**:
    Ensure the Runtime type is set to **T4 GPU** (Menu: *Runtime > Change runtime type*).

4.  **Run**:
    Execute all cells in the notebook. The bot will automatically install dependencies and go online.

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
-   `colab_runner.ipynb`: "Launcher" notebook for Google Colab.
-   `requirements.txt`: List of required Python libraries.
-   `TTBv1.py`: (Legacy) Old monolithic version, kept as reference.

## 🛠 Advanced Configuration

You can change default settings inside `main.py` (`Config` class):
-   `MODEL_SIZE`: `large-v3`, `medium`, `small`, etc.
-   `MAX_AUDIO_DURATION_MINUTES`: Audio duration limit (default: 90 minutes).
-   `ENABLE_IDLE_MONITOR`: Automatically turn off Colab runtime if idle (saves resources).
