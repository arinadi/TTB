# TTB (Telegram Transcription Bot) - Agent Context

> **Role**: Act as an Equal Pair Programmer. Maintain consistent coding standards and structures.

## Project Overview
TTB is a Telegram bot designed for **Google Colab** that performs:
1.  **Transcription**: Using `faster-whisper` (GPU-accelerated).
2.  **Summarization**: Using **Google Gemini** (Indonesian context).
3.  **Large File Handling**: Via Gradio Web UI bypass for files >20MB.

## Architecture
-   **Vibe Coding**: This repository is the *Source of Truth*. Colab pulls this code at runtime.
-   **Async First**: Built on `python-telegram-bot` (async/await) and `asyncio`.
-   **Hybrid Interaction**: Telegram for commands/results, Gradio for large uploads.

## Key Files
| File | Purpose |
| :--- | :--- |
| **`main.py`** | **Entry Point**. Contains `Application` setup, `queue_processor`, and Model initialization. |
| **`config.py`** | **Configuration**. Manages Secrets (`os.environ`) and Bot Settings (Whisper model, VAD, timeouts). |
| **`utils.py`** | **Utilities**. Logging, Time formatting, and **Gemini Summarization Logic**. |
| **`bot_classes.py`**| **Data Structures**. `TranscriptionJob`, `JobManager`, `IdleMonitor`, `FilesHandler`. |
| **`gradio_handler.py`** | **Web UI**. Handles large file uploads and queues them to `JobManager`. |
| **`setup_uv.sh`** | **Installation**. Uses `uv` for ultra-fast dependency installation in Colab. |

## Critical Workflows

### 1. Startup & Initialization
-   **`main.post_init`**: Sends initial "Bot Online" message (with unique `STARTUP_MESSAGE_ID`).
-   **Background Tasks**: `initialize_models_background` and `initialize_gradio_background` run concurrently.
-   **Dynamic Update**: As services load, they **edit** the single `STARTUP_MESSAGE_ID` message instead of spamming new ones.

### 2. Transcription Pipeline (`main.queue_processor`)
1.  **Receive**: File via Telegram or Gradio -> Added to `JobManager` queue.
2.  **Transcribe**: `faster-whisper` processes audio.
3.  **Immediate Result**: Send "Done" message + `TS_...` (Transcript) file **immediately**.
4.  **AI Summary**: Call `utils.summarize_text` (Gemini).
5.  **Final Result**: Send `AI_...` (Summary) file when ready.
6.  **Cleanup**: Local files removed.

## Configuration (Environment Variables)
| Variable | Description | Default |
| :--- | :--- | :--- |
| `TELEGRAM_BOT_TOKEN` | Bot Token | **Required** |
| `TELEGRAM_CHAT_ID` | Admin Chat ID | **Required** |
| `GEMINI_API_KEY` | Google AI Studio Key | Optional (but recommended) |
| `WHISPER_MODEL` | Model Size | `large-v2` |
| `WHISPER_PRECISION` | FP16/INT8 | `auto` |
| `BOT_FILESIZE_LIMIT` | Max MB for Telegram | `20` |

## Development Rules
-   **Language**: English for all docs/comments.
-   **Verification**: Always ensure `npm run build` (if JS) or syntax checks (Python) pass.
-   **Commit**: Document changes in `walkthrough.md`.
