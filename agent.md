# TTB (Telegram Transcription Bot) - Agent Context

> **Role**: Act as an Equal Pair Programmer. Maintain consistent coding standards and structures.

## Project Overview
TTB is a Telegram bot designed for **Google Colab** and local systems that performs:
1.  **Transcription**: Using `faster-whisper` (GPU) or **Google Gemini API** (CPU fallback).
2.  **Summarization**: Using **Google Gemini** (Indonesian context).
3.  **Large File Handling**: Via Gradio Web UI (Whisper Mode only).

## Architecture
-   **Smart Runner**: `start.py` detects CPU/GPU and sets `TRANSCRIPTION_MODE` ('WHISPER' or 'GEMINI').
-   **Vibe Coding**: This repository is the *Source of Truth*. Colab pulls this code at runtime.
-   **Async First**: Built on `python-telegram-bot` (async/await) and `asyncio`.

## Key Files
| File | Purpose |
| :--- | :--- |
| **`start.py`** | **Smart Entry Point**. Environment detection and bot launcher. |
| **`main.py`** | **Core Logic**. Telegram bot handlers and job worker. |
| **`config.py`** | **Configuration**. Manages Secrets and Settings. |
| **`utils.py`** | **Utilities**. Gemini Transcription & Summarization logic. |
| **`bot_classes.py`**| **Data Structures**. `JobManager`, `FilesHandler` (with duration enforcement). |
| **`setup_uv.sh`** | **Installation**. Smart multi-requirements installer using `uv`. |

## Critical Workflows

### 1. Startup & Initialization
-   **`start.py`**: Checks `torch.cuda` -> Sets mode -> Runs `main.py`.
-   **`main.post_init`**: Adjusts idle timers (5x longer in Gemini mode) and notifies admins.

### 2. Transcription Pipeline
1.  **Receive**: `FilesHandler` checks limits (10 mins for Gemini mode).
2.  **Transcribe**: 
    - **WHISPER Mode**: Local processing via `faster-whisper`.
    - **GEMINI Mode**: Cloud processing via Google Gemini File API.
3.  **Immediate Result**: Send "Done" message + `TS_...` file.
4.  **AI Summary**: Call `utils.summarize_text`.
5.  **Final Result**: Send `AI_...` file.
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
