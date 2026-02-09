---
description: TTB (Telegram Transcription Bot) - Development workflow and project context
---

# 🤖 TTB (Telegram Transcription Bot) Agent Guide

## Project Overview

**TTB** is an advanced Telegram bot for audio/video transcription using **OpenAI Whisper** (local GPU via Colab) and **Google Gemini** for automatic summarization in Indonesian.

### Tech Stack
| Component | Technology |
|-----------|------------|
| Transcription | `faster-whisper` (Model: `large-v3`) |
| Summarization | `google-generativeai` (Gemini 2.5 Flash) |
| Bot Framework | `python-telegram-bot` |
| Web UI (Large Files) | `gradio` (handles files >20MB) |
| Package Manager | `uv` (~30s install vs 2+ min pip) |
| Runtime | Google Colab (T4 GPU) |

---

## 📁 File Structure

```
TTB/
├── main.py              # Entry point, Telegram handlers, job queue, Whisper init
├── utils.py             # Helper functions: summarize_text, format_duration, format_transcription
├── log_utils.py         # Centralized logging with timestamp + runtime format
├── gradio_handler.py    # Web interface for large files (>20MB bypass)
├── requirements.txt     # Python dependencies
├── setup_uv.sh          # Fast dependency installer using uv
└── TTBv1.py             # [Legacy] Monolithic version for reference
```

---

## 🏗️ Architecture

### Core Classes (main.py)

| Class | Purpose |
|-------|---------|
| `Config` | Configuration constants (env vars with defaults) |
| `TranscriptionJob` | Dataclass: job info (message_id, filepath, duration, status) |
| `JobManager` | Queue system (asyncio.Queue), registry, cancel/complete jobs |
| `IdleMonitor` | Auto-shutdown on idle (configurable: alert, warn, shutdown) |
| `FilesHandler` | File handling: single files, ZIP, multi-part .zip.001 archives |

### Key Functions

| Function | Location | Purpose |
|----------|----------|---------|
| `queue_processor()` | main.py | Worker loop: processes jobs from queue |
| `run_transcription_process()` | main.py | Blocking Whisper transcription (runs in thread) |
| `initialize_models_background()` | main.py | Loads Whisper + Gemini async |
| `summarize_text()` | utils.py | Gemini API call for Indonesian summary |
| `format_transcription_with_pauses()` | utils.py | Adds timestamps at speech pauses |
| `log()` | log_utils.py | Consistent logging: `[HH:MM:SS] [+Runtime] [CATEGORY]` |

---

## 🔧 Configuration (Environment Variables)

```bash
# Required
TELEGRAM_BOT_TOKEN      # Bot token from @BotFather
TELEGRAM_CHAT_ID        # Chat ID (bot only responds to this ID)

# Optional
GEMINI_API_KEY          # Google AI Studio key (for summarization)
HF_TOKEN                # Hugging Face token (avoid rate limits)
GITHUB_TOKEN            # For private repos

# Customizable (with defaults)
MODEL_SIZE              # Default: "large-v3"
BEAM_SIZE               # Default: 10
PAUSE_THRESHOLD         # Default: 0.3 (seconds)
MAX_AUDIO_DURATION_MINUTES  # Default: 90

# Idle Monitor
ENABLE_IDLE_MONITOR     # Default: "true"
IDLE_FIRST_ALERT_MINUTES    # Default: 1
IDLE_FINAL_WARNING_MINUTES  # Default: 2
IDLE_SHUTDOWN_MINUTES       # Default: 3
```

---

## 🚀 Running the Bot

### Google Colab (Recommended)
```python
# Set env vars in Colab Secrets, then run:
!git clone --depth 1 https://github.com/arinadi/TTB.git
%cd TTB
!bash setup_uv.sh
!python main.py
```

### Local Development
```bash
git clone https://github.com/arinadi/TTB.git && cd TTB
pip install -r requirements.txt  # or: bash setup_uv.sh
export TELEGRAM_BOT_TOKEN="..." TELEGRAM_CHAT_ID="..." GEMINI_API_KEY="..."
python main.py
```

---

## 📝 Development Guidelines

### Logging Convention
Use `log_utils.log()` for consistency:
```python
from log_utils import log

log("INIT", "Starting bot...")      # Categories: INIT, JOB, IDLE, WORKER
log("WHISPER", "Transcribing...")   # GEMINI, WHISPER, FILE, GRADIO, ERROR
```

Output format:
```
[HH:MM:SS] [+Xm XXs] [CATEGORY] message
```

### Adding New Features

1. **New Handler**: Add to `main.py` Section 6 (Telegram UI Commands)
2. **New Utility**: Add to `utils.py` with proper async/sync handling
3. **Gradio Feature**: Modify `gradio_handler.py`

### Testing Flow

1. Send audio/video file to Telegram bot
2. Bot will queue, transcribe, then send results (Transcript + Summary)
3. For files >20MB, use Gradio web UI

---

## 🔄 Common Tasks

// turbo-all

### Update Dependencies
```bash
# Edit requirements.txt, then:
pip install -r requirements.txt
```

### Debug Mode
```bash
# Check logs in console with format:
# [HH:MM:SS] [+Runtime] [CATEGORY] message
```

### Test Gradio Upload
1. Start bot: `python main.py`
2. Open Gradio URL (will appear in Telegram)
3. Upload file, results will be sent to Telegram

---

## ⚠️ Limits & Quotas

| Component | Type | Limit |
|-----------|------|-------|
| Whisper | Local (Colab GPU) | **Unlimited** |
| Hugging Face | Model Download | Rate limit only (use HF_TOKEN) |
| Google Gemini | Cloud API | Free tier quota (~15 RPM, 1500/day) |
| Telegram File | Bot Download | 20MB (bypass via Gradio) |

---

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| `TELEGRAM_BOT_TOKEN not set` | Set in Colab Secrets or env var |
| Whisper OOM | Use smaller model (`medium`, `small`) |
| Gemini rate limit | Reduce usage or upgrade API tier |
| File download failed | Check file size (<20MB), or use Gradio |
| Multipart ZIP error | Ensure all .zip.xxx parts are sent within COMBINE_TIMEOUT (30s) |
