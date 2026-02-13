# 🚀 Run Transcription Bot (Telegram Version - Modular)
# ------------------------------------------------------------------------------
# SECTION 1: CONFIGURATION AND SECRETS
# ------------------------------------------------------------------------------

# 🚀 Run Transcription Bot (Telegram Version - Modular)
# ------------------------------------------------------------------------------
# SECTION 1: IMPORT & CONFIGURATION
# ------------------------------------------------------------------------------

import sys
import os
import asyncio
import gc
import time
from typing import Optional

# --- Local Imports ---
import config
from config import Config
from utils import (
    summarize_text, 
    format_duration, 
    log, 
    get_runtime
)
from bot_classes import TranscriptionJob, IdleMonitor, JobManager, FilesHandler

# --- External Libraries ---
try:
    from faster_whisper import WhisperModel
    import torch

    from google import genai
    import telegram
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
    from telegram.constants import ParseMode
    from telegram.request import HTTPXRequest
    from telegram.ext import (Application, CallbackQueryHandler, CommandHandler,
                              ContextTypes, MessageHandler, filters)
    from werkzeug.utils import secure_filename
    import nest_asyncio
except ImportError as e:
    sys.exit(f"❌ Critical Dependency Missing: {e}\nPlease run: pip install -r requirements.txt")

# Optional: Gradio
try:
    import gradio_handler
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    log("INIT", "Gradio not available. Web interface disabled.")

# --- Secrets Alias ---
TELEGRAM_BOT_TOKEN = config.TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID = config.TELEGRAM_CHAT_ID
GEMINI_API_KEY = config.GEMINI_API_KEY

# Detect Colab
try:
    from google.colab import runtime
    IS_COLAB = True
except ImportError:
    IS_COLAB = False
    class MockRuntime:
        def unassign(self): print("🔌 Local Runtime Shutdown Executed")
    runtime = MockRuntime()

# Validation
if not all([TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID]):
    print("❌ ERROR: Core secrets (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID) are missing.")

if not GEMINI_API_KEY:
    print("⚠️ WARNING: GEMINI_API_KEY not set. Summarization features will be disabled.")

# Constants
TRANSCRIPT_FILENAME_PREFIX = "TS"
SUMMARY_FILENAME_PREFIX = "SU"

# ------------------------------------------------------------------------------
# SECTION 2: ENVIRONMENT SETUP
# ------------------------------------------------------------------------------

nest_asyncio.apply()

# --- Filesystem Setup ---
UPLOAD_FOLDER = 'uploads'
TRANSCRIPT_FOLDER = 'transcripts'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TRANSCRIPT_FOLDER, exist_ok=True)

# ------------------------------------------------------------------------------
# SECTION 3: AI AND HARDWARE INITIALIZATION
# ------------------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
fp16_enabled = str(Config.WHISPER_PRECISION).lower() == 'true' or (str(Config.WHISPER_PRECISION).lower() == 'auto' and device == 'cuda')

# Global State
model = None
gemini_client = None
models_ready_event = asyncio.Event()



# ------------------------------------------------------------------------------
# SECTION 5: GLOBAL OBJECTS & WORKER
# ------------------------------------------------------------------------------

# --- Global State Variables ---
application: Optional[Application] = None
idle_monitor: Optional[IdleMonitor] = None
job_manager: Optional[JobManager] = None
files_handler: Optional[FilesHandler] = None
SHUTDOWN_IN_PROGRESS = False

async def send_telegram_notification(app: Application, message: str):
    """Sends a formatted message to the designated admin chat."""
    try:
        await app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        log("ERROR", f"Telegram notification failed: {e}")

async def perform_shutdown(reason: str):
    """Notifies admins and safely terminates the Colab runtime."""
    global SHUTDOWN_IN_PROGRESS
    if SHUTDOWN_IN_PROGRESS:
        return
    SHUTDOWN_IN_PROGRESS = True
    uptime_str = get_runtime()
    log("SHUTDOWN", f"Initiated. Reason: {reason}")
    try:
        if application:
            await send_telegram_notification(application, f"🔌 *Shutdown*\nReason: {reason}\nUptime: `{uptime_str}`")
            log("SHUTDOWN", "Notification sent")
    except Exception as e:
        log("ERROR", f"Final notification failed: {e}")
    finally:
        log("SHUTDOWN", "Terminating runtime...")
        try:
            if runtime:
                runtime.unassign()
            else:
                log("SHUTDOWN", "Runtime object not found (local execution?)")
        except Exception as e:
            log("ERROR", f"Runtime shutdown failed: {e}")

async def initialize_models_background():
    """Loads Whisper and initializes Gemini client in a background task."""
    global model, gemini_client
    try:
        log("INIT", f"Loading Whisper ({Config.WHISPER_MODEL}, {device})...")
        # Logic for compute_type
        compute_type = "float16" if device == "cuda" else "int8"
        
        # User override logic (backward compatibility with 'True'/'False' strings)
        prec_cfg = str(Config.WHISPER_PRECISION).lower()
        if prec_cfg == 'false' or prec_cfg == 'float32':
            compute_type = "float32"
        elif prec_cfg == 'float16':
            compute_type = "float16"
        elif prec_cfg == 'int8':
            compute_type = "int8"

        model = await asyncio.to_thread(
            WhisperModel, 
            Config.WHISPER_MODEL, 
            device=device, 
            compute_type=compute_type
        )
        log("INIT", f"Whisper loaded ({compute_type})")
        
        if GEMINI_API_KEY:
            log("INIT", "Initializing Gemini...")
            gemini_client = genai.Client(api_key=GEMINI_API_KEY)
            log("INIT", "Gemini ready")

        models_ready_event.set()
        gemini_status = "✓" if gemini_client else "✗"

        await send_telegram_notification(application, f"✅ *Ready!* `{Config.WHISPER_MODEL}` loaded in `{get_runtime()}`\nGemini: {gemini_status}")
    except Exception as e:
        log("ERROR", f"Model loading failed: {e}")
        await send_telegram_notification(application, f"❌ *FATAL:* Model loading failed: {e}")
        await perform_shutdown("AI Model Loading Failed")

async def initialize_gradio_background():
    """Launches Gradio web server in background and notifies Telegram with pinned URL."""
    if not GRADIO_AVAILABLE:
        log("GRADIO", "Not available, skipping")
        return
    
    try:
        log("GRADIO", "Starting web interface...")
        main_loop = asyncio.get_running_loop()
        gradio_handler.set_dependencies(job_manager, UPLOAD_FOLDER, main_loop)
        public_url = await gradio_handler.launch_gradio_async(share=True)
        
        if public_url:
            log("GRADIO", f"Online: {public_url}")
            
            # Check if bot has permission to pin messages
            can_pin = False
            try:
                bot_member = await application.bot.get_chat_member(chat_id=TELEGRAM_CHAT_ID, user_id=application.bot.id)
                can_pin = bot_member.status in ["administrator", "creator"] and (
                    bot_member.status == "creator" or bot_member.can_pin_messages
                )
            except Exception:
                pass  # Silently fail permission check
            
            # Send Gradio URL message
            msg = await application.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=f"🌐 *Web UI*\n{public_url}\n_For files >20MB_",
                parse_mode=ParseMode.MARKDOWN
            )
            
            # Only attempt pin/unpin if bot has permission
            if can_pin:
                try:
                    await application.bot.unpin_all_chat_messages(chat_id=TELEGRAM_CHAT_ID)
                except Exception:
                    pass  # Ignore unpin errors (no previous pins is OK)
                
                try:
                    await application.bot.pin_chat_message(
                        chat_id=TELEGRAM_CHAT_ID,
                        message_id=msg.message_id,
                        disable_notification=True
                    )
                    log("GRADIO", "URL pinned")
                except Exception:
                    pass  # Silently skip if pin fails
        else:
            log("GRADIO", "Started but no public URL")
    except Exception as e:
        log("ERROR", f"Gradio failed: {e}")


def run_transcription_process(job: TranscriptionJob) -> tuple[str, str]:
    """Runs the blocking Whisper transcription in a separate thread."""
    # Note: This runs in a thread, so we use print directly (log_utils works here too)
    from utils import log
    log("WHISPER", f"[{job.job_id}] Transcribing {job.original_filename}...")
    
    transcribe_options = {
        "beam_size": Config.WHISPER_BEAM_SIZE,
        "patience": Config.WHISPER_PATIENCE,
        "temperature": Config.WHISPER_TEMPERATURE,
        "repetition_penalty": Config.WHISPER_REPETITION_PENALTY,
        "no_repeat_ngram_size": Config.WHISPER_NO_REPEAT_NGRAM_SIZE
    }
    
    # Run transcription
    # VAD parameters from user research
    vad_parameters = dict(
        threshold=Config.VAD_THRESHOLD,
        min_speech_duration_ms=Config.VAD_MIN_SPEECH_DURATION_MS,
        min_silence_duration_ms=Config.VAD_MIN_SILENCE_DURATION_MS,
        speech_pad_ms=Config.VAD_SPEECH_PAD_MS
    )
    
    segments_generator, info = model.transcribe(
        job.local_filepath, 
        vad_filter=Config.VAD_FILTER,
        vad_parameters=vad_parameters,
        **transcribe_options
    )
    
    # Convert generator to list to ensure full processing
    segments = list(segments_generator)
    
    # Use native formatting (Raw segments from Whisper)
    from utils import format_transcription_native
    formatted_text = format_transcription_native(segments)
    
    # Legacy pause-based formatting (disabled)
    # formatted_text = format_transcription_with_pauses(segments, Config.PAUSE_THRESHOLD)
    
    log("WHISPER", f"[{job.job_id}] Done: {len(segments)} segments, lang={info.language} ({info.language_probability:.0%})")
    
    return formatted_text, info.language if info.language else 'N/A'

async def queue_processor():
    """The main worker loop that processes jobs from the queue one by one."""
    log("WORKER", "Waiting for AI models...")
    await models_ready_event.wait()
    log("WORKER", "Models ready. Processing jobs...")
    while not SHUTDOWN_IN_PROGRESS:
        job: TranscriptionJob = await job_manager.job_queue.get()

        if job.status == 'cancelled':
            log("WORKER", f"[{job.job_id}] Skipped (cancelled)")
            if os.path.exists(job.local_filepath):
                os.remove(job.local_filepath)
            job_manager.job_queue.task_done()
            job_manager.complete_job(job.job_id)
            continue

        job_manager.set_processing_job(job)
        try:
            duration_str = format_duration(job.audio_duration)
            await application.bot.send_message(job.chat_id, f"▶️ Processing `{job.original_filename}` ({duration_str})...", parse_mode=ParseMode.MARKDOWN)
            start_time = time.time()

            transcript_text, detected_language = await asyncio.to_thread(run_transcription_process, job)
            if job.status == 'cancelled': raise asyncio.CancelledError("Job cancelled during transcription.")

            base_name = os.path.splitext(job.original_filename)[0]
            safe_name = secure_filename(base_name)[:50]
            ts_filename = f"{TRANSCRIPT_FILENAME_PREFIX}_({duration_str.replace(' ', '')})_{safe_name}.txt"
            ts_filepath = os.path.join(TRANSCRIPT_FOLDER, ts_filename)
            with open(ts_filepath, "w", encoding="utf-8") as f:
                f.write(transcript_text)

            # PASSING GEMINI CLIENT HERE
            summary_text = await summarize_text(transcript_text, gemini_client)
            
            if job.status == 'cancelled': raise asyncio.CancelledError("Job cancelled during summarization.")
            su_filename = f"{SUMMARY_FILENAME_PREFIX}_({duration_str.replace(' ', '')})_{safe_name}.txt"
            su_filepath = os.path.join(TRANSCRIPT_FOLDER, su_filename)
            with open(su_filepath, "w", encoding="utf-8") as f:
                f.write(summary_text)

            processing_duration_str = format_duration(time.time() - start_time)
            log("JOB", f"[{job.job_id}] Done in {processing_duration_str}")
            result_text = (f"✅ *Done!* `{job.original_filename}`\n"
                           f"⏱️ {duration_str} audio → {processing_duration_str} process\n"
                           f"🌐 Lang: {detected_language.upper()}")

            await application.bot.send_message(job.chat_id, result_text, parse_mode=ParseMode.MARKDOWN, reply_to_message_id=job.message_id)
            with open(su_filepath, 'rb') as su_file:
                await application.bot.send_document(job.chat_id, document=su_file, filename=su_filename, reply_to_message_id=job.message_id)
            with open(ts_filepath, 'rb') as ts_file:
                await application.bot.send_document(job.chat_id, document=ts_file, filename=ts_filename, reply_to_message_id=job.message_id)
            job.status = "completed"

        except asyncio.CancelledError as e:
            log("WORKER", f"[{job.job_id}] Aborted: {e}")
        except Exception as e:
            job.status = "failed"
            log("ERROR", f"[{job.job_id}] {e}")
            await application.bot.send_message(job.chat_id, f"❌ *Failed:* `{job.original_filename}`\n`{e}`", parse_mode=ParseMode.MARKDOWN, reply_to_message_id=job.message_id)
        finally:
            if os.path.exists(job.local_filepath):
                os.remove(job.local_filepath)
            
            if 'transcript_text' in locals():
                del transcript_text
                gc.collect()

            job_manager.job_queue.task_done()
            job_manager.complete_job(job.job_id)

# ------------------------------------------------------------------------------
# SECTION 6: TELEGRAM UI COMMANDS
# ------------------------------------------------------------------------------

async def get_status_text_and_keyboard():
    """Builds the dynamic status message text and keyboard."""
    processing_job = job_manager.currently_processing
    if processing_job:
        processing_line = f"▶️ `{processing_job.original_filename}` (by {processing_job.author_display_name})\n"
    else:
        processing_line = ""

    ai_status = "✅" if models_ready_event.is_set() else "⏳"
    text = (
        f"📊 *Status*\n"
        f"{processing_line}"
        f"⏳ Uptime: `{get_runtime()}` | Queue: `{job_manager.job_queue.qsize()}`\n"
        f"🤖 AI: {ai_status}"
    )
    keyboard = [[InlineKeyboardButton("📄 Jobs", callback_data="view_cancel_jobs"), InlineKeyboardButton("🔄", callback_data="refresh_status"), InlineKeyboardButton("🔌", callback_data="shutdown_bot")]]

    return text, InlineKeyboardMarkup(keyboard)

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text, reply_markup = await get_status_text_and_keyboard()
    await update.effective_message.reply_text(text, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN)

async def queue_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    processing_job = job_manager.currently_processing
    queued_jobs = job_manager.get_queued_jobs()
    lines = ["📄 *Job Queue*\n"]
    if processing_job:
        lines.append(f"\n▶️ *Currently Processing*\n`{processing_job.original_filename}`\n(By: {processing_job.author_display_name})")
    if queued_jobs:
        queue_text = [f"*{i}.* `{job.original_filename}` (By: {job.author_display_name})" for i, job in enumerate(queued_jobs, 1)]
        lines.append(f"\n⏳ *In Queue ({len(queued_jobs)})*\n" + "\n".join(queue_text))
    elif not processing_job:
        lines.append("\nThe queue is empty.")
    await update.effective_message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)

async def extend_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not Config.ENABLE_IDLE_MONITOR:
        await update.effective_message.reply_text("Idle monitor disabled.")
        return
    msg = "✅ +5m extended" if idle_monitor.extend_timer(5) else "ℹ️ Bot active, no timer."
    await update.effective_message.reply_text(msg)

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data

    if data == "refresh_status":
        text, reply_markup = await get_status_text_and_keyboard()
        try:
            await query.edit_message_text(text, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN)
        except telegram.error.BadRequest:
            pass
    elif data == "shutdown_bot":
        await query.edit_message_text("🔴 *MANUAL SHUTDOWN INITIATED...*", parse_mode=ParseMode.MARKDOWN)
        await perform_shutdown(f"Manual Shutdown by {query.from_user.first_name}")
    elif data == "view_cancel_jobs":
        queued_jobs = job_manager.get_queued_jobs()
        if not queued_jobs:
            await query.edit_message_text("The queue is empty.", reply_markup=None)
            return
        keyboard = [[InlineKeyboardButton(f"{job.original_filename[:40]}... (ID: {job.job_id})", callback_data=f"cancel_{job.job_id}")] for job in queued_jobs]
        keyboard.append([InlineKeyboardButton("« Back to Status", callback_data="refresh_status")])
        await query.edit_message_text("Select a job below to cancel it:", reply_markup=InlineKeyboardMarkup(keyboard))
    elif data.startswith("cancel_"):
        job_id = data.split("_")[1]
        cancelled, job_name = await job_manager.cancel_job(job_id)
        msg = f"✅ Job `{job_name}` was cancelled." if cancelled else "❌ Could not cancel job."
        await query.edit_message_text(msg, reply_markup=None, parse_mode=ParseMode.MARKDOWN)
    elif data == "extend_idle":
        # Rate limit check (5 minutes = 300 seconds)
        if time.time() - idle_monitor.last_extend_time < 300:
            await query.answer("⏳ Please wait 5 minutes before extending again.", show_alert=True)
            return
        
        if idle_monitor.extend_timer(5):
            idle_monitor.last_extend_time = time.time()
            new_text = f"✅ *Idle Extended*\nTimer added +5 minutes.\n_Action by {query.from_user.first_name}_"
            await query.edit_message_text(new_text, parse_mode=ParseMode.MARKDOWN)
        else:
            await query.edit_message_text("ℹ️ Bot is already active, no need to extend.", parse_mode=ParseMode.MARKDOWN)

# ------------------------------------------------------------------------------
# SECTION 7: MAIN ENTRY POINT
# ------------------------------------------------------------------------------

async def main():
    global application, idle_monitor, job_manager, files_handler

    print("🚀 Starting Main Application...")

    # Longer timeouts and connection pool for network resilience
    request = HTTPXRequest(
        read_timeout=60.0, 
        connect_timeout=20.0,
        write_timeout=30.0,
        pool_timeout=30.0,
        connection_pool_size=8
    )
    
    if not TELEGRAM_BOT_TOKEN:
        sys.exit("❌ FATAL: No TELEGRAM_BOT_TOKEN found. Exiting.")

    async def post_init(application: Application):
        """Initializes background tasks after the application is ready."""
        log("INIT", "Running post-init tasks...")
        
        # Background Tasks - start AFTER bot is ready to receive
        application.create_task(queue_processor())
        application.create_task(initialize_models_background())
        
        # Start Gradio web interface (async, like AI models)
        if GRADIO_AVAILABLE:
            application.create_task(initialize_gradio_background())
        
        if Config.ENABLE_IDLE_MONITOR:
            idle_monitor.start()

        # Send startup notification in background (non-blocking)
        startup_message = (
            f"🚀 *Bot Online*\n\n"
            f"📌 Model: `{Config.WHISPER_MODEL}` | Device: `{device.upper()}`\n"
            f"📂 Max file: `{Config.BOT_FILESIZE_LIMIT}MB`\n"
            f"🖥️ Web UI: Loading..."
        )
        asyncio.create_task(send_telegram_notification(application, startup_message))

    # Build Application with post_init hook
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).request(request).post_init(post_init).build()

    # Initialize components
    idle_monitor = IdleMonitor(application, None, perform_shutdown)
    job_manager = JobManager(application, idle_monitor, models_ready_event)
    idle_monitor.job_manager = job_manager
    files_handler = FilesHandler(job_manager, UPLOAD_FOLDER)
    
    # Filter for approved chat only
    chat_filter = filters.Chat(chat_id=TELEGRAM_CHAT_ID)

    # Handlers
    application.add_handler(CommandHandler(["start", "status"], status_command, filters=chat_filter))
    application.add_handler(CommandHandler("queue", queue_command, filters=chat_filter))
    application.add_handler(CommandHandler("extend", extend_command, filters=chat_filter))
    application.add_handler(CallbackQueryHandler(button_callback))
    application.add_handler(MessageHandler(filters.ATTACHMENT & chat_filter, files_handler.handle_files))
    
    
    # Error Handler with retry tracking
    _transient_error_counts = {}  # Track consecutive transient errors
    MAX_TRANSIENT_RETRIES = 2
    
    async def global_error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):

        error = context.error
        error_name = type(error).__name__
        print(f"❌ Exception while handling an update: {error_name}: {error}")
        
        # List of transient network/connection errors that should NOT trigger shutdown
        transient_errors = (
            # httpx errors
            'ReadError', 'WriteError', 'ConnectError', 'ConnectTimeout', 'ReadTimeout', 'WriteTimeout',
            'PoolTimeout', 'CloseError', 'ProxyError', 'ProtocolError', 'RemoteProtocolError',
            'LocalProtocolError', 'UnsupportedProtocol', 'DecodingError', 
            # SSL errors
            'SSLError', 'SSLCertVerificationError',
            # Telegram-bot errors  
            'TimeoutException', 'NetworkError', 'TimedOut', 'RetryAfter', 'Forbidden',
            # General connection
            'ConnectionError', 'ConnectionResetError', 'ConnectionRefusedError', 'BrokenPipeError',
            'OSError', 'IOError', 'socket.error', 'socket.timeout'
        )
        
        if error_name in transient_errors:
            # Track retry count
            _transient_error_counts[error_name] = _transient_error_counts.get(error_name, 0) + 1
            count = _transient_error_counts[error_name]
            
            if count <= MAX_TRANSIENT_RETRIES:
                print(f"⚠️ [ERROR_HANDLER] Transient error {error_name} ({count}/{MAX_TRANSIENT_RETRIES}) - will retry")
                return  # Don't shutdown, let telegram-bot retry
            else:
                print(f"🔴 [ERROR_HANDLER] Transient error {error_name} exceeded {MAX_TRANSIENT_RETRIES} retries - network may be unstable")
                _transient_error_counts[error_name] = 0  # Reset counter
                return  # Still don't shutdown, but log critical warning
        
        # Reset counters on non-transient error
        _transient_error_counts.clear()
        
        # Notify user if possible (wrapped in try-except)
        try:
            if update and isinstance(update, Update) and update.effective_message:
                text = f"❌ *An error occurred:* `{error}`"
                await update.effective_message.reply_text(text, parse_mode=ParseMode.MARKDOWN)
        except Exception as notify_err:
            print(f"⚠️ [ERROR_HANDLER] Could not send error notification: {notify_err}")
        
        # Trigger safe shutdown only for critical errors
        await perform_shutdown(f"Application Error: {error}")

    application.add_error_handler(global_error_handler)

    # ⚡ FAST INIT: Initialize bot connection FIRST (before background tasks)
    # await application.initialize() -> Managed by run_polling
    # log("INIT", f"Bot online ({get_runtime()})")

    # Run polling - bot starts receiving messages immediately
    await application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("🛑 Bot stopped by user.")
    except Exception as e:
        print(f"❌ Application crashed: {e}")
        # Attempt to notify via Telegram if possible
        if 'application' in globals() and application:
            try:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(send_telegram_notification(application, f"❌ *CRASH REPORT:*\nBot crashed with error: `{e}`"))
            except Exception:
                pass
    finally:
        if IS_COLAB:
            print("🔌 Triggering Colab Runtime Shutdown (Error Safe-mode)...")
            runtime.unassign()
