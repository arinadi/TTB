# 🚀 Run Transcription Bot (Telegram Version - Modular)
# ------------------------------------------------------------------------------
# SECTION 1: CONFIGURATION AND SECRETS
# ------------------------------------------------------------------------------

import sys
import os
import asyncio
import gc
import re
import shutil
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from typing import List, Optional

# --- Global Initialization Timing ---
# This tries to get the start time from Colab's README environment variable.
# If running locally (variable missing), it falls back to current time.
INIT_START = float(os.getenv('INIT_START', time.time()))

# External Layouts
# Note: These must be installed via requirements.txt
try:
    import utils
    from faster_whisper import WhisperModel
    import torch
    import ffmpeg
    from google import genai
    import telegram
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
    from telegram.constants import ParseMode
    from telegram.ext import (Application, CallbackQueryHandler, CommandHandler,
                              ContextTypes, MessageHandler, filters)
    from telegram.request import HTTPXRequest
    from werkzeug.utils import secure_filename
    import nest_asyncio
except ImportError as e:
    sys.exit(f"❌ Critical Dependency Missing: {e}\nPlease run: pip install -r requirements.txt")

# Local Imports
from utils import summarize_text, format_duration, format_transcription_with_pauses

# Optional: Gradio for large file uploads
try:
    import gradio_handler
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    print("⚠️ gradio_handler not available. Web interface disabled.")


# --- 1.1. Load Core Secrets (from environment - set by Colab runner) ---
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

# Detect Colab runtime for shutdown functionality
try:
    from google.colab import runtime
    IS_COLAB = True
except ImportError:
    IS_COLAB = False
    # Mock runtime for local testing
    class MockRuntime:
        def unassign(self): print("🔌 Local Runtime Shutdown Executed")
    runtime = MockRuntime()

if not all([TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID]):
    # Note: For local testing, ensure these env vars are set. 
    # For Colab, ensure they are in the Secrets tab.
    print("❌ ERROR: Core secrets (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID) are missing.")
    # We don't exit here to allow import checks, but main() will fail.

if TELEGRAM_CHAT_ID:
    TELEGRAM_CHAT_ID = int(TELEGRAM_CHAT_ID)

if not GEMINI_API_KEY:
    print("⚠️ WARNING: GEMINI_API_KEY not set. Summarization features will be disabled.")

# --- 1.2. Bot & Model Configuration ---
# Hardcoded defaults for the script version. 
# In Colab, these were widgets. Now they are constants.
class Config:
    MODEL_SIZE = os.getenv('MODEL_SIZE', 'large-v3')
    USE_FP16 = os.getenv('USE_FP16', 'auto') # 'auto', 'True', 'False'
    BEAM_SIZE = int(os.getenv('BEAM_SIZE', 10))
    PAUSE_THRESHOLD = float(os.getenv('PAUSE_THRESHOLD', 0.3))
    MAX_AUDIO_DURATION_MINUTES = int(os.getenv('MAX_AUDIO_DURATION_MINUTES', 90))
    
    ENABLE_IDLE_MONITOR = os.getenv('ENABLE_IDLE_MONITOR', 'True').lower() == 'true'
    IDLE_FIRST_ALERT_MINUTES = int(os.getenv('IDLE_FIRST_ALERT_MINUTES', 1))
    IDLE_FINAL_WARNING_MINUTES = int(os.getenv('IDLE_FINAL_WARNING_MINUTES', 2))
    IDLE_SHUTDOWN_MINUTES = int(os.getenv('IDLE_SHUTDOWN_MINUTES', 3))
    
    BOT_FILESIZE_LIMIT = int(os.getenv('BOT_FILESIZE_LIMIT', 20))

# --- 1.3. Derived Constants ---
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
fp16_enabled = str(Config.USE_FP16).lower() == 'true' or (str(Config.USE_FP16).lower() == 'auto' and device == 'cuda')

# Global State
model = None
gemini_client = None
models_ready_event = asyncio.Event()

# ------------------------------------------------------------------------------
# SECTION 4: DATA CLASSES & MANAGERS
# ------------------------------------------------------------------------------

def get_runtime() -> str:
    """Calculates and formats the total runtime since INIT_START."""
    return format_duration(time.time() - INIT_START)

@dataclass
class TranscriptionJob:
    """A data class to hold all information about a single transcription job."""
    message_id: int
    chat_id: int
    original_filename: str
    local_filepath: str
    audio_duration: float
    author_display_name: str = field(init=False)
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    status: str = "queued"
    _original_message: telegram.Message = field(repr=False, init=False)

    @classmethod
    def from_message(cls, message: telegram.Message, local_path: str, duration: float):
        job = cls(
            message_id=message.message_id,
            chat_id=message.chat_id,
            original_filename=message.effective_attachment.file_name or "unknown_file",
            local_filepath=local_path,
            audio_duration=duration
        )
        job._original_message = message
        if message.from_user:
            job.author_display_name = message.from_user.first_name
        elif message.chat and message.chat.title:
            job.author_display_name = message.chat.title
        else:
            job.author_display_name = "Unknown"
        print(f"[JOB:{job.job_id}] New job created for '{job.original_filename}' by {job.author_display_name}.")
        return job

class IdleMonitor:
    """Monitors bot activity and triggers alerts or shutdown when idle.
    
    Timeline with Config (Notify=1, Warn=5, Shutdown=10):
    - [0m]  Bot idle → shutdown_on = now + 10 minutes
    - [1m]  elapsed=1 → First Alert sent (with Extend button)
    - [5m]  elapsed=5 → Final Warning sent
    - [10m] elapsed=10, remaining=0 → Shutdown
    
    extend_timer() adds minutes to shutdown_on, delaying shutdown.
    """
    def __init__(self, app: Application, job_manager: "JobManager"):
        self.app = app
        self.job_manager = job_manager
        self.shutdown_on: Optional[float] = None  # Absolute timestamp for shutdown
        self.shutdown_imminent = False
        self.alerts_sent = {'first_alert': False, 'final_warning': False}
        self.last_extend_time = 0
        self._task: Optional[asyncio.Task] = None
        print("✅ IdleMonitor initialized.")
        print(f"DEBUG: Idle Config -> First Alert at {Config.IDLE_FIRST_ALERT_MINUTES}m, Final Warning at {Config.IDLE_FINAL_WARNING_MINUTES}m, Shutdown at {Config.IDLE_SHUTDOWN_MINUTES}m")

    def start(self):
        if not self._task or self._task.done():
            self._task = asyncio.create_task(self._monitor_loop())
            print("✅ IdleMonitor task started.")

    def stop(self):
        if self._task:
            self._task.cancel()

    def reset(self):
        if self.shutdown_on is not None:
            print(f"[{get_runtime()}] [IDLE_MONITOR] Bot is active. Timer reset.")
            self.shutdown_on = None
            self.alerts_sent = {'first_alert': False, 'final_warning': False}

    def extend_timer(self, minutes: int) -> bool:
        """Extend shutdown time by adding minutes to shutdown_on."""
        if self.shutdown_on is not None:
            self.shutdown_on += (minutes * 60)
            # Reset alerts so user gets them again with new timing
            self.alerts_sent = {'first_alert': False, 'final_warning': False}
            remaining = (self.shutdown_on - time.time()) / 60
            print(f"[{get_runtime()}] [IDLE_MONITOR] Timer extended +{minutes}m. Shutdown in {remaining:.1f}m.")
            return True
        return False

    async def _monitor_loop(self):
        while True:
            await asyncio.sleep(60)
            
            # Skip if not enabled or shutdown already in progress
            if self.shutdown_imminent or not Config.ENABLE_IDLE_MONITOR:
                continue
            
            # Wait for job_manager to be initialized
            if self.job_manager is None:
                print("[IDLE_MONITOR] Waiting for job_manager...")
                continue

            try:
                if self.job_manager.is_idle():
                    # First time idle - set shutdown_on (absolute time)
                    if self.shutdown_on is None:
                        self.shutdown_on = time.time() + (Config.IDLE_SHUTDOWN_MINUTES * 60)
                        print(f"[{get_runtime()}] [IDLE_MONITOR] Bot is now idle. Shutdown in {Config.IDLE_SHUTDOWN_MINUTES}m.")

                    # Calculate times
                    remaining_minutes = (self.shutdown_on - time.time()) / 60
                    elapsed_minutes = Config.IDLE_SHUTDOWN_MINUTES - remaining_minutes
                    print(f"[{get_runtime()}] [IDLE_MONITOR] Elapsed: {elapsed_minutes:.1f}m, Remaining: {remaining_minutes:.1f}m")
                    
                    # 1. FIRST ALERT when elapsed >= IDLE_FIRST_ALERT_MINUTES (e.g., 1 min idle)
                    if elapsed_minutes >= Config.IDLE_FIRST_ALERT_MINUTES and not self.alerts_sent['first_alert']:
                        alert_msg = f"ℹ️ *IDLE ALERT:*\nBot is idle. Shutdown in *{int(remaining_minutes)}* minutes."
                        keyboard = [[InlineKeyboardButton("⏳ Extend Timer (+5 min)", callback_data="extend_idle")]]
                        await self.app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=alert_msg, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.MARKDOWN)
                        self.alerts_sent['first_alert'] = True
                        print(f"[{get_runtime()}] [IDLE_MONITOR] First alert sent.")
                    
                    # 2. FINAL WARNING when elapsed >= IDLE_FINAL_WARNING_MINUTES (e.g., 5 min idle)
                    if elapsed_minutes >= Config.IDLE_FINAL_WARNING_MINUTES and not self.alerts_sent['final_warning']:
                        warn_msg = f"⚠️ *FINAL WARNING:*\nBot will shut down in approx. *{int(remaining_minutes)}* minutes!"
                        await send_telegram_notification(self.app, warn_msg)
                        self.alerts_sent['final_warning'] = True
                        print(f"[{get_runtime()}] [IDLE_MONITOR] Final warning sent.")
                    
                    # 3. SHUTDOWN when remaining <= 0
                    if remaining_minutes <= 0:
                        self.shutdown_imminent = True
                        shutdown_msg = f"🔴 *AUTO SHUTDOWN:*\nBot has been idle for over *{Config.IDLE_SHUTDOWN_MINUTES}* minutes."
                        await send_telegram_notification(self.app, shutdown_msg)
                        print(f"[{get_runtime()}] [IDLE_MONITOR] Shutdown triggered.")
                        await perform_shutdown("Automatic Idle Shutdown")
                else:
                    self.reset()
            except Exception as e:
                print(f"[{get_runtime()}] [IDLE_MONITOR] Error in loop: {e}")

class JobManager:
    """Manages the queue and state of all transcription jobs."""
    def __init__(self, app: Application, idle_monitor_ref: IdleMonitor):
        self.app = app
        self.idle_monitor = idle_monitor_ref
        self.job_queue = asyncio.Queue()
        self.currently_processing: Optional[TranscriptionJob] = None
        self.job_registry: dict[str, TranscriptionJob] = {}
        print("✅ JobManager initialized.")

    async def add_job(self, job: TranscriptionJob):
        self.idle_monitor.reset()
        self.job_registry[job.job_id] = job
        await self.job_queue.put(job)
        queue_position = self.job_queue.qsize()
        model_status_note = "\n\n_(Note: AI models are still initializing...)_" if not models_ready_event.is_set() else ""
        content = (f"✅ `[ID: {job.job_id}]` File `{job.original_filename}` added to queue (Position: *#{queue_position}*).{model_status_note}")
        
        # Add simpler cancel button
        keyboard = [[InlineKeyboardButton("❌", callback_data=f"cancel_{job.job_id}")]]
        
        await self.app.bot.send_message(job.chat_id, content, parse_mode=ParseMode.MARKDOWN, reply_to_message_id=job.message_id, reply_markup=InlineKeyboardMarkup(keyboard))
        print(f"[JOB:{job.job_id}] Added to queue at position {queue_position}.")

    def complete_job(self, job_id: str):
        if self.currently_processing and self.currently_processing.job_id == job_id:
            self.currently_processing = None
        if job_id in self.job_registry:
            del self.job_registry[job_id]
        print(f"[JOB:{job_id}] Job completed and removed from registry.")
        self.idle_monitor.reset()

    def set_processing_job(self, job: TranscriptionJob):
        self.currently_processing = job
        job.status = "processing"
        print(f"[JOB:{job.job_id}] Status set to 'processing'.")
        self.idle_monitor.reset()

    async def cancel_job(self, job_id: str) -> tuple[bool, str]:
        job = self.job_registry.get(job_id)
        if not job:
            return False, "Unknown"
        job.status = "cancelled"
        print(f"[JOB:{job.job_id}] Status set to 'cancelled'.")
        return True, job.original_filename

    def is_idle(self) -> bool:
        return self.job_queue.empty() and self.currently_processing is None

    def get_queued_jobs(self) -> List[TranscriptionJob]:
        return [job for job in self.job_registry.values() if job.status == 'queued']

class FilesHandler:
    """Handles all incoming file attachments, including multi-part ZIP archives."""
    COMBINE_TIMEOUT_SECONDS = 30

    def __init__(self, job_manager: JobManager, upload_folder: str):
        self.job_manager = job_manager
        self.upload_folder = upload_folder
        self.multipart_archives = {} 
        self.multipart_pattern = re.compile(r'(.+)\.(zip|z)\.(\d{2,3})$', re.IGNORECASE)
        print("✅ FilesHandler initialized with multi-part ZIP support.")

    async def _validate_and_queue_file(self, local_path: str, message: telegram.Message, filename_override: str = None):
        try:
            original_filename = filename_override or message.effective_attachment.file_name or "unknown_file"
            probe = await asyncio.to_thread(ffmpeg.probe, local_path)
            duration = float(probe['format']['duration'])

            max_seconds = Config.MAX_AUDIO_DURATION_MINUTES * 60
            if Config.MAX_AUDIO_DURATION_MINUTES > 0 and duration > max_seconds:
                error_msg = f"File duration ({format_duration(duration)}) exceeds the maximum limit ({format_duration(max_seconds)})."
                await message.reply_text(f"❌ Could not process `{original_filename}`. *Reason:* {error_msg}", parse_mode=ParseMode.MARKDOWN)
                if os.path.exists(local_path): os.remove(local_path)
                return

            job_message = message
            if filename_override:
                fake_attachment = type('obj', (object,), {'file_name': original_filename})
                job_message = type('obj', (object,), {
                    'message_id': message.message_id, 'chat_id': message.chat_id,
                    'from_user': message.from_user, 'chat': message.chat,
                    'effective_attachment': fake_attachment
                })

            job = TranscriptionJob.from_message(job_message, local_path, duration)
            await self.job_manager.add_job(job)

        except Exception as e:
            filename_str = f"`{filename_override or 'the uploaded file'}`"
            await message.reply_text(f"❌ Failed to process {filename_str}. The file may be corrupt or in an unsupported format.")
            print(f"Error validating file {filename_str}: {e}", file=sys.stderr)
            if os.path.exists(local_path): os.remove(local_path)

    async def handle_files(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        message = update.effective_message
        if not (message and message.effective_attachment): return

        attachment = message.effective_attachment

        if attachment.file_size and attachment.file_size > (Config.BOT_FILESIZE_LIMIT * 1024 * 1024):
            file_size_mb = attachment.file_size / (1024 * 1024)
            await message.reply_text(
                f"❌ *File Too Large*\n\n"
                f"The file `{attachment.file_name}` ({file_size_mb:.2f} MB) exceeds the bot's download limit of "
                f"*{Config.BOT_FILESIZE_LIMIT} MB*. Please send a smaller file.",
                parse_mode=ParseMode.MARKDOWN
            )
            return

        file_obj = await attachment.get_file()
        original_filename = attachment.file_name or f"file_{int(time.time())}"

        multipart_match = self.multipart_pattern.match(original_filename)
        if multipart_match:
            await self._handle_multipart_chunk(message, file_obj, multipart_match)
        elif original_filename.lower().endswith('.zip'):
            local_path = os.path.join(self.upload_folder, f"{uuid.uuid4().hex}_{secure_filename(original_filename)}")
            await file_obj.download_to_drive(local_path)
            await self._extract_and_queue_zip(local_path, original_filename, message)
        else:
            local_path = os.path.join(self.upload_folder, f"{uuid.uuid4().hex}_{secure_filename(original_filename)}")
            await file_obj.download_to_drive(local_path)
            await self._validate_and_queue_file(local_path, message)

    async def _handle_multipart_chunk(self, message: telegram.Message, file_obj: telegram.File, match: re.Match):
        base_name, original_filename = match.group(1), message.effective_attachment.file_name
        local_path = os.path.join(self.upload_folder, f"{uuid.uuid4().hex}_{secure_filename(original_filename)}")
        await file_obj.download_to_drive(local_path)
        print(f"Received multipart chunk: {original_filename} (saved to {local_path}) for base '{base_name}'")

        loop = asyncio.get_running_loop()
        archive_data = self.multipart_archives.get(base_name)

        if archive_data:
            archive_data['files'].append((local_path, original_filename))
            if archive_data.get('timer'):
                archive_data['timer'].cancel()
            archive_data['timer'] = loop.call_later(self.COMBINE_TIMEOUT_SECONDS, lambda: asyncio.create_task(self._process_multipart_archive(base_name)))
            await message.reply_text(f"✅ Part {len(archive_data['files'])} for `{base_name}` received. Timer reset.", parse_mode=ParseMode.MARKDOWN)
        else:
            status_message = await message.reply_text(
                f"ℹ️ Received the first part of archive `{base_name}`.\n"
                f"Send the other parts. I will wait {self.COMBINE_TIMEOUT_SECONDS} seconds after the last file is received.",
                parse_mode=ParseMode.MARKDOWN
            )
            self.multipart_archives[base_name] = {
                'files': [(local_path, original_filename)],
                'message': status_message,
                'original_message': message,
                'timer': loop.call_later(self.COMBINE_TIMEOUT_SECONDS, lambda: asyncio.create_task(self._process_multipart_archive(base_name)))
            }

    async def _process_multipart_archive(self, base_name: str):
        archive_data = self.multipart_archives.pop(base_name, None)
        if not archive_data:
            return

        status_message = archive_data['message']
        file_tuples = archive_data['files']
        combined_zip_path = os.path.join(self.upload_folder, f"combined_{secure_filename(base_name)}.zip")
        combined_zip_name = f"{base_name}.zip"

        if len(file_tuples) < 1:
            await status_message.edit_text(f"❌ No files were available to combine for `{base_name}`.")
            return

        await status_message.edit_text(f"⏳ Combining {len(file_tuples)} parts for `{base_name}`...", parse_mode=ParseMode.MARKDOWN)

        file_tuples.sort(key=lambda t: int(self.multipart_pattern.search(t[1]).group(3)))
        sorted_file_paths = [t[0] for t in file_tuples]

        try:
            with open(combined_zip_path, 'wb') as outfile:
                for chunk_path in sorted_file_paths:
                    with open(chunk_path, 'rb') as infile:
                        shutil.copyfileobj(infile, outfile)

            print(f"Successfully combined into: {combined_zip_path}")
            await status_message.edit_text(f"✅ Combination complete! Now extracting `{combined_zip_name}`...", parse_mode=ParseMode.MARKDOWN)
            await self._extract_and_queue_zip(combined_zip_path, combined_zip_name, archive_data['original_message'], status_message)

        except Exception as e:
            await status_message.edit_text(f"❌ Failed to combine archive `{base_name}`: {e}", parse_mode=ParseMode.MARKDOWN)
            print(f"Error combining archive '{base_name}': {e}", file=sys.stderr)
        finally:
            for chunk_path in sorted_file_paths:
                if os.path.exists(chunk_path):
                    os.remove(chunk_path)

    async def _extract_and_queue_zip(self, zip_path: str, zip_name: str, message: telegram.Message, status_message: telegram.Message = None):
        extract_dir = os.path.join(self.upload_folder, f"extract_{uuid.uuid4().hex[:8]}")
        status_message = status_message or await message.reply_text(f"🗜️ Extracting `{zip_name}`...", parse_mode=ParseMode.MARKDOWN)
        queued_files_count = 0
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                file_list = [f for f in zip_ref.namelist() if not f.startswith(('__MACOSX', '.')) and not f.endswith('/')]
                if not file_list:
                    await status_message.edit_text(f"⚠️ No valid files found inside `{zip_name}`."); return

                await status_message.edit_text(f"Found {len(file_list)} files in `{zip_name}`. Validating and queueing...", parse_mode=ParseMode.MARKDOWN)
                await asyncio.to_thread(zip_ref.extractall, extract_dir)

            for root, _, files in os.walk(extract_dir):
                for filename in files:
                    if filename.startswith('.'): continue
                    source_path = os.path.join(root, filename)
                    dest_path = os.path.join(self.upload_folder, f"{uuid.uuid4().hex}_{secure_filename(filename)}")
                    shutil.move(source_path, dest_path)
                    await self._validate_and_queue_file(dest_path, message, filename_override=filename)
                    queued_files_count += 1

            await status_message.edit_text(f"✅ Finished processing `{zip_name}`. Added {queued_files_count} files to the queue.", parse_mode=ParseMode.MARKDOWN)
        except zipfile.BadZipFile:
            await status_message.edit_text(f"❌ Failed: `{zip_name}` is not a valid ZIP archive.", parse_mode=ParseMode.MARKDOWN)
        except Exception as e:
            await status_message.edit_text(f"❌ Error while extracting `{zip_name}`: `{e}`", parse_mode=ParseMode.MARKDOWN)
        finally:
            if os.path.exists(extract_dir): shutil.rmtree(extract_dir)
            if os.path.exists(zip_path): os.remove(zip_path)

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
        print(f"⚠️ Could not send Telegram notification: {e}", file=sys.stderr)

async def perform_shutdown(reason: str):
    """Notifies admins and safely terminates the Colab runtime."""
    global SHUTDOWN_IN_PROGRESS
    if SHUTDOWN_IN_PROGRESS: return
    SHUTDOWN_IN_PROGRESS = True
    uptime_str = get_runtime()
    print(f"[{uptime_str}] 🛑 SHUTDOWN INITIATED. Reason: {reason}")
    try:
        if application:
            await send_telegram_notification(application, f"🔌 *Bot is shutting down.*\nReason: {reason}\nRuntime: `{uptime_str}`")
            print(f"[{uptime_str}] ✅ Shutdown notification sent.")
    except Exception as e:
        print(f"⚠️ Could not send final notification, but shutting down anyway: {e}", file=sys.stderr)
    finally:
        print(f"[{uptime_str}] 🔌 Terminating Runtime...")
        runtime.unassign()

async def initialize_models_background():
    """Loads Whisper and initializes Gemini client in a background task."""
    global model, gemini_client
    try:
        print("⏳ [BG Task] Loading Whisper model...");
        # Logic for compute_type based on Config and Device
        compute_type = "float16" if device == "cuda" else "int8"
        if Config.USE_FP16 != 'auto':
             if str(Config.USE_FP16).lower() == 'false':
                 compute_type = "float32"
        
        print(f"   - Model: {Config.MODEL_SIZE}")
        print(f"   - Device: {device}")
        print(f"   - Compute Type: {compute_type}")

        model = await asyncio.to_thread(
            WhisperModel, 
            Config.MODEL_SIZE, 
            device=device, 
            compute_type=compute_type
        )
        print(f"✅ [BG Task] Whisper model '{Config.MODEL_SIZE}' loaded.")
        if GEMINI_API_KEY:
            print("⏳ [BG Task] Initializing Gemini client...")
            gemini_client = genai.Client(api_key=GEMINI_API_KEY)
            print("✅ [BG Task] Gemini client initialized.")

        models_ready_event.set()
        await send_telegram_notification(application, f"✅ *AI Models Online (Faster-Whisper)*\n- Reference Init: `{get_runtime()}`\n- Whisper: `{Config.MODEL_SIZE}`\n- Device: `{device}`\n- Gemini: `{'Enabled' if gemini_client else 'Disabled'}`\nBot is fully operational.")
    except Exception as e:
        error_msg = f"❌ *FATAL ERROR:*\nFailed to load AI models: {e}"
        await send_telegram_notification(application, error_msg)
        await perform_shutdown("AI Model Loading Failed")

async def initialize_gradio_background():
    """Launches Gradio web server in background and notifies Telegram with URL."""
    if not GRADIO_AVAILABLE:
        print("⚠️ [BG Task] Gradio not available, skipping web interface.")
        return
    
    try:
        print("⏳ [BG Task] Starting Gradio web interface...")
        main_loop = asyncio.get_running_loop()
        gradio_handler.set_dependencies(job_manager, UPLOAD_FOLDER, main_loop)
        public_url = await gradio_handler.launch_gradio_async(share=True)
        
        if public_url:
            await send_telegram_notification(
                application,
                f"🌐 *Web Interface Online*\n"
                f"Upload large files (>20MB) via:\n{public_url}\n\n"
                f"_Results will be sent to this chat._"
            )
        else:
            print("⚠️ [BG Task] Gradio started but no public URL available.")
    except Exception as e:
        print(f"⚠️ [BG Task] Failed to start Gradio: {e}")


def run_transcription_process(job: TranscriptionJob) -> tuple[str, str]:
    """Runs the blocking Whisper transcription in a separate thread."""
    print(f"[JOB:{job.job_id}] Transcribing '{job.original_filename}'...")
    
    transcribe_options = {"beam_size": Config.BEAM_SIZE}
    
    # Run transcription
    # faster-whisper returns a generator, so we must iterate to process
    segments_generator, info = model.transcribe(job.local_filepath, **transcribe_options)
    
    # Convert generator to list to ensure full processing
    segments = list(segments_generator)
    
    formatted_text = format_transcription_with_pauses(segments, Config.PAUSE_THRESHOLD)
    print(f"[JOB:{job.job_id}] Transcription complete. Detected language: {info.language} ({info.language_probability:.2f})")
    
    return formatted_text, info.language if info.language else 'N/A'

async def queue_processor():
    """The main worker loop that processes jobs from the queue one by one."""
    print("🛠️ [Worker] Waiting for AI models to load..."); await models_ready_event.wait()
    print("🟢 [Worker] AI Models ready. Starting to process jobs.")
    while not SHUTDOWN_IN_PROGRESS:
        job: TranscriptionJob = await job_manager.job_queue.get()

        if job.status == 'cancelled':
            if os.path.exists(job.local_filepath): os.remove(job.local_filepath)
            job_manager.job_queue.task_done()
            job_manager.complete_job(job.job_id)
            continue

        job_manager.set_processing_job(job)
        try:
            duration_str = format_duration(job.audio_duration)
            await application.bot.send_message(job.chat_id, f"▶️ `[ID: {job.job_id}]` Processing `{job.original_filename}` (*{duration_str}*)...", parse_mode=ParseMode.MARKDOWN)
            start_time = time.time()

            transcript_text, detected_language = await asyncio.to_thread(run_transcription_process, job)
            if job.status == 'cancelled': raise asyncio.CancelledError("Job cancelled during transcription.")

            base_name = os.path.splitext(job.original_filename)[0]
            safe_name = secure_filename(base_name)[:50]
            ts_filename = f"{TRANSCRIPT_FILENAME_PREFIX}_({duration_str.replace(' ', '')})_{safe_name}.txt"
            ts_filepath = os.path.join(TRANSCRIPT_FOLDER, ts_filename)
            with open(ts_filepath, "w", encoding="utf-8") as f: f.write(transcript_text)

            # PASSING GEMINI CLIENT HERE
            summary_text = await summarize_text(transcript_text, gemini_client)
            
            if job.status == 'cancelled': raise asyncio.CancelledError("Job cancelled during summarization.")
            su_filename = f"{SUMMARY_FILENAME_PREFIX}_({duration_str.replace(' ', '')})_{safe_name}.txt"
            su_filepath = os.path.join(TRANSCRIPT_FOLDER, su_filename)
            with open(su_filepath, "w", encoding="utf-8") as f: f.write(summary_text)

            processing_duration_str = format_duration(time.time() - start_time)
            result_text = (f"🎉 *Transcription & Summary Complete!*\n\n"
                           f"*File:* `{job.original_filename}`\n"
                           f"*Duration:* {duration_str} | *Processing:* {processing_duration_str}\n"
                           f"*Language:* {detected_language.upper()}")

            await application.bot.send_message(job.chat_id, result_text, parse_mode=ParseMode.MARKDOWN, reply_to_message_id=job.message_id)
            with open(su_filepath, 'rb') as su_file:
                await application.bot.send_document(job.chat_id, document=su_file, filename=su_filename, reply_to_message_id=job.message_id)
            with open(ts_filepath, 'rb') as ts_file:
                await application.bot.send_document(job.chat_id, document=ts_file, filename=ts_filename, reply_to_message_id=job.message_id)
            job.status = "completed"

        except asyncio.CancelledError as e:
            print(f"[WORKER] Aborting cancelled job {job.job_id}. Reason: {e}")
        except Exception as e:
            job.status = "failed"
            print(f"❌ [JOB:{job.job_id}] An error occurred: {e}", file=sys.stderr)
            await application.bot.send_message(job.chat_id, f"❌ *Failed to Process: {job.original_filename}*\n```\n{e}\n```", parse_mode=ParseMode.MARKDOWN, reply_to_message_id=job.message_id)
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
        processing_status_line = (
            f"⚡️ *Processing:* `{processing_job.original_filename}`\n"
            f"👤 *By:* {processing_job.author_display_name}\n"
        )
        bot_activity = "Active"
    else:
        processing_status_line = ""
        bot_activity = "Idle"

    text = (
        f"📊 *Bot Status & Health*\n\n"
        f"{processing_status_line}"
        f"⏳ *Session Uptime:* `{get_runtime()}`\n"
        f"*Jobs in Queue:* `{job_manager.job_queue.qsize()}`\n"
        f"*Bot Activity:* `{bot_activity}`\n"
        f"*AI Model Status:* `{'✅ Online' if models_ready_event.is_set() else '⏳ Initializing...'}`\n\n"
        f"_Last updated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}_"
    )
    keyboard = [[InlineKeyboardButton("📄 View & Cancel Jobs", callback_data="view_cancel_jobs")],
                [InlineKeyboardButton("🔄 Refresh", callback_data="refresh_status"), InlineKeyboardButton("🔌 Shutdown", callback_data="shutdown_bot")]]

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
        await update.effective_message.reply_text("Idle monitor is disabled.")
        return
    msg = "✅ Idle shutdown timer extended by 5 minutes." if idle_monitor.extend_timer(5) else "ℹ️ The bot is active, so the idle timer is not running."
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
        msg = f"✅ Job `{job_name}` was cancelled." if cancelled else f"❌ Could not cancel job."
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

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).request(request).build()

    # Initialize components
    idle_monitor = IdleMonitor(application, None)
    job_manager = JobManager(application, idle_monitor)
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
        nonlocal _transient_error_counts
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
    await application.initialize()
    print(f"\n▶️ Bot is running (Init: {get_runtime()}). Receiving messages now!")
    print("⏳ Loading AI models and services in background...")

    # Background Tasks - start AFTER bot is ready to receive
    application.create_task(queue_processor())
    application.create_task(initialize_models_background())
    
    # Start Gradio web interface (async, like AI models)
    if GRADIO_AVAILABLE:
        application.create_task(initialize_gradio_background())
    
    if Config.ENABLE_IDLE_MONITOR:
        idle_monitor.start()

    # Send startup notification in background (non-blocking)
    gradio_status = "⏳ Loading..." if GRADIO_AVAILABLE else "❌ Disabled"
    startup_message = (
        f"🚀 *Bot is starting up... (Modular Version)*\n\n"
        f"*Model:* `{Config.MODEL_SIZE}` on `{device.upper()}`\n"
        f"*Idle Monitor:* `{'Enabled' if Config.ENABLE_IDLE_MONITOR else 'Disabled'}`\n"
        f"*Bot Handle Limit:* `{Config.BOT_FILESIZE_LIMIT} MB` per file.\n"
        f"*Web UI:* `{gradio_status}`\n\n"
        f"ℹ️ *Usage Tips:*\n"
        f"- All audio & video formats supported by FFmpeg are accepted.\n"
        f"- For files >20MB, use Web UI (URL menyusul)."
    )
    asyncio.create_task(send_telegram_notification(application, startup_message))
    
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
            except:
                pass
    finally:
        if IS_COLAB:
            print("🔌 Triggering Colab Runtime Shutdown (Error Safe-mode)...")
            runtime.unassign()
