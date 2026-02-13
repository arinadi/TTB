import os
import time
import uuid
import asyncio
import re
import shutil
import zipfile
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Any, Dict

import telegram
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import Application, ContextTypes
from werkzeug.utils import secure_filename
import ffmpeg

from config import Config, TELEGRAM_CHAT_ID
from utils import log

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
        log("JOB", f"[{job.job_id}] Created: {job.original_filename} (by {job.author_display_name})")
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
    def __init__(self, app: Application, job_manager: "JobManager", shutdown_callback: Callable[[str], Any]):
        self.app = app
        self.job_manager = job_manager
        self.shutdown_callback = shutdown_callback
        self.shutdown_on: Optional[float] = None  # Absolute timestamp for shutdown
        self.shutdown_imminent = False
        self.alerts_sent = {'first_alert': False, 'final_warning': False}
        self.last_extend_time = 0
        self._task: Optional[asyncio.Task] = None
        log("INIT", f"IdleMonitor ready (alert={Config.IDLE_FIRST_ALERT_MINUTES}m, warn={Config.IDLE_FINAL_WARNING_MINUTES}m, shutdown={Config.IDLE_SHUTDOWN_MINUTES}m)")

    def start(self):
        if not self._task or self._task.done():
            self._task = asyncio.create_task(self._monitor_loop())
            log("IDLE", "Monitor started")

    def stop(self):
        if self._task:
            self._task.cancel()

    def reset(self):
        if self.shutdown_on is not None:
            log("IDLE", "Bot active. Timer reset.")
            self.shutdown_on = None
            self.alerts_sent = {'first_alert': False, 'final_warning': False}

    def extend_timer(self, minutes: int) -> bool:
        """Extend shutdown time by adding minutes to shutdown_on."""
        if self.shutdown_on is not None:
            self.shutdown_on += (minutes * 60)
            # Reset alerts so user gets them again with new timing
            self.alerts_sent = {'first_alert': False, 'final_warning': False}
            remaining = (self.shutdown_on - time.time()) / 60
            log("IDLE", f"Extended +{minutes}m. Shutdown in {remaining:.1f}m")
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
                log("IDLE", "Waiting for job_manager...")
                continue

            try:
                if self.job_manager.is_idle():
                    # First time idle - set shutdown_on (absolute time)
                    if self.shutdown_on is None:
                        self.shutdown_on = time.time() + (Config.IDLE_SHUTDOWN_MINUTES * 60)
                        log("IDLE", f"Bot idle. Shutdown in {Config.IDLE_SHUTDOWN_MINUTES}m")

                    # Calculate times
                    remaining_minutes = (self.shutdown_on - time.time()) / 60
                    elapsed_minutes = Config.IDLE_SHUTDOWN_MINUTES - remaining_minutes
                    log("IDLE", f"Elapsed: {elapsed_minutes:.1f}m, Remaining: {remaining_minutes:.1f}m")
                    
                    # 1. FIRST ALERT when elapsed >= IDLE_FIRST_ALERT_MINUTES (e.g., 1 min idle)
                    if elapsed_minutes >= Config.IDLE_FIRST_ALERT_MINUTES and not self.alerts_sent['first_alert']:
                        alert_msg = f"⏸️ Idle. Shutdown in `{int(remaining_minutes)}m`"
                        keyboard = [[InlineKeyboardButton("⏳ +5m", callback_data="extend_idle")]]
                        await self.app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=alert_msg, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.MARKDOWN)
                        self.alerts_sent['first_alert'] = True
                        log("IDLE", "First alert sent")
                    
                    # 2. FINAL WARNING when elapsed >= IDLE_FINAL_WARNING_MINUTES (e.g., 5 min idle)
                    if elapsed_minutes >= Config.IDLE_FINAL_WARNING_MINUTES and not self.alerts_sent['final_warning']:
                        warn_msg = f"⚠️ Shutdown in `{int(remaining_minutes)}m`!"
                        try:
                            await self.app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=warn_msg, parse_mode=ParseMode.MARKDOWN)
                        except Exception as e:
                            log("ERROR", f"Telegram notification failed: {e}")
                        self.alerts_sent['final_warning'] = True
                        log("IDLE", "Final warning sent")
                    
                    # 3. SHUTDOWN when remaining <= 0
                    if remaining_minutes <= 0:
                        self.shutdown_imminent = True
                        shutdown_msg = f"🔴 Shutting down (idle {Config.IDLE_SHUTDOWN_MINUTES}m)"
                        try:
                            await self.app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=shutdown_msg, parse_mode=ParseMode.MARKDOWN)
                        except Exception as e:
                            log("ERROR", f"Telegram notification failed: {e}")
                        log("IDLE", "Shutdown triggered")
                        if self.shutdown_callback:
                            if asyncio.iscoroutinefunction(self.shutdown_callback):
                                await self.shutdown_callback("Automatic Idle Shutdown")
                            else:
                                self.shutdown_callback("Automatic Idle Shutdown")
                else:
                    self.reset()
            except Exception as e:
                log("ERROR", f"IdleMonitor: {e}")
                import traceback
                traceback.print_exc()

class JobManager:
    """Manages the queue and state of all transcription jobs."""
    def __init__(self, app: Application, idle_monitor_ref: IdleMonitor, models_ready_event: asyncio.Event):
        self.app = app
        self.idle_monitor = idle_monitor_ref
        self.models_ready_event = models_ready_event
        self.job_queue = asyncio.Queue()
        self.currently_processing: Optional[TranscriptionJob] = None
        self.job_registry: Dict[str, TranscriptionJob] = {}
        log("INIT", "JobManager ready")

    async def add_job(self, job: TranscriptionJob):
        self.idle_monitor.reset()
        self.job_registry[job.job_id] = job
        await self.job_queue.put(job)
        queue_position = self.job_queue.qsize()
        model_status_note = " ⏳" if not self.models_ready_event.is_set() else ""
        content = f"✅ Queued: `{job.original_filename}` (#{queue_position}){model_status_note}"
        
        # Add simpler cancel button
        keyboard = [[InlineKeyboardButton("❌", callback_data=f"cancel_{job.job_id}")]]
        
        await self.app.bot.send_message(job.chat_id, content, parse_mode=ParseMode.MARKDOWN, reply_to_message_id=job.message_id, reply_markup=InlineKeyboardMarkup(keyboard))
        log("JOB", f"[{job.job_id}] Queued at #{queue_position}")

    def complete_job(self, job_id: str):
        if self.currently_processing and self.currently_processing.job_id == job_id:
            self.currently_processing = None
        if job_id in self.job_registry:
            del self.job_registry[job_id]
        log("JOB", f"[{job_id}] Completed")
        self.idle_monitor.reset()

    def set_processing_job(self, job: TranscriptionJob):
        self.currently_processing = job
        job.status = "processing"
        log("JOB", f"[{job.job_id}] Processing...")
        self.idle_monitor.reset()

    async def cancel_job(self, job_id: str) -> tuple[bool, str]:
        job = self.job_registry.get(job_id)
        if not job:
            return False, "Unknown"
        job.status = "cancelled"
        log("JOB", f"[{job.job_id}] Cancelled")
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

            # Duration check removed by user request

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
