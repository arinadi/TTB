# 🌐 Gradio Web Handler for TTB
# ------------------------------------------------------------------------------
# Module for handling large file uploads (>2GB) via Gradio web interface.
# This bypasses Telegram's 20MB file size limit.
# ------------------------------------------------------------------------------

import os
import uuid
import asyncio
from typing import Optional, TYPE_CHECKING

try:
    import gradio as gr
except ImportError:
    gr = None
    print("⚠️ WARNING: Gradio not installed. Web interface will be disabled.")

if TYPE_CHECKING:
    from main import JobManager, TranscriptionJob

# Module-level state
gradio_app: Optional["gr.Blocks"] = None
gradio_ready_event = asyncio.Event()
_job_manager: Optional["JobManager"] = None
_upload_folder: str = "uploads"
_main_loop: Optional[asyncio.AbstractEventLoop] = None


def set_dependencies(job_manager: "JobManager", upload_folder: str, main_loop: asyncio.AbstractEventLoop):
    """Set dependencies from main module."""
    global _job_manager, _upload_folder, _main_loop
    _job_manager = job_manager
    _upload_folder = upload_folder
    _main_loop = main_loop


def _get_telegram_chat_id() -> int:
    """Get TELEGRAM_CHAT_ID from environment variable (set by Colab runner)."""
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')
    if chat_id:
        return int(chat_id)
    raise ValueError("TELEGRAM_CHAT_ID not found in environment variables")


def process_upload(file_paths: list) -> str:
    """
    Process uploaded files and add to job queue.
    Supports multiple files. Returns status message for Gradio UI.
    """
    if not _job_manager:
        return "❌ Error: Job manager not initialized."
    
    if not file_paths:
        return ""
    
    try:
        chat_id = _get_telegram_chat_id()
    except ValueError as e:
        return f"❌ Error: {e}"
    
    import shutil
    results = []
    
    for file_path in file_paths:
        # Get original filename from path
        original_filename = os.path.basename(file_path)
        
        # Generate unique filename for storage
        unique_filename = f"{uuid.uuid4().hex}_{original_filename}"
        dest_path = os.path.join(_upload_folder, unique_filename)
        
        # Copy file to upload folder (Gradio provides temp path)
        shutil.copy2(file_path, dest_path)
        
        # Get file size for display
        file_size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        
        # Queue the job on the main event loop (from sync thread)
        if _main_loop is None:
            print("[GRADIO] Error: Main event loop not set!")
            continue
        
        asyncio.run_coroutine_threadsafe(_queue_gradio_job(dest_path, original_filename, chat_id), _main_loop)
        
        results.append(f"✅ {original_filename} ({file_size_mb:.2f} MB)")
    
    file_count = len(results)
    file_list = "\n".join(results)
    
    return (
        f"📤 **{file_count} file berhasil di-upload!**\n\n"
        f"{file_list}\n\n"
        f"📋 Status: Masuk antrian transkripsi\n"
        f"📱 Hasil akan dikirim ke Telegram."
    )


async def _queue_gradio_job(file_path: str, filename: str, chat_id: int):
    """Add job to queue from Gradio upload."""
    try:
        import ffmpeg
        from main import TranscriptionJob
        
        # Probe audio duration
        probe = await asyncio.to_thread(ffmpeg.probe, file_path)
        duration = float(probe['format']['duration'])
        
        # Create a mock message object for TranscriptionJob
        class GradioMessage:
            def __init__(self, chat_id: int, filename: str):
                self.message_id = 0  # No reply needed for web uploads
                self.chat_id = chat_id
                self.from_user = None
                self.chat = type('obj', (object,), {'title': 'Gradio Web Upload'})
                self.effective_attachment = type('obj', (object,), {'file_name': filename})
        
        mock_message = GradioMessage(chat_id, filename)
        job = TranscriptionJob.from_message(mock_message, file_path, duration)
        job.author_display_name = "Web Upload"
        
        await _job_manager.add_job(job)
        print(f"[GRADIO] Job {job.job_id} added to queue for file: {filename}")
        
    except Exception as e:
        print(f"[GRADIO] Error queuing job: {e}")
        # Clean up file on error
        if os.path.exists(file_path):
            os.remove(file_path)


def create_gradio_interface() -> Optional["gr.Blocks"]:
    """Create and return the Gradio interface."""
    global gradio_app
    
    if gr is None:
        print("❌ Gradio not available. Web interface disabled.")
        return None
    
    with gr.Blocks(
        title="TTB - Transcription Bot",
        theme=gr.themes.Soft(primary_hue="blue"),
        css="""
        .gradio-container { max-width: 600px !important; margin: auto; }
        .upload-box { border: 2px dashed #3b82f6 !important; border-radius: 12px !important; }
        """
    ) as app:
        gr.Markdown(
            """
            # 🎙️ TTB - Transcription Bot (Web)
            
            Upload file audio/video berukuran besar (>20MB) melalui interface ini.
            Hasil transkripsi akan dikirim ke Telegram.
            """
        )
        
        with gr.Column():
            file_input = gr.File(
                label="📁 Upload Audio/Video (bisa pilih banyak file)",
                file_types=["audio", "video", ".mp3", ".mp4", ".wav", ".m4a", ".webm", ".ogg", ".flac", ".mkv"],
                type="filepath",
                file_count="multiple"
            )
            
            status_output = gr.Textbox(
                label="Status",
                interactive=False,
                lines=6
            )
            
            upload_more_btn = gr.Button("🔄 Upload Lagi", variant="secondary", size="lg", visible=False)
        
        gr.Markdown(
            """
            ---
            **Supported formats:** MP3, MP4, WAV, M4A, WEBM, OGG, FLAC, MKV
            
            **Note:** File akan diproses secara berurutan (FIFO queue).
            Hasil akan dikirim ke Telegram secara otomatis.
            """
        )
        
        def handle_upload(files):
            """Process files and return status + show reset button."""
            if not files:
                return "", gr.update(visible=False)
            status = process_upload(files)
            return status, gr.update(visible=True)
        
        def reset_upload():
            """Reset file input for new uploads."""
            return None, "", gr.update(visible=False)
        
        # Auto-process when files are added
        file_input.change(
            fn=handle_upload,
            inputs=[file_input],
            outputs=[status_output, upload_more_btn]
        )
        
        # Reset button to upload more files
        upload_more_btn.click(
            fn=reset_upload,
            outputs=[file_input, status_output, upload_more_btn]
        )
    
    gradio_app = app
    return app


async def launch_gradio_async(share: bool = True) -> Optional[str]:
    """
    Launch Gradio server asynchronously.
    Returns the public URL if share=True.
    """
    global gradio_app
    
    if gr is None:
        return None
    
    app = create_gradio_interface()
    if app is None:
        return None
    
    print("⏳ [BG Task] Starting Gradio server...")
    
    try:
        # Launch in a separate thread to not block
        def _launch():
            app.launch(
                share=share,
                quiet=True,
                prevent_thread_lock=True,
                show_error=True
            )
        
        await asyncio.to_thread(_launch)
        
        # Wait a moment for the server to start
        await asyncio.sleep(3)
        
        # Get the public URL
        public_url = None
        if hasattr(app, 'share_url') and app.share_url:
            public_url = app.share_url
        elif hasattr(app, 'local_url'):
            public_url = app.local_url
        
        gradio_ready_event.set()
        print(f"✅ [BG Task] Gradio server ready: {public_url}")
        
        return public_url
        
    except Exception as e:
        print(f"❌ [BG Task] Failed to start Gradio: {e}")
        return None


async def shutdown_gradio():
    """Shutdown Gradio server gracefully."""
    global gradio_app
    if gradio_app:
        try:
            gradio_app.close()
            print("🔌 Gradio server stopped.")
        except Exception as e:
            print(f"⚠️ Error stopping Gradio: {e}")
