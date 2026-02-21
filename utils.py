from datetime import datetime
import asyncio
import time
import os
import config

# --- Logging Utilities (Merged from log_utils.py) ---

def get_runtime() -> str:
    """Formats total runtime since INIT_START as 'Xm XXs'."""
    elapsed = time.time() - config.INIT_START
    minutes, seconds = divmod(int(elapsed), 60)
    return f"{minutes}m {seconds:02d}s"

def log(category: str, message: str):
    """
    Print log with format: [HH:MM:SS] [+Runtime] [CATEGORY] message
    
    Categories: INIT, JOB, IDLE, WORKER, GEMINI, WHISPER, FILE, GRADIO, ERROR
    """
    timestamp = time.strftime("%H:%M:%S")
    runtime = get_runtime()
    print(f"[{timestamp}] [+{runtime}] [{category}] {message}")

# --- AI & Formatting Utilities ---

async def summarize_text(transcript: str, gemini_client, mode: str = 'GEMINI') -> str:
    """Generates a journalist-friendly summary of the transcript using the Gemini API in Indonesian."""
    if not gemini_client:
        return "Summarization disabled: Gemini API key not configured or client failed to load."

    today_date = datetime.now().strftime("%d %B %Y")
    
    prompt = (
        "Anda adalah AI peringkas untuk jurnalis. "
        "Ringkas transkrip berikut ke dalam Bahasa Indonesia dengan format Plain Text.\n\n"
        "ATURAN PENTING:\n"
        "- JANGAN mengarang atau berasumsi informasi yang tidak ada di transkrip.\n"
        "- Jika informasi tidak ditemukan, KOSONGKAN bagian tersebut atau tulis '-'.\n"
        "- Hanya tulis informasi yang JELAS terlihat di transkrip.\n"
        f"- Jika tanggal tidak disebutkan di transkrip, gunakan: {today_date}\n\n"
        "FORMAT OUTPUT:\n\n"
        "FAKTA BERITA\n"
        f"Tanggal: [tanggal dari transkrip atau {today_date}]\n\n"
        "LEAD (Paragraf Pembuka):\n"
        "[1-2 kalimat inti berita: siapa, apa, kapan, dimana]\n\n"
        "BODY:\n"
        "A. [Topik/Angle 1]\n"
        "   - Detail penting\n"
        "   - Kutipan pendukung (jika ada)\n\n"
        "B. [Topik/Angle 2]\n"
        "   - Detail penting\n\n"
        "C. [Topik/Angle 3, jika ada]\n"
        "   - Detail penting\n\n"
        "D. [Topik/Angle 4, jika ada]\n"
        "   - Detail penting\n\n"
        "NARASUMBER:\n"
        "1. [Nama] - [Jabatan] - \"[Kutipan kunci]\"\n"
        "(Kosongkan jika tidak ada narasumber jelas)\n\n"
        "DATA PENDUKUNG:\n"
        "- [Angka/statistik dari transkrip]\n"
        "(Kosongkan jika tidak ada data)\n\n"
        "PERLU KLARIFIKASI:\n"
        "- [Hal yang tidak jelas atau perlu dicek]\n"
        "(Kosongkan jika tidak ada)\n\n"
        "-----\n"
    )

    # WHISPER mode: append RETOUCH TRANSCRIPT section
    if mode == 'WHISPER':
        prompt += (
            "\n\n"
            "-----\n\n"
            "RETOUCH TRANSCRIPT:\n"
            "! WARNING: Bagian ini adalah hasil perbaikan AI dan mengandung asumsi.\n\n"
            "[Perbaiki typo, kesalahan penulisan, serta tanda baca (seperti tanda tanya) pada transkrip. "
            "Berikan jeda baris (enter) di setiap akhir paragraf agar teks lebih mudah dibaca. "
            "Pastikan urutan kalimat dan struktur asli teks tetap sama.]\n\n"
            "--- TRANSKRIP ASLI [JANGAN KIRIM KEMBALI] ---\n"
            f"```\n{transcript}\n```"
        )
    
    # Gemini models
    PRIMARY_MODEL = "gemini-3-flash-preview"     # Use newer flash as primary
    FALLBACK_MODEL = "gemini-2.5-flash"

    # WHISPER: transcript already embedded in prompt (RETOUCH section)
    # GEMINI: pass transcript separately to avoid embedding it in the prompt
    contents = prompt if mode == 'WHISPER' else [prompt, transcript]
    
    try:
        log("GEMINI", f"Requesting summary ({len(transcript)} chars) with {PRIMARY_MODEL}...")
        response = await asyncio.to_thread(
            gemini_client.models.generate_content,
            model=PRIMARY_MODEL,
            contents=contents
        )
        log("GEMINI", f"Summary received ({len(response.text)} chars)")
        return response.text
    except Exception as e:
        log("ERROR", f"Gemini {PRIMARY_MODEL} failed: {e}")
        log("GEMINI", f"Retrying with fallback model {FALLBACK_MODEL}...")
        try:
            response = await asyncio.to_thread(
                gemini_client.models.generate_content,
                model=FALLBACK_MODEL,
                contents=contents
            )
            log("GEMINI", f"Fallback summary received ({len(response.text)} chars)")
            return response.text
        except Exception as fallback_error:
            log("ERROR", f"Gemini {FALLBACK_MODEL} also failed: {fallback_error}")
            return f"❌ Error generating summary: {fallback_error}"

def format_duration(seconds: float) -> str:
    """Converts a duration in seconds to a human-readable 'Xm XXs' format."""
    if not isinstance(seconds, (int, float)) or seconds < 0:
        return "N/A"
    minutes, remaining_seconds = divmod(int(seconds), 60)
    return f"{minutes}m {remaining_seconds:02d}s"

def format_timestamp(seconds: float) -> str:
    """Formats seconds into [HH:MM:SS] or [MM:SS]."""
    if not isinstance(seconds, (int, float)) or seconds < 0:
        return "[00:00]"
    
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    
    if hours > 0:
        return f"[{hours:02d}:{minutes:02d}:{secs:02d}]"
    return f"{minutes:02d}:{secs:02d}"

def get_val(seg, key, default=0.0):
    """Helper to safely access attributes (handles dict vs object)."""
    if hasattr(seg, key):
        return getattr(seg, key)
    elif isinstance(seg, dict):
        return seg.get(key, default)
    return default

def format_transcription_with_pauses(segments: list, pause_thresh: float = 2.0) -> str:
    """
    Formats Whisper segments with timestamps at significant pauses.
    """
    if not segments:
        return ""

    # 1. Normalize and clean segments
    clean_segments = []
    for seg in segments:
        text = str(get_val(seg, 'text', '')).strip()
        if not text:
            continue
            
        start = float(get_val(seg, 'start', 0.0))
        end = float(get_val(seg, 'end', start))
        
        clean_segments.append({
            'start': start, 
            'end': end, 
            'text': text
        })

    if not clean_segments:
        return ""

    # 2. Build blocks based on pauses
    blocks = []
    current_block_start = clean_segments[0]['start']
    current_text_parts = [clean_segments[0]['text']]
    last_end = clean_segments[0]['end']

    for seg in clean_segments[1:]:
        gap = seg['start'] - last_end
        
        if gap > pause_thresh:
            # Commit previous block
            timestamp = format_timestamp(current_block_start)
            block_content = " ".join(current_text_parts)
            blocks.append(f"{timestamp}\n{block_content}")
            
            # Start new block
            current_block_start = seg['start']
            current_text_parts = [seg['text']]
        else:
            # Continue current block
            current_text_parts.append(seg['text'])
        
        last_end = seg['end']

    # 3. Commit final block
    if current_text_parts:
        timestamp = format_timestamp(current_block_start)
        block_content = " ".join(current_text_parts)
        blocks.append(f"{timestamp}\n{block_content}")

    return "\n\n".join(blocks)

def format_transcription_native(segments: list) -> str:
    """
    Formats Whisper segments exactly as output by the model (with VAD enabled).
    Format: [HH:MM:SS] Text
    """
    if not segments:
        return ""
    
    lines = []
    for seg in segments:
        text = str(get_val(seg, 'text', '')).strip()
        if not text:
            continue
            
        lines.append(f"{text}")
        
    return "\n\n".join(lines)

async def transcribe_with_gemini(local_filepath: str, duration: float, gemini_client) -> tuple[str, str]:
    """Transcribes audio using Gemini API (File API)."""
    if not gemini_client:
        return "Error: Gemini client not initialized.", "N/A"

    try:
        log("GEMINI", f"Uploading {os.path.basename(local_filepath)}...")
        # 1. Upload
        audio_file = await asyncio.to_thread(
            gemini_client.files.upload, 
            file=local_filepath
        )
        
        # 2. Wait for ACTIVE
        log("GEMINI", "Waiting for file processing...")
        while True:
            audio_file = await asyncio.to_thread(
                gemini_client.files.get, 
                name=audio_file.name
            )
            if audio_file.state.name == "ACTIVE":
                break
            elif audio_file.state.name != "PROCESSING":
                raise Exception(f"File failed to process. State: {audio_file.state.name}")
            await asyncio.sleep(2)

        # 3. Generate Transcript
        prompt = (
            "Transcribe this audio file accurately. Identify different speakers if possible. "
            "Output only the transcript.\n"
            "STRICT FORMATTING RULE:\n"
            "- DO NOT include timestamps.\n"
            "- Insert a double newline (\\n\\n) after every sentence/period.\n"
            "- Do not change any words, order, or content.\n"
            "- Simply ensure there is a blank line between every sentence for readability."
        )

        log("GEMINI", f"Generating transcript for {duration:.1f}s audio...")
        response = await asyncio.to_thread(
            gemini_client.models.generate_content,
            model="gemini-2.5-flash", # Use specific model for transcript
            contents=[audio_file, prompt]
        )

        if not response.text:
            return "Warning: Empty transcript returned by Gemini.", "N/A"

        return response.text, "ID" # Assume ID as default or let gemini detect (but return ID for lang label)

    except Exception as e:
        log("ERROR", f"Gemini transcription failed: {e}")
        return f"Error transcribing with Gemini: {e}", "N/A"
