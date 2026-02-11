import asyncio
import sys
from datetime import datetime
from log_utils import log

async def summarize_text(transcript: str, gemini_client) -> str:
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
        "RINGKASAN BERITA\n"
        f"Tanggal: [tanggal dari transkrip atau {today_date}]\n\n"
        "LEAD (Paragraf Pembuka):\n"
        "[1-2 kalimat inti berita: siapa, apa, kapan, dimana]\n\n"
        "BODY:\n"
        "A. [Topik/Angle 1]\n"
        "   - Detail penting\n"
        "   - Kutipan pendukung (jika ada)\n\n"
        "B. [Topik/Angle 2, jika ada]\n"
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
        "--- TRANSKRIP ---\n"
        f"```\n{transcript}\n```"
    )
    # Gemini models: primary and fallback
    PRIMARY_MODEL = "gemini-3-flash-preview"
    FALLBACK_MODEL = "gemini-2.5-flash"
    
    try:
        log("GEMINI", f"Requesting summary ({len(transcript)} chars) with {PRIMARY_MODEL}...")
        response = await asyncio.to_thread(
            gemini_client.models.generate_content,
            model=PRIMARY_MODEL,
            contents=prompt
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
                contents=prompt
            )
            log("GEMINI", f"Fallback summary received ({len(response.text)} chars)")
            return response.text
        except Exception as fallback_error:
            log("ERROR", f"Gemini {FALLBACK_MODEL} also failed: {fallback_error}")
            return f"❌ Error generating summary: {fallback_error}"



def format_duration(seconds: float) -> str:
    """Converts a duration in seconds to a human-readable 'Xm XXs' format."""
    if not isinstance(seconds, (int, float)) or seconds < 0: return "N/A"
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
    return f"[{minutes:02d}:{secs:02d}]"

def format_transcription_with_pauses(segments: list, pause_thresh: float = 2.0) -> str:
    """
    Formats Whisper segments with timestamps at significant pauses.
    
    Optimized to:
    - Handle both dictionary and object-like segments.
    - Support HH:MM:SS timestamps for long audio.
    - Filter empty or whitespace-only segments.
    - Gracefully handle missing attributes.
    """
    if not segments:
        return ""

    # Helper to safely access attributes (handles dict vs object)
    def get_val(seg, key, default=0.0):
        if hasattr(seg, key):
            return getattr(seg, key)
        elif isinstance(seg, dict):
            return seg.get(key, default)
        return default

    # 1. Normalize and clean segments
    clean_segments = []
    for seg in segments:
        text = str(get_val(seg, 'text', '')).strip()
        if not text:
            continue
            
        start = float(get_val(seg, 'start', 0.0))
        end = float(get_val(seg, 'end', start)) # Fallback end to start if missing
        
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
    It simply lists each segment with its timestamp, relying on VAD for segmentation.
    Format: [HH:MM:SS] Text
    """
    if not segments:
        return ""
    
    # Helper to safely access attributes (handles dict vs object)
    def get_val(seg, key, default=0.0):
        if hasattr(seg, key):
            return getattr(seg, key)
        elif isinstance(seg, dict):
            return seg.get(key, default)
        return default

    lines = []
    for seg in segments:
        text = str(get_val(seg, 'text', '')).strip()
        if not text:
            continue
            
        start = float(get_val(seg, 'start', 0.0))
        # Format: [HH:MM:SS] Text
        timestamp = format_timestamp(start)
        lines.append(f"{timestamp}\n{text}")
        
    return "\n".join(lines)
