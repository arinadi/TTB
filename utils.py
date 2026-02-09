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
    try:
        log("GEMINI", f"Requesting summary ({len(transcript)} chars)...")
        response = await asyncio.to_thread(
            gemini_client.models.generate_content,
            model="gemini-2.5-flash",
            contents=prompt
        )
        log("GEMINI", f"Summary received ({len(response.text)} chars)")
        return response.text
    except Exception as e:
        log("ERROR", f"Gemini failed: {e}")
        return f"❌ Error generating summary: {e}"

def format_duration(seconds: float) -> str:
    """Converts a duration in seconds to a human-readable 'Xm XXs' format."""
    if not isinstance(seconds, (int, float)) or seconds < 0: return "N/A"
    minutes, remaining_seconds = divmod(int(seconds), 60)
    return f"{minutes}m {remaining_seconds:02d}s"

def format_transcription_with_pauses(segments: list, pause_thresh: float) -> str:
    """Formats the Whisper segments with [mm:ss] timestamps at significant pauses."""
    if not segments:
        return ""

    def format_timestamp(seconds: float) -> str:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"[{minutes:02d}:{secs:02d}]"

    # Handle both object (faster-whisper) and dict (legacy) access
    def get_attr(seg, attr):
        return getattr(seg, attr) if hasattr(seg, attr) else seg[attr]

    first_segment = segments[0]
    first_start = get_attr(first_segment, 'start')
    first_text = get_attr(first_segment, 'text').strip()
    
    current_block = f"{format_timestamp(first_start)}\n{first_text}"
    completed_blocks = []
    
    # Use 'end' if available, otherwise fallback to 'start' which is not ideal but safe
    previous_end = get_attr(first_segment, 'end')

    for segment in segments[1:]:
        start = get_attr(segment, 'start')
        text = get_attr(segment, 'text').strip()
        
        if (start - previous_end) > pause_thresh:
            completed_blocks.append(current_block)
            current_block = f"{format_timestamp(start)}\n{text}"
        else:
            current_block += " " + text
        
        previous_end = get_attr(segment, 'end')

    completed_blocks.append(current_block)
    return "\n\n".join(completed_blocks)
