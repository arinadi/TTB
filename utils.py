import asyncio
import sys

async def summarize_text(transcript: str, gemini_client) -> str:
    """Generates a summary of the transcript using the Gemini API in Indonesian."""
    if not gemini_client:
        return "Summarisasi dinonaktifkan: Kunci API Gemini tidak dikonfigurasi atau client gagal dimuat."

    prompt = (
        "Anda adalah AI ahli peringkas. "
        "Ringkas transkrip berikut ke dalam Bahasa Indonesia dengan format poin-poin yang ringkas dan jelas.\n\n"
        "Fokus pada:\n"
        "- Ide utama atau topik sentral.\n"
        "- Kesimpulan atau poin-poin penting yang disampaikan.\n"
        "- Jika ada, sebutkan juga keputusan atau tugas (action items) yang jelas.\n\n"
        "--- TRANSKRIP ---\n"
        f"```\n{transcript}\n```"
    )
    try:
        # Using run_in_executor to run the synchronous API call in a separate thread
        # to avoid blocking the asyncio event loop.
        response = await asyncio.to_thread(
            gemini_client.models.generate_content,
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"❌ Terjadi kesalahan saat membuat ringkasan: {e}"

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
