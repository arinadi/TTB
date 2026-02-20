# Install the new SDK if running in Colab
# !pip install -q -U google-genai

import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime
from google import genai
from google.genai import types

# --- Configuration ---
# Detect environment
IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    from google.colab import userdata, files
    try:
        API_KEY = userdata.get('GEMINI_API_KEY')
    except Exception:
        API_KEY = None
        print("Warning: GEMINI_API_KEY not found in Colab secrets.")
else:
    API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("Error: API Key is required. Set 'GEMINI_API_KEY' in environment or Colab secrets.")

# Model: Gemini 2.5 Flash
MODEL_NAME = "gemini-2.5-flash"

# Output Directory
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize Client
client = genai.Client(api_key=API_KEY)

def get_audio_duration(file_path):
    """Gets audio duration in seconds using ffprobe (available in Colab)."""
    try:
        cmd = [
            "ffprobe", 
            "-v", "error", 
            "-show_entries", "format=duration", 
            "-of", "default=noprint_wrappers=1:nokey=1", 
            file_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return float(result.stdout.strip())
    except Exception as e:
        print(f"Warning: Could not determine duration locally. Error: {e}")
        return None

def upload_to_gemini(path, mime_type=None):
    """Uploads the given file to Gemini."""
    try:
        file = client.files.upload(file=path, config={'mime_type': mime_type})
        print(f"Uploaded to Gemini: '{file.display_name}' as: {file.uri}")
        return file
    except Exception as e:
        print(f"Failed to upload to Gemini: {e}")
        return None

def wait_for_files_active(file):
    """Waits for the given file to start processing."""
    print("Waiting for file processing...", end="")
    
    while file.state.name == "PROCESSING":
        print(".", end="", flush=True)
        time.sleep(2)
        file = client.files.get(name=file.name)
        
    if file.state.name != "ACTIVE":
        raise Exception(f"File {file.name} failed to process. State: {file.state.name}")
    
    print("\n...file ready")
    return file

def generate_summary(transcript_text, today_date):
    """Generates a summary based on the transcript."""
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
    
    print(f"Sending Summary Request to {MODEL_NAME}...")
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[prompt, transcript_text],
            config=types.GenerateContentConfig(
                temperature=0.1,
                top_p=0.95,
                top_k=40,
                max_output_tokens=8192,
                response_mime_type="text/plain",
            )
        )
        return response.text
    except Exception as e:
        print(f"\n[ERROR] Generating Summary Failed: {e}")
        return f"Error: Could not generate summary. Details: {e}"

def transcribe_audio(audio_path):
    """Main function to handle transcription and summarization."""
    
    print(f"\nProcessing: {audio_path}")
    today_date = datetime.now().strftime("%Y-%m-%d")

    # 0. Check Duration
    duration = get_audio_duration(audio_path)
    if duration:
        print(f"Audio Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        if duration > 600: # 10 minutes
            print("Error: Audio file exceeds 10 minutes. Please trim the file to avoid token limits.")
            return
    else:
        print("Warning: Skipping duration check (ffprobe not found or failed).")

    # 1. Upload File to Gemini
    audio_file = upload_to_gemini(audio_path)
    if not audio_file:
        return

    # 2. Wait for Processing
    try:
        audio_file = wait_for_files_active(audio_file)
    except Exception as e:
        print(f"Error waiting for processing: {e}")
        return

    # 3. Transcribe Audio (Step 1)
    transcribe_prompt = (
        "Transcribe this audio file accurately. Identify different speakers if possible. "
        "Output only the transcript.\n"
        "STRICT FORMATTING RULE:\n"
        "- DO NOT include timestamps.\n"
        "- Insert a double newline (\\n\\n) after every sentence/period.\n"
        "- Do not change any words, order, or content.\n"
        "- Simply ensure there is a blank line between every sentence for readability."
    )

    print(f"Transcribing with model: {MODEL_NAME}...")
    try:
        # Generate Transcript
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[audio_file, transcribe_prompt],
            config=types.GenerateContentConfig(
                temperature=0.1,
                top_p=0.95,
                top_k=40,
                max_output_tokens=8192,
                response_mime_type="text/plain",
            )
        )
        
        # Check if response is valid but empty/blocked
        if not response.text:
             print("\n[WARNING] Response text is empty. Checking candidates...")
             if response.candidates:
                 print(f"Finish Reason: {response.candidates[0].finish_reason}")
                 print(f"Safety Ratings: {response.candidates[0].safety_ratings}")
             else:
                 print("No candidates returned.")
             return

        transcript_text = response.text
        
        # Save Transcript (TS)
        ts_filename = os.path.join(OUTPUT_DIR, f"TS_{Path(audio_path).stem}.txt")
        with open(ts_filename, "w", encoding="utf-8") as f:
            f.write(transcript_text)
        print(f"Success! Transcript saved to: {ts_filename}")

        # 4. Generate Summary (Step 2)
        print("Generating Jurnalis Summary...")
        summary_text = generate_summary(transcript_text, today_date)

        # Save Summary (AI)
        ai_filename = os.path.join(OUTPUT_DIR, f"AI_{Path(audio_path).stem}.txt")
        with open(ai_filename, "w", encoding="utf-8") as f:
            f.write(summary_text)
        print(f"Success! Summary saved to: {ai_filename}")
        
    except Exception as e:
        print(f"\n[ERROR] Processing Failed: {e}")
        # Try to print more details about the exception if available
        if hasattr(e, 'response'):
             print(f"Response Status: {e.response.status_code}")
             print(f"Response Body: {e.response.text}")

if __name__ == "__main__":
    # Ensure pip install runs via user check
    if IN_COLAB:
        print("--- Colab Mode Detected ---")
        print("Tip: Ensure '!pip install -q -U google-genai' is run before this script.")
        print("Using API Key from Secrets.")
        print("Please upload an audio file (Max 10 mins):")
        
        uploaded = files.upload()
        if uploaded:
            audio_path = next(iter(uploaded))
            transcribe_audio(audio_path)
            print(f"\nAll files saved in '{OUTPUT_DIR}' folder.")
        else:
            print("No file uploaded.")
    else:
        # Local Mode
        audio_path = "input.mp3" 
        if not os.path.exists(audio_path):
            print(f"File '{audio_path}' not found. Please provide a valid audio file.")
        else:
            transcribe_audio(audio_path)
