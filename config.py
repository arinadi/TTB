import os
import time

# --- Global Initialization Timing ---
# This tries to get the start time from Colab's environment variable.
# If running locally (variable missing), it falls back to current time.
INIT_START = float(os.getenv('INIT_START', time.time()))

# --- Core Secrets ---
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')  # Token from BotFather
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')      # Admin Chat ID (integer)
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')          # Google AI Studio Key

if TELEGRAM_CHAT_ID:
    TELEGRAM_CHAT_ID = int(TELEGRAM_CHAT_ID)

# --- Bot Configuration ---
class Config:
    # --- Whisper Core Settings ---
    # Model size: tiny, base, small, medium, large-v2, large-v3. Larger = slower but more accurate.
    WHISPER_MODEL = os.getenv('MODEL_SIZE', 'large-v2')
    
    # Precision (compute_type): 'auto' (default), 'float16', 'int8_float16', 'int8'. 
    # 'auto' selects float16 for CUDA and int8 for CPU.
    WHISPER_PRECISION = os.getenv('USE_FP16', 'auto') 
    
    # Beam Size: Number of paths to search. Higher (5-10) = better accuracy, slower speed.
    WHISPER_BEAM_SIZE = int(os.getenv('BEAM_SIZE', 10))
    
    # --- Whisper Advanced decoding ---
    # Patience: "Time to think". Higher (2.0+) forces deeper beam search. 
    # Requires temperature=0.
    WHISPER_PATIENCE = float(os.getenv('WHISPER_PATIENCE', 2.0))
    
    # Temperature: Sampling randomness. 0 = deterministic (best for long form).
    WHISPER_TEMPERATURE = float(os.getenv('WHISPER_TEMPERATURE', 0.0))
    
    # Repetition Penalty: Penalty for repeating phrases. >1.0 reduces loops.
    WHISPER_REPETITION_PENALTY = float(os.getenv('REPETITION_PENALTY', 1.1))
    
    # No Repeat N-Gram: Prevent repeating sequences of N words.
    WHISPER_NO_REPEAT_NGRAM_SIZE = int(os.getenv('NO_REPEAT_NGRAM_SIZE', 3))

    # --- VAD (Voice Activity Detection) ---
    # Filter: Enable/Disable VAD to remove silence/hallucinations.
    VAD_FILTER = os.getenv('VAD_FILTER', 'False').lower() == 'true'
    
    # Threshold: Probability threshold for speech (0.1-1.0). Higher = strictly speech.
    VAD_THRESHOLD = float(os.getenv('VAD_THRESHOLD', 0.5))
    
    # Min Speech: Shortest audio chunk to keep (ms). Ignores short noises.
    VAD_MIN_SPEECH_DURATION_MS = int(os.getenv('VAD_MIN_SPEECH_DURATION_MS', 250))
    
    # Min Silence: Silence duration to trigger a new segment (ms).
    VAD_MIN_SILENCE_DURATION_MS = int(os.getenv('VAD_MIN_SILENCE_DURATION_MS', 2000))
    
    # Speech Pad: Padding added to speech segments (ms) to prevent cutting words.
    VAD_SPEECH_PAD_MS = int(os.getenv('VAD_SPEECH_PAD_MS', 400))

    # --- System Limits ---
    # File Size Limit: Max file size for processing (MB).
    BOT_FILESIZE_LIMIT = int(os.getenv('BOT_FILESIZE_LIMIT', 20)) 

    # --- Idle Monitor (Colab Optimization) ---
    # Enable: Auto-shutdown runtime to save credits/GPU.
    ENABLE_IDLE_MONITOR = os.getenv('ENABLE_IDLE_MONITOR', 'True').lower() == 'true'
    
    # First Alert: Minutes of idleness before first warning button.
    IDLE_FIRST_ALERT_MINUTES = int(os.getenv('IDLE_FIRST_ALERT_MINUTES', 1))
    
    # Final Warning: Minutes of idleness before final warning.
    IDLE_FINAL_WARNING_MINUTES = int(os.getenv('IDLE_FINAL_WARNING_MINUTES', 5))
    
    # Shutdown: Minutes of idleness before killing runtime.
    IDLE_SHUTDOWN_MINUTES = int(os.getenv('IDLE_SHUTDOWN_MINUTES', 10))
