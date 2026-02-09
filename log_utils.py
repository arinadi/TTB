"""
Centralized logging utilities for TTB with consistent timestamp format.
"""
import time
import os

# Get INIT_START from environment (set by Colab runner) or use current time
INIT_START = float(os.getenv('INIT_START', time.time()))

def get_runtime() -> str:
    """Formats total runtime since INIT_START as 'Xm XXs'."""
    elapsed = time.time() - INIT_START
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
