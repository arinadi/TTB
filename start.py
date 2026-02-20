import os
import sys
import subprocess
import time

def check_cuda():
    """Checks if CUDA is available and faster-whisper is installed."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False, "CUDA not available"
        
        # Check faster-whisper
        import faster_whisper
        return True, "GPU/Whisper Ready"
    except ImportError:
        return False, "Dependencies missing (torch/faster-whisper)"
    except Exception as e:
        return False, f"Check failed: {e}"

def main():
    print("🔍 TTB Smart Runner: Detecting Environment...")
    
    # Default to Gemini
    mode = 'GEMINI'
    reason = "Default fallback"
    
    is_gpu, gpu_reason = check_cuda()
    
    if is_gpu:
        mode = 'WHISPER'
        reason = gpu_reason
        print(f"🚀 {reason}. Transcription Mode: WHISPER")
    else:
        print(f"⚠️ {gpu_reason}. Transcription Mode: GEMINI (CPU)")

    # Set Environment Variable
    os.environ['TRANSCRIPTION_MODE'] = mode
    
    # Launch main.py
    print(f"🚀 Starting TTB in {mode} Mode...")
    
    try:
        # Use sys.executable to ensure we use the same environment
        cmd = [sys.executable, "main.py"]
        # In Colab/Terminal, we want to see the output in real-time
        process = subprocess.Popen(cmd)
        process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Runner stopped by user.")
    except Exception as e:
        print(f"❌ Runner Error: {e}")

if __name__ == "__main__":
    main()
