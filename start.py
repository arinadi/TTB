import os
import sys
import subprocess

def check_cuda():
    """Checks if CUDA is available and faster-whisper is installed."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False, "CUDA not available"
        
        # Check faster-whisper
        import faster_whisper  # noqa: F401
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
        # Check if basic dependencies like google-genai are present
        try:
            import google.genai  # noqa: F401
            import telegram  # noqa: F401
        except ImportError:
            print("❌ Basic dependencies missing. Running setup_uv.sh...")
            try:
                subprocess.run(["bash", "setup_uv.sh"], check=True)
                print("✅ Dependencies installed. Please restart the runner.")
                sys.exit(0)
            except Exception as e:
                print(f"❌ Failed to run setup_uv.sh: {e}")
                print("Please run 'bash setup_uv.sh' manually.")
                sys.exit(1)

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
