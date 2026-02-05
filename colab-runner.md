# 🚀 TTB Runner for Google Colab

Use this guide to run the TTB (Transcription Telegram Bot) in Google Colab.

### Setup Instructions
1.  **Secrets**: Open the 'Secrets' tab (🔑 key icon on the left) in Colab and add:
    *   `TELEGRAM_BOT_TOKEN`
    *   `TELEGRAM_CHAT_ID`
    *   `GEMINI_API_KEY` (Optional, for summarization)
    *   `GITHUB_TOKEN` (Optional, if your repo is private)
2.  **Runtime**: Change runtime type to **GPU** (Runtime > Change runtime type > T4 GPU).
3.  **Run**: Copy the code block below into a single cell and run it.

```python
import os
import sys
from google.colab import userdata

# --- CONFIGURATION ---
# ⚠️ REPLACE THIS WITH YOUR REPOSITORY URL IF NEEDED
REPO_URL = "https://github.com/arinadi/TTB.git" 
REPO_NAME = "TTB"
# ---------------------

print("🔄 Checking environment...")

try:
    # Try to use GITHUB_TOKEN if available for private repos
    token = userdata.get('GITHUB_TOKEN')
    if token and "github.com" in REPO_URL:
        REPO_URL = REPO_URL.replace("https://", f"https://{token}@")
except Exception:
    pass

# 1. Clone or Update Repository
if not os.path.exists(REPO_NAME):
    print(f"⏳ Cloning {REPO_NAME}...")
    !git clone {REPO_URL}
    %cd {REPO_NAME}
else:
    print(f"⏳ Updating {REPO_NAME}...")
    %cd {REPO_NAME}
    !git pull

# 2. Install Dependencies
print("⏳ Installing dependencies...")
!pip install -q -r requirements.txt
print("✅ Dependencies installed.")

# 3. Run the Bot
print("🚀 Starting TTB...")
!python main.py
```
