#!/bin/bash
# 🚀 Fast Dependency Installation using uv
# ------------------------------------------------------------------------------
# This script uses uv (ultrafast Python package installer) to install
# all dependencies in under 30 seconds, compared to 2+ minutes with pip.
# ------------------------------------------------------------------------------

echo "⏳ Installing uv package manager..."
pip install uv -q

echo "⏳ Checking hardware..."
if command -v nvidia-smi &> /dev/null; then
    echo "🚀 GPU detected. Installing full requirements..."
    REQUIREMENTS_FILE="requirements.txt"
else
    echo "⚠️ No GPU detected. Installing minimal CPU requirements..."
    REQUIREMENTS_FILE="requirements_cpu.txt"
fi

echo "⏳ Installing dependencies from $REQUIREMENTS_FILE with uv..."
cat $REQUIREMENTS_FILE
uv pip install -r $REQUIREMENTS_FILE --system

echo "✅ Installation complete!"
