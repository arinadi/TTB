#!/bin/bash
# 🚀 Fast Dependency Installation using uv
# ------------------------------------------------------------------------------
# This script uses uv (ultrafast Python package installer) to install
# all dependencies in under 30 seconds, compared to 2+ minutes with pip.
# ------------------------------------------------------------------------------

echo "⏳ Installing uv package manager..."
pip install uv -q

echo "⏳ Installing dependencies with uv..."
uv pip install -r requirements.txt --system

echo "✅ Installation complete!"
