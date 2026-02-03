"""Configuration for the pipeline."""
import os
from pathlib import Path

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Paths
PROJECT_ROOT = Path(__file__).parent
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Create directories if they don't exist
INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Gemini Settings
GEMINI_MODEL = "gemini-3-pro-image-preview"  # or "gemini-1.5-pro"
