"""
config.py — Central configuration for Empathy Engine.
Set your tokens here OR use environment variables (recommended).
"""

import os

# ── HuggingFace ───────────────────────────────────────────────────────────────
# Option A: set env var:  export HF_TOKEN=hf_xxxxxxxxxxxx
# Option B: paste token directly below (not recommended for git repos)


# Primary emotion model (7-class, English)
# Requires HF token only if the repo is gated; this one is public.
EMOTION_MODEL: str = "j-hartmann/emotion-english-distilroberta-base"

# ── TTS ───────────────────────────────────────────────────────────────────────
TTS_LANGUAGE: str = "en"       # gTTS language code
TTS_SLOW_MODE: bool = False    # gTTS slow mode (overridden by rate modulation)

# ── Audio ─────────────────────────────────────────────────────────────────────
OUTPUT_SAMPLE_RATE: int = 44100   # Hz — normalize all outputs to this
MAX_TEXT_LENGTH: int = 500        # characters

# ── Flask ─────────────────────────────────────────────────────────────────────
FLASK_PORT: int = 5000
FLASK_DEBUG: bool = True

# Keep last N audio files in static/audio (older ones auto-deleted)
MAX_AUDIO_FILES: int = 20
