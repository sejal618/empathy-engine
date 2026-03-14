"""
TTS Synthesizer — gTTS + pydub audio modulation + SSML pre-processing.
"""

import os
import math
import uuid
import tempfile
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "static" / "audio"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    from config import MAX_AUDIO_FILES, OUTPUT_SAMPLE_RATE, TTS_LANGUAGE, TTS_SLOW_MODE
except ImportError:
    MAX_AUDIO_FILES, OUTPUT_SAMPLE_RATE, TTS_LANGUAGE, TTS_SLOW_MODE = 20, 44100, "en", False


def _cleanup_old_files():
    files = sorted(OUTPUT_DIR.glob("*.mp3"), key=lambda f: f.stat().st_mtime)
    while len(files) > MAX_AUDIO_FILES:
        files.pop(0).unlink(missing_ok=True)


def synthesize(text: str, params, ssml_plain_text: str = None, filename: str = None) -> Path:
    from gtts import gTTS
    from pydub import AudioSegment

    if filename is None:
        filename = f"empathy_{uuid.uuid4().hex[:8]}.mp3"

    output_path = OUTPUT_DIR / filename
    tts_text = ssml_plain_text if ssml_plain_text else text

    tts = gTTS(text=tts_text, lang=TTS_LANGUAGE, slow=TTS_SLOW_MODE)
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name
    tts.save(tmp_path)

    audio = AudioSegment.from_mp3(tmp_path)
    os.unlink(tmp_path)

    if params.volume != 1.0:
        db = 20 * math.log10(max(params.volume, 0.01))
        audio = audio + db

    if params.rate != 1.0:
        orig_rate = audio.frame_rate
        audio = audio._spawn(audio.raw_data, overrides={"frame_rate": int(orig_rate * params.rate)})
        audio = audio.set_frame_rate(orig_rate)

    if params.pitch != 0.0:
        factor = 2 ** (params.pitch / 12.0)
        audio = audio._spawn(audio.raw_data, overrides={"frame_rate": int(audio.frame_rate * factor)})
        audio = audio.set_frame_rate(OUTPUT_SAMPLE_RATE)

    audio.export(str(output_path), format="mp3", bitrate="128k")
    _cleanup_old_files()
    return output_path


def process_text(text: str, output_filename: str = None):
    from emotion_engine import detect_emotion, map_emotion_to_voice
    from ssml_engine import build_ssml

    emotion = detect_emotion(text)
    params  = map_emotion_to_voice(emotion)
    ssml    = build_ssml(text, emotion, params)
    path    = synthesize(text, params, ssml_plain_text=ssml.plain_text, filename=output_filename)

    return path, emotion, params, ssml
