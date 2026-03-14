# 🎙 Empathy Engine

> AI-powered Text-to-Speech that speaks with *feeling* — dynamically modulating
> vocal characteristics based on detected emotion.

---

## ✨ Features

| Feature | Details |
|---|---|
| **7-class emotion detection** | joy · surprise · neutral · sadness · fear · anger · disgust |
| **Intensity scaling** | Modulation depth scales with model confidence score |
| **Vocal modulation** | Rate (speed), Pitch (semitones), Volume (amplitude) |
| **SSML pre-processing** | Emphasis on emotional keywords, prosody-aware pauses |
| **Web Interface** | Flask app with live playback, emotion bars, SSML inspector |
| **CLI** | Full-featured command-line tool |
| **HuggingFace auth** | Token support for authenticated model access |
| **VADER fallback** | Works offline if transformers is unavailable |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- **ffmpeg** (for audio processing)

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# Windows — https://ffmpeg.org/download.html
```

### Install

```bash
git clone https://github.com/YOUR_USERNAME/empathy-engine.git
cd empathy-engine
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> **Lightweight install** (no PyTorch — uses VADER fallback):
> ```bash
> pip install flask gtts pydub vaderSentiment
> ```

### Set HuggingFace Token

```bash
# Recommended: environment variable
export HF_TOKEN=hf_xxxxxxxxxxxxxxxx

# Or: edit config.py directly
HUGGINGFACE_TOKEN = "hf_xxxxxxxxxxxxxxxx"
```

### Run

**Web app**
```bash
python app.py
# → http://localhost:5000
```

**CLI**
```bash
python cli.py "I just got promoted! This is the best day ever!"
python cli.py                                    # interactive prompt
python cli.py --output out.mp3 "Your text here"
python cli.py --token hf_xxx "Text"              # pass token via flag
```

---

## 🏗 Architecture

```
Input Text
    │
    ▼
emotion_engine.py
    ├─ HuggingFace distilroberta (7 emotions, authenticated via HF_TOKEN)
    └─ VADER fallback (3 categories, no dependencies)
    │
    ▼
ssml_engine.py
    ├─ Keyword emphasis detection
    ├─ Emotion-specific pause insertion
    └─ Prosody annotation
    │
    ▼
tts_engine.py
    ├─ gTTS → raw MP3
    └─ pydub modulation:
         ├─ Volume  (dB = 20·log₁₀(multiplier))
         ├─ Rate    (frame rate resample trick)
         └─ Pitch   (semitone shift via 2^(st/12))
    │
    ▼
Output MP3
```

---

## 🎛 Emotion → Voice Mapping

| Emotion  | Rate  | Pitch  | Volume | Style |
|----------|-------|--------|--------|-------|
| joy      | 1.15× | +3 st  | 1.10×  | Upbeat, enthusiastic |
| surprise | 1.20× | +4 st  | 1.15×  | Energetic, excited |
| neutral  | 1.00× |  0 st  | 1.00×  | Calm, measured |
| sadness  | 0.85× | -3 st  | 0.85×  | Slow, subdued |
| fear     | 1.10× | -1 st  | 0.90×  | Tense, hushed |
| anger    | 1.25× | -2 st  | 1.20×  | Forceful, clipped |
| disgust  | 0.90× | -2.5st | 0.95×  | Flat, disapproving |

### Intensity Scaling

```python
final_rate = neutral_rate + (emotion_rate - neutral_rate) × confidence
```

"This is good" at 60% joy → mild modulation  
"THIS IS AMAZING!!!" at 97% joy → full modulation

---

## 📁 Structure

```
empathy_engine/
├── app.py              ← Flask web server
├── cli.py              ← Command-line interface
├── config.py           ← HF token & settings
├── emotion_engine.py   ← Emotion detection + voice mapping
├── ssml_engine.py      ← SSML & text pre-processing
├── tts_engine.py       ← TTS synthesis + audio modulation
├── requirements.txt
├── README.md
├── templates/
│   └── index.html      ← Web UI
└── static/audio/       ← Generated audio files
```

---

## Design Decisions

**gTTS over pyttsx3** — gTTS produces far more natural-sounding output. Robotic TTS defeats the purpose of an empathy engine.

**pydub modulation over SSML pitch/rate** — Google Cloud TTS SSML requires a paid API. pydub gives signal-level control with zero cost.

**distilroberta over TextBlob/VADER** — VADER only detects 3 sentiment classes. distilroberta provides 7 distinct emotions enabling much richer, more nuanced voice mapping.

**Intensity scaling** — a flat mapping (joy = always +3st) sounds artificial. Scaling by confidence makes the output feel proportional to the actual emotional content.

---

## License

MIT
