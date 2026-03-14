"""
Emotion Engine — detects emotion and maps it to vocal parameters.
Uses HuggingFace distilroberta model (authenticated via HF_TOKEN).
Falls back to VADER if transformers is unavailable.
"""

from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class EmotionResult:
    label: str          # e.g. "joy", "anger"
    score: float        # 0.0 – 1.0 confidence
    category: str       # "positive" | "negative" | "neutral"
    all_scores: dict = field(default_factory=dict)  # full distribution

@dataclass
class VocalParams:
    rate: float         # speed multiplier  (1.0 = normal, >1 faster, <1 slower)
    pitch: float        # semitone shift     (0 = no change, +N = higher, -N = lower)
    volume: float       # amplitude multiplier (1.0 = normal)
    description: str    # human-readable explanation

# ---------------------------------------------------------------------------
# Emotion → vocal parameter mapping
# ---------------------------------------------------------------------------

# Base configs per emotion label
# rate, pitch, volume
_EMOTION_MAP = {
    "joy":       {"rate": 1.15, "pitch":  3.0, "volume": 1.10},
    "surprise":  {"rate": 1.20, "pitch":  4.0, "volume": 1.15},
    "neutral":   {"rate": 1.00, "pitch":  0.0, "volume": 1.00},
    "sadness":   {"rate": 0.85, "pitch": -3.0, "volume": 0.85},
    "fear":      {"rate": 1.10, "pitch": -1.0, "volume": 0.90},
    "anger":     {"rate": 1.25, "pitch": -2.0, "volume": 1.20},
    "disgust":   {"rate": 0.90, "pitch": -2.5, "volume": 0.95},
}

_CATEGORY_MAP = {
    "joy":      "positive",
    "surprise": "positive",
    "neutral":  "neutral",
    "sadness":  "negative",
    "fear":     "negative",
    "anger":    "negative",
    "disgust":  "negative",
}

_DESCRIPTIONS = {
    "joy":      "Upbeat and enthusiastic — faster pace, higher pitch",
    "surprise": "Excited and energetic — quickened speech, elevated pitch",
    "neutral":  "Calm and measured — standard delivery",
    "sadness":  "Slow and subdued — reduced pace and pitch",
    "fear":     "Tense and hushed — slightly faster, lower volume",
    "anger":    "Forceful and clipped — fast, louder, lower pitch",
    "disgust":  "Flat and disapproving — slightly slower, lower pitch",
}


def map_emotion_to_voice(emotion: EmotionResult) -> VocalParams:
    """
    Apply intensity scaling: the higher the confidence score,
    the more extreme the modulation (interpolated from neutral).
    """
    base = _EMOTION_MAP.get(emotion.label, _EMOTION_MAP["neutral"])
    neutral = _EMOTION_MAP["neutral"]

    intensity = emotion.score  # 0.0 – 1.0

    # Interpolate between neutral and full base config
    rate   = neutral["rate"]   + (base["rate"]   - neutral["rate"])   * intensity
    pitch  = neutral["pitch"]  + (base["pitch"]  - neutral["pitch"])  * intensity
    volume = neutral["volume"] + (base["volume"] - neutral["volume"]) * intensity

    desc = _DESCRIPTIONS.get(emotion.label, "Standard delivery")
    return VocalParams(rate=round(rate, 3), pitch=round(pitch, 2),
                       volume=round(volume, 3), description=desc)


# ---------------------------------------------------------------------------
# Emotion detection  (HuggingFace primary, VADER fallback)
# ---------------------------------------------------------------------------

_pipeline_cache = None  # load model once, reuse across requests


def _detect_with_transformers(text: str) -> Optional[EmotionResult]:
    global _pipeline_cache
    try:
        from transformers import pipeline as hf_pipeline
        try:
            from config import HUGGINGFACE_TOKEN, EMOTION_MODEL
        except ImportError:
            HUGGINGFACE_TOKEN, EMOTION_MODEL = "", "j-hartmann/emotion-english-distilroberta-base"

        if _pipeline_cache is None:
            kwargs = {"task": "text-classification", "model": EMOTION_MODEL, "top_k": None}
            if HUGGINGFACE_TOKEN:
                kwargs["token"] = HUGGINGFACE_TOKEN
            _pipeline_cache = hf_pipeline(**kwargs)

        results = _pipeline_cache(text)[0]
        results.sort(key=lambda x: x["score"], reverse=True)
        top = results[0]
        label = top["label"].lower()
        score = float(top["score"])
        all_scores = {r["label"].lower(): round(float(r["score"]), 4) for r in results}
        category = _CATEGORY_MAP.get(label, "neutral")
        return EmotionResult(label=label, score=score, category=category, all_scores=all_scores)
    except Exception as e:
        print(f"[emotion_engine] HuggingFace unavailable: {e}")
        return None


def _detect_with_vader(text: str) -> EmotionResult:
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        vs = SentimentIntensityAnalyzer().polarity_scores(text)
        compound = vs["compound"]
        if compound >= 0.35:
            label, score = "joy", round(min(1.0, 0.5 + compound / 2), 3)
        elif compound <= -0.35:
            label, score = "sadness", round(min(1.0, 0.5 + abs(compound) / 2), 3)
        else:
            label, score = "neutral", round(1.0 - abs(compound), 3)
        category = _CATEGORY_MAP.get(label, "neutral")
        return EmotionResult(label=label, score=score, category=category,
                             all_scores={label: score})
    except Exception:
        return EmotionResult(label="neutral", score=1.0,
                             category="neutral", all_scores={"neutral": 1.0})


def detect_emotion(text: str) -> EmotionResult:
    """Detect emotion using best available model (HuggingFace → VADER fallback)."""
    result = _detect_with_transformers(text)
    return result if result else _detect_with_vader(text)
