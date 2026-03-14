"""
SSML Engine — builds Speech Synthesis Markup Language tags
to add emphasis, pauses, and prosody to TTS output.

Note: gTTS does NOT support SSML natively. This module serves two purposes:
  1. When using Google Cloud TTS (SSML-capable), it generates full SSML.
  2. For gTTS, it pre-processes the text (inserting punctuation pauses,
     capitalisation cues) to coax better prosody from the engine.
"""

import re
from dataclasses import dataclass
from typing import List
from emotion_engine import EmotionResult, VocalParams


@dataclass
class SSMLResult:
    ssml: str           # Full SSML string (for Cloud TTS)
    plain_text: str     # Pre-processed plain text (for gTTS fallback)
    annotations: List[str]   # Human-readable list of what was applied


# Words that deserve extra emphasis per emotion
_EMPHASIS_TRIGGERS = {
    "joy":      ["amazing", "wonderful", "great", "love", "best", "fantastic",
                 "excellent", "perfect", "brilliant", "incredible", "thrilled"],
    "anger":    ["never", "always", "unacceptable", "terrible", "wrong",
                 "ridiculous", "disgusting", "awful", "worst", "furious"],
    "sadness":  ["sorry", "unfortunately", "regret", "miss", "lost", "alone",
                 "hopeless", "devastated", "heartbroken", "terrible"],
    "fear":     ["danger", "warning", "careful", "risk", "threat", "urgent",
                 "critical", "emergency", "serious", "worried"],
    "surprise": ["suddenly", "unbelievable", "unexpected", "incredible",
                 "shocked", "astonished", "wow", "really", "actually"],
    "disgust":  ["horrible", "disgusting", "awful", "terrible", "gross",
                 "repulsive", "revolting", "nasty"],
    "neutral":  [],
}

# Prosody rate/pitch tags per emotion (for Cloud TTS SSML)
_SSML_PROSODY = {
    "joy":      {"rate": "fast",    "pitch": "+3st",  "volume": "loud"},
    "surprise": {"rate": "fast",    "pitch": "+4st",  "volume": "loud"},
    "neutral":  {"rate": "medium",  "pitch": "+0st",  "volume": "medium"},
    "sadness":  {"rate": "slow",    "pitch": "-3st",  "volume": "soft"},
    "fear":     {"rate": "medium",  "pitch": "-1st",  "volume": "medium"},
    "anger":    {"rate": "fast",    "pitch": "-2st",  "volume": "x-loud"},
    "disgust":  {"rate": "medium",  "pitch": "-2.5st","volume": "medium"},
}

# Pause durations (ms) after sentence-ending punctuation
_PAUSE_MAP = {
    "joy":      300,
    "surprise": 200,
    "neutral":  400,
    "sadness":  600,
    "fear":     350,
    "anger":    200,
    "disgust":  450,
}


def build_ssml(text: str, emotion: EmotionResult, params: VocalParams) -> SSMLResult:
    """Generate SSML markup and pre-processed plain text from emotion analysis."""
    label = emotion.label
    annotations = []

    # ── 1. Emphasis: find trigger words and wrap them ─────────────────────────
    triggers = _EMPHASIS_TRIGGERS.get(label, [])
    emphasis_level = "strong" if emotion.score > 0.75 else "moderate"

    def emphasize(match):
        word = match.group(0)
        annotations.append(f"Emphasis on '{word}'")
        return f'<emphasis level="{emphasis_level}">{word}</emphasis>'

    ssml_text = text
    if triggers:
        pattern = r'\b(' + '|'.join(re.escape(w) for w in triggers) + r')\b'
        ssml_text = re.sub(pattern, emphasize, ssml_text, flags=re.IGNORECASE)

    # ── 2. Sentence breaks → SSML pauses ─────────────────────────────────────
    pause_ms = _PAUSE_MAP.get(label, 400)
    pause_tag = f'<break time="{pause_ms}ms"/>'
    # Replace ". ", "! ", "? " with pause tag
    ssml_text = re.sub(r'([.!?])\s+', rf'\1{pause_tag} ', ssml_text)
    if ssml_text and ssml_text[-1] in '.!?':
        ssml_text += pause_tag
    annotations.append(f"Sentence pauses: {pause_ms}ms ({label})")

    # ── 3. Exclamation marks → say-as for extra punch ─────────────────────────
    if label in ("anger", "joy", "surprise") and "!" in text:
        annotations.append("Exclamation marks detected — increased prosody")

    # ── 4. Wrap in <prosody> and <speak> ─────────────────────────────────────
    prosody = _SSML_PROSODY.get(label, _SSML_PROSODY["neutral"])
    full_ssml = (
        f'<speak>'
        f'<prosody rate="{prosody["rate"]}" pitch="{prosody["pitch"]}" '
        f'volume="{prosody["volume"]}">'
        f'{ssml_text}'
        f'</prosody>'
        f'</speak>'
    )
    annotations.append(
        f"Prosody: rate={prosody['rate']}, pitch={prosody['pitch']}, "
        f"volume={prosody['volume']}"
    )

    # ── 5. Plain-text fallback for gTTS ──────────────────────────────────────
    # Strip all XML tags for gTTS, but keep punctuation cues
    plain = re.sub(r'<[^>]+>', '', full_ssml)
    # For sadness/fear: add "..." pauses manually
    if label in ("sadness", "fear"):
        plain = re.sub(r'([.!?])\s+', r'\1... ', plain)
    plain = plain.strip()

    return SSMLResult(ssml=full_ssml, plain_text=plain, annotations=annotations)
