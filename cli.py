#!/usr/bin/env python3
"""
Empathy Engine — CLI
Usage:
  python cli.py "Your text here"
  python cli.py                         # interactive prompt
  python cli.py --output result.mp3 "Text"
  HF_TOKEN=hf_xxx python cli.py "Text"  # with HuggingFace auth
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(description="Empathy Engine: Emotionally expressive TTS")
    parser.add_argument("text", nargs="?", help="Text to synthesize")
    parser.add_argument("--output", "-o", default=None, help="Output filename")
    parser.add_argument("--no-play", action="store_true", help="Skip auto-play")
    parser.add_argument("--token", default=None, help="HuggingFace token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    # Allow passing token via CLI flag too
    if args.token:
        import os; os.environ["HF_TOKEN"] = args.token

    text = args.text
    if not text:
        print("🎙  Empathy Engine")
        print("─" * 44)
        text = input("Enter text: ").strip()
        if not text:
            print("No text provided. Exiting.")
            sys.exit(1)

    print("\n🔍 Detecting emotion…")
    from emotion_engine import detect_emotion, map_emotion_to_voice
    from ssml_engine import build_ssml
    from tts_engine import synthesize

    emotion = detect_emotion(text)
    params  = map_emotion_to_voice(emotion)
    ssml    = build_ssml(text, emotion, params)

    # ── Display results ───────────────────────────────────────────────────────
    print(f"\n   Emotion  : {emotion.label.upper()}  ({emotion.score*100:.1f}% confidence)")
    print(f"   Category : {emotion.category}")

    if emotion.all_scores and len(emotion.all_scores) > 1:
        print("\n   All scores:")
        sorted_scores = sorted(emotion.all_scores.items(), key=lambda x: x[1], reverse=True)
        for lbl, sc in sorted_scores:
            bar = "█" * int(sc * 20)
            print(f"     {lbl:<10} {bar:<20} {sc*100:.1f}%")

    print(f"\n   Rate     : {params.rate:.3f}×")
    print(f"   Pitch    : {params.pitch:+.2f} semitones")
    print(f"   Volume   : {params.volume:.3f}×")
    print(f"   Style    : {params.description}")

    if ssml.annotations:
        print("\n   SSML applied:")
        for a in ssml.annotations:
            print(f"     ⚙  {a}")

    print("\n🔊 Synthesizing audio…")
    output_path = synthesize(text, params, ssml_plain_text=ssml.plain_text, filename=args.output)
    print(f"   ✅ Saved: {output_path.resolve()}")

    if not args.no_play:
        try:
            import subprocess, platform, os
            plat = platform.system()
            if plat == "Darwin":
                subprocess.run(["afplay", str(output_path)])
            elif plat == "Linux":
                subprocess.run(["mpg123", "-q", str(output_path)])
            elif plat == "Windows":
                os.startfile(str(output_path))
        except Exception:
            print("   (Auto-play unavailable — open the file manually)")

if __name__ == "__main__":
    main()
