"""
Empathy Engine — Flask Web Application
"""

import os
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory

try:
    from config import FLASK_PORT, FLASK_DEBUG, MAX_TEXT_LENGTH
except ImportError:
    FLASK_PORT, FLASK_DEBUG, MAX_TEXT_LENGTH = 5000, True, 500

app = Flask(__name__)
AUDIO_DIR = Path(__file__).parent / "static" / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/synthesize", methods=["POST"])
def synthesize_route():
    data = request.get_json()
    text = (data or {}).get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400
    if len(text) > MAX_TEXT_LENGTH:
        return jsonify({"error": f"Text too long (max {MAX_TEXT_LENGTH} chars)"}), 400

    try:
        from tts_engine import process_text
        output_path, emotion, params, ssml = process_text(text)

        return jsonify({
            "audio_url": f"/audio/{output_path.name}",
            "emotion": {
                "label":    emotion.label,
                "score":    round(emotion.score * 100, 1),
                "category": emotion.category,
                "all_scores": {k: round(v * 100, 1) for k, v in emotion.all_scores.items()},
            },
            "voice_params": {
                "rate":        params.rate,
                "pitch":       params.pitch,
                "volume":      params.volume,
                "description": params.description,
            },
            "ssml": {
                "plain_text":  ssml.plain_text,
                "annotations": ssml.annotations,
            }
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/audio/<filename>")
def serve_audio(filename):
    return send_from_directory(str(AUDIO_DIR), filename)


if __name__ == "__main__":
    app.run(debug=FLASK_DEBUG, port=FLASK_PORT)
