"""
ThermaSense - Bodda side server
--------------------------------
ESP32-CAM theke JPEG image receive kore, Groq-e LLaMA 3.2 Vision call kore,
food name + temperature + time parse kore JSON return kore.

Cholano:
    pip install flask requests
    export GROQ_API_KEY="gsk_..."      # Windows: set GROQ_API_KEY=gsk_...
    python server.py

Ei PC r local IP ta (ipconfig / ifconfig) ESP32-CAM code er serverUrl e boshai dio.
"""

import os
import re
import base64
import requests
from flask import Flask, request, jsonify

# ---------------- Config ----------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# NOTE: 'llama-3.2-11b-vision-preview' Groq-e deprecated hoye gese thakte pare.
# Groq console e giye current vision model dekhe niyo (e.g. meta-llama/llama-4-scout-17b-16e-instruct).
MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# LLM ke ei exact format e reply dite bola hocche jate regex diye parse kora jay.
PROMPT = (
    "You are a food-safety cooking assistant. Look at the food in the image, "
    "identify it, and give the safe cooking temperature in Celsius and the "
    "cooking time in minutes based on WHO/USDA guidelines. "
    "Reply in EXACTLY this one-line format and nothing else:\n"
    "Food: <name> --- Temperature: <number>°C --- Time: <number> minutes"
)

# ---------------- Safety guardrail ----------------
# LLM je value dey seta WHO/USDA range e clamp kora hoy jate hallucination
# direct heater e na jay. Ei tuku thakle report er 'validation' angle o strong hoy.
SAFE_RANGE = {  # (min_temp, max_temp, min_time, max_time)
    "poultry":     (72, 78, 25, 30),
    "chicken":     (72, 78, 25, 30),
    "turkey":      (72, 78, 25, 30),
    "ground":      (69, 74, 20, 25),
    "beef":        (60, 74, 15, 25),
    "pork":        (69, 74, 20, 25),
    "lamb":        (60, 66, 15, 20),
    "steak":       (60, 66, 15, 20),
    "fish":        (60, 66, 10, 15),
    "egg":         (68, 72, 8, 10),
    "vegetable":   (68, 78, 5, 10),
    "rice":        (58, 72, 12, 18),
    "lentil":      (82, 96, 18, 22),
    "dal":         (82, 96, 18, 22),
    "hot dog":     (70, 78, 3, 8),
    "sausage":     (70, 78, 3, 8),
}

app = Flask(__name__)


def clamp_to_safe(food, temp, minutes):
    """LLM output ke ref table er range e boshai. Match na pele default 75C/10min."""
    food_l = food.lower()
    for key, (tmin, tmax, mmin, mmax) in SAFE_RANGE.items():
        if key in food_l:
            temp = max(tmin, min(tmax, temp))
            minutes = max(mmin, min(mmax, minutes))
            return temp, minutes, True
    # unknown food -> conservative safe default
    return max(70, min(85, temp)), max(8, min(20, minutes)), False


def parse_llm_output(text):
    """Regex diye Food / Temperature / Time ber kora."""
    food = re.search(r"Food:\s*(.+?)\s*---", text)
    temp = re.search(r"Temperature:\s*([\d.]+)", text)
    time = re.search(r"Time:\s*([\d.]+)", text)
    if not (food and temp and time):
        return None
    return food.group(1).strip(), float(temp.group(1)), float(time.group(1))


@app.route("/predict", methods=["POST"])
def predict():
    if not GROQ_API_KEY:
        return jsonify(error="GROQ_API_KEY set kora nai"), 500

    img_bytes = request.get_data()
    if not img_bytes:
        return jsonify(error="kono image pai nai"), 400

    b64 = base64.b64encode(img_bytes).decode("utf-8")

    payload = {
        "model": MODEL,
        "temperature": 0.2,
        "max_tokens": 100,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                ],
            }
        ],
    }

    try:
        r = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json=payload,
            timeout=30,
        )
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return jsonify(error=f"Groq call fail: {e}"), 502

    print("LLM raw:", raw)

    parsed = parse_llm_output(raw)
    if not parsed:
        # format match na korle safe fallback -> heater jate misfire na kore
        return jsonify(food="unknown", temp=75, time=10, matched=False, raw=raw)

    food, temp, minutes = parsed
    temp, minutes, matched = clamp_to_safe(food, temp, minutes)

    # ESP/Arduino jate ekdom short ekta line pay: "75,10"
    return jsonify(food=food, temp=round(temp), time=round(minutes), matched=matched)


@app.route("/health")
def health():
    return "ok"


if __name__ == "__main__":
    # Cloud host (Render/Railway) PORT env var dey; na thakle 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
