"""
ThermaSense - Bodda side server
--------------------------------
ESP32-CAM theke JPEG image receive kore, Groq-e LLaMA Vision call kore,
food name + temperature + time parse kore JSON return kore.

Notun ja ache:
  /         -> dashboard: last chobi + LLM ki detect korlo (browser e dekhar jonno)
  /last     -> last chobi ta raw JPEG hisebe
  /predict  -> ESP32-CAM ekhane POST kore (image), JSON reply pay
  /health   -> "ok"

Deploy (Render):
  Start Command: gunicorn app:app
  Env var:       GROQ_API_KEY = gsk_...
"""

import os
import re
import time
import base64
import requests
from flask import Flask, request, jsonify, Response

# ---------------- Config ----------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Groq console e current vision model dekhe niyo, deprecate hole ekhane bodlao.
MODEL = "llama-3.2-90b-vision-preview"

PROMPT = (
    "You are a food-safety cooking assistant. Look at the food in the image, "
    "identify it, and give the safe cooking temperature in Celsius and the "
    "cooking time in minutes based on WHO/USDA guidelines. "
    "Reply in EXACTLY this one-line format and nothing else:\n"
    "Food: <name> --- Temperature: <number>°C --- Time: <number> minutes"
)

# ---------------- Safety guardrail ----------------
SAFE_RANGE = {  # (min_temp, max_temp, min_time, max_time)
    "poultry":   (72, 78, 25, 30),
    "chicken":   (72, 78, 25, 30),
    "turkey":    (72, 78, 25, 30),
    "ground":    (69, 74, 20, 25),
    "beef":      (60, 74, 15, 25),
    "pork":      (69, 74, 20, 25),
    "lamb":      (60, 66, 15, 20),
    "steak":     (60, 66, 15, 20),
    "fish":      (60, 66, 10, 15),
    "egg":       (68, 72, 8, 10),
    "vegetable": (68, 78, 5, 10),
    "rice":      (58, 72, 12, 18),
    "lentil":    (82, 96, 18, 22),
    "dal":       (82, 96, 18, 22),
    "hot dog":   (70, 78, 3, 8),
    "sausage":   (70, 78, 3, 8),
}

app = Flask(__name__)

# ---------------- Last request state (browser e dekhar jonno) ----------------
LAST = {
    "image": None,       # raw JPEG bytes
    "food": None,
    "temp": None,
    "time": None,
    "matched": None,
    "raw": None,
    "at": None,          # kobe eshechilo
}


def clamp_to_safe(food, temp, minutes):
    food_l = food.lower()
    for key, (tmin, tmax, mmin, mmax) in SAFE_RANGE.items():
        if key in food_l:
            temp = max(tmin, min(tmax, temp))
            minutes = max(mmin, min(mmax, minutes))
            return temp, minutes, True
    return max(70, min(85, temp)), max(8, min(20, minutes)), False


def parse_llm_output(text):
    food = re.search(r"Food:\s*(.+?)\s*---", text)
    temp = re.search(r"Temperature:\s*([\d.]+)", text)
    tm = re.search(r"Time:\s*([\d.]+)", text)
    if not (food and temp and tm):
        return None
    return food.group(1).strip(), float(temp.group(1)), float(tm.group(1))


@app.route("/predict", methods=["POST"])
def predict():
    if not GROQ_API_KEY:
        return jsonify(error="GROQ_API_KEY set kora nai"), 500

    img_bytes = request.get_data()
    if not img_bytes:
        return jsonify(error="kono image pai nai"), 400

    # last chobi ta rekhe dei jate /last o / te dekha jay
    LAST["image"] = img_bytes
    LAST["at"] = time.strftime("%Y-%m-%d %H:%M:%S")

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
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
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
        LAST["raw"] = f"Groq call fail: {e}"
        return jsonify(error=f"Groq call fail: {e}"), 502

    print("LLM raw:", raw)
    LAST["raw"] = raw

    parsed = parse_llm_output(raw)
    if not parsed:
        LAST.update(food="unknown", temp=75, time=10, matched=False)
        return jsonify(food="unknown", temp=75, time=10, matched=False, raw=raw)

    food, temp, minutes = parsed
    temp, minutes, matched = clamp_to_safe(food, temp, minutes)
    temp, minutes = round(temp), round(minutes)

    LAST.update(food=food, temp=temp, time=minutes, matched=matched)

    return jsonify(food=food, temp=temp, time=minutes, matched=matched)


@app.route("/last")
def last_image():
    """Last chobi ta raw JPEG hisebe."""
    if LAST["image"] is None:
        return "kono chobi ekhono ashe nai", 404
    return Response(LAST["image"], mimetype="image/jpeg")


@app.route("/")
def dashboard():
    """Browser e last chobi + detection dekhabe. Auto-refresh 5s por por."""
    if LAST["image"] is None:
        body = "<p style='color:#888'>Ekhono kono chobi ashe nai. ESP32-CAM theke ekta chobi pathaao.</p>"
    else:
        matched_txt = "✅ ref table e match korese" if LAST["matched"] else "⚠️ table e match hoy nai (safe default)"
        body = f"""
          <img src="/last?t={int(time.time())}" alt="last food image"
               style="max-width:100%;border-radius:12px;border:1px solid #333"/>
          <div class="grid">
            <div class="card"><span>Food</span><b>{LAST['food']}</b></div>
            <div class="card"><span>Temperature</span><b>{LAST['temp']} &deg;C</b></div>
            <div class="card"><span>Time</span><b>{LAST['time']} min</b></div>
          </div>
          <p class="muted">{matched_txt}</p>
          <p class="muted">Chobi asher somoy: {LAST['at']}</p>
          <details><summary>LLM raw output</summary><pre>{LAST['raw']}</pre></details>
        """

    html = f"""<!doctype html>
    <html><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <meta http-equiv="refresh" content="5">
    <title>ThermaSense</title>
    <style>
      body{{font-family:system-ui,Arial,sans-serif;background:#111;color:#eee;
           max-width:640px;margin:0 auto;padding:20px}}
      h1{{font-size:20px;letter-spacing:.5px}}
      .grid{{display:flex;gap:10px;margin-top:14px;flex-wrap:wrap}}
      .card{{flex:1;min-width:120px;background:#1c1c1c;border:1px solid #333;
             border-radius:12px;padding:12px 14px}}
      .card span{{display:block;font-size:12px;color:#888;margin-bottom:4px}}
      .card b{{font-size:20px}}
      .muted{{color:#888;font-size:13px}}
      pre{{background:#000;padding:10px;border-radius:8px;overflow:auto;font-size:12px}}
      summary{{cursor:pointer;color:#9cf}}
    </style></head>
    <body>
      <h1>🍳 ThermaSense — Last Detection</h1>
      {body}
    </body></html>"""
    return html


@app.route("/health")
def health():
    return "ok"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
