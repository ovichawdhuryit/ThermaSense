"""
ThermaSense - Bodda side server
--------------------------------
ESP32-CAM theke JPEG image receive kore, Groq-e vision call kore,
food name + temperature + time parse kore JSON return kore.

Routes:
  /         -> dashboard (JS diye image + result quietly update hoy, full reload NAI)
  /state    -> last detection JSON (dashboard JS eta fetch kore)
  /last     -> last chobi ta raw JPEG hisebe
  /predict  -> ESP32-CAM ekhane POST kore (image), JSON reply pay
  /health   -> "ok"

Deploy (Render):
  Start Command: gunicorn app:app --workers 1
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

MODEL = "qwen/qwen3.6-27b"

PROMPT = (
    "You are a food-safety cooking assistant. Look at the food in the image, "
    "identify it, and give the safe cooking temperature in Celsius and the "
    "cooking time in minutes based on WHO/USDA guidelines. "
    "Do NOT explain your reasoning. Do NOT think out loud. Do NOT use <think> tags. "
    "Output ONLY one single line in EXACTLY this format and nothing else:\n"
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

# --- unhandled error hole traceback log e, kintu 404/405 normal thakbe ---
import traceback
import html as html_lib
from werkzeug.exceptions import HTTPException


@app.errorhandler(Exception)
def _log_all_errors(e):
    if isinstance(e, HTTPException):
        return e
    print("=== UNHANDLED ERROR ===")
    traceback.print_exc()
    return jsonify(error=str(e)), 500


LAST = {
    "image": None, "food": None, "temp": None, "time": None,
    "matched": None, "raw": None, "at": None,
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
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
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

    LAST["image"] = img_bytes
    LAST["at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    b64 = base64.b64encode(img_bytes).decode("utf-8")

    payload = {
        "model": MODEL,
        "temperature": 0.2,
        "max_tokens": 512,
        "reasoning_effort": "none",
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
            timeout=60,
        )
    except Exception as e:
        msg = f"Groq connection fail: {e}"
        print("!!!", msg)
        LAST["raw"] = msg
        return jsonify(error=msg), 502

    print(">>> Groq status:", r.status_code)
    print(">>> Groq body:", r.text[:800])

    if r.status_code != 200:
        msg = f"Groq returned {r.status_code}: {r.text[:300]}"
        LAST["raw"] = msg
        return jsonify(error=msg), 502

    try:
        raw = r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        msg = f"Groq response parse fail: {e} | body={r.text[:300]}"
        print("!!!", msg)
        LAST["raw"] = msg
        return jsonify(error=msg), 502

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


@app.route("/state")
def state():
    """Dashboard JS eta fetch kore -- full page reload chara update korar jonno."""
    return jsonify(
        food=LAST["food"],
        temp=LAST["temp"],
        time=LAST["time"],
        matched=LAST["matched"],
        at=LAST["at"],
        raw=LAST["raw"],
        has_image=LAST["image"] is not None,
    )


@app.route("/last")
def last_image():
    if LAST["image"] is None:
        return "kono chobi ekhono ashe nai", 404
    return Response(LAST["image"], mimetype="image/jpeg")


@app.route("/favicon.ico")
def favicon():
    return "", 204


@app.route("/")
def dashboard():
    """
    NO meta-refresh. JS 4 second por por /state fetch kore image + text update kore.
    Page ta ar 'continuously reload' hobe na.
    """
    html = """<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ThermaSense</title>
<style>
  body{font-family:system-ui,Arial,sans-serif;background:#111;color:#eee;
       max-width:640px;margin:0 auto;padding:20px}
  h1{font-size:20px;letter-spacing:.5px}
  .grid{display:flex;gap:10px;margin-top:14px;flex-wrap:wrap}
  .card{flex:1;min-width:120px;background:#1c1c1c;border:1px solid #333;
         border-radius:12px;padding:12px 14px}
  .card span{display:block;font-size:12px;color:#888;margin-bottom:4px}
  .card b{font-size:20px}
  .muted{color:#888;font-size:13px}
  img{max-width:100%;border-radius:12px;border:1px solid #333}
  pre{background:#000;padding:10px;border-radius:8px;overflow:auto;font-size:12px}
  summary{cursor:pointer;color:#9cf}
</style></head>
<body>
  <h1>ThermaSense - Last Detection</h1>
  <div id="empty" class="muted">Ekhono kono chobi ashe nai. ESP32-CAM theke ekta chobi pathaao.</div>
  <div id="content" style="display:none">
    <img id="shot" alt="last food image"/>
    <div class="grid">
      <div class="card"><span>Food</span><b id="food">-</b></div>
      <div class="card"><span>Temperature</span><b id="temp">-</b></div>
      <div class="card"><span>Time</span><b id="time">-</b></div>
    </div>
    <p class="muted" id="matched"></p>
    <p class="muted" id="at"></p>
    <details><summary>LLM raw output</summary><pre id="raw"></pre></details>
  </div>

<script>
async function tick(){
  try{
    const r = await fetch('/state', {cache:'no-store'});
    const s = await r.json();
    if(!s.has_image){
      document.getElementById('empty').style.display='block';
      document.getElementById('content').style.display='none';
      return;
    }
    document.getElementById('empty').style.display='none';
    document.getElementById('content').style.display='block';
    // sudhu image ta re-fetch kori, page reload noy
    document.getElementById('shot').src = '/last?t=' + Date.now();
    document.getElementById('food').textContent = s.food;
    document.getElementById('temp').textContent = s.temp + ' \\u00B0C';
    document.getElementById('time').textContent = s.time + ' min';
    document.getElementById('matched').textContent =
      s.matched ? 'ref table e match korese' : 'table e match hoy nai (safe default)';
    document.getElementById('at').textContent = 'Chobi asher somoy: ' + s.at;
    document.getElementById('raw').textContent = s.raw || '';
  }catch(e){ /* network hiccup, next tick e abar try korbe */ }
}
tick();
setInterval(tick, 4000);
</script>
</body></html>"""
    return html


@app.route("/health")
def health():
    return "ok"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
