
"""
ThermaSense - Flask Backend
---------------------------
Receives JPEG images from ESP32-CAM, sends them to Groq Vision,
identifies food, applies the safe-temperature mapping, and
returns JSON for the ESP8266 and dashboard.

Routes:
  /         -> Dashboard
  /state    -> Latest detection JSON
  /last     -> Latest uploaded JPEG image
  /predict  -> Receives image from ESP32-CAM
  /health   -> Health check

Render start command:
  gunicorn app:app --workers 1

Environment variable:
  GROQ_API_KEY = your_groq_api_key
"""

import os
import re
import time
import base64
import traceback

import requests

from flask import Flask, request, jsonify, Response
from werkzeug.exceptions import HTTPException


# =========================================================
# CONFIGURATION
# =========================================================

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

MODEL = "qwen/qwen3.8-27b"

app = Flask(__name__)


# =========================================================
# FOOD RECOGNITION PROMPT
# =========================================================


PROMPT = """
You are ThermaSense, an intelligent food recognition
and cooking parameter estimation assistant.

Your task is to identify the food and provide TWO important
cooking parameters:

1. Target cooking temperature in Celsius.
2. Estimated cooking duration in minutes.

FOOD IDENTIFICATION:

Carefully inspect the uploaded image.

Identify:
- Actual food and ingredient.
- Whether it is raw, partially cooked, or fully cooked.
- Whether it is meat, fish, poultry, vegetables, etc.
- Whether it is a ready-to-eat or processed food.
- Visible thickness and approximate portion size.

Do not confuse:
- Fried fish with chicken.
- Beef with mutton.
- Raw meat with cooked meat.
- Actual food with food advertisements or pictures
  displayed on phones, screens, menus, or packaging.

COOKING PARAMETERS:

If the image clearly shows an ingredient that needs cooking:

- Provide an appropriate target INTERNAL temperature.
- Provide a practical estimated cooking duration.
- Consider the visible food size and thickness.
- Assume one standard portion.
- Assume the food is being cooked using the configured
  ThermaSense heating setup.
- Do not return zero time for an identifiable,
  cookable ingredient merely because cooking time varies.
- Give your best reasonable estimate and clearly treat
  the duration as an estimate, not proof of doneness.

If the food is already cooked, ready to eat, an advertisement,
or not suitable for the configured cooking process,
do not recommend starting the heater.

If identification is uncertain, return zero values.

OUTPUT RULES:

Return ONLY this exact format:

Food: <food name> --- Temperature: <temperature>°C --- Time: <minutes> minutes

For uncertain, ready-to-eat, or non-cookable images:

Food: uncertain --- Temperature: 0°C --- Time: 0 minutes
"""



# =========================================================
# SAFE TEMPERATURE MAPPING
# =========================================================
# Format:
# food keyword: (minimum temperature, maximum temperature,
#                minimum time, maximum time)
#
# These are project configuration ranges, not a guarantee
# that food is safe. Verify food-specific safety guidance
# before using this system for real cooking.

SAFE_RANGE = {
    "poultry":   (72, 78, 25, 30),
    "chicken":   (72, 78, 25, 30),
    "turkey":    (72, 78, 25, 30),

    "ground":    (69, 74, 20, 25),

    "beef":      (60, 74, 15, 25),
    "pork":      (69, 74, 20, 25),
    "lamb":      (60, 66, 15, 20),
    "mutton":    (60, 66, 15, 20),
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


# =========================================================
# LAST DETECTION STORAGE
# =========================================================

LAST = {
    "image": None,
    "food": "unknown",
    "temp": 0,
    "time": 0,
    "matched": False,
    "raw": None,
    "at": None,
}


# =========================================================
# ERROR HANDLING
# =========================================================

@app.errorhandler(Exception)
def handle_all_errors(e):

    if isinstance(e, HTTPException):
        return e

    print("=== UNHANDLED ERROR ===")
    traceback.print_exc()

    return jsonify(error=str(e)), 500


# =========================================================
# RESET / UNKNOWN FOOD STATE
# =========================================================

def set_unknown_state(raw=None):
    """
    Reset cooking values.

    IMPORTANT:
    temp=0 and time=0 prevent the ESP8266 from starting
    a cooking session when food recognition fails.
    """

    LAST.update(
        food="Uncertain",
        temp=0,
        time=0,
        matched=False,
        raw=raw
    )


# =========================================================
# SAFE RANGE VALIDATION
# =========================================================

def clamp_to_safe(food, temp, minutes):

    food_l = food.lower().strip()

    for key, (tmin, tmax, mmin, mmax) in SAFE_RANGE.items():

        if key in food_l:

            temp = max(tmin, min(tmax, temp))
            minutes = max(mmin, min(mmax, minutes))

            return temp, minutes, True

    # IMPORTANT:
    # Never apply a default heating temperature to an
    # unidentified or unmapped food.
    return 0, 0, False


# =========================================================
# PARSE GROQ RESPONSE
# =========================================================

def parse_llm_output(text):

    if not text:
        return None

    # Remove model reasoning tags if present.
    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL
    )

    text = re.sub(
        r"<think>.*",
        "",
        text,
        flags=re.DOTALL
    )

    food = re.search(
        r"Food:\s*(.+?)\s*---",
        text,
        flags=re.IGNORECASE
    )

    temp = re.search(
        r"Temperature:\s*([\d.]+)",
        text,
        flags=re.IGNORECASE
    )

    tm = re.search(
        r"Time:\s*([\d.]+)",
        text,
        flags=re.IGNORECASE
    )

    if not (food and temp and tm):
        return None

    try:

        food_name = food.group(1).strip()
        temperature = float(temp.group(1))
        minutes = float(tm.group(1))

        return food_name, temperature, minutes

    except (ValueError, TypeError):
        return None


# =========================================================
# PREDICTION ROUTE
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():

    # Check API key.
    if not GROQ_API_KEY:

        set_unknown_state("GROQ_API_KEY is missing")

        return jsonify(
            error="GROQ_API_KEY is not configured",
            food="Uncertain",
            temp=0,
            time=0,
            matched=False
        ), 500

    # Receive image bytes from ESP32-CAM.
    img_bytes = request.get_data()

    if not img_bytes:

        set_unknown_state("No image received")

        return jsonify(
            error="No image received",
            food="Uncertain",
            temp=0,
            time=0,
            matched=False
        ), 400

    # Store latest image and timestamp.
    LAST["image"] = img_bytes

    LAST["at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    # Reset previous cooking values immediately.
    # This prevents an old detection from remaining active
    # while a new image is being processed.
    set_unknown_state("Image received; recognition in progress")

    # Convert image to Base64.
    b64 = base64.b64encode(img_bytes).decode("utf-8")

    # Groq request payload.
    payload = {
        "model": MODEL,
        "temperature": 0.2,
        "max_tokens": 512,
        "reasoning_effort": "none",

        "messages": [
            {
                "role": "user",

                "content": [
                    {
                        "type": "text",
                        "text": PROMPT
                    },

                    {
                        "type": "image_url",

                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}"
                        }
                    }
                ]
            }
        ]
    }

    # -----------------------------------------------------
    # SEND IMAGE TO GROQ
    # -----------------------------------------------------

    try:

        r = requests.post(
            GROQ_URL,

            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },

            json=payload,
            timeout=60
        )

    except requests.RequestException as e:

        msg = f"Groq connection failed: {e}"

        print("!!!", msg)

        set_unknown_state(msg)

        return jsonify(
            error=msg,
            food="Uncertain",
            temp=0,
            time=0,
            matched=False
        ), 502

    print(">>> Groq status:", r.status_code)
    print(">>> Groq body:", r.text[:800])

    # -----------------------------------------------------
    # CHECK GROQ RESPONSE STATUS
    # -----------------------------------------------------

    if r.status_code != 200:

        msg = f"Groq returned {r.status_code}: {r.text[:300]}"

        print("!!!", msg)

        set_unknown_state(msg)

        return jsonify(
            error=msg,
            food="Uncertain",
            temp=0,
            time=0,
            matched=False
        ), 502

    # -----------------------------------------------------
    # EXTRACT MODEL OUTPUT
    # -----------------------------------------------------

    try:

        raw = r.json()["choices"][0]["message"]["content"]

        if not isinstance(raw, str):
            raise ValueError("Unexpected Groq response format")

        raw = raw.strip()

    except (KeyError, IndexError, TypeError, ValueError) as e:

        msg = f"Groq response parsing failed: {e}"

        print("!!!", msg)

        set_unknown_state(msg)

        return jsonify(
            error=msg,
            food="Uncertain",
            temp=0,
            time=0,
            matched=False
        ), 502

    print("LLM raw:", raw)

    LAST["raw"] = raw

    # -----------------------------------------------------
    # PARSE FOOD, TEMPERATURE AND TIME
    # -----------------------------------------------------

    parsed = parse_llm_output(raw)

    # Invalid model response.
    if not parsed:

        set_unknown_state(raw)

        return jsonify(
            food="Uncertain",
            temp=0,
            time=0,
            matched=False,
            raw=raw
        )

    food, temp, minutes = parsed

    food_lower = food.lower().strip()

    # -----------------------------------------------------
    # UNCERTAIN / UNKNOWN FOOD
    # -----------------------------------------------------

    if food_lower in ["uncertain", "unknown", "unidentified", "none"]:

        set_unknown_state(raw)

        return jsonify(
            food="Uncertain",
            temp=0,
            time=0,
            matched=False,
            raw=raw
        )

    # -----------------------------------------------------
    # APPLY SAFE RANGE
    # -----------------------------------------------------

    temp, minutes, matched = clamp_to_safe(
        food,
        temp,
        minutes
    )

    temp = round(temp)
    minutes = round(minutes)

    # -----------------------------------------------------
    # UNMAPPED FOOD
    # -----------------------------------------------------

    if not matched:

        set_unknown_state(raw)

        return jsonify(
            food=food,
            temp=0,
            time=0,
            matched=False,
            raw=raw
        )

    # -----------------------------------------------------
    # SAVE SUCCESSFUL DETECTION
    # -----------------------------------------------------

    LAST.update(
        food=food,
        temp=temp,
        time=minutes,
        matched=True,
        raw=raw
    )

    print("=== FINAL RESULT ===")
    print("Food:", food)
    print("Temperature:", temp)
    print("Time:", minutes)
    print("Matched:", matched)

    return jsonify(
        food=food,
        temp=temp,
        time=minutes,
        matched=matched,
        raw=raw
    )


# =========================================================
# STATE ROUTE
# =========================================================

@app.route("/state")
def state():

    return jsonify(
        food=LAST["food"],
        temp=LAST["temp"],
        time=LAST["time"],
        matched=LAST["matched"],
        at=LAST["at"],
        raw=LAST["raw"],
        has_image=LAST["image"] is not None
    )


# =========================================================
# LAST IMAGE ROUTE
# =========================================================

@app.route("/last")
def last_image():

    if LAST["image"] is None:

        return "No image received yet", 404

    return Response(
        LAST["image"],
        mimetype="image/jpeg"
    )


# =========================================================
# FAVICON
# =========================================================

@app.route("/favicon.ico")
def favicon():

    return "", 204


# =========================================================
# DASHBOARD
# =========================================================

@app.route("/")
def dashboard():

    html = """
<!doctype html>

<html>

<head>

<meta charset="utf-8">

<meta name="viewport"
content="width=device-width,initial-scale=1">

<title>ThermaSense</title>

<style>

body {
    font-family: system-ui, Arial, sans-serif;
    background: #111;
    color: #eee;
    max-width: 640px;
    margin: 0 auto;
    padding: 20px;
}

h1 {
    font-size: 20px;
    letter-spacing: .5px;
}

.grid {
    display: flex;
    gap: 10px;
    margin-top: 14px;
    flex-wrap: wrap;
}

.card {
    flex: 1;
    min-width: 120px;
    background: #1c1c1c;
    border: 1px solid #333;
    border-radius: 12px;
    padding: 12px 14px;
}

.card span {
    display: block;
    font-size: 12px;
    color: #888;
    margin-bottom: 4px;
}

.card b {
    font-size: 20px;
}

.muted {
    color: #888;
    font-size: 13px;
}

img {
    max-width: 100%;
    border-radius: 12px;
    border: 1px solid #333;
}

pre {
    background: #000;
    padding: 10px;
    border-radius: 8px;
    overflow: auto;
    font-size: 12px;
}

summary {
    cursor: pointer;
    color: #9cf;
}

</style>

</head>

<body>

<h1>ThermaSense - Last Detection</h1>

<div id="empty" class="muted">
    Waiting for an image from ESP32-CAM...
</div>

<div id="content" style="display:none">

    <img id="shot" alt="Last food image"/>

    <div class="grid">

        <div class="card">
            <span>Food</span>
            <b id="food">-</b>
        </div>

        <div class="card">
            <span>Temperature</span>
            <b id="temp">-</b>
        </div>

        <div class="card">
            <span>Time</span>
            <b id="time">-</b>
        </div>

    </div>

    <p class="muted" id="matched"></p>

    <p class="muted" id="at"></p>

    <details>
        <summary>LLM raw output</summary>
        <pre id="raw"></pre>
    </details>

</div>


<script>

async function tick() {

    try {

        const r = await fetch('/state', {
            cache: 'no-store'
        });

        const s = await r.json();

        if (!s.has_image) {

            document.getElementById('empty').style.display = 'block';

            document.getElementById('content').style.display = 'none';

            return;
        }

        document.getElementById('empty').style.display = 'none';

        document.getElementById('content').style.display = 'block';

        document.getElementById('shot').src =
            '/last?t=' + Date.now();

        document.getElementById('food').textContent =
            s.food || 'Uncertain';

        document.getElementById('temp').textContent =
            (s.temp || 0) + ' °C';

        document.getElementById('time').textContent =
            (s.time || 0) + ' min';

        if (s.matched) {

            document.getElementById('matched').textContent =
                'Food category matched with SAFE_RANGE table';

        } else {

            document.getElementById('matched').textContent =
                'No safe-range mapping: heating should remain disabled';

        }

        document.getElementById('at').textContent =
            'Image received: ' + (s.at || '-');

        document.getElementById('raw').textContent =
            s.raw || '';

    } catch (e) {

        // Retry on next update.

    }

}

tick();

setInterval(tick, 4000);

</script>

</body>

</html>
"""

    return Response(html, mimetype="text/html")


# =========================================================
# HEALTH CHECK
# =========================================================

@app.route("/health")
def health():

    return "ok"


# =========================================================
# LOCAL DEVELOPMENT
# =========================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port
    )
