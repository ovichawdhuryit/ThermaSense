"""
ThermaSense - Bodda Side Server V2
-----------------------------------

ESP32-CAM sends JPEG image.
Flask receives image.
Groq Vision identifies food and generates:
    - Food name
    - Recommended temperature
    - Cooking time

The LLM output is the primary recommendation.

The safety reference table is ONLY for comparison.
It NEVER overrides LLM output.

Routes:
    /          -> Live dashboard
    /state     -> Latest detection JSON
    /last      -> Latest JPEG image
    /predict   -> ESP32-CAM image upload
    /health    -> Server health

Render:
    Build: pip install -r requirements.txt
    Start: gunicorn app:app --workers 1 --threads 4 --timeout 90

Environment:
    GROQ_API_KEY = your_api_key
"""

import os
import io
import json
import time
import base64
import logging
import threading

import requests

from PIL import Image

from flask import (
    Flask,
    request,
    jsonify,
    Response
)

from werkzeug.exceptions import HTTPException


# ============================================================
# 1. CONFIGURATION
# ============================================================

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

MAX_IMAGE_SIZE = 8 * 1024 * 1024  # 8 MB

REQUEST_TIMEOUT = 60

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = MAX_IMAGE_SIZE

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("ThermaSense")


# ============================================================
# 2. LLM PROMPT
# ============================================================

PROMPT = """
You are the AI food recognition and cooking recommendation
engine for ThermaSense.

Analyze the food visible in the provided image.

Your tasks:

1. Identify the food as specifically as possible.
2. Recommend a suitable cooking temperature in Celsius.
3. Recommend a suitable cooking duration in minutes.

IMPORTANT RULES:

- Recognize any recognizable food.
- Do not restrict recognition to predefined food categories.
- Return your own food-specific recommendations.
- Do not use an external predefined safety table.
- Do not return confidence scores.
- Do not explain your reasoning.
- Do not include markdown.
- Do not use <think> tags.
- Temperature must be a numeric Celsius value or null.
- Time must be a numeric minute value or null.
- If the food cannot be identified, use "unknown".
- If a recommendation cannot be responsibly estimated,
  return null for that field.
- Do not claim that image-based estimates represent
  measured internal food temperature.
- Temperature means a recommended cooking target.

Return ONLY valid JSON in EXACTLY this structure:

{
    "food": "food name",
    "temperature": 74,
    "time": 25
}
"""


# ============================================================
# 3. SAFETY REFERENCE TABLE
# ============================================================
#
# IMPORTANT:
# This table is ONLY for dashboard comparison.
# It NEVER changes the original LLM temperature or time.
#
# Reference values are illustrative food-safety targets.
# Actual requirements depend on preparation and food type.
# ============================================================

SAFE_RANGE = {

    "poultry": 74,

    "chicken": 74,

    "turkey": 74,

    "ground beef": 71,

    "ground pork": 71,

    "ground lamb": 71,

    "beef": 63,

    "pork": 63,

    "lamb": 63,

    "steak": 63,

    "fish": 63,

    "egg": 71,

    "leftovers": 74,

}


def get_reference_temperature(food):

    """
    Find a reference temperature for comparison only.

    Never use this value to overwrite LLM output.
    """

    if not food:
        return None

    food_lower = food.lower()

    # More specific terms should be checked first.
    for key in sorted(
        SAFE_RANGE.keys(),
        key=len,
        reverse=True
    ):

        if key in food_lower:

            return SAFE_RANGE[key]

    return None


# ============================================================
# 4. GLOBAL STATE
# ============================================================

STATE_LOCK = threading.RLock()

REQUEST_COUNTER = 0

LAST = {

    "image": None,

    "image_id": 0,

    "food": None,

    "temp": None,

    "time": None,

    "reference_temp": None,

    "matched": None,

    "status": "waiting",

    "error": None,

    "raw": None,

    "at": None

}


# ============================================================
# 5. STATE MANAGEMENT
# ============================================================

def update_state(request_id, **kwargs):

    """
    Prevent an older request from overwriting a newer request.
    Returns True if the update was applied, False if skipped.
    """

    with STATE_LOCK:

        if request_id != REQUEST_COUNTER:

            return False

        LAST.update(kwargs)

        return True


def fail_prediction(request_id, message):

    """
    Write an error state only if this request is still current.
    If update_state returns False (a newer request has taken
    over), we still return the error response to the caller
    but do NOT overwrite the newer state.
    """

    update_state(

        request_id,

        status="error",

        error=message,

        food=None,

        temp=None,

        time=None,

        reference_temp=None,

        matched=None,

        raw=None

    )

    return jsonify(

        status="error",

        error=message

    ), 502


# ============================================================
# 6. JPEG VALIDATION
# ============================================================

def validate_jpeg(image_bytes):

    if not image_bytes:

        raise ValueError("No image received.")

    if len(image_bytes) > MAX_IMAGE_SIZE:

        raise ValueError("Image exceeds 8 MB limit.")

    if not image_bytes.startswith(b"\xff\xd8\xff"):

        raise ValueError("Invalid JPEG image.")

    try:

        with Image.open(io.BytesIO(image_bytes)) as img:

            if img.format != "JPEG":

                raise ValueError(
                    "Only JPEG images are accepted."
                )

            img.verify()

    except Exception as e:

        raise ValueError(
            f"Invalid or corrupted JPEG: {e}"
        )


# ============================================================
# 7. GROQ VISION REQUEST
# ============================================================

def identify_food(image_bytes):

    encoded_image = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    payload = {

        "model": MODEL,

        "temperature": 0.2,

        "max_completion_tokens": 256,

        "response_format": {
            "type": "json_object"
        },

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

                            "url": (
                                "data:image/jpeg;base64,"
                                + encoded_image
                            )

                        }

                    }

                ]

            }

        ]

    }

    response = requests.post(

        GROQ_URL,

        headers={

            "Authorization": (
                f"Bearer {GROQ_API_KEY}"
            ),

            "Content-Type": "application/json"

        },

        json=payload,

        timeout=REQUEST_TIMEOUT

    )

    logger.info(
        "Groq status: %s",
        response.status_code
    )

    if response.status_code != 200:

        logger.error(
            "Groq response: %s",
            response.text[:500]
        )

        raise RuntimeError(
            f"Groq API returned HTTP {response.status_code}"
        )

    data = response.json()

    raw = data["choices"][0]["message"]["content"]

    if not isinstance(raw, str):

        raise ValueError(
            "Invalid Groq response content."
        )

    return raw.strip()


# ============================================================
# 8. LLM JSON PARSER
# ============================================================

def parse_llm_output(raw):

    """
    Parse the LLM JSON.

    No confidence score.
    No safety-table override.
    No temperature/time clamping.

    Original LLM recommendations are preserved.
    """

    result = json.loads(raw)

    if not isinstance(result, dict):

        raise ValueError(
            "LLM did not return a JSON object."
        )

    food = result.get("food")

    temperature = result.get("temperature")

    cooking_time = result.get("time")

    # Validate food name.

    if not isinstance(food, str):

        raise ValueError(
            "Invalid food name."
        )

    food = food.strip()[:100]

    if not food:

        food = "unknown"

    # Validate numeric recommendations.

    def validate_number(value, field):

        if value is None:

            return None

        if isinstance(value, bool):

            raise ValueError(
                f"Invalid {field}."
            )

        try:

            number = float(value)

        except (ValueError, TypeError):

            raise ValueError(
                f"Invalid {field}."
            )

        if not (
            number > 0
            and number < float("inf")
        ):

            raise ValueError(
                f"Invalid {field}."
            )

        return round(number, 1)

    temperature = validate_number(
        temperature,
        "temperature"
    )

    cooking_time = validate_number(
        cooking_time,
        "time"
    )

    return {

        "food": food,

        "temp": temperature,

        "time": cooking_time

    }


# ============================================================
# 9. PREDICT ENDPOINT
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    global REQUEST_COUNTER

    if not GROQ_API_KEY:

        return jsonify(

            status="error",

            error="GROQ_API_KEY is not configured."

        ), 500

    image_bytes = request.get_data(cache=False)

    # Validate image.

    try:

        validate_jpeg(image_bytes)

    except ValueError as e:

        return jsonify(

            status="error",

            error=str(e)

        ), 400

    # Register new prediction.

    with STATE_LOCK:

        REQUEST_COUNTER += 1

        request_id = REQUEST_COUNTER

        LAST.update({

            "image": image_bytes,

            "image_id": request_id,

            "food": None,

            "temp": None,

            "time": None,

            "reference_temp": None,

            "matched": None,

            "status": "processing",

            "error": None,

            "raw": None,

            "at": time.strftime(
                "%Y-%m-%d %H:%M:%S"
            )

        })

    # --------------------------------------------------------
    # All work happens before update_state(success).
    # update_state(success) is the very last mutation.
    # This ensures that if anything throws, the except
    # blocks correctly write an error state rather than
    # overwriting a state that was never set to success.
    # --------------------------------------------------------

    try:

        logger.info(
            "Processing image ID: %s",
            request_id
        )

        # Step 1: Groq Vision identification.

        raw = identify_food(image_bytes)

        logger.info(
            "LLM raw output: %s",
            raw
        )

        # Step 2: Parse LLM JSON.

        llm = parse_llm_output(raw)

        food = llm["food"]

        temperature = llm["temp"]

        cooking_time = llm["time"]

        # Step 3: Reference table comparison.

        reference_temp = get_reference_temperature(food)

        # matched means a reference entry exists.

        matched = reference_temp is not None

        # Step 4: Build response dict before touching state.
        # Nothing below this point can raise an exception
        # from business logic, so update_state(success) is
        # safe to call immediately after.

        response_data = {
            "status":        "success",
            "food":          food,
            "temp":          temperature,
            "time":          cooking_time,
            "reference_temp": reference_temp,
            "matched":       matched,
            "raw":           raw,
        }

        # Step 5: Commit success state — the very last
        # mutation before returning to the caller.

        update_state(

            request_id,

            food=food,

            temp=temperature,

            time=cooking_time,

            reference_temp=reference_temp,

            matched=matched,

            status="success",

            error=None,

            raw=raw

        )

        logger.info(
            "Detection complete: %s | %s C | %s min",
            food,
            temperature,
            cooking_time
        )

        # Step 6: Send response to ESP32.

        return jsonify(response_data)

    except requests.Timeout:

        logger.exception("Groq request timeout.")

        return fail_prediction(

            request_id,

            "Groq request timed out."

        )

    except requests.RequestException:

        logger.exception("Groq connection error.")

        return fail_prediction(

            request_id,

            "Groq connection failed."

        )

    except (ValueError, KeyError, IndexError) as e:

        logger.exception(
            "LLM response parsing failed: %s",
            repr(e)
        )

        return fail_prediction(

            request_id,

            "Invalid LLM response."

        )

    except Exception as e:

        logger.exception(
            "Unexpected prediction error: %s",
            repr(e)
        )

        return fail_prediction(

            request_id,

            "Unexpected server error."

        )


# ============================================================
# 10. STATE ENDPOINT
# ============================================================

@app.route("/state", methods=["GET"])
def state():

    with STATE_LOCK:

        snapshot = {

            "food": LAST["food"],

            "temp": LAST["temp"],

            "time": LAST["time"],

            "reference_temp": LAST["reference_temp"],

            "matched": LAST["matched"],

            "status": LAST["status"],

            "error": LAST["error"],

            "at": LAST["at"],

            "image_id": LAST["image_id"],

            "raw": LAST["raw"],

            "has_image": LAST["image"] is not None

        }

    return jsonify(snapshot)


# ============================================================
# 11. LAST IMAGE ENDPOINT
# ============================================================

@app.route("/last", methods=["GET"])
def last_image():

    with STATE_LOCK:

        image = LAST["image"]

    if image is None:

        return (
            "No image received yet.",
            404
        )

    return Response(

        image,

        mimetype="image/jpeg",

        headers={

            "Cache-Control": (
                "no-store, no-cache, must-revalidate"
            )

        }

    )


# ============================================================
# 12. DASHBOARD
# ============================================================

@app.route("/", methods=["GET"])
def dashboard():

    html = """
<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

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
    font-size: 22px;
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
    padding: 14px;
}

.card span {
    display: block;
    font-size: 12px;
    color: #999;
    margin-bottom: 5px;
}

.card b {
    font-size: 20px;
}

img {
    max-width: 100%;
    border-radius: 12px;
    border: 1px solid #333;
    margin-top: 12px;
}

.muted {
    color: #999;
    font-size: 13px;
}

.status {
    padding: 10px;
    border-radius: 8px;
    background: #222;
    margin-top: 12px;
}

.success {
    color: #7cda91;
}

.warning {
    color: #ffcc66;
}

.error {
    color: #ff7777;
}

.reference {
    background: #202b38;
    border: 1px solid #34475c;
    border-radius: 10px;
    padding: 12px;
    margin-top: 14px;
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
    Waiting for ESP32-CAM image...
</div>

<div id="content" style="display:none">

    <img id="shot" alt="Latest food image">

    <div class="status" id="status">
        Waiting...
    </div>

    <div class="grid">

        <div class="card">
            <span>LLM Food</span>
            <b id="food">-</b>
        </div>

        <div class="card">
            <span>LLM Temperature</span>
            <b id="temp">-</b>
        </div>

        <div class="card">
            <span>LLM Cooking Time</span>
            <b id="time">-</b>
        </div>

    </div>

    <div class="reference">

        <b>Safety Reference (Comparison Only)</b>

        <p id="reference">
            Not available
        </p>

    </div>

    <p class="muted" id="at"></p>

    <details>

        <summary>LLM Raw Output</summary>

        <pre id="raw"></pre>

    </details>

</div>


<script>

let previousImageId = -1;

async function tick() {

    try {

        const response = await fetch(
            "/state",
            { cache: "no-store" }
        );

        if (!response.ok) return;

        const s = await response.json();

        const empty = document.getElementById("empty");

        const content = document.getElementById("content");

        if (!s.has_image) {

            empty.style.display = "block";

            content.style.display = "none";

            return;
        }

        empty.style.display = "none";

        content.style.display = "block";

        // Update image only when a new image arrives.

        if (s.image_id !== previousImageId) {

            document.getElementById("shot").src =
                "/last?id=" + encodeURIComponent(s.image_id);

            previousImageId = s.image_id;

        }

        document.getElementById("food").textContent =
            s.food || "-";

        document.getElementById("temp").textContent =
            s.temp !== null ? s.temp + " \u00b0C" : "Not available";

        document.getElementById("time").textContent =
            s.time !== null ? s.time + " min" : "Not available";

        document.getElementById("at").textContent =
            "Image received: " + (s.at || "-");

        document.getElementById("raw").textContent =
            s.raw || "";

        const reference =
            document.getElementById("reference");

        if (s.reference_temp !== null) {

            reference.textContent =
                "Reference temperature: "
                + s.reference_temp
                + " \u00b0C. This is separate from the LLM recommendation.";

        } else {

            reference.textContent =
                "No matching reference entry.";

        }

        const status =
            document.getElementById("status");

        status.className = "status";

        if (s.status === "processing") {

            status.textContent =
                "Analyzing food image...";

            status.classList.add("warning");

        }

        else if (s.status === "success") {

            status.textContent =
                "LLM food detection completed.";

            status.classList.add("success");

        }

        else if (s.status === "error") {

            status.textContent =
                "Prediction failed: "
                + (s.error || "Unknown error");

            status.classList.add("error");

        }

    }

    catch (error) {

        console.error(
            "Dashboard update failed:",
            error
        );

    }

}

tick();

setInterval(tick, 4000);

</script>

</body>

</html>
"""

    return Response(
        html,
        mimetype="text/html"
    )


# ============================================================
# 13. HEALTH CHECK
# ============================================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({

        "status": "ok",

        "service": "ThermaSense",

        "model": MODEL

    })


# ============================================================
# 14. ERROR HANDLING
# ============================================================

@app.errorhandler(HTTPException)
def handle_http_error(error):

    return jsonify({

        "status": "error",

        "error": error.description

    }), error.code


@app.errorhandler(Exception)
def handle_unexpected_error(error):

    logger.exception(
        "Unhandled server error."
    )

    return jsonify({

        "status": "error",

        "error": "Internal server error."

    }), 500


# ============================================================
# 15. MAIN
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 5000)
    )

    app.run(

        host="0.0.0.0",

        port=port,

        debug=False

    )
