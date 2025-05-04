from flask import Flask, request, jsonify
import google.generativeai as genai
from dotenv import load_dotenv
import os
import base64
import json
from io import BytesIO
from PIL import Image
import logging
import re

app = Flask(__name__)

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.DEBUG)

# ─── Gemini setup ─────────────────────────────────────────────────────────────
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

# ──────────────────────────────────────────────────────────────────────────────
@app.route("/generate-caption", methods=["POST"])
def generate_caption():
    """
    Accepts a multipart/form‑data request:
      • field "image"  – required – image file (png/jpg/…)
      • field "prompt" – optional – extra text to guide the caption

    Returns
      { "caption": "…", "hashtags": ["#one", "#two", …] }
    """
    try:
        # ---- 1. Validate + load the image ------------------------------------
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400

        image_file = request.files["image"]
        image = Image.open(image_file)

        # ---- 2. Read the (optional) prompt -----------------------------------
        user_prompt = (request.form.get("prompt") or "").strip()

        # ---- 3. Convert image ➜ base64 for Gemini ---------------------------
        buf = BytesIO()
        image.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        # ---- 4. Build the prompt for Gemini ---------------------------------
        base_prompt = (
            "Generate a short caption (max 20 words) describing this image for a "
            "social‑media post."
        )
        if user_prompt:
            base_prompt += f' Take this user idea into account: "{user_prompt}".'

        base_prompt += (
            " Also provide 5 relevant hashtags.\n"
            'Return ONLY valid JSON in the form: {"caption": "", "hashtags": []}'
        )

        # ---- 5. Call Gemini --------------------------------------------------
        response = model.generate_content(
            [
                {"mime_type": "image/png", "data": img_b64},
                {"text": base_prompt},
            ]
        )

        logging.debug("Raw Gemini response: %s", response.text)

        # ---- 6. Strip any markdown fencing Gemini may return ----------------
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:].rstrip("```").strip()
        elif text.startswith("```"):
            text = text[3:].rstrip("```").strip()

        # ---- 7. Parse JSON (with a fallback regex extraction) ---------------
        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if not m:
                return jsonify(
                    {"error": "Failed to parse Gemini response", "raw_response": text}
                ), 500
            result = json.loads(m.group())

        # ---- 8. Validate + normalise hashtags -------------------------------
        if (
            not isinstance(result, dict)
            or "caption" not in result
            or "hashtags" not in result
            or not isinstance(result["hashtags"], (list, tuple))
        ):
            return jsonify({"error": "Invalid JSON from Gemini"}), 500

        result["hashtags"] = [
            tag if tag.startswith("#") else f"#{tag}" for tag in result["hashtags"]
        ]

        return jsonify(result)

    except Exception as e:
        logging.exception("Error processing request")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Railway sets $PORT; fall back to 5000 locally
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
