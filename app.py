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

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize Gemini model
model = genai.GenerativeModel("gemini-1.5-flash")

@app.route("/generate-caption", methods=["POST"])
def generate_caption():
    try:
        # Check if an image file is provided
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400

        image_file = request.files["image"]
        image = Image.open(image_file)

        # Convert image to base64 for Gemini API
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Prepare prompt for caption and hashtags
        prompt = (
            "Generate a short caption (max 20 words) describing this image, the caption will be used for social media posting, make it relative to that. "
            "Also provide 5 relevant hashtags. "
            "Return ONLY valid JSON: {\"caption\": \"\", \"hashtags\": []}"
        )

        # Call Gemini API
        response = model.generate_content(
            [
                {"mime_type": "image/png", "data": img_base64},
                {"text": prompt}
            ]
        )

        # Log the raw response for debugging
        logging.debug(f"Raw Gemini API response: {response.text}")

        # Parse response as JSON
        response_text = response.text.strip()
        # Strip markdown if present
        if response_text.startswith("```json"):
            response_text = response_text[7:].rstrip("```")
        elif response_text.startswith("```"):
            response_text = response_text[3:].rstrip("```")

        try:
            result = json.loads(response_text)
            if not isinstance(result, dict) or "caption" not in result or "hashtags" not in result:
                return jsonify({"error": "Invalid response format from Gemini API"}), 500
            # Add # to each hashtag
            result["hashtags"] = [f"#{tag}" if not tag.startswith("#") else tag for tag in result["hashtags"]]
        except json.JSONDecodeError as e:
            # Try manual JSON extraction
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                    if not isinstance(result, dict) or "caption" not in result or "hashtags" not in result:
                        return jsonify({"error": "Invalid response format from Gemini API"}), 500
                    # Add # to each hashtag
                    result["hashtags"] = [f"#{tag}" if not tag.startswith("#") else tag for tag in result["hashtags"]]
                except json.JSONDecodeError as e:
                    return jsonify({"error": f"Failed to parse Gemini API response: {str(e)}", "raw_response": response_text}), 500
            else:
                return jsonify({"error": f"Failed to parse Gemini API response: {str(e)}", "raw_response": response_text}), 500

        return jsonify(result)

    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Use PORT environment variable for Railway, default to 5000 for local
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)