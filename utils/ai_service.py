import os
import json
import base64
import sys
import time
import io
import ast
import requests
from PIL import Image
from dotenv import load_dotenv

try:
    load_dotenv()
except Exception:
    pass

def resize_and_encode_image(image_path, max_size=768):
    """
    Resize image to max_size dimension and encode to JPEG base64.
    Drastically reduces token usage and prevents Groq 429 rate limit issues.
    """
    with Image.open(image_path) as img:
        img = img.convert('RGB')
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

def parse_json_safely(json_str):
    """
    Parse a JSON string safely, handling single quotes, markdown backticks, and loose formatting.
    """
    json_str = json_str.strip()
    if json_str.startswith("```json"):
        json_str = json_str[7:]
    elif json_str.startswith("```"):
        json_str = json_str[3:]
    if json_str.endswith("```"):
        json_str = json_str[:-3]
    json_str = json_str.strip()

    try:
        data = json.loads(json_str)
        if isinstance(data, list):
            return data
    except Exception:
        pass

    try:
        data = ast.literal_eval(json_str)
        if isinstance(data, list):
            return data
    except Exception:
        pass

    return None

def analyze_food_image(image_path):
    """
    Analyze a food image using Groq Qwen 3.6 27B Multimodal Vision API.
    Returns a list of detected food items with nutritional info, or None on failure.
    No dummy/mock fallbacks are used.
    """
    if not os.path.exists(image_path):
        print(f"ERROR: Image file not found at {image_path}", file=sys.stderr)
        return None

    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        print("ERROR: GROQ_API_KEY is not set in .env", file=sys.stderr)
        return None

    try:
        # Resize image for fast processing and optimal token consumption
        encoded_image = resize_and_encode_image(image_path, max_size=768)

        prompt = (
            "Analyze this food image and identify all food items. "
            "For each item, estimate the quantity and provide nutritional information (calories, protein, fat, carbs). "
            "Output ONLY a valid JSON array starting with '[' and ending with ']'. "
            'Example format: [{"name": "food name", "quantity": "1 portion", "calories": 250, "protein": 12, "fat": 8, "carbs": 30}]\n'
            "If NO food is detected, return an empty array []."
        )

        payload = {
            "model": "qwen/qwen3.6-27b",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a concise nutrition API. Write a 1-sentence thought inside <think>, then immediately output </think> followed by the JSON array starting with '['."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                    ]
                }
            ],
            "temperature": 0.0,
            "max_tokens": 1500
        }

        # Attempt up to 3 times for transient network/rate limit issues
        for attempt in range(3):
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=30
            )

            if response.status_code == 429:
                wait_secs = 5 * (attempt + 1)
                print(f"Groq API rate limit reached. Waiting {wait_secs}s before retry...", file=sys.stderr)
                time.sleep(wait_secs)
                continue
            else:
                break

        if response.status_code != 200:
            print(f"Groq API Error ({response.status_code}): {response.text}", file=sys.stderr)
            return None

        raw_content = response.json()['choices'][0]['message']['content']

        # Strip reasoning thoughts if present
        if "</think>" in raw_content:
            clean = raw_content.split("</think>")[-1].strip()
        else:
            clean = raw_content.strip()

        # Extract JSON array between [ and ]
        start = clean.find('[')
        end = clean.rfind(']')
        if start != -1 and end != -1:
            json_str = clean[start:end+1]
            data = parse_json_safely(json_str)
            if data is not None:
                return data

        print(f"Failed to parse JSON array from Groq output: {raw_content}", file=sys.stderr)
        return None

    except Exception as e:
        print(f"Exception during Groq AI analysis: {e}", file=sys.stderr)
        return None
