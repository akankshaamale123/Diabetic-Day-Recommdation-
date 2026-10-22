
import requests
import json
import base64
import mimetypes
from pathlib import Path
from config import Model, Model_URL
from Prompts import Qwen_VL_prompts

# 1. Function for text input
def generate_text(prompt, stream=False):
    payload = {
        "model": Model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.1,
        "max_tokens": 512,
        "stream": stream
    }

    return send_request(payload, stream)


# 2. Function for image input
def generate_image(image_path, prompt="Describe this image in simple words.",
                   stream=False):

    path = Path(image_path)

    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    mime_type = mimetypes.guess_type(str(path))[0] or "image/png"

    with open(path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")

    image_data_url = f"data:{mime_type};base64,{image_base64}"

    payload = {
        "model": Model,
        "messages": [
            {
            "role": "system",
            "content": Qwen_VL_prompts
        },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url
                        }
                    }
                ]
            }
        ],
        "temperature": 0.1,
        "max_tokens": 2000,
        "stream": stream
    }

    return send_request(payload, stream)


# Shared function for sending requests
def send_request(payload, stream=True):
    try:
        with requests.post(
            Model_URL,
            json=payload,
            stream=stream,
            timeout=(15, 300)
        ) as response:

            print("HTTP status:", response.status_code)

            if not response.ok:
                print("Server error:", response.text)
                response.raise_for_status()

            # Streaming response
            if stream:
                for line in response.iter_lines(decode_unicode=True):
                    if not line:
                        continue

                    if line.startswith("data: "):
                        data = line[6:]

                        if data.strip() == "[DONE]":
                            break

                        chunk = json.loads(data)
                        choices = chunk.get("choices", [])

                        if choices:
                            token = choices[0].get(
                                "delta", {}
                            ).get("content", "")

                            if isinstance(token, str) and token:
                                print(token, end="", flush=True)

                print("\n\nStreaming completed.")

            # Non-streaming response
            else:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                return content

    except requests.RequestException as e:
        print(f"\nRequest failed: {e}")
        if getattr(e, "response", None) is not None:
            print(e.response.text)

    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"\nResponse parsing failed: {e}")


# # -----------------------------
# # Example 1: Text input
# # -----------------------------
# generate_text("What is computer vision?")


# # -----------------------------
# # Example 2: Image input
# # -----------------------------
# image_path = "F:\Akanksha\diabetic and Cancer day\Diabetic Diet plan\Fish-Tacos-1337495.jpg"

# generate_image(
#     image_path,
#     prompt="Describe all visible food items in this image."
# )
