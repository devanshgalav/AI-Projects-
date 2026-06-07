from typing import Any
import base64
from pathlib import Path


def image_file_to_data_url(file_path: str) -> str:
    """Convert a local image file to a data URL."""
    path = Path(file_path)
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        media_type = "image/jpeg"
    elif path.suffix.lower() == ".png":
        media_type = "image/png"
    elif path.suffix.lower() == ".gif":
        media_type = "image/gif"
    elif path.suffix.lower() == ".webp":
        media_type = "image/webp"
    else:
        media_type = "image/jpeg"
    
    with open(file_path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")
    return f"data:{media_type};base64,{data}"


def get_uploaded_file_path(file_value: Any) -> str | None:
    """Normalize common Gradio file values into a path string."""
    # TODO 1: if file_value is a string, return it.
    # TODO 2: if file_value is a dict, return file_value["path"] or file_value["name"].
    # TODO 3: otherwise try file_value.path or file_value.name.

    if not file_value:
        return None

    if isinstance(file_value, str):
        return file_value

    if isinstance(file_value, dict):
        return file_value.get("path") or file_value.get("name")

    return getattr(file_value, "path", None) or getattr(file_value, "name", None)

def build_user_content(message: dict[str, Any]) -> str | list[dict[str, Any]]:
    """Convert Gradio multimodal input into OpenRouter user content."""
    text = (message.get("text") or "").strip()
    files = message.get("files") or []

    content: list[dict[str, Any]] = []
    if text:
        content.append({"type": "text", "text": text})

    for file_value in files:
        file_path = get_uploaded_file_path(file_value)
        if file_path:
            # TODO 4: append an image_url content block.
            # Shape:
            # {"type": "image_url", "image_url": {"url": <get url from file path using image_file_to_data_url>}}

            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_file_to_data_url(file_path)
                    },
                }
            )

    if not content:
        return "Please send text or upload an image."
    if len(content) == 1 and content[0]["type"] == "text":
        return content[0]["text"]
    if not text:
        content.insert(0, {"type": "text", "text": "Please analyze this image."})
    return content


def build_multimodal_messages(
    history: list[dict[str, Any]],
    current_message: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build messages for a vision-capable OpenRouter chat model."""
    messages: list[dict[str, Any]] = []

    for item in history:
        role = item.get("role")
        content = item.get("content")
        if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
            # TODO 5: append prior text messages.

            messages.append(
                {
                    "role": role,
                    "content": content.strip(),
                }
            )

    # TODO 9: append the latest user message using build_user_content(current_message).

    messages.append(
        {
            "role": "user",
            "content": build_user_content(current_message),
        }
    )

    return messages