import os
from pathlib import Path

import httpx
from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parent / ".env")


async def ask_qwen(messages):
    base = os.environ["QWEN_BASE_URL"].strip().rstrip("/")
    api_key = os.environ["QWEN_API_KEY"].strip()

    payload = {
        "model": os.environ["QWEN_MODEL"].strip(),
        "messages": messages,
        "max_tokens": 1400,
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            base + "/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
        )

    if response.is_error:
        detail = response.text.replace(api_key, "[REDACTED]")
        raise RuntimeError(
            f"HTTP {response.status_code}: {detail[:1500]}"
        )

    choice = response.json()["choices"][0]
    content = choice["message"].get("content")

    if not content:
        raise RuntimeError(
            "No final text returned. "
            f"Finish reason: {choice.get('finish_reason')}"
        )

    return content.strip()