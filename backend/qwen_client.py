import os  # Reads QWEN_* settings from environment variables.
from pathlib import Path  # Locates the backend directory and its .env file.

import httpx  # Sends the asynchronous HTTP request to the model provider.
from dotenv import load_dotenv  # Loads local settings from backend/.env.


# Load defaults from backend/.env; existing environment variables take priority.
# Never commit .env because it contains the API key.
load_dotenv(Path(__file__).resolve().parent / ".env")


async def ask_qwen(messages):
    """Send chat messages to Qwen and return only its final answer text."""

    # Remove an optional trailing slash so the endpoint is formed correctly.
    base = os.environ["QWEN_BASE_URL"].strip().rstrip("/")

    # Read the API key from the environment; never put it in source code.
    api_key = os.environ["QWEN_API_KEY"].strip()

    payload = {
        # Select the Qwen model configured in backend/.env.
        "model": os.environ["QWEN_MODEL"].strip(),

        # These are the instructions and detector evidence from qwen_agent.py.
        "messages": messages,

        # Cap generation. The final report should be much shorter than this.
        "max_tokens": 4096,

        # Request one complete response instead of a stream of pieces.
        "stream": False,

        # Qwen3.5 thinks by default. For a short report, ask it to answer
        # directly so thinking does not consume the budget before final text.
        "chat_template_kwargs": {"enable_thinking": False},
    }

    # The async client lets FastAPI await the provider without blocking here.
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            base + "/chat/completions",

            # Authenticate without putting the key in the URL.
            headers={"Authorization": f"Bearer {api_key}"},

            # httpx converts this dictionary into a JSON request body.
            json=payload,
        )

    # If the provider rejected the request, show its error without the key.
    if response.is_error:
        detail = response.text.replace(api_key, "[REDACTED]")
        raise RuntimeError(
            f"HTTP {response.status_code}: {detail[:1500]}"
        )

    # Chat-completion responses contain choices; read the first answer.
    choice = response.json()["choices"][0]
    content = choice["message"].get("content")

    # An empty answer is not a usable final report.
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(
            "No final text returned. "
            f"Finish reason: {choice.get('finish_reason')}"
        )

    # Return the report text, leaving detector results unchanged.
    return content.strip()