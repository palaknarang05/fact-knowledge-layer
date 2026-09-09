"""
Run from fact-layer/ root:
    python3 groq_test_v2.py

Prints exactly what model name it's about to call and where it read it
from, before making the call, so there's zero ambiguity about what's
actually being sent to Groq.
"""
import asyncio
import os
import sys
import traceback

sys.path.insert(0, ".")

print("--- ENV CHECK ---")
print("Shell env FKL_EXTRACTION_MODEL (before dotenv load):", repr(os.environ.get("FKL_EXTRACTION_MODEL")))

from backend import config  # noqa: E402  (import after path insert / env check on purpose)

print("config.py file:", config.__file__)
print("config.EXTRACTION_MODEL resolved to:", repr(config.EXTRACTION_MODEL))
print("config.GROQ_API_KEY present:", bool(config.GROQ_API_KEY))
print("------------------")

from openai import AsyncOpenAI  # noqa: E402


async def main():
    if not config.GROQ_API_KEY:
        print("GROQ_API_KEY missing — stop here, fix .env first.")
        return

    client = AsyncOpenAI(api_key=config.GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")

    print(f"\nCalling Groq with model={config.EXTRACTION_MODEL!r} ...")
    try:
        completion = await client.chat.completions.create(
            model=config.EXTRACTION_MODEL,
            messages=[
                {"role": "system", "content": 'Reply with a short JSON object: {"ok": true}'},
                {"role": "user", "content": "test"},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        print("SUCCESS:", completion.choices[0].message.content)
    except Exception:
        print("FAILED — full traceback:")
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())