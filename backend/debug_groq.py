"""
Run this directly from the fact-layer project root:

    python3 debug_groq.py

It calls the Groq extraction client exactly like extractor.py does, but with
NO tenacity retry wrapping it - so the real exception and full traceback
print immediately instead of being hidden inside a RetryError.
"""
import asyncio
import sys
import traceback

sys.path.insert(0, ".")

from backend import config
from openai import AsyncOpenAI


async def main():
    if not config.GROQ_API_KEY:
        print("GROQ_API_KEY is empty/missing in .env — that's your bug.")
        return

    client = AsyncOpenAI(api_key=config.GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")

    print(f"Using model: {config.EXTRACTION_MODEL}")
    try:
        completion = await client.chat.completions.create(
            model=config.EXTRACTION_MODEL,
            messages=[
                {"role": "system", "content": "Reply with a short JSON object: {\"ok\": true}"},
                {"role": "user", "content": "test"},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        print("SUCCESS. Response:")
        print(completion.choices[0].message.content)
    except Exception:
        print("REAL ERROR (full traceback below):")
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())