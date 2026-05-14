"""
Shared LLM caller — all agents go through this.
Model: gpt-4o-mini (fast + cheap, good JSON output)
"""
import json
import os
from openai import AsyncOpenAI
from typing import Type, TypeVar
from pydantic import BaseModel

_api_key = os.getenv("OPENAI_API_KEY", "placeholder")
client = AsyncOpenAI(api_key=_api_key)

T = TypeVar("T", bound=BaseModel)

MODEL = "gpt-4o-mini"


async def call_llm(
    system_prompt: str,
    user_content: str,
    schema: Type[T],
    temperature: float = 0.3,
    max_retries: int = 2,
) -> T:
    """
    Call GPT-4o-mini and parse response into a Pydantic schema.
    Retries once on validation failure with error feedback.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_content},
    ]

    for attempt in range(max_retries):
        response = await client.chat.completions.create(
            model=MODEL,
            temperature=temperature,
            response_format={"type": "json_object"},  # forces JSON mode
            messages=messages,
        )
        raw = response.choices[0].message.content.strip()

        try:
            data = json.loads(raw)
            return schema(**data)
        except Exception as e:
            if attempt == max_retries - 1:
                raise ValueError(
                    f"[{schema.__name__}] Validation failed after {max_retries} attempts.\n"
                    f"Error: {e}\nRaw output: {raw[:500]}"
                )
            # Give the model its error and ask it to fix
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": (
                    f"Your output failed validation with this error: {e}\n"
                    f"Fix the JSON and return only the corrected object."
                )
            })


async def call_llm_text(
    system_prompt: str,
    user_content: str,
    temperature: float = 0.5,
) -> str:
    """Plain text response — used by orchestrator for routing decisions."""
    response = await client.chat.completions.create(
        model=MODEL,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ],
    )
    return response.choices[0].message.content.strip()