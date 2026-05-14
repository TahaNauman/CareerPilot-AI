"""
Profile Agent
Extracts structured profile facts from user messages.
Uses gpt-4o-mini (cheap — pure extraction, no reasoning).
"""
import json
from .llm import call_llm
from models.schema import ProfileOutput

SYSTEM_PROMPT = f"""
{rag_context if rag_context else ''} 
You are a student profile extraction agent.

Your job: extract structured information from a student's message and merge it 
with their existing profile.

Rules:
- Return ONLY valid JSON matching the schema.
- Only include fields explicitly mentioned or strongly implied — never invent data.
- Merge new info with existing profile — do not drop existing fields unless the 
  student explicitly changes them.
- interests: academic/professional interests (e.g. "machine learning", "biology")
- skills: technical or soft skills they already have
- goals: career or life goals mentioned
- If a field is not mentioned at all, carry over the existing value or use null.

Output schema:
{
  "interests": ["string"],
  "skills": ["string"],
  "goals": ["string"],
  "academic_year": int or null,
  "location": "string or null",
  "gpa": float or null
}"""


async def run(user_message: str, existing_profile: dict | None = None) -> dict:
    existing = existing_profile or {}

    user_content = f"""Existing profile:
{json.dumps(existing, indent=2)}

New message from student:
\"{user_message}\"

Extract any new or updated information and return the merged profile JSON."""

    output = await call_llm(SYSTEM_PROMPT, user_content, ProfileOutput)
    merged = existing.copy()
    merged.update({k: v for k, v in output.model_dump().items() if v is not None})
    return merged