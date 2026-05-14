"""
Career Agent
Maps student profile → ranked career paths.
Uses gpt-4o-mini with strong system prompt for reliable JSON.
"""
import json
from .llm import call_llm
from models.schemas import CareerOutput

SYSTEM_PROMPT = """You are a career intelligence agent specializing in student 
career path prediction.

Given a student profile, return the 3 best-fit career paths ranked by confidence.

Rules:
- Return ONLY valid JSON. No markdown, no prose, no code fences.
- confidence: 0.0 to 1.0. Be realistic — do not give >0.92 unless the profile 
  is extremely clear and detailed.
- reasons: 2-3 specific reasons tied to the ACTUAL profile data provided.
  Never hallucinate interests or skills not present in the profile.
- required_skills: what this career actually needs (from domain knowledge).
- salary_range_usd: realistic range for a mid-level professional globally.
- Consider the student's location for relevant market context.
- If the profile is sparse, still return 3 careers but lower confidence scores.

Output schema (return exactly this structure):
{
  "careers": [
    {
      "name": "Career Title",
      "confidence": 0.85,
      "reasons": ["reason 1", "reason 2"],
      "required_skills": ["skill1", "skill2", "skill3"],
      "salary_range_usd": "$60k-$110k"
    }
  ]
}"""


async def run(profile: dict) -> CareerOutput:
    user_content = f"""Student profile:
{json.dumps(profile, indent=2)}

Return the 3 best career paths for this student as JSON."""

    return await call_llm(SYSTEM_PROMPT, user_content, CareerOutput, temperature=0.3)