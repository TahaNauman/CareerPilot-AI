"""
Academic Agent
Recommends major, electives, and online courses based on career target.
Uses gpt-4o-mini.
"""
import json
from .llm import call_llm
from models.schema import AcademicOutput, CareerOutput

SYSTEM_PROMPT = f"""
{rag_context if rag_context else ''}
You are an academic planning agent for university students.

Given a student's profile and their target career, recommend:
- The best-fit university major
- 2 alternative majors
- 4-5 core electives to take
- 3 online courses (real, currently available courses)

Rules:
- Return ONLY valid JSON. No prose.
- Recommend real academic majors that actually exist.
- Online courses must be real and available on Coursera, edX, or freeCodeCamp.
- duration_weeks must be realistic (e.g. a Coursera specialization = 16-20 weeks).
- Tailor recommendations to the student's academic year — don't recommend 
  advanced electives for a 1st-year student.
- Consider location for regional relevance if provided.

Output schema:
{
  "recommended_major": "string",
  "alternative_majors": ["string", "string"],
  "core_electives": ["string", "string", "string", "string"],
  "online_courses": [
    {
      "title": "string",
      "platform": "string",
      "duration_weeks": 12
    }
  ]
}"""


async def run(profile: dict, career_output: CareerOutput) -> AcademicOutput:
    top_career = career_output.careers[0]

    user_content = f"""Student profile:
{json.dumps(profile, indent=2)}

Target career: {top_career.name}
Required skills: {json.dumps(top_career.required_skills)}

Recommend major, electives, and online courses as JSON."""

    return await call_llm(SYSTEM_PROMPT, user_content, AcademicOutput, temperature=0.3)