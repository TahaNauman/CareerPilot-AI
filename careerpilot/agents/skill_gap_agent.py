"""
Skill Gap Agent
Compares current skills vs required skills for top career.
Simple set-diff + prioritization — uses gpt-4o-mini.
"""
import json
from .llm import call_llm
from models.schema import SkillGapOutput, CareerOutput

SYSTEM_PROMPT = f"""
{rag_context if rag_context else ''}

You are a skill gap analysis agent.

Given a student's current skills and their target career's required skills,
identify what is missing and create a prioritized learning plan.

Rules:
- Return ONLY valid JSON.
- missing_skills: skills in required_skills NOT present in current_skills.
  Do a fuzzy match — "Python" and "python basics" count as the same thing.
- priority_order: order missing_skills from most foundational to most advanced.
  Foundational = prerequisite for other skills on the list.
- priority_rationale: one sentence per skill explaining why it's at that priority.
- If no skills are missing, return missing_skills and priority_order as [].

Output schema:
{
  "target_role": "string",
  "current_skills": ["string"],
  "missing_skills": ["string"],
  "priority_order": ["string"],
  "priority_rationale": {
    "skill_name": "one sentence reason"
  }
}"""


async def run(profile: dict, career_output: CareerOutput) -> SkillGapOutput:
    top_career = career_output.careers[0]

    user_content = f"""Student current skills:
{json.dumps(profile.get('skills', []))}

Target career: {top_career.name}
Required skills for target career:
{json.dumps(top_career.required_skills)}

Identify skill gaps and prioritize them."""

    return await call_llm(SYSTEM_PROMPT, user_content, SkillGapOutput, temperature=0.2)