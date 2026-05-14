"""
Roadmap Agent
Synthesizes all agent outputs into a semester-by-semester action plan.
This is the most complex agent — uses gpt-4o-mini with detailed prompt.
"""
import json
from .llm import call_llm
from models.schemas import RoadmapOutput, CareerOutput, AcademicOutput, SkillGapOutput

SYSTEM_PROMPT = """You are a semester-by-semester academic roadmap planner.

Given a student's profile, career target, academic plan, and skill gaps,
create a realistic semester action plan from their current year to graduation.

Rules:
- Return ONLY valid JSON.
- Each semester must have: courses, skills_to_acquire, milestones, internship_target.
- Follow the skill gap priority_order — foundational skills go in early semesters.
- Do NOT front-load: max 2 new technical skills per semester.
- internship_target: null for semester 1, realistic entry-level for semester 2+.
- label: a 1-3 word theme for the semester (e.g. "Foundation", "Domain entry", 
  "Applied work", "Specialization", "Industry ready").
- milestones: concrete, achievable things (e.g. "Complete Coursera ML cert", 
  "Build portfolio project", "Apply to 5 internships").
- Total semesters = 8 minus (academic_year - 1) * 2 
  (e.g. year 2 → 6 semesters remaining).
- courses: use the electives from the academic plan + standard courses for the major.

Output schema:
{
  "total_semesters": 6,
  "roadmap": [
    {
      "semester": 1,
      "label": "Foundation",
      "courses": ["Course A", "Course B"],
      "skills_to_acquire": ["skill1", "skill2"],
      "milestones": ["milestone 1"],
      "internship_target": null
    }
  ]
}"""


async def run(
    profile: dict,
    career_output: CareerOutput,
    academic_output: AcademicOutput,
    skill_gap_output: SkillGapOutput,
) -> RoadmapOutput:
    academic_year = profile.get("academic_year", 2)
    remaining_semesters = max(2, 8 - (academic_year - 1) * 2)

    user_content = f"""Student profile:
{json.dumps(profile, indent=2)}

Target career: {career_output.careers[0].name}

Academic plan:
- Major: {academic_output.recommended_major}
- Electives: {json.dumps(academic_output.core_electives)}
- Online courses: {json.dumps([c.model_dump() for c in academic_output.online_courses])}

Skill gaps (in priority order):
{json.dumps(skill_gap_output.priority_order)}

Remaining semesters until graduation: {remaining_semesters}

Generate a {remaining_semesters}-semester roadmap as JSON."""

    return await call_llm(SYSTEM_PROMPT, user_content, RoadmapOutput, temperature=0.4)