from pydantic import BaseModel, Field
from typing import Optional


# --- Career Agent ---
class CareerItem(BaseModel):
    name: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasons: list[str]
    required_skills: list[str]
    salary_range_usd: str


class CareerOutput(BaseModel):
    careers: list[CareerItem]


# --- Skill Gap Agent ---
class SkillGapOutput(BaseModel):
    target_role: str
    current_skills: list[str]
    missing_skills: list[str]
    priority_order: list[str]
    priority_rationale: dict[str, str]


# --- Academic Agent ---
class OnlineCourse(BaseModel):
    title: str
    platform: str
    duration_weeks: int


class AcademicOutput(BaseModel):
    recommended_major: str
    alternative_majors: list[str]
    core_electives: list[str]
    online_courses: list[OnlineCourse]


# --- Roadmap Agent ---
class RoadmapSemester(BaseModel):
    semester: int
    label: str
    courses: list[str]
    skills_to_acquire: list[str]
    milestones: list[str]
    internship_target: Optional[str] = None


class RoadmapOutput(BaseModel):
    total_semesters: int
    roadmap: list[RoadmapSemester]


# --- Profile Agent ---
class ProfileOutput(BaseModel):
    interests: list[str]
    skills: list[str]
    goals: list[str]
    academic_year: Optional[int] = None
    location: Optional[str] = None
    gpa: Optional[float] = None