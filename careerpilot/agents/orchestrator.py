"""
Orchestrator Agent
Coordinates the full pipeline: profile → career → [academic + skillgap] → roadmap
Runs academic and skill_gap agents in parallel for speed.
"""
import asyncio
import json
from typing import Optional
from .llm import call_llm_text
from . import profile_agent, career_agent, academic_agent, skill_gap_agent, roadmap_agent
from memory.store import MemoryStore
from agents.rag_agent import get_career_context, get_course_context, get_user_doc_context

careers = career_agent.run(profile, rag_context=career_ctx + "\n" + user_ctx)

# After skill gaps are identified:


INTENT_PROMPT = """You are a routing agent for a career guidance system.

Classify the user's intent into ONE of these categories:
- FULL_PIPELINE: first-time query, career exploration, "what should I do", general advice
- UPDATE_GOAL: user is changing their career goal or interests
- ROADMAP_ONLY: user already has careers, just wants the plan updated
- QUESTION: a specific question not requiring full pipeline re-run
- PROFILE_UPDATE: updating skills, year, location only

Return ONLY one of these exact strings. Nothing else."""


async def classify_intent(message: str, has_profile: bool) -> str:
    if not has_profile:
        return "FULL_PIPELINE"
    intent = await call_llm_text(INTENT_PROMPT, f'User message: "{message}"')
    # Normalize — strip any accidental whitespace or quotes
    intent = intent.strip().strip('"').upper()
    valid = {"FULL_PIPELINE", "UPDATE_GOAL", "ROADMAP_ONLY", "QUESTION", "PROFILE_UPDATE"}
    return intent if intent in valid else "FULL_PIPELINE"


async def answer_question(message: str, profile: dict, store: MemoryStore, user_id: str) -> dict:
    """Handle a specific Q without running the full pipeline."""
    context = {
        "profile": profile,
        "roadmap": store.get_roadmap(user_id),
    }
    answer = await call_llm_text(
        system_prompt=(
            "You are CareerPilot, a helpful career advisor for university students. "
            "Answer the student's question using their profile and roadmap context. "
            "Be concise, specific, and encouraging. 3-5 sentences max."
        ),
        user_content=(
            f"Student context:\n{json.dumps(context, indent=2)}\n\n"
            f"Student question: {message}"
        ),
    )
    return {"type": "answer", "message": answer}


async def run_pipeline(
    user_message: str,
    user_id: str,
    session_id: str,
    existing_profile: Optional[dict],
    store: MemoryStore,
) -> dict:
    """
    Main pipeline. Returns a dict with all agent outputs.
    """

    # Step 1: Classify intent
    intent = await classify_intent(user_message, has_profile=bool(existing_profile))
    print(f"[Orchestrator] Intent: {intent}")

    # Handle simple Q&A without full pipeline
    if intent == "QUESTION" and existing_profile:
        return await answer_question(user_message, existing_profile, store, user_id)

    # Step 2: Profile Agent — always runs to keep profile fresh
    print("[Orchestrator] Running Profile Agent...")
    profile = await profile_agent.run(user_message, existing_profile)
    profile["user_id"] = user_id
    store.save_profile(user_id, profile)
    print(f"[Orchestrator] Profile updated: {list(profile.keys())}")

    career_ctx = get_career_context(profile)
    user_ctx   = get_user_doc_context(profile.get("user_id",""), _profile_to_query(profile))

    # Step 3: Career Agent
    print("[Orchestrator] Running Career Agent...")
    career_out = await career_agent.run(profile)
    print(f"[Orchestrator] Top career: {career_out.careers[0].name} ({career_out.careers[0].confidence:.2f})")

    # Step 4: Academic + Skill Gap in PARALLEL
    print("[Orchestrator] Running Academic + Skill Gap agents in parallel...")
    academic_out, skillgap_out = await asyncio.gather(
        academic_agent.run(profile, career_out),
        skill_gap_agent.run(profile, career_out),
    )
    print(f"[Orchestrator] Missing skills: {skillgap_out.missing_skills[:3]}...")

    course_ctx = get_course_context(skill_gaps.missing_skills)
    academics  = academic_agent.run(profile, skill_gaps, rag_context=course_ctx)
    
    # Step 5: Roadmap Agent
    print("[Orchestrator] Running Roadmap Agent...")
    roadmap_out = await roadmap_agent.run(profile, career_out, academic_out, skillgap_out)
    print(f"[Orchestrator] Roadmap: {roadmap_out.total_semesters} semesters generated")

    # Persist roadmap
    store.save_roadmap(user_id, roadmap_out.model_dump())

    # Step 6: Build summary message
    top = career_out.careers[0]
    summary = (
        f"Based on your profile, your top career match is **{top.name}** "
        f"(confidence: {int(top.confidence * 100)}%). "
        f"I've built you a {roadmap_out.total_semesters}-semester roadmap. "
        f"Your first priority skill to learn: **{skillgap_out.priority_order[0] if skillgap_out.priority_order else 'N/A'}**."
    )

    return {
        "type": "full_analysis",
        "message": summary,
        "profile": profile,
        "careers": [c.model_dump() for c in career_out.careers],
        "skill_gap": skillgap_out.model_dump(),
        "academic": academic_out.model_dump(),
        "roadmap": roadmap_out.model_dump(),
    }