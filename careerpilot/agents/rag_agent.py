"""
RAG Agent — retrieves grounding context from ChromaDB
and injects it into other agents' prompts.

Public API:
    get_career_context(profile)          → str  for career_agent / skill_gap_agent
    get_course_context(skills)           → str  for academic_agent
    get_user_doc_context(user_id, query) → str  for any agent needing resume context
    get_general_context(query)           → str  for open-ended questions
    ingest_user_resume(user_id, path)    → int  (chunk count)
    rag_status()                         → dict (collection sizes)
"""

from rag.chroma_store import query, collection_stats
from rag.ingest import ingest_resume, chunk_text


# ── Public retrieval functions ────────────────────────────────

def get_career_context(profile: dict, k: int = 4) -> str:
    """
    Retrieve real O*NET career data relevant to this user's profile.
    Call before career_agent and skill_gap_agent.
    """
    q = _profile_to_query(profile)
    hits = query("onet_careers", q, k=k)
    if not hits:
        return ""
    lines = ["=== REAL CAREER DATA (O*NET) ==="]
    for h in hits:
        lines.append(h["text"])
        lines.append(f"(relevance score: {h['score']})")
        lines.append("")
    return "\n".join(lines)


def get_course_context(skills_needed: list[str], level: str = "", k: int = 6) -> str:
    """
    Retrieve real courses for the given skill gaps.
    Call before academic_agent.
    """
    q = "Courses teaching: " + ", ".join(skills_needed)
    where = {"level": level} if level else None
    hits = query("course_catalog", q, k=k, where=where)
    if not hits:
        return ""
    lines = ["=== REAL COURSES (Coursera / edX) ==="]
    for h in hits:
        m = h["metadata"]
        lines.append(
            f"• {m.get('title', 'Unknown')} [{m.get('platform', '')}]"
            f"\n  Skills: {m.get('skills', '')}"
            f"\n  Level: {m.get('level', '')}"
            f"\n  URL: {m.get('url', '')}"
        )
    return "\n".join(lines)


def get_user_doc_context(user_id: str, query_text: str, k: int = 4) -> str:
    """
    Retrieve chunks from this user's uploaded resume / transcript.
    Call when you need grounding from what they've told you about themselves.
    """
    hits = query(
        "user_documents",
        query_text,
        k=k,
        where={"user_id": user_id}
    )
    if not hits:
        return ""
    lines = ["=== FROM USER'S UPLOADED DOCUMENTS ==="]
    for h in hits:
        lines.append(h["text"])
        lines.append("")
    return "\n".join(lines)


def get_general_context(query_text: str, k: int = 4) -> str:
    """
    Retrieve from the general docs/ folder (anything you drop in docs/general/).
    Good for custom handbooks, company info, extra reference material.
    """
    hits = query("general_docs", query_text, k=k)
    if not hits:
        return ""
    lines = ["=== REFERENCE DOCUMENTS ==="]
    for h in hits:
        m = h["metadata"]
        lines.append(f"[{m.get('filename', 'doc')}]\n{h['text']}")
        lines.append("")
    return "\n".join(lines)


def get_full_context(profile: dict, skills_needed: list[str] = None) -> dict:
    """
    Convenience: fetch all relevant context in one call.
    Returns a dict you can unpack into agent prompts.
    """
    user_id = profile.get("user_id", "")
    query_text = _profile_to_query(profile)
    skills_needed = skills_needed or []

    return {
        "career_context": get_career_context(profile),
        "course_context": get_course_context(skills_needed) if skills_needed else "",
        "user_doc_context": get_user_doc_context(user_id, query_text) if user_id else "",
        "general_context": get_general_context(query_text),
    }


# ── Resume ingestion (called at upload time) ──────────────────

def ingest_user_resume(user_id: str, file_path: str) -> int:
    """
    Call this from your /api/upload endpoint when a user uploads a resume.
    Returns number of chunks stored.
    """
    ingest_resume(user_id=user_id, file_path=file_path)
    from rag.chroma_store import get_collection
    col = get_collection("user_documents")
    results = col.get(where={"user_id": user_id})
    return len(results["ids"])


# ── Status ────────────────────────────────────────────────────

def rag_status() -> dict:
    """Returns how many items are in each collection. Useful for /api/status."""
    return collection_stats()


# ── Internal helpers ──────────────────────────────────────────

def _profile_to_query(profile: dict) -> str:
    parts = []
    if profile.get("interests"):
        parts.append("interests: " + ", ".join(profile["interests"]))
    if profile.get("skills"):
        parts.append("skills: " + ", ".join(profile["skills"]))
    if profile.get("goal"):
        parts.append("goal: " + profile["goal"])
    if profile.get("major"):
        parts.append("studying: " + profile["major"])
    if profile.get("year"):
        parts.append("year: " + str(profile["year"]))
    return " | ".join(parts) if parts else "career planning student"