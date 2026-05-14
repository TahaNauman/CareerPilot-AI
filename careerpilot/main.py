from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import uuid
import os

from agents.orchestrator import run_pipeline
from memory.store import MemoryStore

app = FastAPI(title="CareerPilot AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

store = MemoryStore()


class ChatRequest(BaseModel):
    user_id: Optional[str] = None
    message: str
    session_id: Optional[str] = None


class OnboardingRequest(BaseModel):
    interests: list[str]
    skills: list[str]
    academic_year: int
    goals: list[str]
    location: str = "Pakistan"
    gpa: Optional[float] = None


@app.post("/api/onboard")
async def onboard(req: OnboardingRequest):
    user_id = str(uuid.uuid4())
    profile = {
        "user_id": user_id,
        "interests": req.interests,
        "skills": req.skills,
        "academic_year": req.academic_year,
        "goals": req.goals,
        "location": req.location,
        "gpa": req.gpa,
    }
    store.save_profile(user_id, profile)
    return {"user_id": user_id, "profile": profile}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    user_id = req.user_id or str(uuid.uuid4())
    session_id = req.session_id or str(uuid.uuid4())

    profile = store.get_profile(user_id)

    result = await run_pipeline(
        user_message=req.message,
        user_id=user_id,
        session_id=session_id,
        existing_profile=profile,
        store=store,
    )

    return {
        "session_id": session_id,
        "user_id": user_id,
        **result,
    }


@app.get("/api/profile/{user_id}")
async def get_profile(user_id: str):
    profile = store.get_profile(user_id)
    if not profile:
        raise HTTPException(404, "Profile not found")
    return profile


@app.get("/api/roadmap/{user_id}")
async def get_roadmap(user_id: str):
    roadmap = store.get_roadmap(user_id)
    if not roadmap:
        raise HTTPException(404, "No roadmap yet")
    return roadmap


# Serve frontend
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

    @app.get("/")
    async def serve_frontend():
        return FileResponse("static/index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)